"""The ordered rule table. This is the decision.

Every rule is evaluated and every fired rule is recorded, so a clerk sees all of a
letter's defects rather than only the first. The **decisive** rule is the first fired
route-bearing rule in table order; informational `I*` rules are excluded from that choice,
because "the IBAN checksum was valid" can never be the reason a letter went anywhere.

**Table order.** All `R*` first, so reject dominance is structural rather than conditional.
Then the `H*` blocks, ordered by what a clerk most needs to see first: whether we could
read the letter at all, then whose money it is, then how much, then how well-evidenced the
reading was. Then `A00`. Then the informational rules.

**Candidate-property rules versus amount-value rules.** H04, H13 and H14 describe the
figure the letter *pointed at*, so they read the considered candidate and fire even when
that figure never parsed. H12, H15 and R05 describe the *resolved* amount, so they carry
explicit non-null guards and simply do not fire when there is no number to compare. This
split is why one prose figure can report both "stated approximately" and "could not be
read": two true statements about one figure, neither of which should silence the other.

**Why every reject rule also requires a successful reading.** `reject` is a statement about
the letter. When our own reader fails, the fields it did not produce are evidence about us,
not about the court, so R01-R05 are guarded on `extraction_status == OK` and a malformed
response escalates under H01 instead. Without the guard, an outage would emit a stream of
refused court orders and corrupt the audit trail with invalidity findings we invented.

The same guard sits on H02, H05, H06 and H17 for the same reason. Each makes a claim about
the *letter* — what kind of document it is, how the debtor is named, whether an IBAN is
given, whether two readings of it cohere — and after a reading failure every one of them
would be a sentence we made up about a letter nobody read. H01 stays unguarded because it
is the one rule whose subject is the failure itself. H09 also stays unguarded: its subject
is our own confidence, not the court's letter, and "we had none" is true after a failure.

**Why R01 also requires a coherent document type.** `document_type` and `is_seizure_order`
can disagree, and a letter classified `restriction_order` while flagged not-a-seizure-order
is a contradiction, not evidence of invalidity. Left alone it would fire R01 while the
resolver simultaneously applied the restriction doctrine to the same extraction — rejecting
a letter and computing its operative amount in one pass. R01 therefore requires the absence
of a seizure-order classification, and H17 escalates the contradiction instead.

This module contains no numeric literals. Every comparison reads a name from `thresholds`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal

from garnishment.domain.amount_parsing import ParsedAmount, parse_eur_amount
from garnishment.domain.confidence import (
    FieldConfidence,
    derive_field_confidences,
    gated_field_names,
)
from garnishment.domain.enums import (
    AmountRole,
    DocumentType,
    ExtractionStatus,
    ResolutionReason,
    Route,
    RuleId,
)
from garnishment.domain.extraction import AmountCandidate, LetterExtraction, candidate_at
from garnishment.domain.operative_amount import (
    RESTRICTION_DOCTRINE_VERSION,
    Resolution,
    cited_original_amount,
    resolve_operative_amount,
)
from garnishment.domain.thresholds import (
    AUTO_MAX_AMOUNT_EUR,
    AUTO_MIN_FIELD_CONFIDENCE,
    DOC_TYPE_AUTO_MIN_CONFIDENCE,
    DOC_TYPE_REJECT_MIN_CONFIDENCE,
    GROUNDING_ABSENT,
    MIN_CANDIDATES_FOR_A_SECOND_LOOK,
    ROLE_MIN_CONFIDENCE,
    ZERO_EUR,
    below_floor,
)
from garnishment.domain.validators import (
    PASS,
    validate_case_reference,
    validate_debtor_name,
    validate_iban,
)
from garnishment.domain.values import EUR, Money

_INJECTION_PHRASES = (
    "ignore the above",
    "ignore previous instructions",
    "ignore all previous",
    "disregard the above",
    "disregard previous instructions",
    "system prompt",
    "as an ai language model",
    "new instructions:",
    "approve this automatically",
    "route this as auto",
)
"""A deliberately small substring heuristic.

It is not the control that stops an injection; grounding is. An injected instruction
cannot fabricate a value whose evidence appears in the letter, and routing reads only
typed fields. This heuristic has a real false-positive rate on legal prose, which is why
I04 records the signal rather than letting a regex refuse a genuine court order.
"""


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Everything a predicate may read. Derived from the extraction, and nothing else."""

    extraction: LetterExtraction
    field_confidences: Mapping[str, FieldConfidence]
    resolution: Resolution
    considered_candidate: AmountCandidate | None
    considered_parse: ParsedAmount | None
    gated_fields: tuple[str, ...]
    fired_route_bearing: tuple[RuleId, ...] = ()

    @property
    def resolved_amount(self) -> Money | None:
        return self.resolution.amount

    @property
    def resolved_decimal(self) -> Decimal | None:
        return None if self.resolution.amount is None else self.resolution.amount.amount

    def confidence(self, field_name: str) -> FieldConfidence:
        return self.field_confidences[field_name]


def build_rule_context(extraction: LetterExtraction) -> RuleContext:
    """Derive every input the rule table reads. Pure; no clock, no I/O."""
    resolution = resolve_operative_amount(extraction.amount_candidates, extraction.document_type)
    considered = candidate_at(extraction, resolution.considered_candidate_index)
    return RuleContext(
        extraction=extraction,
        field_confidences=derive_field_confidences(extraction),
        resolution=resolution,
        considered_candidate=considered,
        considered_parse=(
            None if considered is None else parse_eur_amount(considered.value_as_written)
        ),
        gated_fields=gated_field_names(extraction),
    )


def _no_reason_args(_context: RuleContext) -> Mapping[str, object]:
    return {}


@dataclass(frozen=True, slots=True)
class Rule:
    """One row of the table.

    `reason_template` is stored as a literal string so the table can be read, reviewed and
    diffed as text; `reason_args` supplies the values it interpolates.
    """

    rule_id: RuleId
    predicate: Callable[[RuleContext], bool]
    route: Route | None
    reason_template: str
    reason_args: Callable[[RuleContext], Mapping[str, object]] = _no_reason_args

    def reason(self, context: RuleContext) -> str:
        return self.reason_template.format(**self.reason_args(context))


# --- helpers shared by the predicates ---------------------------------------------------


_SEIZURE_ORDER_DOCUMENT_TYPES = frozenset(
    {DocumentType.ATTACHMENT_ORDER, DocumentType.RESTRICTION_ORDER}
)
"""Document types that name a specific kind of seizure order.

A letter classified as one of these while `is_seizure_order` is False is not a letter we
have read; it is two readings that contradict each other (H17). R01 requires the absence
of such a classification for exactly that reason: a contradiction must never be able to
satisfy a reject predicate.
"""


def _reading_succeeded(ctx: RuleContext) -> bool:
    return ctx.extraction.extraction_status is ExtractionStatus.OK


def _document_type_contradicts_seizure_flag(ctx: RuleContext) -> bool:
    # Guarded like H02/H05/H06: two fields we never read cannot contradict each other,
    # and after a reading failure H01 already says the only true thing there is to say.
    return (
        _reading_succeeded(ctx)
        and ctx.extraction.document_type in _SEIZURE_ORDER_DOCUMENT_TYPES
        and not ctx.extraction.is_seizure_order
    )


def _display(value: str | None) -> str:
    """Render a value for a reason line, without printing a bare `None`."""
    return repr(value) if value else "(absent)"


def _debtor_raw(ctx: RuleContext) -> str | None:
    value = ctx.extraction.debtor_name.value
    return None if value is None else value.raw


def _case_raw(ctx: RuleContext) -> str | None:
    value = ctx.extraction.case_reference.value
    return None if value is None else value.raw


def _iban_raw(ctx: RuleContext) -> str | None:
    value = ctx.extraction.account_iban.value
    return None if value is None else value.raw


def _candidate_label(candidate: AmountCandidate) -> str:
    return candidate.raw_text or candidate.value_as_written


def _considered_label(ctx: RuleContext) -> str | None:
    """How the figure the letter pointed at was written, for a reason line."""
    candidate = ctx.considered_candidate
    return None if candidate is None else _candidate_label(candidate)


def _first_ungrounded_field(ctx: RuleContext) -> str | None:
    for name in ctx.gated_fields:
        confidence = ctx.confidence(name)
        if confidence.value_present and confidence.grounding == GROUNDING_ABSENT:
            return name
    return None


def _first_low_confidence_field(ctx: RuleContext) -> str | None:
    # `below_floor`, never a raw `<`: a NaN self-report compares False against every
    # bound, so `final < floor` would let a degenerate confidence past the gate that
    # exists to catch it.
    for name in ctx.gated_fields:
        if below_floor(ctx.confidence(name).final, AUTO_MIN_FIELD_CONFIDENCE):
            return name
    return None


def _first_value_only_grounded_field(ctx: RuleContext) -> str | None:
    for name, confidence in ctx.field_confidences.items():
        if confidence.grounded_on_value_only:
            return name
    return None


def _lowest_offending_confidence(ctx: RuleContext) -> float:
    """The confidence of the field H09 names, not the minimum across all of them."""
    offending = _first_low_confidence_field(ctx)
    return AUTO_MIN_FIELD_CONFIDENCE if offending is None else ctx.confidence(offending).final


def _lowest_gated_confidence(ctx: RuleContext) -> float:
    return min(ctx.confidence(name).final for name in ctx.gated_fields)


def _weak_role_note(ctx: RuleContext) -> str:
    """Name the low role label when that is why the figures could not be told apart."""
    scores = [
        c.role_confidence
        for c in ctx.extraction.amount_candidates
        if c.role in {AmountRole.OPERATIVE, AmountRole.RESTRICTED}
        and below_floor(c.role_confidence, ROLE_MIN_CONFIDENCE)
    ]
    if not scores:
        return ""
    return (
        f"; the role label for one figure scored {max(scores):.2f}, "
        f"below the {ROLE_MIN_CONFIDENCE:.2f} floor"
    )


# --- the table --------------------------------------------------------------------------

RULES: tuple[Rule, ...] = (
    # ===== Reject: evidence that the letter itself is invalid ===========================
    Rule(
        RuleId.R01_NOT_SEIZURE_ORDER,
        lambda ctx: (
            _reading_succeeded(ctx)
            and not ctx.extraction.is_seizure_order
            and ctx.extraction.document_type not in _SEIZURE_ORDER_DOCUMENT_TYPES
            and ctx.extraction.is_seizure_order_confidence >= DOC_TYPE_REJECT_MIN_CONFIDENCE
        ),
        Route.REJECT,
        "Document is not a seizure order (model confidence {c:.2f} ≥ {floor:.2f}).",
        lambda ctx: {
            "c": ctx.extraction.is_seizure_order_confidence,
            "floor": DOC_TYPE_REJECT_MIN_CONFIDENCE,
        },
    ),
    Rule(
        RuleId.R02_CASE_REFERENCE_MISSING,
        lambda ctx: _reading_succeeded(ctx) and ctx.extraction.case_reference.value is None,
        Route.REJECT,
        "No court case reference found.",
    ),
    Rule(
        RuleId.R03_CASE_REFERENCE_MALFORMED,
        lambda ctx: (
            _reading_succeeded(ctx)
            and _case_raw(ctx) is not None
            and validate_case_reference(_case_raw(ctx)).score != PASS
        ),
        Route.REJECT,
        "Case reference {v} does not match the required format XX-0000/00.",
        lambda ctx: {"v": _display(_case_raw(ctx))},
    ),
    Rule(
        RuleId.R04_AMOUNT_MISSING,
        lambda ctx: _reading_succeeded(ctx) and not ctx.extraction.amount_candidates,
        Route.REJECT,
        "No monetary amount found anywhere in the letter.",
    ),
    Rule(
        RuleId.R05_AMOUNT_NON_POSITIVE,
        lambda ctx: (
            _reading_succeeded(ctx)
            and ctx.resolved_decimal is not None
            and ctx.resolved_decimal <= ZERO_EUR
        ),
        Route.REJECT,
        "Amount {v} is not positive.",
        lambda ctx: {"v": ctx.resolved_amount},
    ),
    # ===== Escalate: could we read the letter at all? ===================================
    Rule(
        RuleId.H01_EXTRACTION_FAILED,
        lambda ctx: ctx.extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY,
        Route.NEEDS_HUMAN,
        "Automated reading failed (malformed model response after one retry); "
        "routed for human reading.",
    ),
    Rule(
        RuleId.H17_DOCUMENT_TYPE_CONTRADICTION,
        _document_type_contradicts_seizure_flag,
        Route.NEEDS_HUMAN,
        "Reading is self-contradictory: classified {t} yet reported not to be a seizure "
        "order (confidence {c:.2f}).",
        lambda ctx: {
            "t": ctx.extraction.document_type.value,
            "c": ctx.extraction.is_seizure_order_confidence,
        },
    ),
    Rule(
        RuleId.H02_DOC_TYPE_UNCERTAIN,
        lambda ctx: _reading_succeeded(ctx)
        and (
            (
                not ctx.extraction.is_seizure_order
                and below_floor(
                    ctx.extraction.is_seizure_order_confidence, DOC_TYPE_REJECT_MIN_CONFIDENCE
                )
            )
            or (
                ctx.extraction.is_seizure_order
                and below_floor(
                    ctx.extraction.is_seizure_order_confidence, DOC_TYPE_AUTO_MIN_CONFIDENCE
                )
            )
        ),
        Route.NEEDS_HUMAN,
        "Uncertain whether this is a seizure order (confidence {c:.2f}).",
        lambda ctx: {"c": ctx.extraction.is_seizure_order_confidence},
    ),
    # ===== Escalate: identity, whose money is this ======================================
    Rule(
        RuleId.H05_DEBTOR_NAME_PARTIAL,
        lambda ctx: _reading_succeeded(ctx)
        and (
            ctx.extraction.debtor_name_is_partial
            or validate_debtor_name(_debtor_raw(ctx)).score != PASS
        ),
        Route.NEEDS_HUMAN,
        "Debtor name {v} is partial or abbreviated (fewer than two full name tokens).",
        lambda ctx: {"v": _display(_debtor_raw(ctx))},
    ),
    Rule(
        RuleId.H06_IBAN_ABSENT,
        lambda ctx: _reading_succeeded(ctx) and ctx.extraction.account_iban.value is None,
        Route.NEEDS_HUMAN,
        "No IBAN given; the target account must be identified by a human.",
    ),
    Rule(
        RuleId.H07_IBAN_INVALID,
        lambda ctx: (
            _iban_raw(ctx) is not None
            and validate_iban(_iban_raw(ctx)).score != PASS
        ),
        Route.NEEDS_HUMAN,
        "IBAN {v} fails checksum or format validation.",
        lambda ctx: {"v": _display(_iban_raw(ctx))},
    ),
    # ===== Escalate: amount, how much ===================================================
    Rule(
        RuleId.H11_OPERATIVE_AMOUNT_UNRESOLVED,
        lambda ctx: ctx.resolution.reason is ResolutionReason.RESTRICTION_UNRESOLVED,
        Route.NEEDS_HUMAN,
        "Letter restricts a prior order but the currently effective amount "
        "could not be identified.",
    ),
    Rule(
        RuleId.H12_RESTRICTION_NOT_REDUCING,
        lambda ctx: (
            ctx.resolution.reason is ResolutionReason.RESTRICTION_SUPERSEDES
            and ctx.resolved_decimal is not None
            and cited_original_amount(ctx.extraction.amount_candidates) is not None
            and ctx.resolved_decimal >= _original_or_zero(ctx)
        ),
        Route.NEEDS_HUMAN,
        "Restricted amount {new} is not lower than the original {old}; internally inconsistent.",
        lambda ctx: {
            "new": ctx.resolved_decimal,
            "old": cited_original_amount(ctx.extraction.amount_candidates),
        },
    ),
    Rule(
        RuleId.H03_AMOUNT_AMBIGUOUS,
        lambda ctx: ctx.resolution.reason is ResolutionReason.AMBIGUOUS,
        Route.NEEDS_HUMAN,
        "{n} monetary figures found and none is unambiguously the effective one{role_note}.",
        lambda ctx: {
            "n": len(ctx.extraction.amount_candidates),
            "role_note": _weak_role_note(ctx),
        },
    ),
    Rule(
        RuleId.H18_MULTIPLE_FIGURES_NO_RESTRICTION_CONTEXT,
        lambda ctx: (
            len(ctx.extraction.amount_candidates) >= MIN_CANDIDATES_FOR_A_SECOND_LOOK
            and ctx.extraction.document_type is not DocumentType.RESTRICTION_ORDER
        ),
        Route.NEEDS_HUMAN,
        "{n} monetary figures found and no restriction context to tell them apart "
        "structurally; which one is effective rests on the reader's role labels alone.",
        lambda ctx: {"n": len(ctx.extraction.amount_candidates)},
    ),
    Rule(
        RuleId.H04_AMOUNT_APPROXIMATE,
        lambda ctx: ctx.considered_candidate is not None
        and ctx.considered_candidate.is_approximate,
        Route.NEEDS_HUMAN,
        "Amount is stated approximately ({q}), not as an exact figure.",
        lambda ctx: {
            "q": _display(_considered_label(ctx))
        },
    ),
    Rule(
        RuleId.H16_AMOUNT_VALUE_UNPARSEABLE,
        lambda ctx: ctx.resolution.reason is ResolutionReason.UNPARSEABLE,
        Route.NEEDS_HUMAN,
        "Amount {raw} could not be read as a monetary figure.",
        lambda ctx: {
            "raw": _display(_considered_label(ctx))
        },
    ),
    Rule(
        RuleId.H13_AMOUNT_FORMAT_AMBIGUOUS,
        lambda ctx: (
            ctx.considered_parse is not None
            and ctx.considered_parse.value is not None
            and ctx.considered_parse.is_format_ambiguous
        ),
        Route.NEEDS_HUMAN,
        "Amount {raw} has an ambiguous decimal format.",
        lambda ctx: {
            "raw": _display(_considered_label(ctx))
        },
    ),
    Rule(
        RuleId.H14_CURRENCY_NOT_EUR,
        lambda ctx: ctx.considered_candidate is not None
        and ctx.considered_candidate.currency != EUR,
        Route.NEEDS_HUMAN,
        "Amount is in {cur}, not EUR.",
        lambda ctx: {
            "cur": "" if ctx.considered_candidate is None else ctx.considered_candidate.currency
        },
    ),
    Rule(
        RuleId.H15_AMOUNT_IMPLAUSIBLE,
        lambda ctx: ctx.resolved_decimal is not None
        and ctx.resolved_decimal > AUTO_MAX_AMOUNT_EUR,
        Route.NEEDS_HUMAN,
        "Amount {v} exceeds this deployment's automatic-handling ceiling of EUR {ceiling}.",
        lambda ctx: {"v": ctx.resolved_amount, "ceiling": AUTO_MAX_AMOUNT_EUR},
    ),
    # ===== Escalate: how well evidenced was the reading? ================================
    Rule(
        RuleId.H08_UNGROUNDED_EVIDENCE,
        lambda ctx: _first_ungrounded_field(ctx) is not None,
        Route.NEEDS_HUMAN,
        "Evidence for {field} does not appear in the letter; the value is not grounded.",
        lambda ctx: {"field": _first_ungrounded_field(ctx)},
    ),
    Rule(
        RuleId.H09_LOW_CONFIDENCE_FIELD,
        lambda ctx: _first_low_confidence_field(ctx) is not None,
        Route.NEEDS_HUMAN,
        "Field {field} confidence {c:.2f} is below the automatic threshold of {floor:.2f}.",
        lambda ctx: {
            "field": _first_low_confidence_field(ctx),
            "c": _lowest_offending_confidence(ctx),
            "floor": AUTO_MIN_FIELD_CONFIDENCE,
        },
    ),
    # ===== The automatic default ========================================================
    Rule(
        RuleId.A00_AUTO_DEFAULT,
        lambda ctx: not ctx.fired_route_bearing,
        Route.AUTO,
        "All required fields present, grounded, validated and consistent; lowest final "
        "confidence across the gated fields (debtor_name, amount, case_reference, plus "
        "account_iban when present) is {c:.2f} ≥ {floor:.2f}.",
        lambda ctx: {
            "c": _lowest_gated_confidence(ctx),
            "floor": AUTO_MIN_FIELD_CONFIDENCE,
        },
    ),
    # ===== Informational: recorded, never decisive ======================================
    Rule(
        RuleId.I01_OPERATIVE_AMOUNT_RESOLVED,
        lambda ctx: ctx.resolution.amount is not None,
        None,
        "Effective amount {v} taken from the {role} figure ({reason}, doctrine {doctrine}).",
        lambda ctx: {
            "v": ctx.resolved_amount,
            "role": (
                AmountRole.UNKNOWN.value
                if ctx.considered_candidate is None
                else ctx.considered_candidate.role.value
            ),
            "reason": ctx.resolution.reason.name,
            "doctrine": RESTRICTION_DOCTRINE_VERSION,
        },
    ),
    Rule(
        RuleId.I02_IBAN_VALIDATED,
        lambda ctx: _iban_raw(ctx) is not None
        and validate_iban(_iban_raw(ctx)).score == PASS,
        None,
        "IBAN checksum valid.",
    ),
    Rule(
        RuleId.I03_GROUNDED_ON_VALUE,
        lambda ctx: _first_value_only_grounded_field(ctx) is not None,
        None,
        "Evidence quote for {field} was not found verbatim, but the extracted value "
        "appears in the letter.",
        lambda ctx: {"field": _first_value_only_grounded_field(ctx)},
    ),
    Rule(
        RuleId.I04_INJECTION_SUSPECTED,
        lambda ctx: any(
            phrase in ctx.extraction.letter_text.casefold() for phrase in _INJECTION_PHRASES
        ),
        None,
        "Letter contains text resembling instructions to the automated reader.",
    ),
)


def _original_or_zero(ctx: RuleContext) -> Decimal:
    """The cited original, guarded by the caller's non-null check."""
    original = cited_original_amount(ctx.extraction.amount_candidates)
    return ZERO_EUR if original is None else original


ROUTE_BEARING_RULE_IDS: tuple[RuleId, ...] = tuple(
    rule.rule_id for rule in RULES if rule.route is not None
)
