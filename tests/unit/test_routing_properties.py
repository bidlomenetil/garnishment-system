"""Four properties that hold for every extraction, not just the ones we thought of.

Golden cases prove the rules do what we intended on shapes we imagined. These prove the
boundary's structural guarantees hold on shapes we did not.
"""

from __future__ import annotations

import inspect
import math

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from garnishment.domain.confidence import (
    ACCOUNT_IBAN,
    AMOUNT,
    CASE_REFERENCE,
    DEBTOR_NAME,
)
from garnishment.domain.enums import (
    AmountRole,
    DocumentType,
    ExtractionStatus,
    ResolutionReason,
    Route,
    RuleId,
)
from garnishment.domain.extraction import AmountCandidate, ExtractedField, LetterExtraction
from garnishment.domain.operative_amount import resolve_operative_amount
from garnishment.domain.routing import route
from garnishment.domain.rules import RULES, build_rule_context
from garnishment.domain.thresholds import AUTO_MIN_FIELD_CONFIDENCE, ROLE_MIN_CONFIDENCE
from garnishment.domain.values import CaseReference, DebtorName, EvidenceQuote, Iban

EXAMPLES = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

WRITTEN_FIGURES = [
    "4,250.00",
    "4.250,00",
    "1267,00",
    "4.250",
    "4200",
    "0.00",
    "-5",
    "20,000,000.00",
    "1,2,3",
    "four thousand two hundred",
    "",
    "   ",
]

NAME_FRAGMENTS = ["Jordan A. Meyer", "M. Meyer", "Meyer", "", "J. A. M.", "Anna-Lena Müller"]
CASE_FRAGMENTS = ["AB-1042/24", "AB1042-24", "", "ZZ-0000/00", "not a reference"]
IBAN_FRAGMENTS = [
    "DE44 5001 0517 5407 3249 31",
    "DE45 5001 0517 5407 3249 31",
    "DE44500105175407324931",
    "",
    "nonsense",
]

confidences = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
quotes = st.one_of(st.none(), st.text(max_size=40), st.sampled_from(NAME_FRAGMENTS))


@st.composite
def arbitrary_candidates(draw: st.DrawFn) -> tuple[AmountCandidate, ...]:
    count = draw(st.integers(min_value=0, max_value=3))
    return tuple(
        AmountCandidate(
            value_as_written=draw(st.sampled_from(WRITTEN_FIGURES)),
            currency=draw(st.sampled_from(["EUR", "USD", "", "eur"])),
            raw_text=draw(st.text(max_size=30)),
            role=draw(st.sampled_from(AmountRole)),
            role_confidence=draw(confidences),
            is_approximate=draw(st.booleans()),
            evidence_quote=draw(
                st.one_of(st.none(), st.text(max_size=30).map(EvidenceQuote))
            ),
            llm_confidence=draw(confidences),
        )
        for _ in range(count)
    )


@st.composite
def arbitrary_extractions(draw: st.DrawFn) -> LetterExtraction:
    """Degenerate shapes included on purpose: all-null fields, empty candidate lists."""
    return LetterExtraction(
        letter_text=draw(st.text(max_size=200)),
        document_type=draw(st.sampled_from(DocumentType)),
        is_seizure_order=draw(st.booleans()),
        is_seizure_order_confidence=draw(confidences),
        debtor_name=ExtractedField[DebtorName](
            draw(st.one_of(st.none(), st.sampled_from(NAME_FRAGMENTS).map(DebtorName))),
            draw(quotes.map(lambda q: None if q is None else EvidenceQuote(q))),
            draw(confidences),
        ),
        debtor_name_is_partial=draw(st.booleans()),
        case_reference=ExtractedField[CaseReference](
            draw(st.one_of(st.none(), st.sampled_from(CASE_FRAGMENTS).map(CaseReference))),
            draw(quotes.map(lambda q: None if q is None else EvidenceQuote(q))),
            draw(confidences),
        ),
        account_iban=ExtractedField[Iban](
            draw(st.one_of(st.none(), st.sampled_from(IBAN_FRAGMENTS).map(Iban))),
            draw(quotes.map(lambda q: None if q is None else EvidenceQuote(q))),
            draw(confidences),
        ),
        amount_candidates=draw(arbitrary_candidates()),
        model_id=draw(st.text(max_size=20)),
        prompt_version=draw(st.text(max_size=10)),
        extraction_status=draw(st.sampled_from(ExtractionStatus)),
    )


# --- a well-formed letter that should route auto -----------------------------------------


def iban_with_valid_checksum(bban: str, country: str = "DE") -> str:
    rearranged = f"{bban}{country}00"
    numeric = "".join(str(int(ch, 36)) if ch.isalpha() else ch for ch in rearranged)
    return f"{country}{98 - int(numeric) % 97:02d}{bban}"


SAFE_FIGURES = ["4,250.00", "1,267.00", "1267,00", "900.00", "12.50"]


@st.composite
def auto_routing_extractions(draw: st.DrawFn) -> LetterExtraction:
    """A letter with nothing wrong with it, and confidences that clear every floor."""
    first = draw(st.sampled_from(["Jordan", "Anna", "Mikael", "Rosa"]))
    last = draw(st.sampled_from(["Meyer", "Novak", "Fernandes", "Okoro"]))
    name = f"{first} {last}"
    case_reference = "{}-{:04d}/{:02d}".format(
        draw(st.sampled_from(["AB", "CD", "ZQ"])),
        draw(st.integers(min_value=0, max_value=9999)),
        draw(st.integers(min_value=0, max_value=99)),
    )
    iban = iban_with_valid_checksum(
        "".join(str(draw(st.integers(min_value=0, max_value=9))) for _ in range(18))
    )
    figure = draw(st.sampled_from(SAFE_FIGURES))

    debtor_quote = f"Debtor: {name}"
    case_quote = f"Case reference: {case_reference}"
    iban_quote = f"Account (IBAN): {iban}"
    amount_quote = f"Amount to seize: EUR {figure}"
    letter = "\n".join(
        [
            "DISTRICT COURT OF EXAMPLETON",
            "Order of Attachment / Garnishment",
            "",
            case_quote,
            debtor_quote,
            iban_quote,
            amount_quote,
            "",
            "Registrar, District Court of Exampleton",
        ]
    )
    high = st.floats(
        min_value=AUTO_MIN_FIELD_CONFIDENCE,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
    return LetterExtraction(
        letter_text=letter,
        document_type=DocumentType.ATTACHMENT_ORDER,
        is_seizure_order=True,
        is_seizure_order_confidence=draw(high),
        debtor_name=ExtractedField[DebtorName](
            DebtorName(name), EvidenceQuote(debtor_quote), draw(high)
        ),
        case_reference=ExtractedField[CaseReference](
            CaseReference(case_reference), EvidenceQuote(case_quote), draw(high)
        ),
        account_iban=ExtractedField[Iban](Iban(iban), EvidenceQuote(iban_quote), draw(high)),
        amount_candidates=(
            AmountCandidate(
                value_as_written=figure,
                currency="EUR",
                raw_text=f"EUR {figure}",
                role=AmountRole.OPERATIVE,
                role_confidence=draw(high),
                evidence_quote=EvidenceQuote(amount_quote),
                llm_confidence=draw(high),
            ),
        ),
        model_id="claude-sonnet-5",
        prompt_version="v1",
    )


CONFIDENCE_COMPONENTS = ("document_type", DEBTOR_NAME, CASE_REFERENCE, ACCOUNT_IBAN, AMOUNT)


def lower_one_component(
    extraction: LetterExtraction, component: str, value: float
) -> LetterExtraction:
    """Set exactly one self-reported confidence beneath the automatic floor."""
    import dataclasses

    if component == "document_type":
        return dataclasses.replace(extraction, is_seizure_order_confidence=value)
    if component == DEBTOR_NAME:
        return dataclasses.replace(
            extraction,
            debtor_name=dataclasses.replace(extraction.debtor_name, llm_confidence=value),
        )
    if component == CASE_REFERENCE:
        return dataclasses.replace(
            extraction,
            case_reference=dataclasses.replace(extraction.case_reference, llm_confidence=value),
        )
    if component == ACCOUNT_IBAN:
        return dataclasses.replace(
            extraction,
            account_iban=dataclasses.replace(extraction.account_iban, llm_confidence=value),
        )
    candidate = extraction.amount_candidates[0]
    return dataclasses.replace(
        extraction,
        amount_candidates=(dataclasses.replace(candidate, llm_confidence=value),),
    )


# --- property 1: determinism --------------------------------------------------------------


@EXAMPLES
@given(arbitrary_extractions())
def test_property_1_routing_is_deterministic(extraction: LetterExtraction) -> None:
    """Same extraction, same everything: route, confidences, fired rules, decisive rule."""
    first = route(extraction)
    second = route(extraction)
    assert first == second
    assert [f.rule_id for f in first.fired_rules] == [f.rule_id for f in second.fired_rules]
    assert first.field_confidences == second.field_confidences


def test_property_1_route_takes_exactly_one_parameter() -> None:
    """The signature *is* the guarantee: no clock, no I/O, no pre-computed confidences.

    Anything else in this list would mean part of the decision was made somewhere it
    cannot be replayed from the stored record.
    """
    parameters = inspect.signature(route).parameters
    assert list(parameters) == ["extraction"]
    assert parameters["extraction"].annotation in {LetterExtraction, "LetterExtraction"}


# --- property 2: no silent promotion to auto -----------------------------------------------


@EXAMPLES
@given(
    auto_routing_extractions(),
    st.sampled_from(CONFIDENCE_COMPONENTS),
    st.floats(min_value=0.0, max_value=0.84, allow_nan=False, allow_infinity=False),
)
def test_property_2_lowering_any_single_confidence_removes_auto(
    extraction: LetterExtraction, component: str, lowered: float
) -> None:
    """Two-valued, and deliberately not three-level.

    The three-level form -- "severity never decreases" -- is FALSE BY DESIGN. Lowering the
    model's confidence that a document is *not* a seizure order moves it from `reject` to
    `needs_human`, a decrease in severity and exactly what the design intends, because
    escalation requires less evidence than rejection.

    The two-valued form is the one that protects money, and it is true: nothing can be
    approved automatically once any single confidence component falls beneath the floor.
    """
    assume(route(extraction).route is Route.AUTO)
    weakened = lower_one_component(extraction, component, lowered)
    assert route(weakened).route is not Route.AUTO


@EXAMPLES
@given(auto_routing_extractions())
def test_property_2_a_well_formed_letter_does_route_auto(extraction: LetterExtraction) -> None:
    """Guards property 2 against passing vacuously on a generator that never reaches auto."""
    outcome = route(extraction)
    assert outcome.route is Route.AUTO
    assert outcome.decisive_rule_id is RuleId.A00_AUTO_DEFAULT


# --- property 3: reject dominance -----------------------------------------------------------


@EXAMPLES
@given(arbitrary_extractions())
def test_property_3_any_true_reject_predicate_forces_reject(
    extraction: LetterExtraction,
) -> None:
    """Reject dominance is structural: the reject rules sit first in the table."""
    context = build_rule_context(extraction)
    reject_fires = any(
        rule.predicate(context) for rule in RULES if rule.route is Route.REJECT
    )
    outcome = route(extraction)
    if reject_fires:
        assert outcome.route is Route.REJECT
    else:
        assert outcome.route is not Route.REJECT


@EXAMPLES
@given(arbitrary_extractions())
def test_property_3_our_own_reading_failure_never_rejects(
    extraction: LetterExtraction,
) -> None:
    """`reject` is a statement about the letter, never about our reader malfunctioning."""
    assume(extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY)
    assert route(extraction).route is Route.NEEDS_HUMAN


# --- property 4: totality ---------------------------------------------------------------------


@EXAMPLES
@given(arbitrary_extractions())
def test_property_4_routing_is_total(extraction: LetterExtraction) -> None:
    outcome = route(extraction)
    assert outcome.route in set(Route)
    assert outcome.decisive_rule_id in {r.rule_id for r in RULES if r.route is not None}
    assert outcome.fired_rules
    assert len(outcome.reasons) == len(outcome.fired_rules)
    assert outcome.operative_amount_reason in set(ResolutionReason)


@EXAMPLES
@given(arbitrary_candidates(), st.sampled_from(DocumentType))
def test_property_4_the_resolver_is_total(
    candidates: tuple[AmountCandidate, ...], document_type: DocumentType
) -> None:
    """Every candidate list reaches a named reason; there is no unmatched branch."""
    resolution = resolve_operative_amount(candidates, document_type)
    assert resolution.reason in set(ResolutionReason)
    if resolution.source_candidate_index is not None:
        assert resolution.amount is not None
        assert resolution.considered_candidate_index == resolution.source_candidate_index
    if resolution.considered_candidate_index is not None:
        assert 0 <= resolution.considered_candidate_index < len(candidates)


def test_property_4_the_named_degenerate_shapes() -> None:
    degenerate = [
        LetterExtraction(letter_text=""),
        LetterExtraction(letter_text="", amount_candidates=()),
        # The shape that matched no resolution row before the catch-all existed.
        LetterExtraction(
            letter_text="an order citing a prior one",
            document_type=DocumentType.ATTACHMENT_ORDER,
            amount_candidates=(
                AmountCandidate(value_as_written="4,250.00", role=AmountRole.ORIGINAL),
            ),
        ),
        LetterExtraction(
            letter_text="x" * 5000,
            extraction_status=ExtractionStatus.MALFORMED_AFTER_RETRY,
        ),
    ]
    for extraction in degenerate:
        outcome = route(extraction)
        assert outcome.decisive_rule_id in {r.rule_id for r in RULES if r.route is not None}


def test_property_4_a_degenerate_confidence_escalates_rather_than_slipping_through() -> None:
    """A NaN self-report must fail toward escalation, not past the gate."""
    extraction = LetterExtraction(
        letter_text="Case reference: AB-1042/24",
        is_seizure_order=True,
        is_seizure_order_confidence=math.nan,
        debtor_name=ExtractedField[DebtorName](
            DebtorName("Jordan Meyer"), EvidenceQuote("Jordan Meyer"), math.nan
        ),
        case_reference=ExtractedField[CaseReference](
            CaseReference("AB-1042/24"), EvidenceQuote("Case reference: AB-1042/24"), math.nan
        ),
        amount_candidates=(
            AmountCandidate(value_as_written="1.00", role=AmountRole.OPERATIVE),
        ),
    )
    assert route(extraction).route is not Route.AUTO


@given(auto_routing_extractions())
@EXAMPLES
def test_a_nan_self_report_is_caught_by_the_confidence_gate_itself(
    extraction: LetterExtraction,
) -> None:
    """Which rule catches a NaN matters, not just that some rule does.

    The test above escalates on several NaNs at once and passes through H02 whatever H09
    does, so it stayed green while H09's own comparison — a raw `final < floor`, False
    for every NaN — let a degenerate confidence straight past the gate. This narrows it
    to one NaN on one self-report, with nothing else wrong, and names the rule that must
    fire.

    Constructed rather than drawn: every confidence strategy in this module sets
    `allow_nan=False`, so Hypothesis structurally cannot find this class of bug here.
    """
    import dataclasses

    degenerate = dataclasses.replace(
        extraction,
        debtor_name=dataclasses.replace(extraction.debtor_name, llm_confidence=math.nan),
    )
    outcome = route(degenerate)
    assert outcome.route is Route.NEEDS_HUMAN
    assert outcome.decisive_rule_id is RuleId.H09_LOW_CONFIDENCE_FIELD


# --- property 5: a lone figure on a restriction order may be the superseded one ------------


@EXAMPLES
@given(
    st.sampled_from(SAFE_FIGURES),
    st.sampled_from(AmountRole),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property_5_a_lone_figure_resolves_on_a_restriction_order_only_on_a_trusted_label(
    figure: str, role: AmountRole, role_confidence: float
) -> None:
    """The sole-candidate exemption is an exemption from the role gate, not from judgement.

    Off a restriction order a lone figure is the only figure and its label decides
    nothing. On a restriction order the letter is superseding a prior figure, so the one
    figure in front of us may be the old one; only a label that says "this is the
    effective one", from a reader that is sure of it, may resolve it.
    """
    candidate = AmountCandidate(
        value_as_written=figure,
        currency="EUR",
        raw_text=f"EUR {figure}",
        role=role,
        role_confidence=role_confidence,
    )
    resolution = resolve_operative_amount((candidate,), DocumentType.RESTRICTION_ORDER)
    may_resolve = (
        role in {AmountRole.OPERATIVE, AmountRole.RESTRICTED}
        and role_confidence >= ROLE_MIN_CONFIDENCE
    )
    if may_resolve:
        assert resolution.reason is ResolutionReason.SOLE_CANDIDATE
    else:
        assert resolution.reason is not ResolutionReason.SOLE_CANDIDATE
        assert resolution.amount is None


@EXAMPLES
@given(
    auto_routing_extractions(),
    st.sampled_from([AmountRole.UNKNOWN, AmountRole.ORIGINAL]),
    st.floats(min_value=0.0, max_value=0.84, allow_nan=False, allow_infinity=False),
)
def test_property_5_such_a_letter_never_routes_auto(
    extraction: LetterExtraction, role: AmountRole, role_confidence: float
) -> None:
    """The same shape end to end: it must not reach `auto` through the whole table."""
    import dataclasses

    restriction = dataclasses.replace(
        extraction,
        document_type=DocumentType.RESTRICTION_ORDER,
        amount_candidates=(
            dataclasses.replace(
                extraction.amount_candidates[0], role=role, role_confidence=role_confidence
            ),
        ),
    )
    assert route(restriction).route is not Route.AUTO
