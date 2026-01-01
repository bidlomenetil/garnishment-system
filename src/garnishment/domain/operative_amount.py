"""Which figure in the letter is currently effective?

This is the legal judgement the design refuses to hand to the model. The reader's job is
reading comprehension: enumerate every figure and label the part each one plays. The
judgement that a restriction supersedes the original it cites is made here, by a versioned
doctrine, in code that can be unit-tested and recorded as a fired rule.

The resolver is **total**: the final branch is an unconditional `else`, so every possible
candidate list reaches a named `ResolutionReason`. A shape no earlier row claims is
ambiguous, and ambiguous means a human looks.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from garnishment.domain.amount_parsing import parse_eur_amount
from garnishment.domain.enums import AmountRole, DocumentType, ResolutionReason
from garnishment.domain.extraction import AmountCandidate
from garnishment.domain.thresholds import ROLE_MIN_CONFIDENCE, below_floor
from garnishment.domain.values import Money

RESTRICTION_DOCTRINE_V1 = (
    "a restriction order's restricted figure supersedes the original it cites"
)
"""The doctrine, stated as a sentence rather than buried in an `if`.

Its evidence base is **n = 1**: one sample letter. It is a hypothesis about how restriction
orders work, versioned so that a future correction is visible in stored records rather
than inferred from git history.
"""

RESTRICTION_DOCTRINE_VERSION = "RESTRICTION_DOCTRINE_V1"
"""Recorded on every decision, including decisions the doctrine did not decide."""

_ROLES_A_SOLE_CANDIDATE_MAY_CARRY = frozenset(
    {AmountRole.OPERATIVE, AmountRole.RESTRICTED, AmountRole.UNKNOWN}
)

_ROLES_A_SOLE_CANDIDATE_MAY_CARRY_UNDER_RESTRICTION = frozenset(
    {AmountRole.OPERATIVE, AmountRole.RESTRICTED}
)
"""The narrower set a lone figure must fall in when the letter restricts a prior order.

On any other document a lone figure is the only figure, so its role label decides nothing
and an unsure label costs nothing. A restriction order is the one context where a second,
superseded figure is *expected* to exist: a lone figure there may well be the old one the
letter is superseding, and acting on it would pay out the wrong amount. So the label has
to say the figure is the effective one, and the reader has to be sure of that label.
"""


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of asking which figure is operative.

    Two indices, because "which figure did we mean" and "which figure did we get" are
    different questions. `considered_candidate_index` is set whenever the candidate set
    identifies one figure -- a sole candidate, or a unique winner by role -- independently
    of whether that figure parsed. `source_candidate_index` is set only when that figure
    also parsed into a usable number.

    The distinction matters because a candidate's properties (approximate, ambiguous
    format, non-EUR currency) are facts about the figure the letter pointed at, and they
    must stay reportable when the figure itself could not be read.
    """

    amount: Money | None
    source_candidate_index: int | None
    considered_candidate_index: int | None
    reason: ResolutionReason
    doctrine_version: str = RESTRICTION_DOCTRINE_VERSION


def resolve_operative_amount(
    candidates: tuple[AmountCandidate, ...],
    document_type: DocumentType,
) -> Resolution:
    """Resolve the operative figure. Pure, total, and never raises."""
    if not candidates:
        return _unresolved(ResolutionReason.NO_AMOUNT)

    if len(candidates) == 1:
        return _resolve_single(candidates[0], document_type)

    winner_index, winner_reason = _winner_by_role(candidates, document_type)
    if winner_index is not None and winner_reason is not None:
        return _resolve_winner(candidates[winner_index], winner_index, winner_reason)

    if _is_unresolved_restriction(candidates, document_type):
        return _unresolved(ResolutionReason.RESTRICTION_UNRESOLVED)

    # Catch-all. Any shape the rows above do not claim is ambiguous by construction:
    # several figures with no unique winner, a winner-by-role whose label we do not
    # trust, or a lone figure whose role rules it out.
    return _unresolved(ResolutionReason.AMBIGUOUS)


def cited_original_amount(candidates: tuple[AmountCandidate, ...]) -> Decimal | None:
    """The figure a restriction order supersedes, when exactly one is labelled `original`.

    Feeds H12's consistency check: a restriction that raises the amount is internally
    inconsistent. Returns None when there is no unique original, or when it did not parse,
    so the rule can guard on a real number rather than compare against nothing.
    """
    originals = [c for c in candidates if c.role is AmountRole.ORIGINAL]
    if len(originals) != 1:
        return None
    return parse_eur_amount(originals[0].value_as_written).value


def _resolve_single(only: AmountCandidate, document_type: DocumentType) -> Resolution:
    parsed = parse_eur_amount(only.value_as_written)
    if parsed.value is None:
        # The letter pointed at a figure we could not read. Record which one.
        return Resolution(None, None, 0, ResolutionReason.UNPARSEABLE)
    if _a_sole_candidate_may_resolve(only, document_type):
        return Resolution(
            Money(parsed.value, only.currency), 0, 0, ResolutionReason.SOLE_CANDIDATE
        )
    if document_type is DocumentType.RESTRICTION_ORDER and only.role is AmountRole.ORIGINAL:
        return _unresolved(ResolutionReason.RESTRICTION_UNRESOLVED)
    return _unresolved(ResolutionReason.AMBIGUOUS)


def _a_sole_candidate_may_resolve(only: AmountCandidate, document_type: DocumentType) -> bool:
    """May a lone figure be taken as the effective one on its own?

    Off a restriction order, yes for any role the letter might plausibly be pointing at:
    with one figure the label decides nothing, so the role gate does not apply. On a
    restriction order it does apply, because that is the one document where the figure in
    front of us may be the superseded one.
    """
    if document_type is DocumentType.RESTRICTION_ORDER:
        return only.role in _ROLES_A_SOLE_CANDIDATE_MAY_CARRY_UNDER_RESTRICTION and not (
            below_floor(only.role_confidence, ROLE_MIN_CONFIDENCE)
        )
    return only.role in _ROLES_A_SOLE_CANDIDATE_MAY_CARRY


def _winner_by_role(
    candidates: tuple[AmountCandidate, ...],
    document_type: DocumentType,
) -> tuple[int | None, ResolutionReason | None]:
    """A unique, confidently-labelled figure, or nothing."""
    explicit = _unique_trusted_index(candidates, AmountRole.OPERATIVE)
    if explicit is not None:
        return explicit, ResolutionReason.EXPLICIT_OPERATIVE

    if document_type is DocumentType.RESTRICTION_ORDER:
        restricted = _unique_trusted_index(candidates, AmountRole.RESTRICTED)
        if restricted is not None:
            return restricted, ResolutionReason.RESTRICTION_SUPERSEDES

    return None, None


def _unique_trusted_index(candidates: tuple[AmountCandidate, ...], role: AmountRole) -> int | None:
    matches = [index for index, c in enumerate(candidates) if c.role is role]
    if len(matches) != 1:
        return None
    if below_floor(candidates[matches[0]].role_confidence, ROLE_MIN_CONFIDENCE):
        return None
    return matches[0]


def _resolve_winner(
    winner: AmountCandidate, index: int, reason: ResolutionReason
) -> Resolution:
    parsed = parse_eur_amount(winner.value_as_written)
    if parsed.value is None:
        return Resolution(None, None, index, ResolutionReason.UNPARSEABLE)
    return Resolution(Money(parsed.value, winner.currency), index, index, reason)


def _is_unresolved_restriction(
    candidates: tuple[AmountCandidate, ...], document_type: DocumentType
) -> bool:
    """The letter restricts a prior order but carries only the superseded figure."""
    return document_type is DocumentType.RESTRICTION_ORDER and all(
        c.role is AmountRole.ORIGINAL for c in candidates
    )


def _unresolved(reason: ResolutionReason) -> Resolution:
    return Resolution(None, None, None, reason)
