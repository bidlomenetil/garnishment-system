"""The resolver decides which figure in a letter is currently effective.

This is the judgement the design deliberately refuses to let the model make. The model
enumerates figures and labels roles; twenty lines of tested code under a versioned
doctrine decide which one the bank acts on.
"""

from __future__ import annotations

from decimal import Decimal

from garnishment.domain.enums import AmountRole, DocumentType, ResolutionReason
from garnishment.domain.extraction import AmountCandidate
from garnishment.domain.operative_amount import (
    RESTRICTION_DOCTRINE_V1,
    RESTRICTION_DOCTRINE_VERSION,
    cited_original_amount,
    resolve_operative_amount,
)
from garnishment.domain.thresholds import ROLE_MIN_CONFIDENCE

ATTACHMENT = DocumentType.ATTACHMENT_ORDER
RESTRICTION = DocumentType.RESTRICTION_ORDER


def candidate(
    written: str,
    role: AmountRole = AmountRole.UNKNOWN,
    role_confidence: float = 0.95,
    currency: str = "EUR",
) -> AmountCandidate:
    return AmountCandidate(
        value_as_written=written,
        currency=currency,
        raw_text=f"{currency} {written}",
        role=role,
        role_confidence=role_confidence,
    )


# --- row 1: no candidates --------------------------------------------------------------


def test_row_1_no_candidates() -> None:
    result = resolve_operative_amount((), ATTACHMENT)
    assert result.reason is ResolutionReason.NO_AMOUNT
    assert result.amount is None
    assert result.considered_candidate_index is None
    assert result.source_candidate_index is None


# --- row 2: a sole candidate that does not parse ---------------------------------------


def test_row_2_sole_unparseable_candidate_is_still_considered() -> None:
    """Sample 2's shape: the letter points at one figure, written as prose.

    `considered` records which figure the letter pointed at; `source` records which figure
    we actually managed to read. Keeping them apart is what lets the record say both
    "stated approximately" and "could not be read" about the same figure.
    """
    result = resolve_operative_amount(
        (candidate("four thousand two hundred", AmountRole.OPERATIVE),), ATTACHMENT
    )
    assert result.reason is ResolutionReason.UNPARSEABLE
    assert result.considered_candidate_index == 0
    assert result.source_candidate_index is None
    assert result.amount is None


def test_row_2_applies_to_any_role_including_original() -> None:
    result = resolve_operative_amount((candidate("prose", AmountRole.ORIGINAL),), ATTACHMENT)
    assert result.reason is ResolutionReason.UNPARSEABLE
    assert result.considered_candidate_index == 0


# --- row 3: a winner by role that does not parse ----------------------------------------


def test_row_3_winner_by_role_that_does_not_parse() -> None:
    result = resolve_operative_amount(
        (
            candidate("4,250.00", AmountRole.ORIGINAL),
            candidate("one thousand two hundred", AmountRole.RESTRICTED),
        ),
        RESTRICTION,
    )
    assert result.reason is ResolutionReason.UNPARSEABLE
    assert result.considered_candidate_index == 1
    assert result.source_candidate_index is None


# --- rows 4 and 5: sole candidate and explicit operative --------------------------------


def test_row_4_sole_candidate() -> None:
    result = resolve_operative_amount((candidate("4,250.00", AmountRole.UNKNOWN),), ATTACHMENT)
    assert result.reason is ResolutionReason.SOLE_CANDIDATE
    assert result.amount is not None
    assert result.amount.amount == Decimal("4250.00")
    assert result.considered_candidate_index == 0
    assert result.source_candidate_index == 0


def test_row_4_sole_candidate_is_exempt_from_the_role_confidence_gate() -> None:
    """With one figure the role label decides nothing, so an unsure label costs nothing."""
    result = resolve_operative_amount(
        (candidate("4,250.00", AmountRole.OPERATIVE, role_confidence=0.10),), ATTACHMENT
    )
    assert result.reason is ResolutionReason.SOLE_CANDIDATE


def test_row_4_the_exemption_does_not_extend_to_a_restriction_order() -> None:
    """The exact wrong-amount failure this design exists to prevent.

    A restriction order is the one document where a second, superseded figure is expected
    to exist. A lone figure there may well be the old one, so an unsure `unknown` label is
    not "a label that decides nothing" — it is the only thing standing between the bank
    and paying out the amount the court has just reduced.
    """
    result = resolve_operative_amount(
        (candidate("4,250.00", AmountRole.UNKNOWN, role_confidence=0.20),), RESTRICTION
    )
    assert result.reason is ResolutionReason.AMBIGUOUS
    assert result.amount is None
    assert result.source_candidate_index is None


def test_row_4_a_lone_restricted_figure_the_reader_is_sure_of_still_resolves() -> None:
    result = resolve_operative_amount(
        (candidate("1,267.00", AmountRole.RESTRICTED, role_confidence=0.95),), RESTRICTION
    )
    assert result.reason is ResolutionReason.SOLE_CANDIDATE
    assert result.amount is not None
    assert result.amount.amount == Decimal("1267.00")


def test_row_4_a_lone_effective_label_below_the_floor_does_not_resolve() -> None:
    result = resolve_operative_amount(
        (candidate("1,267.00", AmountRole.RESTRICTED, role_confidence=0.60),), RESTRICTION
    )
    assert result.reason is ResolutionReason.AMBIGUOUS


def test_row_5_explicit_operative_wins_among_several() -> None:
    result = resolve_operative_amount(
        (
            candidate("9,999.00", AmountRole.UNKNOWN),
            candidate("4,250.00", AmountRole.OPERATIVE, role_confidence=0.94),
        ),
        ATTACHMENT,
    )
    assert result.reason is ResolutionReason.EXPLICIT_OPERATIVE
    assert result.source_candidate_index == 1
    assert result.amount is not None
    assert result.amount.amount == Decimal("4250.00")


def test_row_5_operative_beats_restricted_on_a_restriction_order() -> None:
    result = resolve_operative_amount(
        (
            candidate("1,267.00", AmountRole.RESTRICTED),
            candidate("900.00", AmountRole.OPERATIVE),
        ),
        RESTRICTION,
    )
    assert result.reason is ResolutionReason.EXPLICIT_OPERATIVE
    assert result.source_candidate_index == 1


def test_two_operative_candidates_have_no_unique_winner() -> None:
    result = resolve_operative_amount(
        (
            candidate("4,250.00", AmountRole.OPERATIVE),
            candidate("1,267.00", AmountRole.OPERATIVE),
        ),
        ATTACHMENT,
    )
    assert result.reason is ResolutionReason.AMBIGUOUS


# --- row 6: the restriction doctrine ----------------------------------------------------


def test_row_6_restriction_supersedes_the_original() -> None:
    result = resolve_operative_amount(
        (
            candidate("4,250.00", AmountRole.ORIGINAL, role_confidence=0.94),
            candidate("1,267.00", AmountRole.RESTRICTED, role_confidence=0.95),
        ),
        RESTRICTION,
    )
    assert result.reason is ResolutionReason.RESTRICTION_SUPERSEDES
    assert result.amount is not None
    assert result.amount.amount == Decimal("1267.00")
    assert result.considered_candidate_index == 1
    assert result.source_candidate_index == 1
    assert result.doctrine_version == RESTRICTION_DOCTRINE_VERSION


def test_the_doctrine_only_applies_to_a_restriction_order() -> None:
    result = resolve_operative_amount(
        (
            candidate("4,250.00", AmountRole.ORIGINAL),
            candidate("1,267.00", AmountRole.RESTRICTED),
        ),
        ATTACHMENT,
    )
    assert result.reason is ResolutionReason.AMBIGUOUS


# --- the role-confidence gate -----------------------------------------------------------


def test_low_role_confidence_degrades_an_explicit_operative_to_ambiguous() -> None:
    result = resolve_operative_amount(
        (
            candidate("9,999.00", AmountRole.UNKNOWN),
            candidate("4,250.00", AmountRole.OPERATIVE, role_confidence=0.60),
        ),
        ATTACHMENT,
    )
    assert result.reason is ResolutionReason.AMBIGUOUS
    assert result.considered_candidate_index is None


def test_low_role_confidence_degrades_a_restriction_to_ambiguous() -> None:
    """Sample 3's fourth failure mode."""
    result = resolve_operative_amount(
        (
            candidate("4,250.00", AmountRole.ORIGINAL, role_confidence=0.94),
            candidate("1,267.00", AmountRole.RESTRICTED, role_confidence=0.60),
        ),
        RESTRICTION,
    )
    assert result.reason is ResolutionReason.AMBIGUOUS


def test_the_gate_admits_a_label_exactly_at_the_floor() -> None:
    result = resolve_operative_amount(
        (
            candidate("9,999.00", AmountRole.UNKNOWN),
            candidate("4,250.00", AmountRole.OPERATIVE, role_confidence=ROLE_MIN_CONFIDENCE),
        ),
        ATTACHMENT,
    )
    assert result.reason is ResolutionReason.EXPLICIT_OPERATIVE


# --- row 7: a restriction we could not resolve ------------------------------------------


def test_row_7_restriction_order_carrying_only_the_original_figure() -> None:
    """Sample 3's second failure mode: the model returned only the superseded figure."""
    result = resolve_operative_amount(
        (candidate("4,250.00", AmountRole.ORIGINAL),), RESTRICTION
    )
    assert result.reason is ResolutionReason.RESTRICTION_UNRESOLVED
    assert result.considered_candidate_index is None
    assert result.source_candidate_index is None


def test_row_7_applies_when_every_candidate_is_the_original() -> None:
    result = resolve_operative_amount(
        (
            candidate("4,250.00", AmountRole.ORIGINAL),
            candidate("4,250.00", AmountRole.ORIGINAL),
        ),
        RESTRICTION,
    )
    assert result.reason is ResolutionReason.RESTRICTION_UNRESOLVED


def test_unlabelled_candidates_on_a_restriction_order_are_ambiguous_not_unresolved() -> None:
    """Sample 3's first failure mode.

    H11 says "this letter restricts a prior order and we could not find the new figure",
    which is a claim about a letter carrying only the superseded figure. Two unlabelled
    figures is a different and weaker situation: we found figures, none is identifiably
    effective. That is H03.
    """
    result = resolve_operative_amount(
        (
            candidate("4,250.00", AmountRole.UNKNOWN),
            candidate("1,267.00", AmountRole.UNKNOWN),
        ),
        RESTRICTION,
    )
    assert result.reason is ResolutionReason.AMBIGUOUS


# --- row 8: the catch-all ----------------------------------------------------------------


def test_row_8_catches_a_lone_original_figure_on_an_attachment_order() -> None:
    """A real shape: a letter citing a prior order without restricting it.

    Before the catch-all existed this matched no row at all. Escalation is the right
    default for an unclassified shape.
    """
    result = resolve_operative_amount((candidate("4,250.00", AmountRole.ORIGINAL),), ATTACHMENT)
    assert result.reason is ResolutionReason.AMBIGUOUS
    assert result.considered_candidate_index is None
    assert result.source_candidate_index is None


def test_row_8_catches_several_unlabelled_figures() -> None:
    result = resolve_operative_amount(
        (
            candidate("4,250.00", AmountRole.UNKNOWN),
            candidate("1,267.00", AmountRole.UNKNOWN),
            candidate("99.00", AmountRole.UNKNOWN),
        ),
        ATTACHMENT,
    )
    assert result.reason is ResolutionReason.AMBIGUOUS


# --- invariants -------------------------------------------------------------------------


def test_every_resolution_records_the_doctrine_version() -> None:
    inputs = [
        ((), ATTACHMENT),
        ((candidate("4,250.00", AmountRole.OPERATIVE),), ATTACHMENT),
        ((candidate("prose"),), RESTRICTION),
        ((candidate("1.00", AmountRole.ORIGINAL),), RESTRICTION),
    ]
    for candidates, document_type in inputs:
        assert resolve_operative_amount(candidates, document_type).doctrine_version == (
            RESTRICTION_DOCTRINE_VERSION
        )


def test_a_source_index_always_implies_a_considered_index_and_an_amount() -> None:
    inputs = [
        ((candidate("4,250.00", AmountRole.OPERATIVE),), ATTACHMENT),
        (
            (
                candidate("4,250.00", AmountRole.ORIGINAL),
                candidate("1,267.00", AmountRole.RESTRICTED),
            ),
            RESTRICTION,
        ),
    ]
    for candidates, document_type in inputs:
        result = resolve_operative_amount(candidates, document_type)
        assert result.source_candidate_index is not None
        assert result.considered_candidate_index == result.source_candidate_index
        assert result.amount is not None


def test_currency_is_carried_through_from_the_winning_candidate() -> None:
    result = resolve_operative_amount(
        (candidate("4,250.00", AmountRole.OPERATIVE, currency="USD"),), ATTACHMENT
    )
    assert result.amount is not None
    assert result.amount.currency == "USD"


def test_the_doctrine_is_a_stated_sentence_not_a_hidden_conditional() -> None:
    assert "supersedes" in RESTRICTION_DOCTRINE_V1


# --- the consistency check feeding H12 ---------------------------------------------------


def test_cited_original_amount_reads_the_unique_original_candidate() -> None:
    candidates = (
        candidate("4,250.00", AmountRole.ORIGINAL),
        candidate("1,267.00", AmountRole.RESTRICTED),
    )
    assert cited_original_amount(candidates) == Decimal("4250.00")


def test_cited_original_amount_is_none_when_there_is_no_unique_original() -> None:
    assert cited_original_amount(()) is None
    assert (
        cited_original_amount(
            (
                candidate("4,250.00", AmountRole.ORIGINAL),
                candidate("9,000.00", AmountRole.ORIGINAL),
            )
        )
        is None
    )


def test_cited_original_amount_is_none_when_the_original_does_not_parse() -> None:
    assert cited_original_amount((candidate("prose", AmountRole.ORIGINAL),)) is None
