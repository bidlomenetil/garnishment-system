"""Grounding is the verifiable half of confidence: does the letter actually say this?

It is the control that stops a fabricated value — including one injected into the letter
text — from reaching `auto`, so its tiers and its per-field normalisers are load-bearing.
"""

from __future__ import annotations

from garnishment.domain.grounding import (
    grounding_score,
    normalise_iban,
    normalise_text,
    normaliser_for,
)
from garnishment.domain.thresholds import (
    GROUNDING_ABSENT,
    GROUNDING_CASE_INSENSITIVE,
    GROUNDING_EXACT,
    GROUNDING_VALUE_ONLY,
)

LETTER = (
    "DISTRICT COURT OF EXAMPLETON\n"
    "Case reference: AB-1042/24\n"
    "Debtor:        Jordan A. Meyer\n"
    "Account (IBAN): DE44 5001 0517 5407 3249 31\n"
    "Amount to seize: EUR 4,250.00\n"
)


def _score(quote: str | None, value: str | None, field: str = "debtor_name") -> float:
    return grounding_score(
        letter_text=LETTER,
        evidence_quote=quote,
        value_text=value,
        normalise=normaliser_for(field),
    ).score


def test_tier_exact_match() -> None:
    assert _score("Debtor:        Jordan A. Meyer", "Jordan A. Meyer") == GROUNDING_EXACT


def test_tier_exact_match_survives_whitespace_differences() -> None:
    """The default normaliser collapses whitespace runs, so re-spacing is still exact."""
    assert _score("Debtor: Jordan A. Meyer", "Jordan A. Meyer") == GROUNDING_EXACT


def test_tier_case_insensitive_only() -> None:
    assert _score("debtor: jordan a. meyer", "Jordan A. Meyer") == GROUNDING_CASE_INSENSITIVE


def test_tier_value_present_quote_absent() -> None:
    result = grounding_score(
        letter_text=LETTER,
        evidence_quote="The court names the debtor as follows",
        value_text="Jordan A. Meyer",
        normalise=normalise_text,
    )
    assert result.score == GROUNDING_VALUE_ONLY
    assert result.grounded_on_value_only is True


def test_the_value_only_tier_tolerates_a_case_difference_too() -> None:
    """A reader that gets the value right but retypes its case (all-caps here) has still
    found the right value — this used to score 0.0 (H08), an ungrounded false alarm on a
    value plainly present in the letter."""
    result = grounding_score(
        letter_text=LETTER,
        evidence_quote="The court names the debtor as follows",
        value_text="JORDAN A. MEYER",
        normalise=normalise_text,
    )
    assert result.score == GROUNDING_VALUE_ONLY
    assert result.grounded_on_value_only is True


def test_tier_absent_when_neither_quote_nor_value_occurs() -> None:
    assert _score("Debtor: Alex Fabricated", "Alex Fabricated") == GROUNDING_ABSENT


def test_tier_absent_when_both_are_missing() -> None:
    assert _score(None, None) == GROUNDING_ABSENT


def test_empty_quote_is_treated_as_absent_not_as_a_match() -> None:
    """An empty string is a substring of everything; it must not score 1.00."""
    assert _score("", None) == GROUNDING_ABSENT
    assert _score("   ", None) == GROUNDING_ABSENT


def test_value_only_tier_requires_a_non_empty_value() -> None:
    assert _score("not in the letter at all", "") == GROUNDING_ABSENT


def test_nfkc_normalisation_is_applied() -> None:
    """A full-width digit run normalises to the ASCII form the letter uses."""
    full_width = "Amount to seize: EUR ４,250.00"  # noqa: RUF001 - the point of the test
    assert _score(full_width, None, field="amount") == GROUNDING_EXACT


# --- PM-B: the compact-IBAN-against-spaced-letter case ---------------------------------


def test_compact_iban_value_grounds_against_the_spaced_letter_form() -> None:
    """The model commonly returns a compact IBAN while the letter writes it in groups.

    Without an IBAN-specific normaliser this scores 0.00 and wrongly escalates Samples 1
    and 3, which is load-bearing now that a present IBAN sits inside the confidence gate.
    """
    result = grounding_score(
        letter_text=LETTER,
        evidence_quote=None,
        value_text="DE44500105175407324931",
        normalise=normaliser_for("account_iban"),
    )
    assert result.score == GROUNDING_VALUE_ONLY


def test_compact_iban_quote_grounds_exactly_against_the_spaced_letter_form() -> None:
    result = grounding_score(
        letter_text=LETTER,
        evidence_quote="Account (IBAN): DE44500105175407324931",
        value_text="DE44500105175407324931",
        normalise=normaliser_for("account_iban"),
    )
    assert result.score == GROUNDING_EXACT


def test_lower_case_iban_grounds_exactly() -> None:
    result = grounding_score(
        letter_text=LETTER,
        evidence_quote=None,
        value_text="de44 5001 0517 5407 3249 31",
        normalise=normaliser_for("account_iban"),
    )
    assert result.score == GROUNDING_VALUE_ONLY


def test_the_default_normaliser_would_fail_the_pm_b_case() -> None:
    """Documents why the IBAN field needs its own normaliser rather than the default."""
    result = grounding_score(
        letter_text=LETTER,
        evidence_quote=None,
        value_text="DE44500105175407324931",
        normalise=normalise_text,
    )
    assert result.score == GROUNDING_ABSENT


def test_normalisers() -> None:
    assert normalise_text("  a \n b  ") == "a b"
    assert normalise_iban("de44 5001 0517") == "DE4450010517"
    assert normaliser_for("account_iban") is normalise_iban
    assert normaliser_for("debtor_name") is normalise_text
    assert normaliser_for("a_field_with_no_special_handling") is normalise_text
