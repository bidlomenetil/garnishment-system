"""Validators check *form*, and say so in binary.

Each returns 1.0 or 0.0 with a named code. Soft signals deliberately get their own rules
in the table rather than being blended into a number where nobody can see them.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from garnishment.domain.amount_parsing import parse_eur_amount
from garnishment.domain.validators import (
    validate_amount,
    validate_case_reference,
    validate_debtor_name,
    validate_iban,
)

SAMPLE_IBAN = "DE44 5001 0517 5407 3249 31"


# --- case reference -------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["AB-1042/24", "  AB-1042/24  ", "ZZ-0000/00"])
def test_valid_case_references(raw: str) -> None:
    assert validate_case_reference(raw).score == 1.0


@pytest.mark.parametrize(
    "raw",
    [
        "AB1042/24",  # missing hyphen
        "AB-104/24",  # too few digits
        "AB-10425/24",  # too many digits
        "ABC-1042/24",  # too many letters
        "ab-1042/24",  # lower case
        "AB-1042-24",  # wrong separator
        "AB-1042/2024",  # four-digit year
        "",
    ],
)
def test_malformed_case_references(raw: str) -> None:
    result = validate_case_reference(raw)
    assert result.score == 0.0
    assert result.code == "CASE_REFERENCE_MALFORMED"


def test_absent_case_reference_is_reported_separately_from_a_malformed_one() -> None:
    assert validate_case_reference(None).code == "CASE_REFERENCE_ABSENT"


# --- IBAN -----------------------------------------------------------------------------


def test_sample_iban_passes_mod_97() -> None:
    assert validate_iban(SAMPLE_IBAN).score == 1.0


def test_compact_and_lower_case_forms_pass() -> None:
    assert validate_iban("DE44500105175407324931").score == 1.0
    assert validate_iban("de44 5001 0517 5407 3249 31").score == 1.0


def test_single_digit_flip_fails_the_checksum() -> None:
    """The honest claim for mod-97: it catches single-character transcription errors."""
    result = validate_iban("DE44 5001 0517 5407 3249 32")
    assert result.score == 0.0
    assert result.code == "IBAN_CHECKSUM_FAILED"


def test_wrong_check_digits_fail() -> None:
    assert validate_iban("DE45 5001 0517 5407 3249 31").score == 0.0


def test_adjacent_digit_transposition_fails() -> None:
    assert validate_iban("DE44 5001 0517 5407 3249 13").score == 0.0


@pytest.mark.parametrize(
    "raw",
    [
        "DE44 5001",  # too short
        "DE44" + "1" * 40,  # too long
        "1244 5001 0517 5407 3249 31",  # no country letters
        "DEXX 5001 0517 5407 3249 31",  # no check digits
        "DE44-5001-0517-5407-3249-31",  # punctuation is not an IBAN character
        "",
    ],
)
def test_malformed_ibans(raw: str) -> None:
    assert validate_iban(raw).score == 0.0


def test_absent_iban_is_reported_separately() -> None:
    assert validate_iban(None).code == "IBAN_ABSENT"


# --- amount ---------------------------------------------------------------------------


def test_valid_amount() -> None:
    assert validate_amount(Decimal("4250.00"), "EUR").score == 1.0


def test_unparseable_amount_fails() -> None:
    assert validate_amount(None, "EUR").code == "AMOUNT_ABSENT"


@pytest.mark.parametrize("amount", [Decimal("0.00"), Decimal("-5.00")])
def test_non_positive_amount_fails(amount: Decimal) -> None:
    result = validate_amount(amount, "EUR")
    assert result.score == 0.0
    assert result.code == "AMOUNT_NOT_POSITIVE"


def test_scale_is_the_parser_s_invariant_and_not_this_validator_s_business() -> None:
    """`parse_eur_amount` quantizes, so a figure off two places cannot reach the pipeline.

    The validator used to carry a two-decimal-places check. It could only ever be reached
    by calling the validator directly with a `Decimal` the parser would never emit, which
    is green coverage over a branch the real pipeline cannot enter. The invariant is
    asserted where it actually holds: at the parser.
    """
    for raw in ["4,250.00", "4200", "4.250", "1.234.567,89"]:
        value = parse_eur_amount(raw).value
        assert value is not None
        assert -value.as_tuple().exponent == 2, raw
        assert validate_amount(value, "EUR").score == 1.0


def test_non_eur_currency_fails() -> None:
    result = validate_amount(Decimal("4250.00"), "USD")
    assert result.score == 0.0
    assert result.code == "AMOUNT_CURRENCY_NOT_EUR"


# --- debtor name ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["Jordan A. Meyer", "Jordan Meyer", "Anna-Lena Müller", "Jordan A Meyer"],
)
def test_names_with_two_substantive_tokens_pass(raw: str) -> None:
    assert validate_debtor_name(raw).score == 1.0


@pytest.mark.parametrize(
    "raw", ["M. Meyer", "Meyer", "J. A. M.", "J.A.M.", "", "   ", "-- Meyer", "'' 42"]
)
def test_names_without_two_substantive_tokens_fail(raw: str) -> None:
    result = validate_debtor_name(raw)
    assert result.score == 0.0
    assert result.code in {"DEBTOR_NAME_TOO_FEW_TOKENS", "DEBTOR_NAME_ABSENT"}


def test_absent_name_is_reported_separately() -> None:
    assert validate_debtor_name(None).code == "DEBTOR_NAME_ABSENT"


def test_every_validator_is_binary_and_explains_itself() -> None:
    results = [
        validate_case_reference("AB-1042/24"),
        validate_case_reference("nonsense"),
        validate_iban(SAMPLE_IBAN),
        validate_iban("nonsense"),
        validate_amount(Decimal("1.00"), "EUR"),
        validate_amount(Decimal("-1.00"), "EUR"),
        validate_debtor_name("Jordan Meyer"),
        validate_debtor_name("Meyer"),
    ]
    for result in results:
        assert result.score in {0.0, 1.0}
        assert result.code
        assert result.detail
