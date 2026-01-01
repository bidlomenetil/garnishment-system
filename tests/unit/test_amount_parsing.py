"""`parse_eur_amount` is the only place a written figure becomes a number.

It is deliberately conservative: anything it cannot read unambiguously comes back as
`value=None` or `is_format_ambiguous=True`, both of which escalate. It never guesses.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from garnishment.domain.amount_parsing import ParsedAmount, parse_eur_amount

CASES: list[tuple[str, Decimal | None, bool]] = [
    # (raw, expected value, expected is_format_ambiguous)
    # --- both separators present: the LAST one is the decimal separator ---------------
    ("4,250.00", Decimal("4250.00"), False),
    ("4.250,00", Decimal("4250.00"), False),
    ("1.234.567,89", Decimal("1234567.89"), False),
    ("1,234,567.89", Decimal("1234567.89"), False),
    # --- one separator followed by exactly 2 digits: decimal separator ----------------
    ("1267,00", Decimal("1267.00"), False),
    ("1267.00", Decimal("1267.00"), False),
    ("0.00", Decimal("0.00"), False),
    # --- one separator followed by exactly 3 digits: grouping, and we say so ----------
    ("4.250", Decimal("4250.00"), True),
    ("4,250", Decimal("4250.00"), True),
    # --- no separator: integer euros --------------------------------------------------
    ("4200", Decimal("4200.00"), False),
    ("0", Decimal("0.00"), False),
    ("-5", Decimal("-5.00"), False),
    # --- currency decoration is stripped before parsing -------------------------------
    ("EUR 1.267,00", Decimal("1267.00"), False),
    ("€ 1,267.00", Decimal("1267.00"), False),
    ("USD 4,250.00", Decimal("4250.00"), False),
    # --- unreadable: value is None, and we flag it rather than guessing ---------------
    ("four thousand two hundred", None, True),
    ("in the region of four thousand two hundred euros", None, True),
    ("1,2,3", None, True),
    ("1,234,567", None, True),
    ("4.2", None, True),
    ("4.2500", None, True),
    ("", None, True),
    ("   ", None, True),
    ("EUR 4,250.00 approx.", None, True),
    # both separators present, but the integer part still carries the decimal one
    ("1.2,3.45", None, True),
    # both separators present, but the fraction is not two digits
    ("1,234.5", None, True),
    # a fraction with no integer part in front of it
    (".50", None, True),
    # --- grouping that is neither European nor US: groups of three, or unreadable -------
    # 4,25.00 is not 425.00: a two-digit group is no convention this parser knows, and
    # stripping the comma anyway would resolve an ambiguity silently.
    ("4,25.00", None, True),
    ("1,23,45.67", None, True),
    ("1.23.45,67", None, True),
    # the leading group may be 1-3 digits; four is a missing separator, not a figure
    ("1234,567.89", None, True),
]


@pytest.mark.parametrize(("raw", "expected_value", "expected_ambiguous"), CASES)
def test_parse_eur_amount(
    raw: str, expected_value: Decimal | None, expected_ambiguous: bool
) -> None:
    parsed = parse_eur_amount(raw)
    assert parsed.value == expected_value
    assert parsed.is_format_ambiguous is expected_ambiguous


def test_parsed_values_always_carry_two_decimal_places() -> None:
    for raw, expected_value, _ in CASES:
        if expected_value is None:
            continue
        value = parse_eur_amount(raw).value
        assert value is not None
        assert -value.as_tuple().exponent == 2, raw


def test_parse_is_total_over_arbitrary_text() -> None:
    """The parser is called on model output, so it must never raise."""
    for raw in ["\x00", "∞", "NaN", "Infinity", "1e10", "--5", "..", ",", "-", "1" * 200]:
        assert isinstance(parse_eur_amount(raw), ParsedAmount)


def test_well_formed_grouping_still_parses_exactly_as_before() -> None:
    """The grouping check must refuse malformed runs without touching correct ones."""
    unchanged = {
        "4,250.00": Decimal("4250.00"),
        "4.250,00": Decimal("4250.00"),
        "1267,00": Decimal("1267.00"),
        "4200": Decimal("4200.00"),
        "EUR 1.267,00": Decimal("1267.00"),
        "1.234.567,89": Decimal("1234567.89"),
        "1,234,567.89": Decimal("1234567.89"),
    }
    for raw, expected in unchanged.items():
        parsed = parse_eur_amount(raw)
        assert parsed.value == expected, raw
        assert parsed.is_format_ambiguous is False, raw


def test_nan_and_infinity_are_not_parsed_as_numbers() -> None:
    """Decimal() happily accepts 'NaN' and 'Infinity'; the rule table must never see one."""
    for raw in ["NaN", "Infinity", "-Infinity", "1e10"]:
        assert parse_eur_amount(raw).value is None
