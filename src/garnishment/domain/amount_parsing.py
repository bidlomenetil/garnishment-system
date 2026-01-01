"""Turn a figure as written in a letter into a `Decimal`, or refuse to.

`"4,250.00"` and `"4.250,00"` are the same figure under different locale conventions, and
`"4.250"` is genuinely ambiguous between the two. The parser never resolves an ambiguity
silently: it either returns a value it is sure of, or flags the reading so a named rule
can escalate it.

This function is the *only* place a written amount becomes a number, and it runs at
routing time on the verbatim `value_as_written` string. No parsed `Decimal` is ever
stored, so a change here changes every replayed decision rather than disagreeing with a
number frozen into an old record.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_DIGITS = frozenset("0123456789")
_SEPARATORS = frozenset(",.")
_PERMITTED = _DIGITS | _SEPARATORS

# Stripped before parsing. Upper-cased comparison, longest forms first.
_CURRENCY_TOKENS = ("EUR", "USD", "GBP", "CHF", "€", "$", "£")

_TWO_PLACES = Decimal("0.01")
_DECIMAL_DIGIT_COUNT = 2
_GROUPING_DIGIT_COUNT = 3


@dataclass(frozen=True, slots=True)
class ParsedAmount:
    """The outcome of reading one written figure.

    `value is None` means the figure could not be read at all (rule H16).
    `is_format_ambiguous` means a value was produced but its decimal convention was
    guessed from digit counts alone (rule H13). Both escalate.
    """

    value: Decimal | None
    is_format_ambiguous: bool


_UNREADABLE = ParsedAmount(value=None, is_format_ambiguous=True)


def parse_eur_amount(raw: str) -> ParsedAmount:
    """Read one figure as written. Total: never raises, for any input."""
    compact, negative = _strip_decoration(raw)
    if not compact or any(ch not in _PERMITTED for ch in compact):
        return _UNREADABLE

    has_comma = "," in compact
    has_dot = "." in compact
    if has_comma and has_dot:
        parsed = _parse_with_both_separators(compact)
    elif has_comma or has_dot:
        parsed = _parse_with_one_separator(compact, "," if has_comma else ".")
    else:
        parsed = _parse_without_separator(compact)

    if parsed.value is None or not negative:
        return parsed
    return ParsedAmount(value=-parsed.value, is_format_ambiguous=parsed.is_format_ambiguous)


def _strip_decoration(raw: str) -> tuple[str, bool]:
    """Remove currency decoration, whitespace and a leading sign."""
    text = unicodedata.normalize("NFKC", raw).upper()
    for token in _CURRENCY_TOKENS:
        text = text.replace(token, "")
    compact = "".join(text.split())
    negative = compact.startswith("-")
    if compact[:1] in {"+", "-"}:
        compact = compact[1:]
    return compact, negative


def _parse_with_both_separators(compact: str) -> ParsedAmount:
    """Both conventions present, so the last separator is unambiguously the decimal one."""
    decimal_separator = "," if compact.rfind(",") > compact.rfind(".") else "."
    grouping_separator = "." if decimal_separator == "," else ","
    integer_part, _, fraction = compact.rpartition(decimal_separator)
    if decimal_separator in integer_part:
        return _UNREADABLE
    if len(fraction) != _DECIMAL_DIGIT_COUNT:
        return _UNREADABLE
    if not _groups_are_well_formed(integer_part, grouping_separator):
        return _UNREADABLE
    return _build(
        integer_part.replace(grouping_separator, ""), fraction, is_format_ambiguous=False
    )


def _groups_are_well_formed(integer_part: str, grouping_separator: str) -> bool:
    """Do the digit runs either side of the grouping separator form real thousands groups?

    Both conventions this parser handles group in threes, so ``4,25.00`` is neither
    European nor US — it is a figure we do not know how to read. Stripping the separator
    anyway would turn it into 425.00 with no flag raised, which is the one thing the
    module promises never to do: resolve an ambiguity silently. A run of the wrong length
    is treated as unreadable so H16 escalates it.
    """
    leading, *rest = integer_part.split(grouping_separator)
    if not rest:
        return True
    if not 1 <= len(leading) <= _GROUPING_DIGIT_COUNT:
        return False
    return all(len(group) == _GROUPING_DIGIT_COUNT for group in rest)


def _parse_with_one_separator(compact: str, separator: str) -> ParsedAmount:
    if compact.count(separator) != 1:
        return _UNREADABLE
    head, _, tail = compact.partition(separator)
    if len(tail) == _DECIMAL_DIGIT_COUNT:
        return _build(head, tail, is_format_ambiguous=False)
    if len(tail) == _GROUPING_DIGIT_COUNT:
        # EUR carries two decimal places, so three trailing digits cannot be a fraction.
        # The figure is readable, but its convention was inferred — H13 escalates it.
        return _build(head + tail, "", is_format_ambiguous=True)
    return _UNREADABLE


def _parse_without_separator(compact: str) -> ParsedAmount:
    return _build(compact, "", is_format_ambiguous=False)


def _build(integer_part: str, fraction: str, *, is_format_ambiguous: bool) -> ParsedAmount:
    if not integer_part or not _all_digits(integer_part):
        return _UNREADABLE
    # Defensive: no separator can follow the decimal one, so the fraction is always
    # digits by the time it reaches here.
    if fraction and not _all_digits(fraction):  # pragma: no cover
        return _UNREADABLE
    try:
        value = Decimal(f"{integer_part}.{fraction or '0'}").quantize(_TWO_PLACES)
    except InvalidOperation:  # pragma: no cover - guarded by the digit checks above
        return _UNREADABLE
    return ParsedAmount(value=value, is_format_ambiguous=is_format_ambiguous)


def _all_digits(text: str) -> bool:
    # str.isdigit() accepts superscripts and other numeric forms Decimal would reject.
    return all(ch in _DIGITS for ch in text)
