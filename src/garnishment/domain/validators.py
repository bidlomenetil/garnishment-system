"""Mechanical checks on the *form* of an extracted value.

Each validator is a named pure function returning a binary score with a code and a
human-readable detail. Binary by design: a soft signal earns its own rule in the table,
where a clerk can see it, rather than being blended into a number where it disappears.

**What these prove, and what they do not.** Mod-97 admits roughly 1 in 97 random strings
of the right shape, so it is not a near-zero false-accept test in the abstract. The honest
claim is narrower and still strong: it catches essentially every single-character
transcription error and every adjacent-digit transposition, which are the realistic ways
an IBAN goes wrong in a typed court letter. It does not prove the account exists or that
it belongs to the debtor. Nothing available at this boundary could.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from garnishment.domain.thresholds import MIN_SUBSTANTIVE_NAME_TOKENS, ZERO_EUR
from garnishment.domain.values import EUR, DebtorName, Iban

PASS = 1.0
FAIL = 0.0

_CASE_REFERENCE_PATTERN = re.compile(r"^[A-Z]{2}-\d{4}/\d{2}$")
_IBAN_PATTERN = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]+$")
_IBAN_MIN_LENGTH = 15
_IBAN_MAX_LENGTH = 34
_IBAN_CHECKSUM_REMAINDER = 1


@dataclass(frozen=True, slots=True)
class ValidatorResult:
    score: float
    code: str
    detail: str


def validate_case_reference(raw: str | None) -> ValidatorResult:
    """Passes when the reference matches ``XX-0000/00`` after stripping."""
    if raw is None:
        return ValidatorResult(FAIL, "CASE_REFERENCE_ABSENT", "No case reference was extracted.")
    if _CASE_REFERENCE_PATTERN.match(raw.strip()) is None:
        return ValidatorResult(
            FAIL,
            "CASE_REFERENCE_MALFORMED",
            f"{raw!r} does not match the required format XX-0000/00.",
        )
    return ValidatorResult(PASS, "CASE_REFERENCE_OK", "Case reference matches XX-0000/00.")


def validate_iban(raw: str | None) -> ValidatorResult:
    """Passes on shape, length 15-34, and a mod-97 remainder of 1."""
    if raw is None:
        return ValidatorResult(FAIL, "IBAN_ABSENT", "No IBAN was extracted.")
    compact = Iban(raw).compact
    if not _IBAN_MIN_LENGTH <= len(compact) <= _IBAN_MAX_LENGTH:
        return ValidatorResult(
            FAIL, "IBAN_MALFORMED", f"{raw!r} is not between 15 and 34 characters long."
        )
    if _IBAN_PATTERN.match(compact) is None:
        return ValidatorResult(
            FAIL, "IBAN_MALFORMED", f"{raw!r} is not two letters, two digits, then alphanumerics."
        )
    if _mod_97(compact) != _IBAN_CHECKSUM_REMAINDER:
        return ValidatorResult(FAIL, "IBAN_CHECKSUM_FAILED", f"{raw!r} fails the mod-97 checksum.")
    return ValidatorResult(PASS, "IBAN_OK", "IBAN checksum valid.")


def validate_amount(value: Decimal | None, currency: str) -> ValidatorResult:
    """Passes when the figure is positive and in EUR.

    There is deliberately no decimal-places check. `parse_eur_amount` is the only way a
    figure becomes a `Decimal` in this system and it ends by quantizing to two places, so
    a scale check here could never be False through the real pipeline. A branch that can
    only be reached by calling this function directly is not a control; it is a test
    surface that reports coverage it does not have.
    """
    if value is None:
        return ValidatorResult(FAIL, "AMOUNT_ABSENT", "The figure could not be read as a number.")
    if not value.is_finite():  # pragma: no cover - the parser never emits NaN or infinity
        return ValidatorResult(FAIL, "AMOUNT_NOT_FINITE", "The figure is not a finite number.")
    if value <= ZERO_EUR:
        return ValidatorResult(FAIL, "AMOUNT_NOT_POSITIVE", f"{value} is not positive.")
    if currency != EUR:
        return ValidatorResult(FAIL, "AMOUNT_CURRENCY_NOT_EUR", f"{currency} is not EUR.")
    return ValidatorResult(PASS, "AMOUNT_OK", "Amount is a positive EUR figure.")


def validate_debtor_name(raw: str | None) -> ValidatorResult:
    """Passes on at least ``MIN_SUBSTANTIVE_NAME_TOKENS`` substantive name tokens."""
    if raw is None:
        return ValidatorResult(FAIL, "DEBTOR_NAME_ABSENT", "No debtor name was extracted.")
    tokens = DebtorName(raw).substantive_tokens
    if len(tokens) < MIN_SUBSTANTIVE_NAME_TOKENS:
        return ValidatorResult(
            FAIL,
            "DEBTOR_NAME_TOO_FEW_TOKENS",
            f"{raw!r} has {len(tokens)} substantive token(s); "
            f"{MIN_SUBSTANTIVE_NAME_TOKENS} are required.",
        )
    return ValidatorResult(
        PASS, "DEBTOR_NAME_OK", f"{raw!r} has {len(tokens)} substantive name tokens."
    )


def _mod_97(compact_iban: str) -> int:
    rearranged = compact_iban[4:] + compact_iban[:4]
    digits = "".join(str(int(ch, 36)) if ch.isalpha() else ch for ch in rearranged)
    return int(digits) % 97
