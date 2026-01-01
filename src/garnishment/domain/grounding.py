"""Does the letter actually contain what the reader claims it contains?

Grounding is one of the three inputs to a field's confidence, and the strongest of them,
because it is checkable by hand against the stored letter. It is computed here — inside
the domain, from `LetterExtraction.letter_text` — and never handed in by an adapter, so
replaying a decision genuinely re-derives this number instead of re-reading a conclusion
somebody else drew.

Normalisers are **field-typed**. The default collapses whitespace and applies NFKC; the
IBAN normaliser additionally strips all whitespace and upper-cases both sides, because a
reader that returns the compact `DE44500105175407324931` must still ground against the
letter's grouped `DE44 5001 0517 5407 3249 31`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from garnishment.domain.thresholds import (
    GROUNDING_ABSENT,
    GROUNDING_CASE_INSENSITIVE,
    GROUNDING_EXACT,
    GROUNDING_VALUE_ONLY,
)

Normaliser = Callable[[str], str]

_WHITESPACE_RUN = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Default normaliser: NFKC, whitespace runs collapsed to one space, stripped."""
    return _WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def normalise_iban(text: str) -> str:
    """IBAN normaliser: NFKC, **all** whitespace removed, upper-cased."""
    return "".join(unicodedata.normalize("NFKC", text).split()).upper()


_FIELD_NORMALISERS: Mapping[str, Normaliser] = {
    "account_iban": normalise_iban,
}


def normaliser_for(field_name: str) -> Normaliser:
    return _FIELD_NORMALISERS.get(field_name, normalise_text)


@dataclass(frozen=True, slots=True)
class GroundingResult:
    score: float
    grounded_on_value_only: bool


def grounding_score(
    *,
    letter_text: str,
    evidence_quote: str | None,
    value_text: str | None,
    normalise: Normaliser,
) -> GroundingResult:
    """Score how well a claimed reading is supported by the letter.

    Four tiers, in order of decreasing evidence:

    ==========================================================  =====  ========
    Condition, after the field's normaliser                     Score  Recorded
    ==========================================================  =====  ========
    Normalised quote occurs verbatim in the normalised letter    1.00  --
    Occurs case-insensitively only                               0.95  --
    Quote absent, but the extracted value occurs (exact or       0.90  I03
    case-insensitive) in the letter
    Neither quote nor value occurs, or both are absent           0.00  H08
    ==========================================================  =====  ========

    The value-only tier accepts a case-insensitive match too — a reader that copies the
    value correctly but retypes its case (`"JORDAN A. MEYER"` for a letter that reads
    `"Jordan A. Meyer"`) has still found the right value; nothing here should escalate on
    a difference no clerk would call a defect. The quote-vs-value distinction is what
    `grounded_on_value_only` and `I03` exist to preserve.
    """
    letter = normalise(letter_text)
    quote = _prepared(evidence_quote, normalise)
    if quote is not None:
        if quote in letter:
            return GroundingResult(GROUNDING_EXACT, grounded_on_value_only=False)
        if quote.casefold() in letter.casefold():
            return GroundingResult(GROUNDING_CASE_INSENSITIVE, grounded_on_value_only=False)

    value = _prepared(value_text, normalise)
    if value is not None and (value in letter or value.casefold() in letter.casefold()):
        return GroundingResult(GROUNDING_VALUE_ONLY, grounded_on_value_only=True)

    return GroundingResult(GROUNDING_ABSENT, grounded_on_value_only=False)


def _prepared(text: str | None, normalise: Normaliser) -> str | None:
    """Normalise, treating blank input as absent — an empty needle matches everything."""
    if text is None:
        return None
    normalised = normalise(text)
    return normalised or None
