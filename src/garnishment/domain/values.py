"""Value objects. Hand-written frozen dataclasses — the domain may not import pydantic."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

EUR = "EUR"
"""The only currency this deployment handles automatically; anything else trips H14."""

_TOKEN_EDGE_PUNCTUATION = ".,;:()[]{}\"'`"
# Both the straight and the typographic apostrophe appear in real names.
_TOKEN_INNER_ALLOWED = "-'’"  # noqa: RUF001


@dataclass(frozen=True, slots=True)
class Money:
    """A monetary figure.

    The constructor performs **no** validation. A non-positive figure, a figure with the
    wrong number of decimal places and a figure in the wrong currency must all reach the
    rule table as data so a named rule can report them; raising here would turn a routing
    outcome into an exception and lose the audit trail.
    """

    amount: Decimal
    currency: str = EUR

    def __str__(self) -> str:
        return f"{self.currency} {self.amount}"


@dataclass(frozen=True, slots=True)
class CaseReference:
    raw: str


@dataclass(frozen=True, slots=True)
class Iban:
    raw: str

    @property
    def compact(self) -> str:
        """Whitespace-stripped, upper-cased. The only form the checksum is computed over."""
        return "".join(self.raw.split()).upper()


@dataclass(frozen=True, slots=True)
class DebtorName:
    raw: str

    @property
    def substantive_tokens(self) -> tuple[str, ...]:
        """Name tokens that could identify a person.

        A token counts when, after stripping edge punctuation, it is at least two
        characters long and made only of letters plus internal hyphens or apostrophes.
        ``"Jordan A. Meyer"`` yields ``("Jordan", "Meyer")``; ``"M. Meyer"`` yields
        ``("Meyer",)``.
        """
        tokens: list[str] = []
        for raw_token in self.raw.split():
            token = raw_token.strip(_TOKEN_EDGE_PUNCTUATION)
            if len(token) < 2:
                continue
            if not all(ch.isalpha() or ch in _TOKEN_INNER_ALLOWED for ch in token):
                continue
            if not any(ch.isalpha() for ch in token):
                continue
            tokens.append(token)
        return tuple(tokens)


@dataclass(frozen=True, slots=True)
class EvidenceQuote:
    """A span the reader claims to have copied out of the letter."""

    text: str
