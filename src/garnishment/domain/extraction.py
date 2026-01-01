"""The stored, replayable input to the decision boundary.

A `LetterExtraction` is everything the probabilistic half of the system produced, and it
is the *only* argument `route()` takes. It therefore carries `letter_text`, so grounding
can be recomputed inside the domain on every routing run rather than being handed in by
an adapter as a finished number.

**No parsed `Decimal` appears anywhere in this module.** `AmountCandidate.value_as_written`
is the single stored representation of a figure; `parse_eur_amount` derives the number at
routing time. Storing both a verbatim string and a parsed value would give one quantity
two representations that could drift apart across a schema change, and replay would then
be asserting against the wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from garnishment.domain.enums import AmountRole, DocumentType, ExtractionStatus
from garnishment.domain.values import EUR, CaseReference, DebtorName, EvidenceQuote, Iban


@dataclass(frozen=True, slots=True)
class ExtractedField[T]:
    """One value the reader claims to have found, with its citation and self-report."""

    value: T | None
    evidence_quote: EvidenceQuote | None = None
    llm_confidence: float = 0.0

    @property
    def quote_text(self) -> str | None:
        return None if self.evidence_quote is None else self.evidence_quote.text


@dataclass(frozen=True, slots=True)
class AmountCandidate:
    """One monetary figure found in the letter, with the part the reader thinks it plays.

    The reader enumerates every figure and labels each one's role; it is never asked which
    figure is legally operative. That judgement belongs to a versioned doctrine in
    `operative_amount`, where it is testable and recorded.
    """

    value_as_written: str
    """Verbatim from the letter, e.g. ``"4,250.00"``. The only stored form of this figure."""

    currency: str = EUR
    raw_text: str = ""
    """The full figure as written, e.g. ``"EUR 4,250.00"``."""

    role: AmountRole = AmountRole.UNKNOWN
    role_confidence: float = 0.0
    is_approximate: bool = False
    evidence_quote: EvidenceQuote | None = None
    llm_confidence: float = 0.0

    @property
    def quote_text(self) -> str | None:
        return None if self.evidence_quote is None else self.evidence_quote.text


@dataclass(frozen=True, slots=True)
class LetterExtraction:
    """Everything the reader produced for one letter."""

    letter_text: str
    document_type: DocumentType = DocumentType.OTHER
    is_seizure_order: bool = False
    is_seizure_order_confidence: float = 0.0
    debtor_name: ExtractedField[DebtorName] = field(
        default_factory=lambda: ExtractedField[DebtorName](None)
    )
    debtor_name_is_partial: bool = False
    """The reader's own signal, OR-ed with our substantive-token rule in H05."""

    case_reference: ExtractedField[CaseReference] = field(
        default_factory=lambda: ExtractedField[CaseReference](None)
    )
    account_iban: ExtractedField[Iban] = field(default_factory=lambda: ExtractedField[Iban](None))
    amount_candidates: tuple[AmountCandidate, ...] = ()
    model_id: str = ""
    prompt_version: str = ""
    extraction_status: ExtractionStatus = ExtractionStatus.OK


def candidate_at(
    extraction: LetterExtraction, index: int | None
) -> AmountCandidate | None:
    """Look a candidate up by index, tolerating None and out-of-range values.

    Indices travel with a `Resolution`, which may have been built from a different
    candidate list after a schema change; a lookup that raises would turn a routing
    outcome into an exception.
    """
    if index is None:
        return None
    if index < 0 or index >= len(extraction.amount_candidates):
        return None
    return extraction.amount_candidates[index]
