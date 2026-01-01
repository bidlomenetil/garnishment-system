"""Reshape a tool-call payload into a `LetterExtraction`. Nothing else happens here.

This module is the boundary line itself, and the discipline it keeps is the point of the
whole design: it does **no** grounding, **no** validation, **no** amount parsing and
**no** confidence arithmetic. `value_as_written` passes through as the string the model
wrote; a malformed IBAN is wrapped in `Iban` unexamined; a `certain` self-report becomes
0.97 by table lookup and is compared to nothing.

The one translation that does happen here is the bin-to-float lookup below, and it is
placed here precisely because it is not a judgement: the model answers in four words, the
domain speaks floats, and this is the seam between the two vocabularies.

All of that judgement happens inside `route()`, recomputed from the stored extraction on
every routing run. If it happened here and only the *result* were stored, replaying a
decision would re-read a number this adapter had already decided, and a bug in the
grounding function would be frozen into the record instead of caught by the replay test.

A test asserts this file imports none of `domain.grounding`, `domain.validators` or
`domain.amount_parsing`. That assertion is the structural half of the rule; this docstring
is the half that explains why breaking it would be a design change rather than a shortcut.
"""

from __future__ import annotations

from collections.abc import Mapping

from garnishment.adapters.llm.prompt import PROMPT_VERSION
from garnishment.adapters.llm.schema import (
    AmountCandidateModel,
    ConfidenceBin,
    ExtractedFieldModel,
    LlmExtraction,
)
from garnishment.domain.enums import AmountRole, DocumentType, ExtractionStatus
from garnishment.domain.extraction import AmountCandidate, ExtractedField, LetterExtraction
from garnishment.domain.values import CaseReference, DebtorName, EvidenceQuote, Iban

BIN_TO_CONFIDENCE: Mapping[ConfidenceBin, float] = {
    "certain": 0.97,
    "probable": 0.88,
    "uncertain": 0.60,
    "guess": 0.30,
}
"""The float each verbalised bin becomes, chosen against the thresholds it will meet.

The model picks the word; we pick the number. That split is the whole point: a threshold
now moves only when someone edits this table, never because a model sampled 0.84 on a
Tuesday and 0.86 on a Wednesday (DESIGN.md §11.7).

Every value is a prior in the sense `domain.thresholds` means it, and the spread matters
more than any single one, because `final = min(llm, grounding, validator)` — a bin only
ever decides a field when it is the smallest of the three.

`certain` 0.97 sits clear of the 0.85 floors and just under the grounding tiers
(0.90/0.95/1.00), so a confident self-report never becomes the binding term ahead of the
verifiable evidence — which is the ordering `domain.confidence` argues for.

`probable` 0.88 clears the 0.85 floor, deliberately with headroom rather than landing on
it. Placing it at exactly 0.85 would make `below_floor(0.85, 0.85)` a float-equality
coin-flip and would rebuild the knife edge the bins exist to remove — the same letter
reported the same way would route two ways. The substantive call is that `probable` is a
*pass*: the self-report is a coarse detector of doubt the validators cannot see, and real
doubt has its own word. A reading the model would not call doubtful, whose quote grounds
verbatim and whose validator passes, does not need a clerk.

`uncertain` 0.60 fails every 0.85 floor, so a stated doubt escalates on its own — and it
also fails `ROLE_MIN_CONFIDENCE`, so an unsure role label degrades resolution to
AMBIGUOUS instead of choosing a figure.

`guess` 0.30 fails the same floors with room to spare, and stays distinguishable from
`uncertain` in the stored record so a clerk can see which of the two they were handed. It
is not 0.00: that value is spoken for by `GROUNDING_ABSENT`, where it means *unsupported
by the letter*, which is a different and stronger claim than *the reader was unsure*.
"""


def bin_to_confidence(bin_value: ConfidenceBin) -> float:
    """The one place a verbalised self-report becomes a number the domain can compare."""
    return BIN_TO_CONFIDENCE[bin_value]


def _quote(text: str | None) -> EvidenceQuote | None:
    return None if text is None else EvidenceQuote(text)


def _candidate(model: AmountCandidateModel) -> AmountCandidate:
    return AmountCandidate(
        value_as_written=model.value_as_written,
        currency=model.currency,
        raw_text=model.raw_text,
        role=AmountRole(model.role),
        role_confidence=bin_to_confidence(model.role_confidence),
        is_approximate=model.is_approximate,
        evidence_quote=_quote(model.evidence_quote),
        llm_confidence=bin_to_confidence(model.confidence),
    )


def _case_reference(model: ExtractedFieldModel) -> ExtractedField[CaseReference]:
    return ExtractedField[CaseReference](
        None if model.value is None else CaseReference(model.value),
        _quote(model.evidence_quote),
        bin_to_confidence(model.confidence),
    )


def _iban(model: ExtractedFieldModel) -> ExtractedField[Iban]:
    return ExtractedField[Iban](
        None if model.value is None else Iban(model.value),
        _quote(model.evidence_quote),
        bin_to_confidence(model.confidence),
    )


def map_llm_extraction(
    llm: LlmExtraction,
    *,
    letter_text: str,
    model_id: str,
    prompt_version: str = PROMPT_VERSION,
    extraction_status: ExtractionStatus = ExtractionStatus.OK,
) -> LetterExtraction:
    """Wrap a validated payload in domain value objects, field for field."""
    return LetterExtraction(
        letter_text=letter_text,
        document_type=DocumentType(llm.document_type),
        is_seizure_order=llm.document_is_seizure_order,
        is_seizure_order_confidence=bin_to_confidence(
            llm.document_is_seizure_order_confidence
        ),
        debtor_name=ExtractedField[DebtorName](
            None if llm.debtor_name.value is None else DebtorName(llm.debtor_name.value),
            _quote(llm.debtor_name.evidence_quote),
            bin_to_confidence(llm.debtor_name.confidence),
        ),
        debtor_name_is_partial=llm.debtor_name.is_partial_or_abbreviated,
        case_reference=_case_reference(llm.case_reference),
        account_iban=_iban(llm.account_iban),
        amount_candidates=tuple(_candidate(c) for c in llm.amount_candidates),
        model_id=model_id,
        prompt_version=prompt_version,
        extraction_status=extraction_status,
    )


def malformed_extraction(
    *,
    letter_text: str,
    model_id: str,
    prompt_version: str = PROMPT_VERSION,
) -> LetterExtraction:
    """The extraction to route when the model answered but the answer never fit the schema.

    Every field is left at its empty default and `extraction_status` carries the failure,
    because we genuinely read nothing. The rule table guards every reject on a successful
    reading, so this escalates under H01 rather than refusing a court order we could not
    read — our own malfunction is never evidence that the letter is invalid.
    """
    return LetterExtraction(
        letter_text=letter_text,
        model_id=model_id,
        prompt_version=prompt_version,
        extraction_status=ExtractionStatus.MALFORMED_AFTER_RETRY,
    )
