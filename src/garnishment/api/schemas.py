"""Request and response DTOs.

This module is a **serialisation layer over `Decision`**, and nothing more. It derives no
value, applies no threshold and makes no judgement: every number it emits was already
computed by `route()` and stored on the record. The one thing it does decide is shape.

`amount.value` is the resolved operative amount, not `value_as_written`. The verbatim
figure stays in the `extraction` block, where it belongs -- it is the replay input -- and
the response also carries the figure the doctrine actually settled on, because that is the
number a clerk acts against.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator

from garnishment.domain.confidence import (
    ACCOUNT_IBAN,
    AMOUNT,
    CASE_REFERENCE,
    DEBTOR_NAME,
    FieldConfidence,
)
from garnishment.domain.decision import Decision
from garnishment.domain.extraction import AmountCandidate, candidate_at

MAX_LETTER_CHARS = 100_000
"""Beyond this a letter is not a letter. Over-length is 413, not 422: the body is
well-formed, there is simply too much of it.

The limit is *not* declared as the field's `max_length`, deliberately: Pydantic would
reject an over-long letter as a schema violation and the response would become a 422,
which is the wrong statement about a well-formed body and would contradict the published
error table."""

MAX_REQUEST_BYTES = MAX_LETTER_CHARS * 4
"""The coarse ceiling the middleware enforces on `Content-Length`, before any body is read.

Four bytes per character is UTF-8's worst case, so nothing this rejects could have been a
letter within `MAX_LETTER_CHARS`. The precise check still happens after parsing; this one
exists only so a multi-megabyte body is refused at the door rather than decoded first."""


class LetterRequest(BaseModel):
    """One garnishment letter, as plain text. OCR and PDF are out of scope."""

    letter_text: str = Field(description="The full text of the garnishment letter.")

    @field_validator("letter_text")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("letter_text must not be empty or whitespace-only.")
        return value


class ConfidenceBreakdown(BaseModel):
    """The four parts of a field's confidence. `final` is the `min()` of the other three."""

    llm: float
    grounding: float
    validator: float
    final: float

    @classmethod
    def of(cls, confidence: FieldConfidence) -> Self:
        return cls(
            llm=confidence.llm,
            grounding=confidence.grounding,
            validator=confidence.validator,
            final=confidence.final,
        )


class FieldReport(BaseModel):
    value: str | None
    evidence_quote: str | None
    confidence: ConfidenceBreakdown


class AmountResolutionReport(BaseModel):
    """Which figure the doctrine settled on, and under which doctrine.

    `considered` and `source` differ whenever the letter pointed at a figure that could
    not be read, so a clerk sees both that a figure was meant and that it did not parse.
    """

    reason: str
    role: str | None
    role_confidence: float | None
    considered_candidate_index: int | None
    source_candidate_index: int | None
    doctrine_version: str


class AmountReport(FieldReport):
    currency: str | None
    resolution: AmountResolutionReport


class DecisionFields(BaseModel):
    debtor_name: FieldReport
    case_reference: FieldReport
    amount: AmountReport
    account_iban: FieldReport


class FiredRuleReport(BaseModel):
    """`route` is null for informational rules, which are recorded but never decisive."""

    rule_id: str
    route: str | None
    reason: str


class AmountCandidateReport(BaseModel):
    raw_text: str
    value_as_written: str
    currency: str
    role: str
    role_confidence: float
    is_approximate: bool
    confidence: float

    @classmethod
    def of(cls, candidate: AmountCandidate) -> Self:
        return cls(
            raw_text=candidate.raw_text,
            value_as_written=candidate.value_as_written,
            currency=candidate.currency,
            role=str(candidate.role),
            role_confidence=candidate.role_confidence,
            is_approximate=candidate.is_approximate,
            confidence=candidate.llm_confidence,
        )


class ExtractionReport(BaseModel):
    """The replay input, echoed. Every figure appears exactly once, as written."""

    document_type: str
    is_seizure_order: bool
    is_seizure_order_confidence: float
    extraction_status: str
    amount_candidates: list[AmountCandidateReport]
    model_id: str
    prompt_version: str
    routing_version: str


class DecisionResponse(BaseModel):
    """What `POST /decisions` returns."""

    id: str
    created_at: datetime
    route: str
    decisive_rule_id: str
    reasons: list[str]
    fired_rules: list[FiredRuleReport]
    fields: DecisionFields
    extraction: ExtractionReport

    @classmethod
    def of(cls, decision: Decision) -> Self:
        return cls(**_common_members(decision))


class DecisionDetailResponse(DecisionResponse):
    """What `GET /decisions/{id}` returns: the same body plus the audit tail.

    The tail is what makes the record defensible months later -- the exact bytes decided
    on, the model's verbatim answer, and which reader produced it.
    """

    letter_text: str
    letter_sha256: str
    raw_llm_response: dict[str, Any]
    extractor_provider: str
    doctrine_version: str

    @classmethod
    def of(cls, decision: Decision) -> Self:
        return cls(
            **_common_members(decision),
            letter_text=decision.letter_text,
            letter_sha256=decision.letter_sha256,
            raw_llm_response=dict(decision.raw_llm_response),
            extractor_provider=decision.extractor_provider,
            doctrine_version=decision.doctrine_version,
        )


def _common_members(decision: Decision) -> dict[str, Any]:
    extraction = decision.extraction
    return {
        "id": decision.id,
        "created_at": decision.created_at,
        "route": str(decision.route),
        "decisive_rule_id": str(decision.decisive_rule_id),
        "reasons": list(decision.reasons),
        "fired_rules": [
            FiredRuleReport(
                rule_id=str(fired.rule_id),
                route=None if fired.route is None else str(fired.route),
                reason=fired.reason,
            )
            for fired in decision.fired_rules
        ],
        "fields": _fields_of(decision),
        "extraction": ExtractionReport(
            document_type=str(extraction.document_type),
            is_seizure_order=extraction.is_seizure_order,
            is_seizure_order_confidence=extraction.is_seizure_order_confidence,
            extraction_status=str(extraction.extraction_status),
            amount_candidates=[
                AmountCandidateReport.of(candidate) for candidate in extraction.amount_candidates
            ],
            model_id=extraction.model_id,
            prompt_version=extraction.prompt_version,
            routing_version=decision.routing_version,
        ),
    }


def _fields_of(decision: Decision) -> DecisionFields:
    extraction = decision.extraction
    confidences = decision.field_confidences
    name = extraction.debtor_name.value
    case = extraction.case_reference.value
    iban = extraction.account_iban.value
    return DecisionFields(
        debtor_name=FieldReport(
            value=None if name is None else name.raw,
            evidence_quote=extraction.debtor_name.quote_text,
            confidence=ConfidenceBreakdown.of(confidences[DEBTOR_NAME]),
        ),
        case_reference=FieldReport(
            value=None if case is None else case.raw,
            evidence_quote=extraction.case_reference.quote_text,
            confidence=ConfidenceBreakdown.of(confidences[CASE_REFERENCE]),
        ),
        amount=_amount_report(decision),
        account_iban=FieldReport(
            value=None if iban is None else iban.compact,
            evidence_quote=extraction.account_iban.quote_text,
            confidence=ConfidenceBreakdown.of(confidences[ACCOUNT_IBAN]),
        ),
    )


def _amount_report(decision: Decision) -> AmountReport:
    """The amount field, reported against the figure the letter pointed at.

    Value and currency come from the resolved amount when there is one; role, quote and
    the candidate indices come from the *considered* candidate, which exists even when the
    figure never parsed.
    """
    considered = candidate_at(decision.extraction, decision.considered_candidate_index)
    amount = decision.operative_amount
    currency = amount.currency if amount is not None else (
        considered.currency if considered is not None else None
    )
    return AmountReport(
        value=None if amount is None else str(amount.amount),
        currency=currency,
        evidence_quote=None if considered is None else considered.quote_text,
        confidence=ConfidenceBreakdown.of(decision.field_confidences[AMOUNT]),
        resolution=AmountResolutionReport(
            reason=str(decision.operative_amount_reason),
            role=None if considered is None else str(considered.role),
            role_confidence=None if considered is None else considered.role_confidence,
            considered_candidate_index=decision.considered_candidate_index,
            source_candidate_index=decision.source_candidate_index,
            doctrine_version=decision.doctrine_version,
        ),
    )
