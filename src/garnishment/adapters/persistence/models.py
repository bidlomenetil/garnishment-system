"""The stored shape of a `Decision`, and the codec that keeps the round trip lossless.

One row per decision. The nested structures — the extraction, the per-field confidences,
the fired-rule list and the verbatim model payload — are JSON columns; SQLite keeps them as
opaque text, which is the accepted consequence of ADR-005.

Every figure is stored **as a string**, never as a float. `value_as_written` is the only
representation of a candidate figure inside the extraction JSON, and the derived
`operative_amount` — which lives below §7.7's marked line, as output rather than input —
is stored as its `Decimal` text plus its currency. `float()` anywhere on this path would
silently rewrite ``1267.00`` into something that is nearly but not exactly itself.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, Boolean, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from garnishment.domain.confidence import FieldConfidence
from garnishment.domain.decision import Decision
from garnishment.domain.enums import (
    AmountRole,
    DocumentType,
    ExtractionStatus,
    ResolutionReason,
    Route,
    RuleId,
)
from garnishment.domain.extraction import AmountCandidate, ExtractedField, LetterExtraction
from garnishment.domain.routing import FiredRule
from garnishment.domain.values import CaseReference, DebtorName, EvidenceQuote, Iban, Money


class Base(DeclarativeBase):
    pass


class DecisionRow(Base):
    """One decision, append-only. `id` is a ULID the caller generated once."""

    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(64))
    """ISO 8601 text: SQLite's DATETIME affinity would drop the timezone and microseconds."""

    letter_text: Mapped[str] = mapped_column(Text)
    letter_sha256: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(32))
    extractor_provider: Mapped[str] = mapped_column(String(32))
    raw_llm_response: Mapped[dict[str, Any]] = mapped_column(JSON)
    extraction: Mapped[dict[str, Any]] = mapped_column(JSON)

    # --- everything below is DERIVED OUTPUT of route(), stored as evidence --------------
    field_confidences: Mapped[dict[str, Any]] = mapped_column(JSON)
    operative_amount_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operative_amount_currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    operative_amount_reason: Mapped[str] = mapped_column(String(64))
    considered_candidate_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_candidate_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doctrine_version: Mapped[str] = mapped_column(String(32))
    fired_rules: Mapped[list[Any]] = mapped_column(JSON)
    decisive_rule_id: Mapped[str] = mapped_column(String(64))
    route: Mapped[str] = mapped_column(String(32))
    reasons: Mapped[list[Any]] = mapped_column(JSON)
    routing_version: Mapped[str] = mapped_column(String(32))
    notes: Mapped[str] = mapped_column(Text, default="")
    has_operative_amount: Mapped[bool] = mapped_column(Boolean, default=False)
    """Distinguishes "no amount" from "an amount that happened to serialise as null"."""

    @classmethod
    def from_decision(cls, decision: Decision) -> DecisionRow:
        amount = decision.operative_amount
        return cls(
            id=decision.id,
            created_at=decision.created_at.isoformat(),
            letter_text=decision.letter_text,
            letter_sha256=decision.letter_sha256,
            model_id=decision.model_id,
            prompt_version=decision.prompt_version,
            extractor_provider=decision.extractor_provider,
            raw_llm_response=dict(decision.raw_llm_response),
            extraction=encode_extraction(decision.extraction),
            field_confidences={
                name: _encode_field_confidence(fc)
                for name, fc in decision.field_confidences.items()
            },
            has_operative_amount=amount is not None,
            operative_amount_value=None if amount is None else str(amount.amount),
            operative_amount_currency=None if amount is None else amount.currency,
            operative_amount_reason=decision.operative_amount_reason.value,
            considered_candidate_index=decision.considered_candidate_index,
            source_candidate_index=decision.source_candidate_index,
            doctrine_version=decision.doctrine_version,
            fired_rules=[_encode_fired_rule(f) for f in decision.fired_rules],
            decisive_rule_id=decision.decisive_rule_id.value,
            route=decision.route.value,
            reasons=list(decision.reasons),
            routing_version=decision.routing_version,
            notes=decision.notes,
        )

    def to_decision(self) -> Decision:
        return Decision(
            id=self.id,
            created_at=datetime.fromisoformat(self.created_at),
            letter_text=self.letter_text,
            letter_sha256=self.letter_sha256,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            extractor_provider=self.extractor_provider,
            raw_llm_response=dict(self.raw_llm_response),
            extraction=decode_extraction(self.extraction),
            field_confidences={
                name: _decode_field_confidence(payload)
                for name, payload in self.field_confidences.items()
            },
            operative_amount=self._decode_operative_amount(),
            operative_amount_reason=ResolutionReason(self.operative_amount_reason),
            considered_candidate_index=self.considered_candidate_index,
            source_candidate_index=self.source_candidate_index,
            doctrine_version=self.doctrine_version,
            fired_rules=tuple(_decode_fired_rule(f) for f in self.fired_rules),
            decisive_rule_id=RuleId(self.decisive_rule_id),
            route=Route(self.route),
            reasons=tuple(self.reasons),
            routing_version=self.routing_version,
            notes=self.notes,
        )

    def _decode_operative_amount(self) -> Money | None:
        if not self.has_operative_amount or self.operative_amount_value is None:
            return None
        return Money(
            amount=Decimal(self.operative_amount_value),
            currency=self.operative_amount_currency or "",
        )


def encode_extraction(extraction: LetterExtraction) -> dict[str, Any]:
    """The replay input, as JSON. No parsed `Decimal` appears here — criterion 14."""
    return {
        "letter_text": extraction.letter_text,
        "document_type": extraction.document_type.value,
        "is_seizure_order": extraction.is_seizure_order,
        "is_seizure_order_confidence": extraction.is_seizure_order_confidence,
        "debtor_name": _encode_extracted_field(extraction.debtor_name),
        "debtor_name_is_partial": extraction.debtor_name_is_partial,
        "case_reference": _encode_extracted_field(extraction.case_reference),
        "account_iban": _encode_extracted_field(extraction.account_iban),
        "amount_candidates": [_encode_candidate(c) for c in extraction.amount_candidates],
        "model_id": extraction.model_id,
        "prompt_version": extraction.prompt_version,
        "extraction_status": extraction.extraction_status.value,
    }


def decode_extraction(payload: dict[str, Any]) -> LetterExtraction:
    return LetterExtraction(
        letter_text=payload["letter_text"],
        document_type=DocumentType(payload["document_type"]),
        is_seizure_order=payload["is_seizure_order"],
        is_seizure_order_confidence=payload["is_seizure_order_confidence"],
        debtor_name=_decode_extracted_field(payload["debtor_name"], DebtorName),
        debtor_name_is_partial=payload["debtor_name_is_partial"],
        case_reference=_decode_extracted_field(payload["case_reference"], CaseReference),
        account_iban=_decode_extracted_field(payload["account_iban"], Iban),
        amount_candidates=tuple(_decode_candidate(c) for c in payload["amount_candidates"]),
        model_id=payload["model_id"],
        prompt_version=payload["prompt_version"],
        extraction_status=ExtractionStatus(payload["extraction_status"]),
    )


def _encode_extracted_field(field: ExtractedField[Any]) -> dict[str, Any]:
    return {
        "value": None if field.value is None else field.value.raw,
        "evidence_quote": field.quote_text,
        "llm_confidence": field.llm_confidence,
    }


def _decode_extracted_field[T](payload: dict[str, Any], value_type: type[T]) -> ExtractedField[T]:
    raw = payload["value"]
    quote = payload["evidence_quote"]
    return ExtractedField[T](
        None if raw is None else value_type(raw),  # type: ignore[call-arg]
        None if quote is None else EvidenceQuote(quote),
        payload["llm_confidence"],
    )


def _encode_candidate(candidate: AmountCandidate) -> dict[str, Any]:
    return {
        "value_as_written": candidate.value_as_written,
        "currency": candidate.currency,
        "raw_text": candidate.raw_text,
        "role": candidate.role.value,
        "role_confidence": candidate.role_confidence,
        "is_approximate": candidate.is_approximate,
        "evidence_quote": candidate.quote_text,
        "llm_confidence": candidate.llm_confidence,
    }


def _decode_candidate(payload: dict[str, Any]) -> AmountCandidate:
    quote = payload["evidence_quote"]
    return AmountCandidate(
        value_as_written=payload["value_as_written"],
        currency=payload["currency"],
        raw_text=payload["raw_text"],
        role=AmountRole(payload["role"]),
        role_confidence=payload["role_confidence"],
        is_approximate=payload["is_approximate"],
        evidence_quote=None if quote is None else EvidenceQuote(quote),
        llm_confidence=payload["llm_confidence"],
    )


def _encode_field_confidence(confidence: FieldConfidence) -> dict[str, Any]:
    """`final` is deliberately not stored: it is `min()` of the three parts, and a stored
    total could disagree with them after a reload."""
    return {
        "llm": confidence.llm,
        "grounding": confidence.grounding,
        "validator": confidence.validator,
        "value_present": confidence.value_present,
        "grounded_on_value_only": confidence.grounded_on_value_only,
        "validator_code": confidence.validator_code,
    }


def _decode_field_confidence(payload: dict[str, Any]) -> FieldConfidence:
    return FieldConfidence(
        llm=payload["llm"],
        grounding=payload["grounding"],
        validator=payload["validator"],
        value_present=payload["value_present"],
        grounded_on_value_only=payload["grounded_on_value_only"],
        validator_code=payload["validator_code"],
    )


def _encode_fired_rule(fired: FiredRule) -> dict[str, Any]:
    return {
        "rule_id": fired.rule_id.value,
        "route": None if fired.route is None else fired.route.value,
        "reason": fired.reason,
    }


def _decode_fired_rule(payload: dict[str, Any]) -> FiredRule:
    return FiredRule(
        rule_id=RuleId(payload["rule_id"]),
        route=None if payload["route"] is None else Route(payload["route"]),
        reason=payload["reason"],
    )
