"""The audit record: what was decided, from what, by which rule, under which doctrine.

Everything below the marked line is **derived output** of `route()`, recomputed from
`extraction` on every routing run and stored only so a colleague can read the "why"
without re-running anything. Nothing below the line is ever fed back in as an input.

The replay guarantee follows from that split: `route(decision.extraction)` must reproduce
`route`, `field_confidences`, `fired_rules` and `decisive_rule_id` exactly.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from garnishment.domain.confidence import FieldConfidence
from garnishment.domain.enums import ResolutionReason, Route, RuleId
from garnishment.domain.extraction import LetterExtraction
from garnishment.domain.routing import FiredRule, RoutingOutcome
from garnishment.domain.values import Money


def sha256_of(text: str) -> str:
    """Content hash of the letter, so a record can be tied to the exact bytes decided on."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Decision:
    """One decision, append-only. There is no amendment and no reversal."""

    id: str
    created_at: datetime
    letter_text: str
    letter_sha256: str
    model_id: str
    prompt_version: str
    extractor_provider: str
    raw_llm_response: Mapping[str, Any]
    extraction: LetterExtraction
    """THE replay input. Everything below is derived from it."""

    # --- everything below is DERIVED OUTPUT of route(), stored as evidence --------------
    field_confidences: Mapping[str, FieldConfidence]
    operative_amount: Money | None
    operative_amount_reason: ResolutionReason
    considered_candidate_index: int | None
    """Which figure the letter pointed at, whether or not it could be read."""

    source_candidate_index: int | None
    """Which figure actually parsed into the amount above."""

    doctrine_version: str
    fired_rules: tuple[FiredRule, ...]
    decisive_rule_id: RuleId
    route: Route
    reasons: tuple[str, ...]
    routing_version: str
    notes: str = field(default="")
    """Free text the reader recorded and routing never read."""

    @classmethod
    def from_routing(
        cls,
        *,
        decision_id: str,
        created_at: datetime,
        extraction: LetterExtraction,
        outcome: RoutingOutcome,
        extractor_provider: str,
        raw_llm_response: Mapping[str, Any],
        notes: str = "",
    ) -> Decision:
        """Assemble a record from an extraction, its routing outcome and call metadata.

        `decision_id` and `created_at` are supplied by the caller rather than generated
        here, so that record construction is deterministic in tests — which is what makes
        the replay assertion clean.
        """
        return cls(
            id=decision_id,
            created_at=created_at,
            letter_text=extraction.letter_text,
            letter_sha256=sha256_of(extraction.letter_text),
            model_id=extraction.model_id,
            prompt_version=extraction.prompt_version,
            extractor_provider=extractor_provider,
            raw_llm_response=raw_llm_response,
            extraction=extraction,
            field_confidences=outcome.field_confidences,
            operative_amount=outcome.operative_amount,
            operative_amount_reason=outcome.operative_amount_reason,
            considered_candidate_index=outcome.considered_candidate_index,
            source_candidate_index=outcome.source_candidate_index,
            doctrine_version=outcome.doctrine_version,
            fired_rules=outcome.fired_rules,
            decisive_rule_id=outcome.decisive_rule_id,
            route=outcome.route,
            reasons=outcome.reasons,
            routing_version=outcome.routing_version,
            notes=notes,
        )

    @property
    def fired_rule_ids(self) -> tuple[RuleId, ...]:
        return tuple(f.rule_id for f in self.fired_rules)
