"""The four seams between the pure decision and the world that feeds it.

Each is a `Protocol`, so an adapter conforms by shape rather than by inheritance and the
domain never learns that a database or a model provider exists.

`ClockPort` and `IdGeneratorPort` are separate ports for one reason: `Decision` takes its
id and its timestamp as arguments instead of generating them, so record construction is
deterministic in tests. That is what makes the replay assertion clean — a replayed record
differs from the original in nothing at all, not even its identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from garnishment.domain.decision import Decision
from garnishment.domain.extraction import LetterExtraction


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Everything one call to the reader produced.

    `LetterExtraction` deliberately carries no provenance — it is the replay input, and
    which provider answered is a fact about the call, not about the letter. The three
    fields below are exactly what `Decision.from_routing` needs and the domain does not.
    """

    extraction: LetterExtraction
    """The replay input. Carries `extraction_status`, so a malformed reading arrives here
    as data rather than as an exception."""

    raw_llm_response: Mapping[str, Any]
    """The tool-call payload, verbatim, so a stored decision can be read against what the
    model actually said rather than against our reshaping of it.

    When the reader never obtained a payload at all — every attempt was a refusal, a
    truncated reply, or no tool call — this instead carries a small diagnostic mapping
    naming what happened, under a `_diagnostic` key. An empty mapping means the reader
    recorded nothing, which no adapter in this repository does."""

    provider: str
    """`"anthropic"` or `"fake"`; recorded so a replayed decision names its reader."""

    notes: str = field(default="")
    """Free text the reader recorded and routing never read."""


class ExtractorPort(Protocol):
    """Turn a letter into an extraction. The only probabilistic step in the system."""

    @property
    def provider(self) -> str:
        """Identifies this reader in every record it produces."""
        ...

    async def extract(self, letter_text: str) -> ExtractionResult:
        """Read one letter.

        Raises `ExtractorUnavailable` when the reader never answered — transport failure,
        authentication failure, or rate limiting that survived the client's own retries.

        Returns, and never raises, when the reader answered but the answer did not fit the
        schema after one identical retry: the result then carries `extraction_status =
        MALFORMED_AFTER_RETRY`, which escalates under H01. Our own malfunction must never
        reject a court order.
        """
        ...


class DecisionRepositoryPort(Protocol):
    """Append-only storage for decisions. There is no update and no delete."""

    async def save(self, decision: Decision) -> None: ...

    async def get(self, decision_id: str) -> Decision | None: ...


class ClockPort(Protocol):
    def now(self) -> datetime: ...


class IdGeneratorPort(Protocol):
    def new_id(self) -> str: ...
