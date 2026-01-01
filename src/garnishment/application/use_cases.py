"""The two operations the service offers, each as thin as it can be.

`DecideLetter` is deliberately boring: read, route, record, return. Every judgement it
could have made lives in `route()` instead, which is why this file has no branches on
confidence, no thresholds and no knowledge of what any rule means. The one thing it does
decide is what to do when the reader fails, and that decision is made by *not* catching
`ExtractorUnavailable`: no record is written for a reading that never happened.

It also times its own two outbound calls. That is measurement rather than judgement --
nothing here branches on a duration -- and it lives here because this is the only layer
that can see where the boundary between the reader and the database falls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from garnishment.application.errors import DecisionNotFound
from garnishment.application.ports import (
    ClockPort,
    DecisionRepositoryPort,
    ExtractorPort,
    IdGeneratorPort,
)
from garnishment.domain.decision import Decision
from garnishment.domain.routing import route


@dataclass(frozen=True, slots=True)
class DecidedLetter:
    """A decision, plus how long the two calls outside this process took.

    The timings travel with the decision rather than being logged here, because the log
    line is emitted once at the edge and the edge is the only place that knows the total.
    """

    decision: Decision
    llm_ms: float
    db_ms: float


@dataclass(frozen=True, slots=True)
class DecideLetter:
    extractor: ExtractorPort
    repository: DecisionRepositoryPort
    clock: ClockPort
    id_generator: IdGeneratorPort

    async def execute(self, letter_text: str) -> DecidedLetter:
        """Read a letter, route it, store the record, hand it back.

        A malformed reading takes exactly this path too: it routes `needs_human` under
        H01 and is stored like any other decision. Only an *absent* reading skips the
        store, and it does so by raising out of here rather than by being handled.
        """
        started = time.perf_counter()
        result = await self.extractor.extract(letter_text)
        llm_ms = (time.perf_counter() - started) * 1000

        outcome = route(result.extraction)
        decision = Decision.from_routing(
            decision_id=self.id_generator.new_id(),
            created_at=self.clock.now(),
            extraction=result.extraction,
            outcome=outcome,
            extractor_provider=result.provider,
            raw_llm_response=result.raw_llm_response,
            notes=result.notes,
        )
        started = time.perf_counter()
        await self.repository.save(decision)
        db_ms = (time.perf_counter() - started) * 1000

        return DecidedLetter(decision=decision, llm_ms=llm_ms, db_ms=db_ms)


@dataclass(frozen=True, slots=True)
class GetDecision:
    repository: DecisionRepositoryPort

    async def execute(self, decision_id: str) -> Decision:
        decision = await self.repository.get(decision_id)
        if decision is None:
            raise DecisionNotFound(decision_id)
        return decision
