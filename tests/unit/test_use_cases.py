"""The use case decides one thing only: whether a failed reading leaves a record behind.

Everything else it does is assembly. The asymmetry it enforces — a malformed reading is
stored and escalated, an absent reading is stored nowhere — is the sharpest expression of
the design's rule that our own malfunction is never a statement about the letter.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from garnishment.adapters.llm.fake_extractor import REPLAYED_MODEL_ID, FakeExtractor
from garnishment.adapters.llm.prompt import PROMPT_VERSION
from garnishment.application.errors import DecisionNotFound, ExtractorUnavailable
from garnishment.application.ports import DecisionRepositoryPort
from garnishment.application.use_cases import DecideLetter, GetDecision
from garnishment.domain.decision import Decision, sha256_of
from garnishment.domain.enums import ExtractionStatus, Route, RuleId
from garnishment.domain.routing import route
from tests.fixtures.extractions import SAMPLE_1, SAMPLE_2, SAMPLE_3

FIXED_TIME = datetime(2026, 9, 22, 14, 3, 11, 482000, tzinfo=UTC)


class InMemoryRepository:
    def __init__(self) -> None:
        self.saved: list[Decision] = []

    async def save(self, decision: Decision) -> None:
        self.saved.append(decision)

    async def get(self, decision_id: str) -> Decision | None:
        return next((d for d in self.saved if d.id == decision_id), None)


class FixedClock:
    def now(self) -> datetime:
        return FIXED_TIME


class CountingIdGenerator:
    def __init__(self) -> None:
        self.issued = 0

    def new_id(self) -> str:
        self.issued += 1
        return f"decision-{self.issued:03d}"


def build(extractor: FakeExtractor) -> tuple[DecideLetter, InMemoryRepository]:
    repository = InMemoryRepository()
    use_case = DecideLetter(
        extractor=extractor,
        repository=repository,
        clock=FixedClock(),
        id_generator=CountingIdGenerator(),
    )
    return use_case, repository


async def test_a_decision_is_routed_stored_and_returned() -> None:
    use_case, repository = build(FakeExtractor())

    decision = (await use_case.execute(SAMPLE_3)).decision

    assert decision.route is Route.AUTO
    assert decision.decisive_rule_id is RuleId.A00_AUTO_DEFAULT
    assert repository.saved == [decision]


async def test_the_record_carries_its_own_provenance() -> None:
    use_case, _ = build(FakeExtractor())

    decision = (await use_case.execute(SAMPLE_1)).decision

    assert decision.id == "decision-001"
    assert decision.created_at == FIXED_TIME
    assert decision.letter_text == SAMPLE_1
    assert decision.letter_sha256 == sha256_of(SAMPLE_1)
    assert decision.extractor_provider == "fake"
    assert decision.model_id == REPLAYED_MODEL_ID
    assert decision.prompt_version == PROMPT_VERSION
    assert decision.raw_llm_response


async def test_the_stored_record_replays_to_the_same_decision() -> None:
    """Everything below the derived-output line is recomputed, never fed back in."""
    use_case, _ = build(FakeExtractor())

    decision = (await use_case.execute(SAMPLE_3)).decision
    replayed = route(decision.extraction)

    assert replayed.route is decision.route
    assert replayed.decisive_rule_id is decision.decisive_rule_id
    assert replayed.fired_rules == decision.fired_rules
    assert dict(replayed.field_confidences) == dict(decision.field_confidences)


async def test_each_call_gets_its_own_identity() -> None:
    use_case, repository = build(FakeExtractor())

    first = (await use_case.execute(SAMPLE_1)).decision
    second = (await use_case.execute(SAMPLE_1)).decision

    assert first.id != second.id
    assert len(repository.saved) == 2


async def test_a_malformed_reading_is_recorded_and_escalated_never_rejected() -> None:
    use_case, repository = build(FakeExtractor(malformed=True))

    decision = (await use_case.execute(SAMPLE_1)).decision

    assert decision.extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY
    assert decision.route is Route.NEEDS_HUMAN
    assert decision.decisive_rule_id is RuleId.H01_EXTRACTION_FAILED
    assert repository.saved == [decision]


async def test_an_absent_reading_raises_and_leaves_no_record() -> None:
    """A record must record a real reading; an outage is not an ambiguous letter."""
    use_case, repository = build(FakeExtractor(unavailable=True))

    with pytest.raises(ExtractorUnavailable):
        await use_case.execute(SAMPLE_1)

    assert repository.saved == []


async def test_the_labelled_samples_route_end_to_end_as_the_design_says() -> None:
    use_case, _ = build(FakeExtractor())

    routes = [
        (await use_case.execute(letter)).decision
        for letter in (SAMPLE_1, SAMPLE_2, SAMPLE_3)
    ]

    assert [d.route for d in routes] == [Route.AUTO, Route.NEEDS_HUMAN, Route.AUTO]
    assert [d.decisive_rule_id for d in routes] == [
        RuleId.A00_AUTO_DEFAULT,
        RuleId.H05_DEBTOR_NAME_PARTIAL,
        RuleId.A00_AUTO_DEFAULT,
    ]


async def test_the_two_outbound_calls_are_timed_separately() -> None:
    """The log line splits a slow reader from a slow disk, so the use case must measure
    each call rather than the request as a whole."""
    use_case, _ = build(FakeExtractor())

    decided = await use_case.execute(SAMPLE_1)

    assert decided.llm_ms >= 0.0
    assert decided.db_ms >= 0.0
    assert decided.decision.id == "decision-001"


async def test_a_stored_decision_can_be_read_back() -> None:
    use_case, repository = build(FakeExtractor())
    stored = (await use_case.execute(SAMPLE_1)).decision

    assert await GetDecision(repository).execute(stored.id) == stored


async def test_an_unknown_id_is_a_named_failure_rather_than_a_none() -> None:
    with pytest.raises(DecisionNotFound) as raised:
        await GetDecision(InMemoryRepository()).execute("no-such-decision")

    assert raised.value.decision_id == "no-such-decision"


def test_the_in_memory_repository_conforms_to_the_port() -> None:
    """Checked by mypy at the assignment; asserted here so the intent is visible."""
    repository: DecisionRepositoryPort = InMemoryRepository()

    assert repository is not None
