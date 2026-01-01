"""The replay guarantee, through a real SQLite round trip (plan §7.7, criterion 8).

For every golden case: build a decision, save it, reload it **out of the database**, and
re-run `route()` on the reloaded extraction. The freshly computed route, field
confidences, fired rules and decisive rule must match what the reloaded record carries.

Re-routing the in-memory extraction would only re-test purity, which the unit suite
already covers. Routing the *reloaded* extraction is what proves serialisation is
lossless, because a field silently dropped or coarsened on the way through JSON changes
what the rules see and the two answers diverge.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from garnishment.adapters.persistence.repository import SqlAlchemyDecisionRepository
from garnishment.adapters.persistence.session import (
    build_engine,
    build_session_factory,
    init_models,
)
from garnishment.domain.routing import route
from tests.fixtures.extractions import GOLDEN_CASES, GoldenCase
from tests.integration.test_repository import build_decision


@pytest.fixture
async def repository(
    tmp_path_factory: pytest.TempPathFactory,
) -> AsyncIterator[SqlAlchemyDecisionRepository]:
    """A real SQLite file on disk: the round trip crosses a process boundary's worth of
    serialisation, not just an ORM identity map."""
    database_path = tmp_path_factory.mktemp("replay") / "decisions.db"
    engine = build_engine(f"sqlite+aiosqlite:///{database_path}")
    await init_models(engine)
    try:
        yield SqlAlchemyDecisionRepository(build_session_factory(engine))
    finally:
        await engine.dispose()


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.name)
async def test_replay_reproduces_the_decision_after_a_sqlite_round_trip(
    repository: SqlAlchemyDecisionRepository, case: GoldenCase
) -> None:
    stored = build_decision(case.extraction, decision_id=f"01{case.name[:24].upper():0>24}")

    await repository.save(stored)
    reloaded = await repository.get(stored.id)
    assert reloaded is not None

    replayed = route(reloaded.extraction)

    assert replayed.route == reloaded.route
    assert replayed.field_confidences == reloaded.field_confidences
    assert replayed.fired_rules == reloaded.fired_rules
    assert replayed.decisive_rule_id == reloaded.decisive_rule_id

    # And the record still says what the golden table expects it to say.
    assert reloaded.route is case.expected_route
    assert reloaded.decisive_rule_id is case.expected_decisive_rule
    assert {f.rule_id for f in reloaded.fired_rules} == set(case.expected_fired_rules)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.name)
async def test_replay_reproduces_the_derived_amount_after_a_sqlite_round_trip(
    repository: SqlAlchemyDecisionRepository, case: GoldenCase
) -> None:
    """The amount is derived from `value_as_written` at routing time, so a coarsened
    string in the stored extraction would show up here as a changed `Decimal`."""
    stored = build_decision(case.extraction, decision_id=f"02{case.name[:24].upper():0>24}")

    await repository.save(stored)
    reloaded = await repository.get(stored.id)
    assert reloaded is not None

    replayed = route(reloaded.extraction)

    assert replayed.operative_amount == reloaded.operative_amount
    assert replayed.operative_amount_reason is reloaded.operative_amount_reason
    assert replayed.considered_candidate_index == reloaded.considered_candidate_index
    assert replayed.source_candidate_index == reloaded.source_candidate_index
    assert replayed.doctrine_version == reloaded.doctrine_version
    assert replayed.reasons == reloaded.reasons
