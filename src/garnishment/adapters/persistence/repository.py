"""The `DecisionRepositoryPort` implementation, over async SQLAlchemy."""

from __future__ import annotations

from typing import Any

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from garnishment.adapters.persistence.models import DecisionRow
from garnishment.application.ports import DecisionRepositoryPort
from garnishment.domain.decision import Decision


class SqlAlchemyDecisionRepository(DecisionRepositoryPort):
    """Records are append-only: there is no update and no delete, and `save` cannot
    become one by accident.

    `Decision.id` is a ULID the caller generated once, so a save is an insert. It is
    written as an insert that *ignores* a primary-key conflict rather than as a `merge`,
    which is an upsert: a retry after a partial failure must be harmless, but a second
    save under an id already taken must not silently replace the record stored under it.
    A decision is a dated judgement, and the first one made under an id is the one the
    audit trail has to keep.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def save(self, decision: Decision) -> None:
        row = DecisionRow.from_decision(decision)
        values: dict[str, Any] = {
            column.name: getattr(row, column.name) for column in DecisionRow.__table__.columns
        }
        statement = sqlite_insert(DecisionRow).values(values).on_conflict_do_nothing(
            index_elements=["id"]
        )
        async with self._session_factory() as session:
            await session.execute(statement)
            await session.commit()

    async def get(self, decision_id: str) -> Decision | None:
        async with self._session_factory() as session:
            row = await session.get(DecisionRow, decision_id)
            return None if row is None else row.to_decision()
