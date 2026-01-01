"""Async engine and session factory. Nothing here knows what a `Decision` is."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from garnishment.adapters.persistence.models import Base


def build_engine(database_url: str) -> AsyncEngine:
    """Build an async engine for a URL. Takes a plain string so any caller can use it.

    An in-memory SQLite database lives inside a single connection, so the default pool —
    which opens a fresh connection per checkout — would hand back an empty schema on the
    next statement. `StaticPool` keeps one connection for the engine's lifetime.
    """
    kwargs: dict[str, Any] = {}
    if ":memory:" in database_url:
        kwargs["poolclass"] = StaticPool
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_async_engine(database_url, **kwargs)


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_models(engine: AsyncEngine) -> None:
    """Create the schema. Called once from the API's lifespan hook."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
