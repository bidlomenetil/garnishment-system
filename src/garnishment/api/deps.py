"""Dependency wiring.

Everything expensive -- settings, the engine, the extractor -- is built once in
`create_app`/lifespan and read off `app.state` here. The provider functions below are
therefore trivial, which is the point: each is a seam a test can replace with one line,

    app.dependency_overrides[get_extractor] = lambda: FakeExtractor(unavailable=True)

without touching the rest of the graph.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from garnishment.adapters.persistence.repository import SqlAlchemyDecisionRepository
from garnishment.adapters.system import SystemClock, UlidGenerator
from garnishment.application.ports import (
    ClockPort,
    DecisionRepositoryPort,
    ExtractorPort,
    IdGeneratorPort,
)
from garnishment.application.use_cases import DecideLetter, GetDecision
from garnishment.config import Settings


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_extractor(request: Request) -> ExtractorPort:
    """The one probabilistic component, and the one tests most often replace."""
    return cast(ExtractorPort, request.app.state.extractor)


def get_repository(request: Request) -> DecisionRepositoryPort:
    session_factory = cast(
        "async_sessionmaker[AsyncSession]", request.app.state.session_factory
    )
    return SqlAlchemyDecisionRepository(session_factory)


def get_clock() -> ClockPort:
    return SystemClock()


def get_id_generator() -> IdGeneratorPort:
    return UlidGenerator()


def get_decide_letter(
    extractor: Annotated[ExtractorPort, Depends(get_extractor)],
    repository: Annotated[DecisionRepositoryPort, Depends(get_repository)],
    clock: Annotated[ClockPort, Depends(get_clock)],
    id_generator: Annotated[IdGeneratorPort, Depends(get_id_generator)],
) -> DecideLetter:
    return DecideLetter(
        extractor=extractor, repository=repository, clock=clock, id_generator=id_generator
    )


def get_get_decision(
    repository: Annotated[DecisionRepositoryPort, Depends(get_repository)],
) -> GetDecision:
    return GetDecision(repository=repository)


DecideLetterDep = Annotated[DecideLetter, Depends(get_decide_letter)]
GetDecisionDep = Annotated[GetDecision, Depends(get_get_decision)]
