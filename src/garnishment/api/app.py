"""The FastAPI application factory and its error contract.

Every error leaves here as an RFC 9457 problem detail, and the status codes draw one
distinction that matters more than the rest: a model that answered *badly* is a 201 with
route `needs_human`, while a model that never answered at all is a 503 with no record
written. Our own malfunction must never be reported as the letter's invalidity, and an
outage must never fill a clerk's queue with empty work items.

The handlers that report a failure also log it. A 503 that leaves nothing in the database
and nothing in the log is an outage no one can reconstruct afterwards, which is the one
state worse than the outage itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from garnishment.adapters.llm.factory import build_extractor
from garnishment.adapters.persistence.session import (
    build_engine,
    build_session_factory,
    init_models,
)
from garnishment.api.routers import decisions
from garnishment.api.schemas import MAX_REQUEST_BYTES
from garnishment.application.errors import DecisionNotFound, ExtractorUnavailable
from garnishment.config import Settings, get_settings
from garnishment.observability import configure_logging, log_failure

PROBLEM_JSON = "application/problem+json"
PROBLEM_BASE = "https://garnishment.example/problems"


def problem_body(
    path: str,
    *,
    status: int,
    code: str,
    title: str,
    detail: str,
    **extra: Any,
) -> dict[str, Any]:
    """One RFC 9457 body, built in one place so every error looks the same."""
    body: dict[str, Any] = {
        "type": f"{PROBLEM_BASE}/{code}",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": path,
    }
    body.update(extra)
    return body


def problem_response(
    request: Request,
    *,
    status: int,
    code: str,
    title: str,
    detail: str,
    **extra: Any,
) -> JSONResponse:
    body = problem_body(
        str(request.url.path), status=status, code=code, title=title, detail=detail, **extra
    )
    return JSONResponse(status_code=status, content=body, media_type=PROBLEM_JSON)


class ContentLengthLimitMiddleware:
    """Refuse an absurd body on its declared size, before anything reads it.

    This is a doorman, not the length rule: the rule that a letter is at most
    `MAX_LETTER_CHARS` still lives in the router, still counts characters rather than
    bytes, and still produces the 413 the error table promises. What this adds is that a
    fifty-megabyte body is turned away on its `Content-Length` header rather than being
    received, decoded and parsed into memory first.

    A request with no `Content-Length` -- a chunked upload -- passes through untouched.
    Enforcing a ceiling on a stream means counting it as it arrives, which is a different
    piece of machinery and one this service, whose only body is a text field, does not
    need.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            declared = Headers(scope=scope).get("content-length")
            # Compare decimal strings without converting untrusted, arbitrarily long
            # headers to int (Python limits that conversion to a few thousand digits).
            size = declared.lstrip("0") or "0" if declared is not None else ""
            limit = str(self._max_bytes)
            if (
                declared is not None
                and declared.isascii()
                and declared.isdigit()
                and (len(size) > len(limit) or (len(size) == len(limit) and size > limit))
            ):
                body = problem_body(
                    scope.get("path", ""),
                    status=413,
                    code="request-too-large",
                    title="Request body too large",
                    detail=(
                        f"The declared request body exceeds the maximum of {self._max_bytes} bytes."
                    ),
                )
                response = JSONResponse(
                    status_code=413, content=body, media_type=PROBLEM_JSON
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Zero-argument by default, so `uvicorn --factory` can call it.

    The extractor is built here rather than per request, so an unknown `LLM_PROVIDER`
    fails the deployment instead of failing one letter at a time in production.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = build_engine(resolved.database_url)
        await init_models(engine)
        app.state.session_factory = build_session_factory(engine)
        try:
            yield
        finally:
            await engine.dispose()

    app = FastAPI(
        title="Garnishment decision service",
        version="0.1.0",
        summary="Reads a garnishment letter and routes it to auto, needs_human or reject.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.extractor = build_extractor(resolved)
    app.add_middleware(ContentLengthLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)
    app.include_router(decisions.router)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return problem_response(
            request,
            status=422,
            code="invalid-request",
            title="Invalid request body",
            detail="The request body did not satisfy the schema for this endpoint.",
            errors=[
                {"loc": [str(part) for part in error["loc"]], "msg": error["msg"]}
                for error in exc.errors()
            ],
        )

    @app.exception_handler(DecisionNotFound)
    async def _not_found(request: Request, exc: DecisionNotFound) -> JSONResponse:
        return problem_response(
            request,
            status=404,
            code="decision-not-found",
            title="Decision not found",
            detail=str(exc),
            decision_id=exc.decision_id,
        )

    @app.exception_handler(ExtractorUnavailable)
    async def _unavailable(request: Request, exc: ExtractorUnavailable) -> JSONResponse:
        # No record is written on this path, deliberately: a decision record must record a
        # real reading, and an outage is not one. The log line is therefore the only trace
        # this request leaves anywhere, which is why it is not optional.
        log_failure(
            exc,
            message="extractor unavailable",
            path=str(request.url.path),
            status=503,
        )
        return problem_response(
            request,
            status=503,
            code="extractor-unavailable",
            title="Extraction service unavailable",
            detail=(
                "The letter could not be read because the extraction provider did not "
                "respond. No decision was recorded; retry the request."
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem_response(
            request,
            status=exc.status_code,
            code=f"http-{exc.status_code}",
            title=str(exc.detail),
            detail=str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        """The one path where we do not know what went wrong, and say so to no one.

        The client gets a fixed sentence. An unanticipated exception's message is the one
        string in this service that could contain anything -- a connection URL, a fragment
        of the letter, a provider's echo of the request -- and a 500 body is not the place
        to find out. The traceback goes to the log instead, where it is ours.
        """
        log_failure(
            exc,
            message="unhandled exception",
            path=str(request.url.path),
            status=500,
            exc_info=True,
        )
        return problem_response(
            request,
            status=500,
            code="internal-error",
            title="Internal server error",
            detail="An unexpected error occurred.",
        )

    return app
