"""The two endpoints. Both are thin: every judgement already happened in `route()`."""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from garnishment.api.deps import DecideLetterDep, GetDecisionDep
from garnishment.api.schemas import (
    MAX_LETTER_CHARS,
    DecisionDetailResponse,
    DecisionResponse,
    LetterRequest,
)
from garnishment.observability import log_decision

router = APIRouter(prefix="/decisions", tags=["decisions"])

PLAIN_TEXT = "text/plain"


async def _letter_from(request: Request) -> str:
    """The letter, from either body shape.

    `application/json` is the contract a caller integrates against. `text/plain` exists
    because a court letter is multi-line text, and JSON forbids a literal newline inside a
    string — so pasting a real letter into a JSON body fails on the first line break, which
    is a property of the format rather than anything about the letter. Accepting the raw
    body removes that step for anyone driving the service by hand, from `/docs` or `curl`.

    Both paths converge on the same string and the same rules: blank is 422, over-length is
    413, and nothing downstream can tell which content type was used. Parsing is manual
    because the body shape depends on a header, so failures are re-raised as the same
    `RequestValidationError` FastAPI would have produced — the 422 problem detail is then
    identical whichever content type was sent.
    """
    if request.headers.get("content-type", "").split(";")[0].strip() == PLAIN_TEXT:
        raw = (await request.body()).decode("utf-8", errors="replace")
        return _validated(lambda: LetterRequest(letter_text=raw))

    try:
        body = await request.json()
    except ValueError as error:
        raise RequestValidationError(
            [
                {
                    "type": "json_invalid",
                    "loc": ("body",),
                    "msg": "JSON decode error",
                    "input": None,
                    "ctx": {"error": str(error)},
                }
            ]
        ) from error
    return _validated(lambda: LetterRequest.model_validate(body))


def _validated(build: Callable[[], LetterRequest]) -> str:
    try:
        return build().letter_text
    except ValidationError as error:
        raise RequestValidationError(error.errors()) from error


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=DecisionResponse,
    summary="Decide one garnishment letter",
    openapi_extra={
        "requestBody": {
            "required": True,
            "description": (
                "The letter, either as JSON or as raw text. `text/plain` takes the letter "
                "exactly as written, which is the easier one to paste by hand: JSON "
                "forbids a literal newline inside a string, so a multi-line letter has to "
                "have its line breaks escaped as `\\n` first."
            ),
            "content": {
                "application/json": {
                    "schema": LetterRequest.model_json_schema(),
                    "example": {"letter_text": "DISTRICT COURT OF EXAMPLETON\n..."},
                },
                PLAIN_TEXT: {
                    "schema": {"type": "string"},
                    "example": (
                        "DISTRICT COURT OF EXAMPLETON\n"
                        "Order of Attachment / Garnishment\n\n"
                        "Case reference: AB-1042/24\n\n"
                        "Debtor:        Jordan A. Meyer\n"
                        "Account (IBAN): DE44 5001 0517 5407 3249 31\n"
                        "Amount to seize: EUR 4,250.00\n"
                    ),
                },
            },
        }
    },
)
async def create_decision(
    request: Request,
    response: Response,
    decide: DecideLetterDep,
) -> DecisionResponse:
    """Read, route, record, return.

    Takes the letter as `{"letter_text": "..."}` (`application/json`) or as the raw body
    (`text/plain`); see `_letter_from`.

    A malformed model response comes back through here as a normal 201 carrying route
    `needs_human` under H01. Only an *absent* response escapes as `ExtractorUnavailable`,
    which the application-level handler turns into a 503 with nothing persisted.

    Submitting the same letter twice creates two records: idempotency is out of scope, and
    a decision is a dated judgement rather than a cached lookup.
    """
    letter_text = await _letter_from(request)
    if len(letter_text) > MAX_LETTER_CHARS:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"letter_text is {len(letter_text)} characters; the maximum is "
                f"{MAX_LETTER_CHARS}."
            ),
        )

    started = time.perf_counter()
    decided = await decide.execute(letter_text)
    log_decision(
        decided.decision,
        latency_ms=(time.perf_counter() - started) * 1000,
        llm_ms=decided.llm_ms,
        db_ms=decided.db_ms,
    )

    response.headers["Location"] = f"/decisions/{decided.decision.id}"
    return DecisionResponse.of(decided.decision)


@router.get(
    "/{decision_id}",
    response_model=DecisionDetailResponse,
    summary="Read one decision back, with its audit tail",
)
async def read_decision(decision_id: str, get: GetDecisionDep) -> DecisionDetailResponse:
    """Raises `DecisionNotFound`, which the application handler renders as a 404."""
    return DecisionDetailResponse.of(await get.execute(decision_id))
