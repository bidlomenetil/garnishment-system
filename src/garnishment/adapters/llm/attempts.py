"""What one bad answer from a model looks like, for both adapters.

A model can answer badly in more than one way, and the four causes below land in the same
place — `MALFORMED_AFTER_RETRY` — which is exactly why the adapter has to say which one
happened. Two things carry that information out of the retry loop:

*A warning per failed attempt*, naming the attempt number, the provider and the cause.
Without it the record shows only the final outcome, and a schema break, a refusal, a
truncated reply and unparseable JSON are indistinguishable in the logs.

*A diagnostic payload*, stored as `raw_llm_response` when no attempt ever produced a
parseable payload. Two refusals used to leave that field `{}`, which is the one case where
a colleague debugging the break has nothing at all to look at.

The warning deliberately carries no payload: a cause and a count are safe to log, and
letter-derived text is not.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SCHEMA_VALIDATION_ERROR = "schema_validation_error"


@dataclass(frozen=True, slots=True)
class BadAnswer:
    """A completed call that did not yield a payload to validate."""

    cause: str
    """One of a small vocabulary — `refusal`, `incomplete`, `unparseable_json`,
    `no_tool_call` — so logs can be counted by cause."""

    payload: Mapping[str, Any]
    """What to store if no attempt ever parsed. Never empty."""


def log_failed_attempt(
    logger: logging.Logger, *, provider: str, attempt: int, attempts: int, cause: str
) -> None:
    logger.warning(
        "extraction attempt %s/%s failed (provider=%s cause=%s)",
        attempt,
        attempts,
        provider,
        cause,
    )
