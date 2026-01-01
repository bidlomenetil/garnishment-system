"""One JSON line per decision, on the standard library's `logging`.

The decision line is emitted once per decision, at the edge, not once per sub-step: a
decision is the unit anyone searching these logs cares about, and a line per internal
stage would make counting decisions a matter of de-duplicating log records.

The keys are chosen so that the two questions asked of this service after the fact --
"why did this letter go there" and "which version of the system decided it" -- are both
answerable from the log alone, without opening the database. `letter_sha256` is what ties
a line to a letter without ever putting the letter itself in a log, and the three timings
separate a slow model from a slow disk.

Failures use the same logger and the same formatter, so an outage and a decision land in
one stream. A failure line never carries the letter, the request body, or an untruncated
provider message: provider SDK error text routinely echoes the request it failed on.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from garnishment.domain.decision import Decision

LOGGER_NAME = "garnishment.decision"
EVENT_ATTRIBUTE = "decision_event"

ERROR_DETAIL_CHARS = 200
"""How much of an exception message reaches the log. Provider SDK errors quote the
request and the response body back at you, so the message is evidence, not narrative."""

logger = logging.getLogger(LOGGER_NAME)

_handler: logging.Handler | None = None
"""The handler *this module* attached, if any. Tracked by identity so re-configuring
replaces our own handler and leaves a host application's alone."""


class JsonLogFormatter(logging.Formatter):
    """Renders each record as a single-line JSON object.

    Anything attached under `decision_event` is merged into the top level, so the keys a
    log query filters on sit where a query expects them rather than nested under a message.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, EVENT_ATTRIBUTE, None)
        if isinstance(event, dict):
            payload.update(event)
        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def log_decision(
    decision: Decision,
    *,
    latency_ms: float,
    llm_ms: float,
    db_ms: float,
) -> None:
    """Emit the one line that records this decision was made.

    `latency_ms` is the whole request; `llm_ms` and `db_ms` are the two stages inside it
    that can be slow for reasons outside this service. Their sum is deliberately less than
    the total -- the remainder is routing and serialisation, which is the point of
    reporting all three.
    """
    logger.info(
        "decision recorded",
        extra={
            EVENT_ATTRIBUTE: {
                "decision_id": decision.id,
                "route": str(decision.route),
                "decisive_rule": str(decision.decisive_rule_id),
                "letter_sha256": decision.letter_sha256,
                "provider": decision.extractor_provider,
                "model_id": decision.model_id,
                "prompt_version": decision.prompt_version,
                "routing_version": decision.routing_version,
                "extraction_status": str(decision.extraction.extraction_status),
                "latency_ms": round(latency_ms, 3),
                "llm_ms": round(llm_ms, 3),
                "db_ms": round(db_ms, 3),
            }
        },
    )


def log_failure(
    exc: BaseException,
    *,
    message: str,
    path: str,
    status: int,
    exc_info: bool = False,
) -> None:
    """Emit one line for a request that ended in an error rather than a decision.

    Carries the exception's class and a truncated message, never the letter and never the
    request body. `exc_info` sends the traceback to the log for the one path -- an
    unanticipated 500 -- where the class name alone is not enough to act on.
    """
    logger.error(
        message,
        extra={
            EVENT_ATTRIBUTE: {
                "error": type(exc).__name__,
                "error_detail": str(exc)[:ERROR_DETAIL_CHARS],
                "path": path,
                "status": status,
            }
        },
        exc_info=exc_info,
    )


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach the JSON formatter to this logger, replacing our own handler if present.

    Scoped to our own logger rather than the root, so importing this package never
    reformats a host application's logs, and `propagate` is off so a host that called
    `logging.basicConfig()` does not get every line twice -- once as JSON and once in the
    root formatter's shape.

    Idempotency is tracked by the handler *this module* installed, not by asking whether
    the logger has any handler at all: a host that attached its own handler must still get
    the JSON formatter it asked for by calling this.
    """
    global _handler
    if _handler is not None:
        logger.removeHandler(_handler)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    _handler = handler
