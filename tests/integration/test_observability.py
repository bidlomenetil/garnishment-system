"""One JSON line per decision, carrying the keys a log query needs.

A decision is the unit anyone searching these logs cares about, so the assertion is that
exactly one line is emitted -- not at least one. A line per internal stage would make
counting decisions a de-duplication exercise.

The rest of this file is about the logger being safe to embed: a host application that
configured the root logger must not receive every line twice, a host that attached its own
handler must still get the JSON formatter it asked for, and a timestamp must say which
zone it is in.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from garnishment import observability
from garnishment.adapters.llm.fake_extractor import FakeExtractor
from garnishment.api.app import create_app
from garnishment.api.deps import get_extractor
from garnishment.config import Settings
from garnishment.observability import (
    ERROR_DETAIL_CHARS,
    LOGGER_NAME,
    JsonLogFormatter,
    configure_logging,
)

SAMPLE = Path(__file__).resolve().parents[2] / "evals" / "samples" / "sample_01.txt"

REQUIRED_KEYS: dict[str, type | tuple[type, ...]] = {
    "decision_id": str,
    "route": str,
    "decisive_rule": str,
    "model_id": str,
    "prompt_version": str,
    "routing_version": str,
    "extraction_status": str,
    "letter_sha256": str,
    "provider": str,
    "latency_ms": (int, float),
    "llm_ms": (int, float),
    "db_ms": (int, float),
}


@pytest.fixture
def log_stream() -> Iterator[io.StringIO]:
    """Capture what the formatter actually writes, not the unformatted record.

    `caplog` would prove a log call happened; it would not prove the line is valid JSON,
    which is the half of this that a log pipeline depends on.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings(
        llm_provider="fake", database_url=f"sqlite+aiosqlite:///{tmp_path / 'decisions.db'}"
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def emitted_lines(stream: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_one_json_line_per_decision(client: TestClient, log_stream: io.StringIO) -> None:
    response = client.post(
        "/decisions", json={"letter_text": SAMPLE.read_text(encoding="utf-8")}
    )
    assert response.status_code == 201

    lines = emitted_lines(log_stream)
    assert len(lines) == 1

    line = lines[0]
    for key, expected_type in REQUIRED_KEYS.items():
        assert key in line, f"log line is missing {key!r}"
        assert isinstance(line[key], expected_type), f"{key!r} has the wrong type"

    assert line["decision_id"] == response.json()["id"]
    assert line["route"] == "auto"
    assert line["decisive_rule"] == "A00_AUTO_DEFAULT"
    assert line["extraction_status"] == "ok"
    assert line["latency_ms"] >= 0.0


def test_two_decisions_emit_two_lines(client: TestClient, log_stream: io.StringIO) -> None:
    letter = SAMPLE.read_text(encoding="utf-8")

    client.post("/decisions", json={"letter_text": letter})
    client.post("/decisions", json={"letter_text": letter})

    assert len(emitted_lines(log_stream)) == 2


def test_failed_requests_emit_no_decision_line(client: TestClient, log_stream: io.StringIO) -> None:
    """A rejected request is not a decision, so it must not look like one in the logs."""
    assert client.post("/decisions", json={"letter_text": "   "}).status_code == 422

    assert emitted_lines(log_stream) == []


@pytest.fixture
def restored_logger() -> Iterator[logging.Logger]:
    """Hand back the package logger exactly as it was found.

    These tests reconfigure process-global logging state on purpose, which is the thing
    under test; leaving it reconfigured would make every later test's log capture depend
    on execution order.
    """
    logger = logging.getLogger(LOGGER_NAME)
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate
    installed = observability._handler
    try:
        yield logger
    finally:
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate
        observability._handler = installed


def test_the_line_names_the_letter_and_the_reader(
    client: TestClient, log_stream: io.StringIO
) -> None:
    """`letter_sha256` ties a line to a letter without putting the letter in a log, and
    `provider` is what distinguishes a real reading from a replayed one."""
    response = client.post(
        "/decisions", json={"letter_text": SAMPLE.read_text(encoding="utf-8")}
    )
    detail = client.get(f"/decisions/{response.json()['id']}").json()

    line = emitted_lines(log_stream)[0]

    assert line["letter_sha256"] == detail["letter_sha256"]
    assert line["provider"] == "fake"
    assert SAMPLE.read_text(encoding="utf-8") not in json.dumps(line)


def test_the_two_stage_timings_sit_inside_the_total(
    client: TestClient, log_stream: io.StringIO
) -> None:
    """One `latency_ms` cannot tell a slow model from a slow disk, so the line carries
    both stages as well as the total they happened inside."""
    client.post("/decisions", json={"letter_text": SAMPLE.read_text(encoding="utf-8")})

    line = emitted_lines(log_stream)[0]

    assert line["llm_ms"] >= 0.0
    assert line["db_ms"] >= 0.0
    assert line["llm_ms"] + line["db_ms"] <= line["latency_ms"] + 1e-6


def test_the_timestamp_is_utc_with_an_offset(
    client: TestClient, log_stream: io.StringIO
) -> None:
    """A local timestamp with no offset is unreadable from anywhere else."""
    client.post("/decisions", json={"letter_text": SAMPLE.read_text(encoding="utf-8")})

    timestamp = emitted_lines(log_stream)[0]["timestamp"]
    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == UTC.utcoffset(None)


def test_the_logger_does_not_propagate_to_the_root(restored_logger: logging.Logger) -> None:
    """With `logging.basicConfig()` active on the root, propagation prints every decision
    twice: once as JSON and once in the root formatter's shape."""
    configure_logging()

    assert restored_logger.propagate is False


def test_reconfiguring_does_not_stack_handlers(restored_logger: logging.Logger) -> None:
    configure_logging()
    configure_logging()
    configure_logging()

    ours = [h for h in restored_logger.handlers if isinstance(h.formatter, JsonLogFormatter)]
    assert len(ours) == 1


def test_a_host_handler_neither_blocks_nor_is_blocked(
    restored_logger: logging.Logger,
) -> None:
    """The old guard skipped configuration whenever the logger had *any* handler, so a
    host that attached its own silently got no JSON at all."""
    host_handler = logging.NullHandler()
    restored_logger.addHandler(host_handler)

    configure_logging()

    assert host_handler in restored_logger.handlers
    assert any(isinstance(h.formatter, JsonLogFormatter) for h in restored_logger.handlers)


def test_the_configured_level_comes_from_settings(
    restored_logger: logging.Logger, tmp_path: Path
) -> None:
    create_app(
        Settings(
            llm_provider="fake",
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'decisions.db'}",
            log_level="WARNING",
        )
    )

    assert restored_logger.level == logging.WARNING


def test_an_unavailable_extractor_is_logged_as_an_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, restored_logger: logging.Logger
) -> None:
    """A 503 writes no record, so the log line is the only trace the request leaves."""
    settings = Settings(
        llm_provider="fake", database_url=f"sqlite+aiosqlite:///{tmp_path / 'decisions.db'}"
    )
    app = create_app(settings)
    app.dependency_overrides[get_extractor] = lambda: FakeExtractor(unavailable=True)
    # `propagate` is off by design, so `caplog`'s root handler has to be attached here.
    restored_logger.addHandler(caplog.handler)
    caplog.set_level(logging.ERROR, logger=LOGGER_NAME)

    with TestClient(app) as client:
        assert client.post("/decisions", json={"letter_text": "a letter"}).status_code == 503

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    event = errors[0].decision_event
    assert event["error"] == "ExtractorUnavailable"
    assert event["path"] == "/decisions"
    assert event["status"] == 503
    assert len(event["error_detail"]) <= ERROR_DETAIL_CHARS
