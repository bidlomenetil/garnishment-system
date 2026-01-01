"""The HTTP surface, end to end, with a fake reader and a real SQLite file.

The two tests that matter most here are `test_malformed_reading_is_recorded_as_needs_human`
and `test_unavailable_extractor_returns_503_and_writes_nothing`. Together they assert the
asymmetry the whole error design rests on: a reader that answered *badly* produces a
stored decision routed to a human, while a reader that never answered produces no record
at all. Getting these two confused would either reject court orders because of our own
malfunction, or fill a clerk's queue with work items for letters nobody ever read.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from garnishment.adapters.llm.fake_extractor import FakeExtractor
from garnishment.api.app import create_app
from garnishment.api.deps import get_extractor
from garnishment.api.schemas import MAX_LETTER_CHARS, MAX_REQUEST_BYTES
from garnishment.application.ports import ExtractionResult
from garnishment.config import Settings
from garnishment.domain.routing import ROUTING_VERSION
from garnishment.observability import LOGGER_NAME

SAMPLES = Path(__file__).resolve().parents[2] / "evals" / "samples"

SAMPLE_EXPECTATIONS = [
    ("sample_01.txt", "auto", "A00_AUTO_DEFAULT"),
    ("sample_02.txt", "needs_human", "H05_DEBTOR_NAME_PARTIAL"),
    ("sample_03.txt", "auto", "A00_AUTO_DEFAULT"),
]


def sample(name: str) -> str:
    return (SAMPLES / name).read_text(encoding="utf-8")


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "decisions.db"


@pytest.fixture
def settings(database_path: Path) -> Settings:
    return Settings(llm_provider="fake", database_url=f"sqlite+aiosqlite:///{database_path}")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A client over an app whose only fake is the reader; storage and routing are real."""
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def stored_decision_count(database_path: Path) -> int:
    """Count rows with a connection of our own.

    Asking the API whether it saved something would be asking the accused; a separate
    connection to the same file is the only answer worth asserting on.
    """
    if not database_path.exists():
        return 0
    connection = sqlite3.connect(database_path)
    try:
        return int(connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])
    finally:
        connection.close()


@pytest.mark.parametrize(("name", "expected_route", "expected_rule"), SAMPLE_EXPECTATIONS)
def test_sample_letters_route_as_labelled(
    client: TestClient, name: str, expected_route: str, expected_rule: str
) -> None:
    response = client.post("/decisions", json={"letter_text": sample(name)})

    assert response.status_code == 201
    body = response.json()
    assert body["route"] == expected_route
    assert body["decisive_rule_id"] == expected_rule
    assert response.headers["Location"] == f"/decisions/{body['id']}"


def test_created_decision_carries_fields_and_confidence_breakdown(client: TestClient) -> None:
    body = client.post("/decisions", json={"letter_text": sample("sample_03.txt")}).json()

    fields = body["fields"]
    assert fields["debtor_name"]["value"] == "Jordan A. Meyer"
    assert fields["case_reference"]["value"] == "AB-1042/24"
    assert fields["account_iban"]["value"] == "DE44500105175407324931"

    # The restriction doctrine picked the restricted figure over the original it cites.
    assert fields["amount"]["value"] == "1267.00"
    assert fields["amount"]["currency"] == "EUR"
    assert fields["amount"]["resolution"]["reason"] == "restriction_supersedes"
    assert fields["amount"]["resolution"]["source_candidate_index"] == 1

    for field_name in ("debtor_name", "case_reference", "amount", "account_iban"):
        confidence = fields[field_name]["confidence"]
        assert set(confidence) == {"llm", "grounding", "validator", "final"}
        assert confidence["final"] == pytest.approx(
            min(confidence["llm"], confidence["grounding"], confidence["validator"])
        )

    assert len(body["fired_rules"]) == len(body["reasons"])
    assert body["extraction"]["routing_version"] == ROUTING_VERSION
    assert len(body["extraction"]["amount_candidates"]) == 2


def test_get_returns_the_audit_tail(client: TestClient) -> None:
    letter = sample("sample_01.txt")
    created = client.post("/decisions", json={"letter_text": letter}).json()

    response = client.get(f"/decisions/{created['id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["letter_text"] == letter
    assert len(body["letter_sha256"]) == 64
    assert body["extractor_provider"] == "fake"
    assert body["doctrine_version"]
    assert body["raw_llm_response"]["document_type"] == "attachment_order"
    assert body["extraction"]["model_id"]
    assert body["extraction"]["prompt_version"]
    assert body["route"] == created["route"]
    assert body["fired_rules"] == created["fired_rules"]


def test_posting_the_same_letter_twice_creates_two_decisions(client: TestClient) -> None:
    """Idempotency is an explicit scope exclusion: a decision is a dated judgement."""
    letter = sample("sample_01.txt")

    first = client.post("/decisions", json={"letter_text": letter}).json()
    second = client.post("/decisions", json={"letter_text": letter}).json()

    assert first["id"] != second["id"]
    assert first["route"] == second["route"] == "auto"


@pytest.mark.parametrize("letter_text", ["", "   \n\t  "])
def test_blank_letter_is_422(client: TestClient, letter_text: str) -> None:
    response = client.post("/decisions", json={"letter_text": letter_text})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["status"] == 422


def test_missing_field_is_422(client: TestClient) -> None:
    assert client.post("/decisions", json={}).status_code == 422


def test_over_long_letter_is_413(client: TestClient) -> None:
    response = client.post("/decisions", json={"letter_text": "a" * (MAX_LETTER_CHARS + 1)})

    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["status"] == 413


def test_a_raw_letter_can_be_posted_as_text_plain(client: TestClient) -> None:
    """The reason this content type exists.

    A court letter is multi-line, and JSON forbids a literal newline inside a string, so
    pasting a real letter into a JSON body fails on its first line break. Posting it as
    `text/plain` takes it exactly as written.
    """
    letter = sample("sample_03.txt")
    assert "\n" in letter

    response = client.post(
        "/decisions", content=letter.encode("utf-8"), headers={"content-type": "text/plain"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["route"] == "auto"
    assert body["decisive_rule_id"] == "A00_AUTO_DEFAULT"
    assert body["fields"]["amount"]["value"] == "1267.00"


def test_both_content_types_produce_the_same_decision(client: TestClient) -> None:
    """Nothing downstream can tell which body shape was used."""
    letter = sample("sample_01.txt")

    as_json = client.post("/decisions", json={"letter_text": letter}).json()
    as_text = client.post(
        "/decisions", content=letter.encode("utf-8"), headers={"content-type": "text/plain"}
    ).json()

    assert as_json["id"] != as_text["id"]  # two submissions, two records
    for key in ("route", "decisive_rule_id", "reasons", "fields"):
        assert as_json[key] == as_text[key]


def test_a_blank_text_plain_body_is_still_422(client: TestClient) -> None:
    response = client.post(
        "/decisions", content=b"   \n\t ", headers={"content-type": "text/plain"}
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")


def test_an_over_long_text_plain_body_is_still_413(client: TestClient) -> None:
    response = client.post(
        "/decisions",
        content=b"a" * (MAX_LETTER_CHARS + 1),
        headers={"content-type": "text/plain"},
    )

    assert response.status_code == 413
    assert response.json()["status"] == 413


def test_malformed_json_is_still_a_422_problem_detail(client: TestClient) -> None:
    """The failure that motivated `text/plain`: a letter pasted with real newlines.

    It stays a 422 rather than becoming a 201, because a JSON content type still means
    the body has to be JSON.
    """
    response = client.post(
        "/decisions",
        content=b'{"letter_text": "line one\nline two"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["status"] == 422


def test_unknown_id_is_404(client: TestClient) -> None:
    response = client.get("/decisions/01ARZ3NDEKTSV4RRFFQ69G5FAV")

    assert response.status_code == 404
    body = response.json()
    assert body["status"] == 404
    assert body["decision_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"


def test_malformed_reading_is_recorded_as_needs_human(
    settings: Settings, database_path: Path
) -> None:
    """Our own failure escalates; it never rejects and never 503s.

    A rejection is a statement about the letter. Using it for a malformed model response
    would turn an outage into a stream of refused court orders and put false invalidity
    findings in the audit trail.
    """
    app = create_app(settings)
    app.dependency_overrides[get_extractor] = lambda: FakeExtractor(malformed=True)

    with TestClient(app) as client:
        response = client.post("/decisions", json={"letter_text": sample("sample_01.txt")})

    assert response.status_code == 201
    body = response.json()
    assert body["route"] == "needs_human"
    assert body["decisive_rule_id"] == "H01_EXTRACTION_FAILED"
    assert body["extraction"]["extraction_status"] == "malformed_after_retry"
    assert stored_decision_count(database_path) == 1


def test_unavailable_extractor_returns_503_and_writes_nothing(
    settings: Settings, database_path: Path
) -> None:
    """A reading that never happened leaves no record: 503, and the table stays empty."""
    app = create_app(settings)
    app.dependency_overrides[get_extractor] = lambda: FakeExtractor(unavailable=True)

    with TestClient(app) as client:
        response = client.post("/decisions", json={"letter_text": sample("sample_01.txt")})

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["status"] == 503
    assert stored_decision_count(database_path) == 0


SECRET_IN_THE_EXCEPTION = "postgres://admin:hunter2@db.internal/garnishment"


class ExplodingExtractor:
    """A reader whose failure nobody anticipated.

    `ExtractorUnavailable` is the *handled* outage; this raises something the application
    has no case for, which is the only way to reach the catch-all. The message carries a
    credential on purpose: the assertion that matters is that it does not reach the client.
    """

    @property
    def provider(self) -> str:
        return "exploding"

    async def extract(self, letter_text: str) -> ExtractionResult:
        raise RuntimeError(f"connection refused: {SECRET_IN_THE_EXCEPTION}")


def test_an_unhandled_error_is_a_problem_detail_that_leaks_nothing(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """The README promises a problem detail for every error, including the ones we did not
    foresee -- and those are exactly the ones whose message may contain anything."""
    app = create_app(settings)
    app.dependency_overrides[get_extractor] = lambda: ExplodingExtractor()
    # `propagate` is off by design, so `caplog`'s root handler has to be attached here.
    logging.getLogger(LOGGER_NAME).addHandler(caplog.handler)
    caplog.set_level(logging.ERROR, logger=LOGGER_NAME)

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/decisions", json={"letter_text": sample("sample_01.txt")})
    finally:
        logging.getLogger(LOGGER_NAME).removeHandler(caplog.handler)

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 500
    assert body["detail"] == "An unexpected error occurred."
    serialised = json.dumps(body)
    assert SECRET_IN_THE_EXCEPTION not in serialised
    assert "RuntimeError" not in serialised

    # The detail the client never saw did reach the log, with a traceback attached.
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].decision_event["error"] == "RuntimeError"
    assert SECRET_IN_THE_EXCEPTION in errors[0].decision_event["error_detail"]
    assert errors[0].exc_info is not None


def test_an_oversized_body_is_refused_before_it_is_parsed(client: TestClient) -> None:
    """The character limit is enforced after parsing, which means a multi-megabyte body
    would be decoded into memory first. The declared length is refused at the door."""
    response = client.post(
        "/decisions",
        content=json.dumps({"letter_text": "a" * (MAX_REQUEST_BYTES + 1)}),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 413
    assert body["instance"] == "/decisions"
    assert body["type"].endswith("/request-too-large")


def test_the_doorman_does_not_take_over_the_character_limit(client: TestClient) -> None:
    """A letter just over `MAX_LETTER_CHARS` is well under the byte ceiling, so it must
    still reach the router and be refused there, with the router's own detail."""
    over_by_one = "a" * (MAX_LETTER_CHARS + 1)
    assert len(json.dumps({"letter_text": over_by_one})) < MAX_REQUEST_BYTES

    response = client.post("/decisions", json={"letter_text": over_by_one})

    assert response.status_code == 413
    assert str(MAX_LETTER_CHARS) in response.json()["detail"]


def test_an_extremely_large_declared_length_returns_413(client: TestClient) -> None:
    response = client.post(
        "/decisions",
        content="{}",
        headers={"content-type": "application/json", "content-length": "9" * 5000},
    )
    assert response.status_code == 413
    assert response.json()["type"].endswith("/request-too-large")


def test_leading_zeroes_do_not_make_a_small_declared_length_oversized(
    client: TestClient,
) -> None:
    response = client.post(
        "/decisions",
        content="{}",
        headers={"content-type": "application/json", "content-length": "0" * 5000 + "2"},
    )
    assert response.status_code == 422
