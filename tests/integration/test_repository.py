"""Save/load fidelity: a `Decision` that has been through SQLite is the one we saved.

The assertion is field-for-field equality plus explicit checks on the two things a JSON
round trip actually breaks — `Decimal` precision, which `float()` would quietly destroy,
and the verbatim model payload, which must come back byte-identical as JSON.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from garnishment.adapters.persistence.models import encode_extraction
from garnishment.adapters.persistence.repository import SqlAlchemyDecisionRepository
from garnishment.adapters.persistence.session import (
    build_engine,
    build_session_factory,
    init_models,
)
from garnishment.domain.decision import Decision
from garnishment.domain.extraction import LetterExtraction
from garnishment.domain.routing import route
from garnishment.domain.values import Money
from tests.fixtures.extractions import GOLDEN_CASES, GoldenCase

RAW_LLM_RESPONSE: dict[str, Any] = {
    "id": "msg_01ABCDEF",
    "model": "claude-sonnet-5",
    "stop_reason": "tool_use",
    "content": [
        {
            "type": "tool_use",
            "name": "record_extraction",
            "input": {
                "amount_candidates": [
                    {"value_as_written": "4,250.00", "role": "operative", "approximate": False}
                ],
                "is_seizure_order": True,
                "llm_confidence": 0.97,
            },
        }
    ],
    "usage": {"input_tokens": 812, "output_tokens": 233},
}

CREATED_AT = datetime(2026, 3, 14, 9, 30, 15, 123456, tzinfo=UTC)


@pytest.fixture
async def repository() -> AsyncIterator[SqlAlchemyDecisionRepository]:
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    await init_models(engine)
    try:
        yield SqlAlchemyDecisionRepository(build_session_factory(engine))
    finally:
        await engine.dispose()


def build_decision(
    extraction: LetterExtraction,
    *,
    decision_id: str = "01JQKZ8C6W4Q9N2X7T5R3B1MFE",
    notes: str = "",
) -> Decision:
    return Decision.from_routing(
        decision_id=decision_id,
        created_at=CREATED_AT,
        extraction=extraction,
        outcome=route(extraction),
        extractor_provider="fake",
        raw_llm_response=RAW_LLM_RESPONSE,
        notes=notes,
    )


def golden(name: str) -> GoldenCase:
    return next(case for case in GOLDEN_CASES if case.name == name)


ROUND_TRIP_CASES = (
    # A resolved amount, a present IBAN: every optional field populated.
    "sample_1_attachment_order_auto",
    # No operative amount at all, and no IBAN: the null-heavy row.
    "sample_2_partial_name_needs_human",
    # The restriction doctrine's amount, whose trailing zeros are the precision check.
    "sample_3_restriction_order_auto",
)


@pytest.mark.parametrize("case_name", ROUND_TRIP_CASES)
async def test_decision_survives_the_round_trip_field_for_field(
    repository: SqlAlchemyDecisionRepository, case_name: str
) -> None:
    original = build_decision(golden(case_name).extraction, notes="read by clerk 14")

    await repository.save(original)
    reloaded = await repository.get(original.id)

    assert reloaded is not None
    assert reloaded == original
    assert reloaded.extraction == original.extraction
    assert reloaded.field_confidences == original.field_confidences
    assert reloaded.fired_rules == original.fired_rules
    assert reloaded.created_at == original.created_at


@pytest.mark.parametrize("case_name", ROUND_TRIP_CASES)
async def test_tuples_come_back_as_tuples(
    repository: SqlAlchemyDecisionRepository, case_name: str
) -> None:
    """JSON has one sequence type; the domain has two, and `route()` produces tuples."""
    await repository.save(build_decision(golden(case_name).extraction))
    reloaded = await repository.get("01JQKZ8C6W4Q9N2X7T5R3B1MFE")

    assert reloaded is not None
    assert isinstance(reloaded.fired_rules, tuple)
    assert isinstance(reloaded.reasons, tuple)
    assert isinstance(reloaded.extraction.amount_candidates, tuple)


async def test_enums_come_back_as_enum_members(
    repository: SqlAlchemyDecisionRepository,
) -> None:
    case = golden("sample_3_restriction_order_auto")
    await repository.save(build_decision(case.extraction))
    reloaded = await repository.get("01JQKZ8C6W4Q9N2X7T5R3B1MFE")

    assert reloaded is not None
    assert reloaded.route is case.expected_route
    assert reloaded.decisive_rule_id is case.expected_decisive_rule
    assert reloaded.extraction.document_type is case.extraction.document_type
    assert reloaded.extraction.extraction_status is case.extraction.extraction_status
    for fired, original_fired in zip(
        reloaded.fired_rules, build_decision(case.extraction).fired_rules, strict=True
    ):
        assert fired.rule_id is original_fired.rule_id
        assert fired.route is original_fired.route
    for candidate, original_candidate in zip(
        reloaded.extraction.amount_candidates, case.extraction.amount_candidates, strict=True
    ):
        assert candidate.role is original_candidate.role


async def test_decimal_precision_is_exact(repository: SqlAlchemyDecisionRepository) -> None:
    """`1267.00` must not come back as `1267.0` or `1266.9999999999999`."""
    original = build_decision(golden("sample_3_restriction_order_auto").extraction)
    assert original.operative_amount == Money(Decimal("1267.00"))

    await repository.save(original)
    reloaded = await repository.get(original.id)

    assert reloaded is not None
    assert reloaded.operative_amount is not None
    assert reloaded.operative_amount.amount == Decimal("1267.00")
    assert str(reloaded.operative_amount.amount) == "1267.00"
    assert reloaded.operative_amount.amount.as_tuple() == Decimal("1267.00").as_tuple()
    assert reloaded.operative_amount.currency == "EUR"


async def test_absent_operative_amount_reloads_as_none(
    repository: SqlAlchemyDecisionRepository,
) -> None:
    original = build_decision(golden("sample_2_partial_name_needs_human").extraction)
    assert original.operative_amount is None
    assert original.extraction.account_iban.value is None

    await repository.save(original)
    reloaded = await repository.get(original.id)

    assert reloaded is not None
    assert reloaded.operative_amount is None
    assert reloaded.operative_amount_reason is original.operative_amount_reason
    assert reloaded.extraction.account_iban.value is None
    assert reloaded.extraction.account_iban.evidence_quote is None
    assert reloaded.source_candidate_index is None
    assert reloaded.considered_candidate_index == original.considered_candidate_index


async def test_raw_llm_response_is_byte_identical_as_json(
    repository: SqlAlchemyDecisionRepository,
) -> None:
    original = build_decision(golden("sample_1_attachment_order_auto").extraction)

    await repository.save(original)
    reloaded = await repository.get(original.id)

    assert reloaded is not None
    assert json.dumps(reloaded.raw_llm_response) == json.dumps(RAW_LLM_RESPONSE)


async def test_stored_extraction_holds_no_parsed_decimal(
    repository: SqlAlchemyDecisionRepository,
) -> None:
    """Criterion 14: `value_as_written` is the only stored form of a figure.

    The derived `operative_amount` is a separate top-level column, below §7.7's marked
    line; what must not appear is a second representation of a figure *inside* the
    extraction, which is the one that could drift away from the verbatim string.
    """
    payload = encode_extraction(golden("sample_3_restriction_order_auto").extraction)

    assert [c["value_as_written"] for c in payload["amount_candidates"]] == [
        "4,250.00",
        "1,267.00",
    ]
    for candidate in payload["amount_candidates"]:
        assert set(candidate) == {
            "value_as_written",
            "currency",
            "raw_text",
            "role",
            "role_confidence",
            "is_approximate",
            "evidence_quote",
            "llm_confidence",
        }
    serialised = json.dumps(payload)
    assert "1267.00" not in serialised
    assert "4250.00" not in serialised


async def test_get_returns_none_for_an_unknown_id(
    repository: SqlAlchemyDecisionRepository,
) -> None:
    assert await repository.get("01JQKZ8C6W4Q9N2X7T5R3B1MFE") is None


async def test_two_decisions_are_stored_side_by_side(
    repository: SqlAlchemyDecisionRepository,
) -> None:
    first = build_decision(
        golden("sample_1_attachment_order_auto").extraction,
        decision_id="01AAAAAAAAAAAAAAAAAAAAAAAA",
    )
    second = build_decision(
        golden("sample_3_restriction_order_auto").extraction,
        decision_id="01BBBBBBBBBBBBBBBBBBBBBBBB",
    )

    await repository.save(first)
    await repository.save(second)

    assert await repository.get(first.id) == first
    assert await repository.get(second.id) == second


async def test_a_second_save_under_the_same_id_does_not_overwrite_the_first(
    repository: SqlAlchemyDecisionRepository,
) -> None:
    """Append-only means the first judgement recorded under an id is the one that stays.

    A retry after a partial failure must still be harmless, which is why this is an
    ignored conflict rather than a raised one -- but an *upsert* would let a later call
    silently rewrite an audit record, and the port's own docstring says there is no update.
    """
    first = build_decision(golden("sample_1_attachment_order_auto").extraction, notes="first")
    second = build_decision(
        golden("sample_3_restriction_order_auto").extraction, notes="second"
    )
    assert first.id == second.id
    assert second.letter_sha256 != first.letter_sha256

    await repository.save(first)
    await repository.save(second)
    reloaded = await repository.get(first.id)

    assert reloaded == first
    assert reloaded is not None
    assert reloaded.notes == "first"
    assert reloaded.letter_sha256 == first.letter_sha256
    assert reloaded.operative_amount == first.operative_amount


async def test_re_saving_an_identical_record_is_still_harmless(
    repository: SqlAlchemyDecisionRepository,
) -> None:
    """The retry case the old `merge` existed for: no exception, one row, same record."""
    decision = build_decision(golden("sample_1_attachment_order_auto").extraction)

    await repository.save(decision)
    await repository.save(decision)

    assert await repository.get(decision.id) == decision
