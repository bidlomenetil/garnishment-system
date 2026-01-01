"""The tool-call schema accepts the design's worked example and rejects malformed answers.

The worked example is copied verbatim from the plan's §7.1, including its whitespace, so
this test fails if the schema and the document that describes it ever disagree.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from garnishment.adapters.llm.schema import (
    EXTRACTION_TOOL_NAME,
    LlmExtraction,
    extraction_tool,
    extraction_tool_schema,
)

CONFIDENCE_BINS = ("certain", "probable", "uncertain", "guess")

WORKED_EXAMPLE: dict[str, Any] = {
    "document_is_seizure_order": True,
    "document_is_seizure_order_confidence": "certain",
    "document_type": "restriction_order",
    "debtor_name": {
        "value": "Jordan A. Meyer",
        "evidence_quote": "Debtor:          Jordan A. Meyer",
        "confidence": "certain",
        "is_partial_or_abbreviated": False,
    },
    "case_reference": {
        "value": "AB-1042/24",
        "evidence_quote": "Case reference: AB-1042/24",
        "confidence": "certain",
    },
    "account_iban": {
        "value": "DE44 5001 0517 5407 3249 31",
        "evidence_quote": "Account (IBAN):  DE44 5001 0517 5407 3249 31",
        "confidence": "certain",
    },
    "amount_candidates": [
        {
            "raw_text": "EUR 4,250.00",
            "value_as_written": "4,250.00",
            "currency": "EUR",
            "role": "original",
            "role_confidence": "probable",
            "is_approximate": False,
            "evidence_quote": (
                "which directed the attachment of funds in the amount of EUR 4,250.00"
            ),
            "confidence": "certain",
        },
        {
            "raw_text": "EUR 1,267.00",
            "value_as_written": "1,267.00",
            "currency": "EUR",
            "role": "restricted",
            "role_confidence": "certain",
            "is_approximate": False,
            "evidence_quote": "Restricted amount to seize: EUR 1,267.00",
            "confidence": "certain",
        },
    ],
    "notes": "…recorded, never routed on",
}


def test_the_designs_worked_example_validates() -> None:
    extraction = LlmExtraction.model_validate(WORKED_EXAMPLE)

    assert extraction.document_type == "restriction_order"
    assert extraction.debtor_name.value == "Jordan A. Meyer"
    assert [c.role for c in extraction.amount_candidates] == ["original", "restricted"]
    assert [c.role_confidence for c in extraction.amount_candidates] == ["probable", "certain"]
    assert extraction.notes == "…recorded, never routed on"


def test_the_figure_is_carried_as_written_and_never_as_a_number() -> None:
    """The model copies characters; we parse them, later, inside the domain."""
    extraction = LlmExtraction.model_validate(WORKED_EXAMPLE)

    assert extraction.amount_candidates[1].value_as_written == "1,267.00"
    assert isinstance(extraction.amount_candidates[1].value_as_written, str)


def test_an_absent_field_is_null_rather_than_omitted() -> None:
    payload = json.loads(json.dumps(WORKED_EXAMPLE))
    payload["account_iban"] = {"value": None, "evidence_quote": None, "confidence": "guess"}

    extraction = LlmExtraction.model_validate(payload)

    assert extraction.account_iban.value is None
    assert extraction.account_iban.confidence == "guess"


@pytest.mark.parametrize("bin_value", CONFIDENCE_BINS)
def test_every_bin_in_the_vocabulary_is_accepted(bin_value: str) -> None:
    payload = json.loads(json.dumps(WORKED_EXAMPLE))
    payload["document_is_seizure_order_confidence"] = bin_value
    payload["amount_candidates"][0]["role_confidence"] = bin_value

    extraction = LlmExtraction.model_validate(payload)

    assert extraction.document_is_seizure_order_confidence == bin_value
    assert extraction.amount_candidates[0].role_confidence == bin_value


@pytest.mark.parametrize(
    "not_a_bin",
    [
        "very sure",
        "high",
        "CERTAIN",
        "fairly probable",
        0.92,
        0.0,
        1,
        None,
    ],
)
@pytest.mark.parametrize(
    "path",
    [
        ("document_is_seizure_order_confidence",),
        ("debtor_name", "confidence"),
        ("case_reference", "confidence"),
        ("account_iban", "confidence"),
    ],
)
def test_a_confidence_outside_the_four_bins_is_rejected(
    not_a_bin: Any, path: tuple[str, ...]
) -> None:
    """The point of the change: the model cannot express a value between two bins.

    A float here is what an earlier schema asked for, and DESIGN.md §11.7 recorded what
    that bought — the same letter read `true` every time while its confidence wandered
    0.75 to 0.92 across the 0.85 floor, so the route moved while the reading did not.
    Rejecting 0.92 as hard as "very sure" is the whole mechanism: there is no longer a
    way to say a number, so there is no longer a number to jitter.
    """
    payload = json.loads(json.dumps(WORKED_EXAMPLE))
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = not_a_bin

    with pytest.raises(ValidationError):
        LlmExtraction.model_validate(payload)


def test_a_candidates_own_confidences_are_both_bin_constrained() -> None:
    payload = json.loads(json.dumps(WORKED_EXAMPLE))
    payload["amount_candidates"][0]["confidence"] = 0.97

    with pytest.raises(ValidationError):
        LlmExtraction.model_validate(payload)


def test_notes_may_be_omitted() -> None:
    payload = json.loads(json.dumps(WORKED_EXAMPLE))
    del payload["notes"]

    assert LlmExtraction.model_validate(payload).notes == ""


@pytest.mark.parametrize(
    ("mutation", "why"),
    [
        (lambda p: p.pop("case_reference"), "a required field is missing"),
        (lambda p: p.update(document_type="seizure"), "document_type is not in the enum"),
        (
            lambda p: p["amount_candidates"][0].update(role="superseded"),
            "role is not in the enum",
        ),
        (
            lambda p: p.update(document_is_seizure_order_confidence=1.4),
            "a confidence is a number rather than one of the four bins",
        ),
        (
            lambda p: p["amount_candidates"][0].update(role_confidence="unsure"),
            "a role confidence is a word outside the bin vocabulary",
        ),
        (
            lambda p: p["amount_candidates"][0].update(confidence=float("nan")),
            "a confidence is NaN, which no bin spells",
        ),
        (lambda p: p.update(unexpected_key="anything"), "an unexpected key was added"),
    ],
)
def test_a_malformed_answer_is_rejected(mutation: Any, why: str) -> None:
    payload = json.loads(json.dumps(WORKED_EXAMPLE))
    mutation(payload)

    with pytest.raises(ValidationError):
        LlmExtraction.model_validate(payload)


def test_the_tool_definition_names_one_tool_and_carries_the_schema() -> None:
    tool = extraction_tool()

    assert tool["name"] == EXTRACTION_TOOL_NAME
    assert tool["input_schema"] == extraction_tool_schema()
    assert tool["description"]


def test_the_exported_schema_is_serialisable_json() -> None:
    """It travels to the provider as JSON, so anything unserialisable fails at runtime."""
    schema = extraction_tool_schema()

    assert json.loads(json.dumps(schema)) == schema
    assert schema["type"] == "object"
    assert "amount_candidates" in schema["properties"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["document_is_seizure_order_confidence"]["enum"] == list(
        CONFIDENCE_BINS
    )


def test_the_strict_dialect_still_round_trips_with_the_bin_enums() -> None:
    """OpenAI's strict mode is what forces the model to stay inside the enum at all.

    `to_strict_json_schema` is the same conversion `OpenAIExtractor` sends, so if an enum
    field did not survive it, the constraint would exist in Pydantic and nowhere on the
    wire — validated after the fact instead of made unsayable.
    """
    strict = to_strict_json_schema(LlmExtraction)

    assert json.loads(json.dumps(strict)) == strict
    for definition in (strict, *strict.get("$defs", {}).values()):
        if definition.get("type") == "object":
            assert definition["additionalProperties"] is False
            assert set(definition["required"]) == set(definition["properties"])

    assert strict["properties"]["document_is_seizure_order_confidence"]["enum"] == list(
        CONFIDENCE_BINS
    )
    candidate = strict["$defs"]["AmountCandidateModel"]["properties"]
    assert candidate["confidence"]["enum"] == list(CONFIDENCE_BINS)
    assert candidate["role_confidence"]["enum"] == list(CONFIDENCE_BINS)
