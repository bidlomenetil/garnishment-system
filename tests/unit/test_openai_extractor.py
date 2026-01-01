"""The second real adapter, driven against a stubbed client. No network, no key.

This file mirrors `test_anthropic_extractor.py` case for case, on purpose: the two
adapters are two implementations of one port, and the only way that claim stays true is if
the same behavioural list is asserted of both. The cases that are new here are the two
extra ways the Responses API can answer badly — a truncated reply and an outright refusal —
and both are checked to land where a malformed tool call lands, never on the 503 path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, cast

import httpx2
import openai
import pytest
from openai import AsyncOpenAI

from garnishment.adapters.llm.attempts import SCHEMA_VALIDATION_ERROR
from garnishment.adapters.llm.fake_extractor import SAMPLE_1_PAYLOAD, SAMPLE_3_PAYLOAD
from garnishment.adapters.llm.openai_extractor import (
    INCOMPLETE,
    MAX_OUTPUT_TOKENS,
    PROVIDER,
    REFUSAL,
    STRICT_SCHEMA,
    UNPARSEABLE_JSON,
    OpenAIExtractor,
)
from garnishment.adapters.llm.prompt import (
    PROMPT_VERSION,
    STRUCTURED_REPLY_INSTRUCTION,
    SYSTEM_PROMPT,
    render_system_prompt,
)
from garnishment.adapters.llm.schema import EXTRACTION_TOOL_NAME, LlmExtraction
from garnishment.application.errors import ExtractorUnavailable
from garnishment.domain.enums import ExtractionStatus, Route, RuleId
from garnishment.domain.routing import route
from garnishment.domain.thresholds import EXTRACTION_MAX_RETRIES, LLM_TEMPERATURE
from tests.fixtures.extractions import SAMPLE_1, SAMPLE_3

MODEL_ID = "gpt-5-nano-2025-08-07"

MALFORMED_PAYLOAD = {"document_type": "restriction_order", "amount_candidates": "several"}


@dataclass
class StubPart:
    type: str
    text: str = ""
    refusal: str = ""


@dataclass
class StubItem:
    type: str
    content: list[StubPart] = field(default_factory=list)


@dataclass
class StubResponse:
    status: str
    output: list[StubItem]


@dataclass
class StubResponses:
    scripted: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.scripted:
            raise AssertionError("The adapter called the model more times than expected.")
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@dataclass
class StubClient:
    responses: StubResponses


def message(*parts: StubPart) -> StubResponse:
    """A completed reply. The leading reasoning item is real: GPT-5 models emit one, and
    the adapter has to walk past it to reach the message."""
    return StubResponse(
        status="completed",
        output=[StubItem(type="reasoning"), StubItem(type="message", content=list(parts))],
    )


def json_reply(payload: Any) -> StubResponse:
    return message(StubPart(type="output_text", text=json.dumps(payload)))


def refusal(text: str = "I am afraid I cannot do that.") -> StubResponse:
    return message(StubPart(type="refusal", refusal=text))


def not_json(text: str = "Here is what I found:") -> StubResponse:
    return message(StubPart(type="output_text", text=text))


def incomplete(payload: Any = SAMPLE_1_PAYLOAD) -> StubResponse:
    """Cut short by the output-token budget or a filter. The content is deliberately
    well-formed, so the test can only pass if `status` is what the adapter reads."""
    reply = json_reply(payload)
    return StubResponse(status="incomplete", output=reply.output)


def build(
    *scripted: Any, model_id: str = MODEL_ID, reasoning_effort: str | None = "minimal"
) -> tuple[OpenAIExtractor, StubResponses]:
    responses = StubResponses(list(scripted))
    client = cast(AsyncOpenAI, StubClient(responses))
    extractor = OpenAIExtractor(client, model_id=model_id, reasoning_effort=reasoning_effort)
    return extractor, responses


async def test_a_good_answer_is_mapped_and_recorded() -> None:
    extractor, responses = build(json_reply(SAMPLE_3_PAYLOAD))

    result = await extractor.extract(SAMPLE_3)

    assert result.extraction.extraction_status is ExtractionStatus.OK
    assert result.extraction.model_id == MODEL_ID
    assert result.extraction.prompt_version == PROMPT_VERSION
    assert result.raw_llm_response == SAMPLE_3_PAYLOAD
    assert result.provider == PROVIDER
    assert route(result.extraction).route is Route.AUTO
    assert len(responses.calls) == 1


async def test_the_request_pins_the_model_and_demands_the_strict_schema() -> None:
    extractor, responses = build(json_reply(SAMPLE_1_PAYLOAD))

    await extractor.extract(SAMPLE_1)
    sent = responses.calls[0]
    fmt = sent["text"]["format"]

    assert sent["model"] == MODEL_ID
    assert sent["instructions"] == render_system_prompt(STRUCTURED_REPLY_INSTRUCTION)
    assert SYSTEM_PROMPT in sent["instructions"]
    assert SAMPLE_1 in sent["input"]
    assert sent["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == EXTRACTION_TOOL_NAME
    assert fmt["strict"] is True
    assert fmt["schema"] == STRICT_SCHEMA


async def test_no_sampling_parameter_reaches_the_request() -> None:
    """The same documented deviation from §7.8 item 1 the Anthropic adapter records.

    `thresholds.LLM_TEMPERATURE` states the intent, but the GPT-5 family are reasoning
    models for which `temperature` and `top_p` are not meaningful controls. The
    determinism story rests where it always did: on `route()` being pure over a stored
    extraction. Unlike the Anthropic case this asserts the request rather than the SDK
    signature, because the Responses API still accepts both parameters — omitting them is
    our choice, so it is our choice that has to be pinned.
    """
    extractor, responses = build(json_reply(SAMPLE_1_PAYLOAD))

    await extractor.extract(SAMPLE_1)

    assert LLM_TEMPERATURE == 0.0
    assert "temperature" not in responses.calls[0]
    assert "top_p" not in responses.calls[0]


async def test_the_reasoning_effort_is_sent_when_configured() -> None:
    extractor, responses = build(json_reply(SAMPLE_1_PAYLOAD), reasoning_effort="minimal")

    await extractor.extract(SAMPLE_1)

    assert responses.calls[0]["reasoning"] == {"effort": "minimal"}


async def test_the_reasoning_parameter_is_omitted_entirely_when_unset() -> None:
    """Not sent as null: a model that rejects the parameter must be usable by clearing it."""
    extractor, responses = build(json_reply(SAMPLE_1_PAYLOAD), reasoning_effort=None)

    await extractor.extract(SAMPLE_1)

    assert "reasoning" not in responses.calls[0]


async def test_a_malformed_answer_is_retried_once_identically() -> None:
    extractor, responses = build(json_reply(MALFORMED_PAYLOAD), json_reply(SAMPLE_1_PAYLOAD))

    result = await extractor.extract(SAMPLE_1)

    assert result.extraction.extraction_status is ExtractionStatus.OK
    assert len(responses.calls) == 1 + EXTRACTION_MAX_RETRIES
    assert responses.calls[0] == responses.calls[1], (
        "The retry must be identical. A second, different prompt would mean two records "
        "claiming the same prompt version were produced by different instructions."
    )


async def test_a_malformed_answer_that_survives_the_retry_escalates_and_never_rejects() -> None:
    extractor, responses = build(json_reply(MALFORMED_PAYLOAD), json_reply(MALFORMED_PAYLOAD))

    result = await extractor.extract(SAMPLE_1)
    outcome = route(result.extraction)

    assert result.extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY
    assert outcome.route is Route.NEEDS_HUMAN
    assert outcome.decisive_rule_id is RuleId.H01_EXTRACTION_FAILED
    assert outcome.route is not Route.REJECT
    assert len(responses.calls) == 1 + EXTRACTION_MAX_RETRIES


async def test_the_unparseable_payload_is_still_stored_verbatim() -> None:
    """A colleague debugging the schema break needs what the model actually said."""
    extractor, _ = build(json_reply(MALFORMED_PAYLOAD), json_reply(MALFORMED_PAYLOAD))

    result = await extractor.extract(SAMPLE_1)

    assert result.raw_llm_response == MALFORMED_PAYLOAD


async def test_a_reply_that_is_not_json_is_treated_as_malformed() -> None:
    extractor, responses = build(not_json(), not_json())

    result = await extractor.extract(SAMPLE_1)

    assert result.extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY
    assert len(responses.calls) == 1 + EXTRACTION_MAX_RETRIES


async def test_the_request_never_offers_a_tool_so_the_model_is_never_told_to_call_one() -> None:
    """The shared prompt used to say "by calling the record_garnishment_extraction tool",
    which is true of the Messages API request and false of this one: strict JSON schema is
    a reply format, and no `tools` parameter is sent at all."""
    extractor, responses = build(json_reply(SAMPLE_1_PAYLOAD))

    await extractor.extract(SAMPLE_1)
    sent = responses.calls[0]

    assert "tools" not in sent
    assert "tool" not in sent["instructions"].lower()
    assert STRUCTURED_REPLY_INSTRUCTION in sent["instructions"]


async def test_a_refusal_is_a_bad_answer_and_not_an_outage() -> None:
    """The model declining is evidence about the reading, so it escalates under H01."""
    extractor, responses = build(refusal(), refusal())

    result = await extractor.extract(SAMPLE_1)
    outcome = route(result.extraction)

    assert result.extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY
    assert outcome.route is Route.NEEDS_HUMAN
    assert outcome.decisive_rule_id is RuleId.H01_EXTRACTION_FAILED
    assert len(responses.calls) == 1 + EXTRACTION_MAX_RETRIES


async def test_an_incomplete_reply_is_a_bad_answer_and_not_an_outage() -> None:
    """Truncation is our budget running out, not the provider being unreachable."""
    extractor, responses = build(incomplete(), incomplete())

    result = await extractor.extract(SAMPLE_1)
    outcome = route(result.extraction)

    assert result.extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY
    assert outcome.route is Route.NEEDS_HUMAN
    assert outcome.decisive_rule_id is RuleId.H01_EXTRACTION_FAILED
    assert len(responses.calls) == 1 + EXTRACTION_MAX_RETRIES


@pytest.mark.parametrize(
    ("scripted", "cause", "evidence"),
    [
        (refusal("Not this one."), REFUSAL, "Not this one."),
        (incomplete(), INCOMPLETE, "incomplete"),
        (not_json("Here is what I found:"), UNPARSEABLE_JSON, "Here is what I found:"),
    ],
)
async def test_two_bad_answers_of_one_kind_still_leave_something_to_debug(
    scripted: Any, cause: str, evidence: str
) -> None:
    """A refusal and a truncated reply carry no payload at all, so `raw_llm_response` used
    to end up `{}` — the one case where a colleague debugging the break has nothing to
    look at, which is the whole reason the field exists."""
    extractor, _ = build(scripted, scripted)

    result = await extractor.extract(SAMPLE_1)

    assert result.raw_llm_response != {}
    assert result.raw_llm_response["_diagnostic"] == cause
    assert evidence in str(result.raw_llm_response.values())


async def test_a_real_payload_outranks_a_diagnostic_from_a_later_attempt() -> None:
    """What the model actually said beats a note about what it did not say."""
    extractor, _ = build(json_reply(MALFORMED_PAYLOAD), refusal())

    result = await extractor.extract(SAMPLE_1)

    assert result.raw_llm_response == MALFORMED_PAYLOAD


async def test_every_failed_attempt_is_logged_with_its_own_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Four causes land on the same `MALFORMED_AFTER_RETRY`. Only the warning says which
    one happened on which attempt."""
    extractor, _ = build(refusal(), json_reply(MALFORMED_PAYLOAD))

    with caplog.at_level(logging.WARNING, logger="garnishment.adapters.llm"):
        await extractor.extract(SAMPLE_1)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 + EXTRACTION_MAX_RETRIES
    assert "attempt 1/2" in warnings[0].getMessage()
    assert REFUSAL in warnings[0].getMessage()
    assert "attempt 2/2" in warnings[1].getMessage()
    assert SCHEMA_VALIDATION_ERROR in warnings[1].getMessage()
    assert all(PROVIDER in record.getMessage() for record in warnings)


async def test_the_warning_never_carries_the_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A payload is letter-derived; a cause and a count are not."""
    extractor, _ = build(json_reply(MALFORMED_PAYLOAD), json_reply(MALFORMED_PAYLOAD))

    with caplog.at_level(logging.WARNING, logger="garnishment.adapters.llm"):
        await extractor.extract(SAMPLE_1)

    assert all("several" not in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("first", [refusal(), incomplete(), not_json()])
async def test_the_retry_rescues_every_kind_of_bad_first_answer(first: Any) -> None:
    extractor, _ = build(first, json_reply(SAMPLE_1_PAYLOAD))

    result = await extractor.extract(SAMPLE_1)

    assert result.extraction.extraction_status is ExtractionStatus.OK


def request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.openai.com/v1/responses")


@pytest.mark.parametrize(
    ("error", "why"),
    [
        (openai.APIConnectionError(request=request()), "the provider was unreachable"),
        (
            openai.AuthenticationError(
                "bad key", response=httpx2.Response(401, request=request()), body=None
            ),
            "the credentials were rejected",
        ),
        (
            openai.RateLimitError(
                "slow down", response=httpx2.Response(429, request=request()), body=None
            ),
            "we were rate limited past the client's own backoff",
        ),
    ],
)
async def test_an_absent_answer_raises_rather_than_producing_a_record(
    error: Exception, why: str
) -> None:
    """No record is written for a reading that never happened — the caller owes a 503."""
    extractor, _ = build(error)

    with pytest.raises(ExtractorUnavailable):
        await extractor.extract(SAMPLE_1)


async def test_a_transport_failure_is_not_retried_into_a_malformed_record() -> None:
    """The two failure modes must not collapse into one another."""
    extractor, responses = build(openai.APIConnectionError(request=request()))

    with pytest.raises(ExtractorUnavailable):
        await extractor.extract(SAMPLE_1)

    assert len(responses.calls) == 1


def test_an_undated_model_id_is_refused_at_construction() -> None:
    """A record names the model that produced it; a bare name names a different one each
    time OpenAI moves the alias."""
    with pytest.raises(ValueError, match="alias"):
        build(model_id="gpt-5-nano")


def test_a_dated_snapshot_is_accepted() -> None:
    extractor, _ = build(model_id="gpt-5-nano-2025-08-07")

    assert extractor.provider == PROVIDER


def test_the_strict_schema_is_what_structured_outputs_requires() -> None:
    """Pins the private `openai.lib._pydantic` import. If an SDK release moves it, the
    adapter fails to import and this fails loudly rather than sending a laxer schema."""
    assert STRICT_SCHEMA["additionalProperties"] is False
    assert set(STRICT_SCHEMA["required"]) == set(LlmExtraction.model_fields)
    assert STRICT_SCHEMA["properties"].keys() == LlmExtraction.model_fields.keys()
