"""The real adapter, driven against a stubbed client. No network, no key, no exceptions.

The four cases below are the whole contract: a good answer, a bad answer that the retry
rescues, a bad answer that survives the retry, and no answer at all. The last two are the
ones worth reading — they land in completely different places on purpose.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, cast

import anthropic
import httpx2
import pytest
from anthropic import AsyncAnthropic

from garnishment.adapters.llm.anthropic_extractor import (
    NO_TOOL_CALL,
    PROVIDER,
    AnthropicExtractor,
)
from garnishment.adapters.llm.attempts import SCHEMA_VALIDATION_ERROR
from garnishment.adapters.llm.fake_extractor import SAMPLE_1_PAYLOAD, SAMPLE_3_PAYLOAD
from garnishment.adapters.llm.prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    TOOL_CALL_INSTRUCTION,
    render_system_prompt,
)
from garnishment.adapters.llm.schema import EXTRACTION_TOOL_NAME
from garnishment.application.errors import ExtractorUnavailable
from garnishment.domain.enums import ExtractionStatus, Route, RuleId
from garnishment.domain.routing import route
from garnishment.domain.thresholds import EXTRACTION_MAX_RETRIES, LLM_TEMPERATURE
from tests.fixtures.extractions import SAMPLE_1, SAMPLE_3

MODEL_ID = "claude-sonnet-5"

MALFORMED_PAYLOAD = {"document_type": "restriction_order", "amount_candidates": "several"}


@dataclass
class StubBlock:
    type: str
    name: str = ""
    input: Any = None
    text: str = ""


@dataclass
class StubResponse:
    content: list[StubBlock]


@dataclass
class StubMessages:
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
    messages: StubMessages


def tool_call(payload: Any) -> StubResponse:
    return StubResponse([StubBlock(type="tool_use", name=EXTRACTION_TOOL_NAME, input=payload)])


def text_only(text: str = "I am afraid I cannot do that.") -> StubResponse:
    return StubResponse([StubBlock(type="text", text=text)])


def build(*scripted: Any) -> tuple[AnthropicExtractor, StubMessages]:
    messages = StubMessages(list(scripted))
    client = cast(AsyncAnthropic, StubClient(messages))
    return AnthropicExtractor(client, model_id=MODEL_ID), messages


async def test_a_good_answer_is_mapped_and_recorded() -> None:
    extractor, messages = build(tool_call(SAMPLE_3_PAYLOAD))

    result = await extractor.extract(SAMPLE_3)

    assert result.extraction.extraction_status is ExtractionStatus.OK
    assert result.extraction.model_id == MODEL_ID
    assert result.extraction.prompt_version == PROMPT_VERSION
    assert result.raw_llm_response == SAMPLE_3_PAYLOAD
    assert result.provider == PROVIDER
    assert route(result.extraction).route is Route.AUTO
    assert len(messages.calls) == 1


async def test_the_request_pins_the_model_and_forces_the_single_tool() -> None:
    extractor, messages = build(tool_call(SAMPLE_1_PAYLOAD))

    await extractor.extract(SAMPLE_1)
    sent = messages.calls[0]

    assert sent["model"] == MODEL_ID
    assert sent["tool_choice"] == {"type": "tool", "name": EXTRACTION_TOOL_NAME}
    assert [tool["name"] for tool in sent["tools"]] == [EXTRACTION_TOOL_NAME]
    assert sent["system"] == render_system_prompt(TOOL_CALL_INSTRUCTION)
    assert SYSTEM_PROMPT in sent["system"]
    assert SAMPLE_1 in sent["messages"][0]["content"]


async def test_the_model_is_told_to_call_the_tool_it_is_actually_offered() -> None:
    """This transport forces a tool call, so this is the adapter that may say so. The
    shared prompt says nothing about how to answer, because the other adapter offers the
    model no tools at all."""
    extractor, messages = build(tool_call(SAMPLE_1_PAYLOAD))

    await extractor.extract(SAMPLE_1)

    assert TOOL_CALL_INSTRUCTION in messages.calls[0]["system"]


def test_the_messages_api_no_longer_accepts_a_sampling_parameter() -> None:
    """Pinned so the reason the adapter sends no `temperature` stays visible.

    `thresholds.LLM_TEMPERATURE` states the intent, but this generation of models removed
    `temperature`, `top_p` and `top_k`, and the SDK dropped them from the signature —
    sending one is a call-time error. If a future SDK restores the parameter, this test
    fails and the adapter should pin sampling again. Until then the determinism story
    rests where it always did: on `route()` being pure over a stored extraction.
    """
    accepted = inspect.signature(AsyncAnthropic(api_key="unused").messages.create).parameters

    assert LLM_TEMPERATURE == 0.0
    assert "temperature" not in accepted
    assert "top_p" not in accepted


async def test_a_malformed_answer_is_retried_once_identically() -> None:
    extractor, messages = build(tool_call(MALFORMED_PAYLOAD), tool_call(SAMPLE_1_PAYLOAD))

    result = await extractor.extract(SAMPLE_1)

    assert result.extraction.extraction_status is ExtractionStatus.OK
    assert len(messages.calls) == 1 + EXTRACTION_MAX_RETRIES
    assert messages.calls[0] == messages.calls[1], (
        "The retry must be identical. A second, different prompt would mean two records "
        "claiming the same prompt version were produced by different instructions."
    )


async def test_a_malformed_answer_that_survives_the_retry_escalates_and_never_rejects() -> None:
    extractor, messages = build(tool_call(MALFORMED_PAYLOAD), tool_call(MALFORMED_PAYLOAD))

    result = await extractor.extract(SAMPLE_1)
    outcome = route(result.extraction)

    assert result.extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY
    assert outcome.route is Route.NEEDS_HUMAN
    assert outcome.decisive_rule_id is RuleId.H01_EXTRACTION_FAILED
    assert outcome.route is not Route.REJECT
    assert len(messages.calls) == 1 + EXTRACTION_MAX_RETRIES


async def test_the_unparseable_payload_is_still_stored_verbatim() -> None:
    """A colleague debugging the schema break needs what the model actually said."""
    extractor, _ = build(tool_call(MALFORMED_PAYLOAD), tool_call(MALFORMED_PAYLOAD))

    result = await extractor.extract(SAMPLE_1)

    assert result.raw_llm_response == MALFORMED_PAYLOAD


async def test_an_answer_with_no_tool_call_is_treated_as_malformed() -> None:
    extractor, messages = build(text_only(), text_only())

    result = await extractor.extract(SAMPLE_1)

    assert result.extraction.extraction_status is ExtractionStatus.MALFORMED_AFTER_RETRY
    assert len(messages.calls) == 1 + EXTRACTION_MAX_RETRIES


async def test_two_answers_with_no_tool_call_still_leave_something_to_debug() -> None:
    """The stored payload used to be `{}` here, which defeats the reason it is stored: a
    colleague opening the record for a schema break would find nothing at all."""
    extractor, _ = build(text_only("I will not do that."), text_only("Still no."))

    result = await extractor.extract(SAMPLE_1)

    assert result.raw_llm_response != {}
    assert result.raw_llm_response["_diagnostic"] == NO_TOOL_CALL
    assert result.raw_llm_response["text"] == "Still no."


async def test_a_real_payload_outranks_a_diagnostic_from_a_later_attempt() -> None:
    """What the model actually said beats a note about what it did not say."""
    extractor, _ = build(tool_call(MALFORMED_PAYLOAD), text_only())

    result = await extractor.extract(SAMPLE_1)

    assert result.raw_llm_response == MALFORMED_PAYLOAD


async def test_a_tool_call_the_retry_rescues_after_no_tool_call_at_all() -> None:
    extractor, _ = build(text_only(), tool_call(SAMPLE_1_PAYLOAD))

    result = await extractor.extract(SAMPLE_1)

    assert result.extraction.extraction_status is ExtractionStatus.OK


def request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("error", "why"),
    [
        (anthropic.APIConnectionError(request=request()), "the provider was unreachable"),
        (
            anthropic.AuthenticationError(
                "bad key", response=httpx2.Response(401, request=request()), body=None
            ),
            "the credentials were rejected",
        ),
        (
            anthropic.RateLimitError(
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
    extractor, messages = build(anthropic.APIConnectionError(request=request()))

    with pytest.raises(ExtractorUnavailable):
        await extractor.extract(SAMPLE_1)

    assert len(messages.calls) == 1


def test_an_aliased_model_id_is_refused_at_construction() -> None:
    """A record names the model that produced it; an alias names a different one each day."""
    with pytest.raises(ValueError, match="alias"):
        AnthropicExtractor(
            cast(AsyncAnthropic, StubClient(StubMessages([]))), model_id="claude-latest"
        )


def test_the_bare_generation_id_is_the_pinned_form_anthropic_publishes() -> None:
    """The mirror of `test_a_dated_snapshot_is_accepted` in the OpenAI suite, and
    deliberately not the same predicate. OpenAI ships the bare name as the moving alias
    and freezes snapshots with a date; Anthropic ships the bare generation id as the
    pinned one and moves `-latest`. Both adapters refuse the name that moves — which name
    that is, is a fact about the vendor.
    """
    extractor = AnthropicExtractor(
        cast(AsyncAnthropic, StubClient(StubMessages([]))), model_id=MODEL_ID
    )

    assert extractor.provider == PROVIDER


async def test_every_failed_attempt_is_logged_with_its_cause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Four causes land on the same `MALFORMED_AFTER_RETRY`, so the record alone cannot
    say which one happened on which attempt. The warning can."""
    extractor, _ = build(text_only(), tool_call(MALFORMED_PAYLOAD))

    with caplog.at_level(logging.WARNING, logger="garnishment.adapters.llm"):
        await extractor.extract(SAMPLE_1)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 + EXTRACTION_MAX_RETRIES
    assert "attempt 1/2" in warnings[0].getMessage()
    assert NO_TOOL_CALL in warnings[0].getMessage()
    assert "attempt 2/2" in warnings[1].getMessage()
    assert SCHEMA_VALIDATION_ERROR in warnings[1].getMessage()
    assert all(PROVIDER in record.getMessage() for record in warnings)


async def test_the_warning_never_carries_the_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A payload is letter-derived; a cause and a count are not."""
    extractor, _ = build(tool_call(MALFORMED_PAYLOAD), tool_call(MALFORMED_PAYLOAD))

    with caplog.at_level(logging.WARNING, logger="garnishment.adapters.llm"):
        await extractor.extract(SAMPLE_1)

    assert all("several" not in record.getMessage() for record in caplog.records)
