"""The same reader against a second provider: one strict-schema reply, one retry.

This adapter is the Anthropic one with a different transport bolted on, and that is the
whole point of it. `ExtractorPort` claims the provider is replaceable; two conformant
implementations behind it, tested against the same behavioural list, are what turn that
claim into evidence. The schema and the mapper are shared verbatim, and so is the prompt
bar one line: this transport offers the model no tools, so it is told to answer with a
structured object rather than to call one. Nothing here reshapes an answer, and nothing in
`domain/` or `application/` knows this file exists.

The failure split is identical to `anthropic_extractor.py` and matters for the same reason.
*The model answered badly* — a refusal, a truncated reply, unparseable JSON, or arguments
that do not fit the schema — is retried once identically and then recorded as
`MALFORMED_AFTER_RETRY`. We never raise and the letter is never rejected: a schema break is
evidence about us, not about the court. *The model never answered* — transport failure, bad
credentials, rate limiting that survived the SDK's own backoff — raises
`ExtractorUnavailable`, writes no record, and the caller owes the client a 503.

The Responses API adds two ways to answer badly that the Messages API does not have. A
`status` other than `"completed"` means the reply was cut short (the output-token budget,
or a content filter), and a `refusal` content item means the model declined. Both are the
model answering badly, so both take the retry-then-H01 path rather than the 503 path.

Every attempt that fails logs one warning naming its cause, and the last one keeps a small
diagnostic mapping. Four causes land on the same `MALFORMED_AFTER_RETRY`, and a refusal or
a truncation leaves no payload to store, so without those two the record says only that
the reading failed and never which way.

Structured output is requested as `text.format`, a strict JSON schema generated from
`LlmExtraction` itself, rather than as a forced function call. The distinction is idiomatic
only — the strict-schema rules are the same either way — but the reply *is* the payload
here, and the SDK's `responses.parse` convenience is avoided deliberately: it validates
inside the call and would raise a schema miss through the same `except` as a transport
failure, collapsing the two failure modes this module exists to keep apart.

**Sampling is not pinned here either**, the same documented deviation from the design's
§7.8 item 1 that `anthropic_extractor.py` records at length. The GPT-5 family are reasoning
models and `temperature` / `top_p` are not meaningful controls on them, so neither is sent.
What is sent instead is `reasoning.effort`, configurable and omitted entirely when unset,
because the cheapest setting is the one this extraction wants: the reading is shallow and
the schema does the thinking. The determinism guarantee is unaffected — it never rested on
the model repeating itself, but on `route()` being a pure function of a stored extraction.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, cast

from openai import AsyncOpenAI, OpenAIError
from openai.lib._pydantic import to_strict_json_schema
from pydantic import ValidationError

from garnishment.adapters.llm.attempts import (
    SCHEMA_VALIDATION_ERROR,
    BadAnswer,
    log_failed_attempt,
)
from garnishment.adapters.llm.mapping import malformed_extraction, map_llm_extraction
from garnishment.adapters.llm.model_pinning import require_pinned_openai_model
from garnishment.adapters.llm.prompt import (
    PROMPT_VERSION,
    STRUCTURED_REPLY_INSTRUCTION,
    render_system_prompt,
    render_user_message,
)
from garnishment.adapters.llm.schema import EXTRACTION_TOOL_NAME, LlmExtraction
from garnishment.application.errors import ExtractorUnavailable
from garnishment.application.ports import ExtractionResult
from garnishment.domain.thresholds import EXTRACTION_MAX_RETRIES

logger = logging.getLogger(__name__)

PROVIDER = "openai"

REFUSAL = "refusal"
INCOMPLETE = "incomplete"
UNPARSEABLE_JSON = "unparseable_json"
NON_OBJECT_REPLY = "non_object_reply"
NO_OUTPUT_MESSAGE = "no_output_message"
"""The five ways this transport can complete a call without yielding a payload. Each is
named in the warning logged for the attempt and in what is stored if none ever parses."""

MAX_OUTPUT_TOKENS = 4096
"""The same budget the Anthropic adapter allows; the schema caps how much there is to say.
A reply that exhausts it comes back `incomplete`, which is handled as a bad answer."""

STRICT_SCHEMA: dict[str, Any] = to_strict_json_schema(LlmExtraction)
"""`LlmExtraction` in the dialect strict structured outputs require — every key `required`
and `additionalProperties: false` throughout. Computed once at import, from the one schema
the whole service shares, so the two providers cannot be asked different questions.

`openai.lib._pydantic` is a private path, which the dependency bound and a unit test both
guard: if an SDK release moves it, the import fails loudly at startup instead of quietly
sending a laxer schema."""


class OpenAIExtractor:
    """Reads a letter as one strict-schema reply from a pinned model snapshot."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model_id: str,
        reasoning_effort: str | None = None,
    ) -> None:
        require_pinned_openai_model(model_id)
        self._client = client
        self._model_id = model_id
        self._reasoning_effort = reasoning_effort

    @property
    def provider(self) -> str:
        return PROVIDER

    async def extract(self, letter_text: str) -> ExtractionResult:
        attempts = 1 + EXTRACTION_MAX_RETRIES
        last_payload: Mapping[str, Any] = {}
        last_diagnostic: Mapping[str, Any] = {}

        for attempt in range(1, attempts + 1):
            outcome = await self._call(letter_text)
            if isinstance(outcome, BadAnswer):
                last_diagnostic = outcome.payload
                log_failed_attempt(
                    logger,
                    provider=PROVIDER,
                    attempt=attempt,
                    attempts=attempts,
                    cause=outcome.cause,
                )
                continue

            last_payload = outcome
            try:
                llm = LlmExtraction.model_validate(outcome)
            except ValidationError:
                log_failed_attempt(
                    logger,
                    provider=PROVIDER,
                    attempt=attempt,
                    attempts=attempts,
                    cause=SCHEMA_VALIDATION_ERROR,
                )
                continue

            return ExtractionResult(
                extraction=map_llm_extraction(
                    llm,
                    letter_text=letter_text,
                    model_id=self._model_id,
                    prompt_version=PROMPT_VERSION,
                ),
                raw_llm_response=outcome,
                provider=PROVIDER,
                notes=llm.notes,
            )

        return ExtractionResult(
            extraction=malformed_extraction(
                letter_text=letter_text,
                model_id=self._model_id,
                prompt_version=PROMPT_VERSION,
            ),
            # Verbatim, even though it did not parse: the stored payload is what a
            # colleague debugging the schema break actually needs to look at. A refusal
            # and a truncated reply carry no payload at all, so the diagnostic stands in
            # for one rather than leaving the field empty.
            raw_llm_response=last_payload or last_diagnostic,
            provider=PROVIDER,
        )

    async def _call(self, letter_text: str) -> Mapping[str, Any] | BadAnswer:
        """One request. Returns the parsed reply, or why no usable answer came back.

        Transport retries belong to the SDK client, which already backs off on connection
        errors, 429 and 5xx. Anything that still reaches here has exhausted that budget,
        so it is an outage rather than a blip.
        """
        extra: dict[str, Any] = {}
        if self._reasoning_effort:
            extra["reasoning"] = {"effort": self._reasoning_effort}

        try:
            response = await self._client.responses.create(
                model=self._model_id,
                instructions=render_system_prompt(STRUCTURED_REPLY_INSTRUCTION),
                input=render_user_message(letter_text),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                text=cast(
                    Any,
                    {
                        "format": {
                            "type": "json_schema",
                            "name": EXTRACTION_TOOL_NAME,
                            "schema": STRICT_SCHEMA,
                            "strict": True,
                        }
                    },
                ),
                **extra,
            )
        except OpenAIError as error:
            raise ExtractorUnavailable(str(error)) from error

        if response.status != "completed":
            # "incomplete": the output-token budget ran out, or a filter stopped it. The
            # model answered badly, which is the retry path, not the outage path.
            return BadAnswer(
                cause=INCOMPLETE,
                payload={"_diagnostic": INCOMPLETE, "status": response.status},
            )

        for item in response.output:
            if item.type != "message":
                continue
            for part in item.content:
                if part.type == "refusal":
                    return BadAnswer(
                        cause=REFUSAL,
                        payload={"_diagnostic": REFUSAL, "refusal": part.refusal},
                    )
                if part.type == "output_text":
                    try:
                        parsed = json.loads(part.text)
                    except json.JSONDecodeError:
                        return BadAnswer(
                            cause=UNPARSEABLE_JSON,
                            payload={"_diagnostic": UNPARSEABLE_JSON, "text": part.text},
                        )
                    # Strict mode makes a non-object reply near-impossible, but a payload
                    # that is not a mapping cannot be validated or stored as one.
                    if isinstance(parsed, dict):
                        return cast(Mapping[str, Any], parsed)
                    return BadAnswer(
                        cause=NON_OBJECT_REPLY,
                        payload={"_diagnostic": NON_OBJECT_REPLY, "text": part.text},
                    )

        return BadAnswer(
            cause=NO_OUTPUT_MESSAGE,
            payload={"_diagnostic": NO_OUTPUT_MESSAGE, "status": response.status},
        )
