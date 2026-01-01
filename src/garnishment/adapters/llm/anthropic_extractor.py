"""The real reader: one forced tool call to the Messages API, one retry, no cleverness.

Two failures are handled here and they are handled differently on purpose.

*The model answered badly* — no tool call, or arguments that do not fit the schema. We
retry once, identically, and if the second attempt also fails we return an extraction
carrying `MALFORMED_AFTER_RETRY`. We never raise, and the letter is never rejected: a
schema break is evidence about us, not about the court. Each failed attempt logs one
warning naming its cause, because both causes land on the same recorded outcome.

*The model never answered* — transport failure, bad credentials, rate limiting that
survived the SDK's own backoff. We raise `ExtractorUnavailable`, no record is written and
the caller owes the client a 503. A `needs_human` record for an outage would put an empty
work item in a clerk's queue and make an infrastructure problem indistinguishable from an
ambiguous letter.

Determinism levers, in the order they matter: forced structured output removes free-form
drift, the model id is pinned rather than aliased, and `PROMPT_VERSION` is recorded.

**Sampling is not one of the levers, and cannot be. This is a documented deviation from
the design's §7.8, item 1** ("temperature = 0.0; top_p pinned in the adapter").
`thresholds.LLM_TEMPERATURE` still states the intent, but the Messages API removed
`temperature`, `top_p` and `top_k` for this generation of models and the SDK dropped them
from the signature: sending one is a call-time error, not a request the model would
decline. Forcing the call anyway would break every extraction, so the parameter is simply
not sent.

The determinism story survives intact, because it never rested on the model repeating
itself — §7.8 item 5 is explicit that the guarantee sits downstream, in `route()` being a
pure function of a stored extraction, precisely so that model variance cannot reach it.
What is lost is one of the four *reducers* of that variance, not the guarantee. The eval
harness measures the residual rather than assuming it away.

`tests/unit/test_anthropic_extractor.py` pins the SDK signature, so if a future release
restores the parameter the test fails and sampling should be pinned here again.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

from anthropic import AnthropicError, AsyncAnthropic
from pydantic import ValidationError

from garnishment.adapters.llm.attempts import (
    SCHEMA_VALIDATION_ERROR,
    BadAnswer,
    log_failed_attempt,
)
from garnishment.adapters.llm.mapping import malformed_extraction, map_llm_extraction
from garnishment.adapters.llm.model_pinning import require_pinned_anthropic_model
from garnishment.adapters.llm.prompt import (
    PROMPT_VERSION,
    TOOL_CALL_INSTRUCTION,
    render_system_prompt,
    render_user_message,
)
from garnishment.adapters.llm.schema import (
    EXTRACTION_TOOL_NAME,
    LlmExtraction,
    extraction_tool,
)
from garnishment.application.errors import ExtractorUnavailable
from garnishment.application.ports import ExtractionResult
from garnishment.domain.thresholds import EXTRACTION_MAX_RETRIES

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"

NO_TOOL_CALL = "no_tool_call"
"""The one way this transport can answer badly without producing arguments at all."""

MAX_TOKENS = 4096
"""Ample for one extraction payload; the schema caps how much there is to say."""


class AnthropicExtractor:
    """Reads a letter with a single forced tool call against a pinned model."""

    def __init__(self, client: AsyncAnthropic, *, model_id: str) -> None:
        require_pinned_anthropic_model(model_id)
        self._client = client
        self._model_id = model_id

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
            # colleague debugging the schema break actually needs to look at. When no
            # attempt produced one, the diagnostic says what happened instead, which is
            # still more than the empty mapping this used to store.
            raw_llm_response=last_payload or last_diagnostic,
            provider=PROVIDER,
        )

    async def _call(self, letter_text: str) -> Mapping[str, Any] | BadAnswer:
        """One request. Returns the tool arguments, or why no tool call came back.

        Transport retries belong to the SDK client, which already backs off on connection
        errors, 429 and 5xx. Anything that still reaches here has exhausted that budget,
        so it is an outage rather than a blip.
        """
        try:
            response = await self._client.messages.create(
                model=self._model_id,
                max_tokens=MAX_TOKENS,
                system=render_system_prompt(TOOL_CALL_INSTRUCTION),
                messages=[{"role": "user", "content": render_user_message(letter_text)}],
                tools=[cast(Any, extraction_tool())],
                tool_choice={"type": "tool", "name": EXTRACTION_TOOL_NAME},
            )
        except AnthropicError as error:
            raise ExtractorUnavailable(str(error)) from error

        for block in response.content:
            if block.type == "tool_use" and block.name == EXTRACTION_TOOL_NAME:
                return cast(Mapping[str, Any], block.input)

        said = " ".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        return BadAnswer(
            cause=NO_TOOL_CALL,
            payload={
                "_diagnostic": NO_TOOL_CALL,
                "block_types": [block.type for block in response.content],
                "text": said,
            },
        )
