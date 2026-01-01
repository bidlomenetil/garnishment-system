"""Choose a reader from configuration, once, at startup.

An unknown `LLM_PROVIDER` raises here rather than on the first request, so a typo in the
environment fails the deployment instead of failing one letter at a time in production.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from garnishment.adapters.llm.anthropic_extractor import AnthropicExtractor
from garnishment.adapters.llm.fake_extractor import FakeExtractor
from garnishment.adapters.llm.openai_extractor import OpenAIExtractor
from garnishment.application.ports import ExtractorPort
from garnishment.config import Settings

FAKE = "fake"
ANTHROPIC = "anthropic"
OPENAI = "openai"
PROVIDERS = (FAKE, ANTHROPIC, OPENAI)

TRANSPORT_RETRIES = 1
"""How many times the SDK client retries a connection error, 429 or 5xx before the adapter
calls it an outage. One retry over the adapter's own identical retry is the whole budget:
a schema miss is a completed call and is never charged twice for transport reasons.

Passed explicitly to *both* clients. Left unset, the Anthropic SDK applies its own default
of two, which would make the same malformed letter cost up to six upstream calls on one
provider and two on the other — the adapters would no longer be the matched pair the port
claims they are, and a budget nobody wrote down is not a budget."""


def build_extractor(settings: Settings) -> ExtractorPort:
    provider = settings.llm_provider.strip().lower()
    if provider == FAKE:
        return FakeExtractor()
    if provider == ANTHROPIC:
        return AnthropicExtractor(
            AsyncAnthropic(api_key=settings.anthropic_api_key, max_retries=TRANSPORT_RETRIES),
            model_id=settings.anthropic_model,
        )
    if provider == OPENAI:
        return OpenAIExtractor(
            AsyncOpenAI(api_key=settings.openai_api_key, max_retries=TRANSPORT_RETRIES),
            model_id=settings.openai_model,
            reasoning_effort=settings.openai_reasoning_effort,
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER {settings.llm_provider!r}; expected one of {PROVIDERS}."
    )
