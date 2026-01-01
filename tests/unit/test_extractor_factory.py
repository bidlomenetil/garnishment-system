"""Choosing a reader is a startup decision, and a wrong one fails there rather than later."""

from __future__ import annotations

import pytest

from garnishment.adapters.llm.anthropic_extractor import AnthropicExtractor
from garnishment.adapters.llm.factory import PROVIDERS, TRANSPORT_RETRIES, build_extractor
from garnishment.adapters.llm.fake_extractor import FakeExtractor
from garnishment.adapters.llm.openai_extractor import OpenAIExtractor
from garnishment.config import Settings


def settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "llm_provider": "fake",
        "anthropic_api_key": "test-key-not-used",
        "anthropic_model": "claude-sonnet-5",
        "openai_api_key": "test-key-not-used",
        "openai_model": "gpt-5-nano-2025-08-07",
    }
    return Settings(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_the_fake_provider_builds() -> None:
    assert isinstance(build_extractor(settings(llm_provider="fake")), FakeExtractor)


def test_the_anthropic_provider_builds() -> None:
    extractor = build_extractor(settings(llm_provider="anthropic"))

    assert isinstance(extractor, AnthropicExtractor)
    assert extractor.provider == "anthropic"


def test_the_provider_name_is_read_case_insensitively() -> None:
    assert isinstance(build_extractor(settings(llm_provider=" Fake ")), FakeExtractor)


def test_the_openai_provider_builds() -> None:
    extractor = build_extractor(settings(llm_provider="openai"))

    assert isinstance(extractor, OpenAIExtractor)
    assert extractor.provider == "openai"


def test_both_clients_keep_the_same_transport_retry_budget() -> None:
    """The SDK owns transport retries; the adapter owns the one identical schema retry.

    The budget has to be passed to both, not just to OpenAI. The Anthropic SDK defaults to
    two retries of its own, so leaving it unset made one malformed letter cost up to six
    upstream calls on one provider and two on the other — the two adapters would not be
    the matched pair `ExtractorPort` claims they are.
    """
    openai_extractor = build_extractor(settings(llm_provider="openai"))
    anthropic_extractor = build_extractor(settings(llm_provider="anthropic"))

    assert isinstance(openai_extractor, OpenAIExtractor)
    assert isinstance(anthropic_extractor, AnthropicExtractor)
    assert openai_extractor._client.max_retries == TRANSPORT_RETRIES == 1
    assert anthropic_extractor._client.max_retries == TRANSPORT_RETRIES


def test_an_unknown_provider_fails_at_startup_rather_than_per_request() -> None:
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_extractor(settings(llm_provider="gemini"))


def test_the_unknown_provider_message_names_every_provider_that_would_have_worked() -> None:
    with pytest.raises(ValueError) as caught:
        build_extractor(settings(llm_provider="gemini"))

    assert all(provider in str(caught.value) for provider in PROVIDERS)
    assert PROVIDERS == ("fake", "anthropic", "openai")


def test_an_aliased_model_id_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="alias"):
        build_extractor(settings(llm_provider="anthropic", anthropic_model="claude-latest"))


def test_an_undated_openai_model_id_fails_at_startup() -> None:
    """`gpt-5-nano` is a moving alias; only the dated snapshot names one model."""
    with pytest.raises(ValueError, match="alias"):
        build_extractor(settings(llm_provider="openai", openai_model="gpt-5-nano"))
