"""The three assignment letters, read by whichever real provider is configured.

Everything else in the suite runs with no key, which is the point of `FakeExtractor`: the
recorded payloads travel the same schema and the same mapper a live reply does, so only
the transport is fake. What no offline test can establish is whether a real model fills
the fields the way the recorded payloads assume — in particular `role` and
`role_confidence` on each amount candidate, the one schema assumption the whole
restriction doctrine depends on. That is what this file is for.

It is provider-generic on purpose. `LLM_PROVIDER` chooses the reader, the matching key
decides whether the test runs at all, and the assertions name no provider: two adapters
behind one port should be indistinguishable here, and a divergence is exactly the finding
worth having. Without a key for the configured provider every case skips, so `make check`
from a fresh clone stays green; `make test` never selects them at all, because they are
marked `live`.

    export LLM_PROVIDER=openai OPENAI_API_KEY=sk-...
    uv run pytest -m live -q          # or: make live

The expected routes come from `evals/expectations.yaml`, filtered to the three letters the
assignment itself labelled. The five adversarial letters are our own reading rather than
given truth, so they belong in the eval report where instability is information, not in a
test whose failure would be a red build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from garnishment.adapters.llm.factory import ANTHROPIC, FAKE, OPENAI, build_extractor
from garnishment.config import Settings, get_settings
from garnishment.domain.enums import ExtractionStatus, Route, RuleId
from garnishment.domain.routing import route

EVALS = Path(__file__).resolve().parents[2] / "evals"

KEY_FOR_PROVIDER = {ANTHROPIC: "anthropic_api_key", OPENAI: "openai_api_key"}
"""Every provider that talks to a network, and the setting that lets it. `FAKE` is absent
because a fake reader would make these assertions prove nothing."""


def _configured_key(settings: Settings) -> str | None:
    provider = settings.llm_provider.strip().lower()
    field = KEY_FOR_PROVIDER.get(provider)
    return None if field is None else getattr(settings, field)


SETTINGS = get_settings()
LIVE_KEY = _configured_key(SETTINGS)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_KEY,
        reason=(
            f"LLM_PROVIDER={SETTINGS.llm_provider!r} has no key set. Set one of "
            f"{sorted(KEY_FOR_PROVIDER)} as LLM_PROVIDER with its API key to run these. "
            f"(LLM_PROVIDER={FAKE} can never run them: a recorded payload proves nothing "
            f"about a model.)"
        ),
    ),
]


def assignment_samples() -> list[Any]:
    """The three letters whose routes the assignment gave us, as pytest params."""
    expectations = yaml.safe_load((EVALS / "expectations.yaml").read_text())
    return [
        pytest.param(
            sample["file"],
            Route(sample["route"]),
            RuleId(sample["decisive_rule"]),
            id=sample["file"],
        )
        for sample in expectations["samples"]
        if sample["label_source"] == "assignment"
    ]


@pytest.mark.parametrize(("filename", "expected_route", "expected_rule"), assignment_samples())
async def test_a_real_model_reads_the_letter_into_the_labelled_route(
    filename: str, expected_route: Route, expected_rule: RuleId
) -> None:
    """Asserts the route (what the assignment actually labelled), not the decisive rule.

    The decisive rule for a labelled `expected_rule` came from our own `FakeExtractor`
    fixture, which is a hypothesis about what a real model would report, not a given
    truth the way the route is. Live evidence: gpt-5-nano puts sample 2's decisive rule at
    `H02_DOC_TYPE_UNCERTAIN` in most runs and `H05_DEBTOR_NAME_PARTIAL` in the rest, because
    its self-reported confidence that a genuinely terse letter is a seizure order lands on
    both sides of the 0.85 floor across runs — both rules are true and both fire every
    time; only which one is *decisive* (first in table order) moves. Asserting the exact
    rule here would fail a correct system for reporting an equally correct reason. That
    question belongs to `make eval`'s stability report, which counts it rather than
    red-building on it.

    What we DO assert as a hard failure is the one thing the whole design calls
    unacceptable: a wrong `auto`. A sample labelled `needs_human` must never read `auto`.
    A sample labelled `auto` may legitimately read `needs_human` on a run where the model
    is honestly less certain than usual — that is the system failing toward a human, which
    is safe — but must never read `reject`, since nothing about these three letters is
    invalid.
    """
    extractor = build_extractor(SETTINGS)
    letter = (EVALS / "samples" / filename).read_text()

    result = await extractor.extract(letter)
    outcome = route(result.extraction)
    as_read = [
        (c.value_as_written, c.role, c.role_confidence)
        for c in result.extraction.amount_candidates
    ]
    seizure_conf = result.extraction.is_seizure_order_confidence
    context = (
        f"{filename}: expected {expected_route.value} ({expected_rule.value}), got "
        f"{outcome.route.value} on {outcome.decisive_rule_id.value}. is_seizure_order="
        f"{result.extraction.is_seizure_order}@{seizure_conf:.2f}. Amount candidates as "
        f"read (value, role, role_confidence): {as_read}"
    )

    assert result.extraction.extraction_status is ExtractionStatus.OK, (
        f"The model's answer did not fit the schema after a retry. Raw payload: "
        f"{result.raw_llm_response}"
    )
    assert result.provider == SETTINGS.llm_provider.strip().lower()

    if expected_route is Route.AUTO:
        assert outcome.route is not Route.REJECT, (
            f"A letter labelled auto must never be rejected — that is the one direction "
            f"of error the design treats as unacceptable in either direction. {context}"
        )
        if outcome.route is not Route.AUTO:
            pytest.skip(
                f"Route de-escalated from auto to {outcome.route.value} on this run — "
                f"the safe direction, not a failure, but worth seeing rather than silently "
                f"passing. {context}"
            )
    else:
        assert outcome.route is expected_route, (
            f"A wrong auto is the one outcome this system exists to prevent. {context}"
        )
