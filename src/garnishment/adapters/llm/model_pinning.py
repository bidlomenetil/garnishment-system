"""One rule for both adapters: a decision record must name exactly one model.

Every record carries the model id that produced it. An id that resolves to a different
model on different days makes two records claiming the same id incomparable, so both
adapters refuse one at construction — a typo or a lazy default fails the deployment
rather than one letter at a time.

The rule is shared; the *predicate* differs, because what counts as a moving alias is a
fact about each vendor's naming rather than a preference of ours.

  OpenAI    ships the bare family name as the moving alias and freezes each snapshot with
            a date suffix, so a date is required: `gpt-5-nano` moves, `gpt-5-nano-2025-08-07`
            does not.

  Anthropic ships the bare generation id as the pinned id and publishes the moving alias
            under a `-latest` suffix, so a date must *not* be required: `claude-sonnet-5`
            is the pinned id and `claude-sonnet-5-20260101` names no model at all.

Requiring a date of both would be consistency in the spelling of the check at the cost of
correctness in what it enforces — it would reject the only ids Anthropic serves. What is
shared instead is the rule, the refusal message, and the place to change either.
"""

from __future__ import annotations

import re
from typing import NoReturn

DATED_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")
"""A date suffix, which is how OpenAI distinguishes a frozen snapshot from a moving name."""

MOVING_ALIAS_SUFFIX = "-latest"
"""How Anthropic publishes a name that points at whatever is newest in a generation."""


def _refuse(model_id: str, hint: str) -> NoReturn:
    raise ValueError(
        f"Model id {model_id!r} is an alias. Every decision record names the model "
        f"that produced it, and an alias makes that name mean different things on "
        f"different days. {hint}"
    )


def require_pinned_openai_model(model_id: str) -> None:
    if not DATED_SNAPSHOT.search(model_id):
        _refuse(model_id, "Pin a dated snapshot, e.g. 'gpt-5-nano-2025-08-07'.")


def require_pinned_anthropic_model(model_id: str) -> None:
    if model_id.endswith(MOVING_ALIAS_SUFFIX):
        _refuse(model_id, "Pin an exact model id, e.g. 'claude-sonnet-5'.")
