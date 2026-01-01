"""`route()` — the whole decision, as one pure function of one argument.

It takes a `LetterExtraction` and nothing else: no clock, no randomness, no I/O, no
pre-computed confidences handed in by an adapter. Grounding, validation and confidence
derivation all happen *here*, computed from the stored letter text, which is what makes
the replay guarantee real. If an adapter computed grounding and stored only its result,
replaying a decision would re-read a number somebody else had already decided; because it
is recomputed, a bug in the grounding function is caught by the replay test instead of
being frozen into the record.

The guarantee this service actually makes: identical extraction implies identical route,
identical confidences, identical fired rules and identical decisive rule. Model output may
still vary run to run, which is exactly why the guarantee sits downstream of it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from garnishment.domain.confidence import FieldConfidence
from garnishment.domain.enums import ResolutionReason, Route, RuleId
from garnishment.domain.extraction import LetterExtraction
from garnishment.domain.rules import RULES, build_rule_context
from garnishment.domain.values import Money

ROUTING_VERSION = "v1"
"""Bumped whenever the rule table changes, so a stored record names the table that decided it."""


@dataclass(frozen=True, slots=True)
class FiredRule:
    rule_id: RuleId
    route: Route | None
    reason: str


@dataclass(frozen=True, slots=True)
class RoutingOutcome:
    """The decision, plus every piece of evidence behind it.

    Everything here is derived from the extraction. It is persisted so a colleague can
    read the "why" without re-running anything, and never fed back in as an input.
    """

    route: Route
    decisive_rule_id: RuleId
    fired_rules: tuple[FiredRule, ...]
    reasons: tuple[str, ...]
    field_confidences: dict[str, FieldConfidence]
    operative_amount: Money | None
    operative_amount_reason: ResolutionReason
    considered_candidate_index: int | None
    source_candidate_index: int | None
    doctrine_version: str
    routing_version: str = ROUTING_VERSION


def route(extraction: LetterExtraction) -> RoutingOutcome:
    """Decide a route. Pure, total, and deterministic.

    Every rule in the table is evaluated and every fired rule is recorded, so the record
    carries all of a letter's defects rather than only the first. The decisive rule is the
    first fired **route-bearing** rule in table order; `I*` rules are excluded, because an
    informational observation can never be the reason a letter went anywhere.
    """
    context = build_rule_context(extraction)

    fired: list[FiredRule] = []
    fired_route_bearing: list[RuleId] = []
    for rule in RULES:
        evaluation_context = replace(context, fired_route_bearing=tuple(fired_route_bearing))
        if not rule.predicate(evaluation_context):
            continue
        fired.append(FiredRule(rule.rule_id, rule.route, rule.reason(evaluation_context)))
        if rule.route is not None:
            fired_route_bearing.append(rule.rule_id)

    decisive = next(f for f in fired if f.route is not None)
    resolution = context.resolution
    return RoutingOutcome(
        route=decisive.route if decisive.route is not None else Route.NEEDS_HUMAN,
        decisive_rule_id=decisive.rule_id,
        fired_rules=tuple(fired),
        reasons=tuple(f.reason for f in fired),
        field_confidences=dict(context.field_confidences),
        operative_amount=resolution.amount,
        operative_amount_reason=resolution.reason,
        considered_candidate_index=resolution.considered_candidate_index,
        source_candidate_index=resolution.source_candidate_index,
        doctrine_version=resolution.doctrine_version,
    )
