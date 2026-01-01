"""The golden table: how we would know the routing rules are right.

Every rule appears as fired in at least one case, and every route-bearing rule appears as
decisive wherever it can be, so a rule-table change cannot silently alter behaviour. This
is the deterministic half of the answer; the eval harness measures the probabilistic half.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from garnishment.domain import thresholds
from garnishment.domain.decision import Decision, sha256_of
from garnishment.domain.enums import ResolutionReason, Route, RuleId
from garnishment.domain.routing import ROUTING_VERSION, route
from garnishment.domain.rules import ROUTE_BEARING_RULE_IDS, RULES
from tests.fixtures.extractions import GOLDEN_CASES, GoldenCase

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "garnishment"
DOMAIN_ROOT = SRC_ROOT / "domain"


# --- the golden cases --------------------------------------------------------------------


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=str)
def test_golden_case(case: GoldenCase) -> None:
    outcome = route(case.extraction)
    assert outcome.route is case.expected_route
    assert outcome.decisive_rule_id is case.expected_decisive_rule
    assert frozenset(f.rule_id for f in outcome.fired_rules) == case.expected_fired_rules


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=str)
def test_every_fired_rule_carries_a_rendered_reason(case: GoldenCase) -> None:
    outcome = route(case.extraction)
    assert len(outcome.reasons) == len(outcome.fired_rules)
    for fired in outcome.fired_rules:
        assert fired.reason
        assert "{" not in fired.reason, "a reason template was left uninterpolated"
        assert "None" not in fired.reason, "a reason line printed a bare None"


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=str)
def test_the_decisive_rule_is_the_first_fired_route_bearing_rule(case: GoldenCase) -> None:
    outcome = route(case.extraction)
    route_bearing = [f for f in outcome.fired_rules if f.route is not None]
    assert route_bearing[0].rule_id is outcome.decisive_rule_id
    assert route_bearing[0].route is outcome.route


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=str)
def test_informational_rules_are_never_decisive(case: GoldenCase) -> None:
    assert not route(case.extraction).decisive_rule_id.value.startswith("I")


# --- the three labelled samples, by name --------------------------------------------------


def _case(name: str) -> GoldenCase:
    return next(c for c in GOLDEN_CASES if c.name == name)


def test_the_three_labelled_samples_route_for_the_right_reasons() -> None:
    expected = [
        ("sample_1_attachment_order_auto", Route.AUTO, RuleId.A00_AUTO_DEFAULT),
        (
            "sample_2_partial_name_needs_human",
            Route.NEEDS_HUMAN,
            RuleId.H05_DEBTOR_NAME_PARTIAL,
        ),
        ("sample_3_restriction_order_auto", Route.AUTO, RuleId.A00_AUTO_DEFAULT),
    ]
    for name, expected_route, expected_rule in expected:
        outcome = route(_case(name).extraction)
        assert (outcome.route, outcome.decisive_rule_id) == (expected_route, expected_rule)


def test_sample_2_reports_both_that_the_figure_was_approximate_and_unreadable() -> None:
    """The prose-amount path, and the reason both indices exist.

    The letter pointed at one figure that was approximate AND could not be read. Under a
    design with a single candidate index, the parse failure would have silently suppressed
    H04 and the clerk would have lost the more informative of the two facts.
    """
    outcome = route(_case("sample_2_partial_name_needs_human").extraction)
    fired = {f.rule_id for f in outcome.fired_rules}
    assert RuleId.H04_AMOUNT_APPROXIMATE in fired
    assert RuleId.H16_AMOUNT_VALUE_UNPARSEABLE in fired
    assert outcome.considered_candidate_index == 0
    assert outcome.source_candidate_index is None
    assert outcome.operative_amount is None
    assert outcome.operative_amount_reason is ResolutionReason.UNPARSEABLE
    assert RuleId.H13_AMOUNT_FORMAT_AMBIGUOUS not in fired, (
        "format ambiguity is a claim about a figure that parsed"
    )


def test_sample_3_resolves_the_restricted_figure_under_the_doctrine() -> None:
    outcome = route(_case("sample_3_restriction_order_auto").extraction)
    assert outcome.operative_amount is not None
    assert str(outcome.operative_amount) == "EUR 1267.00"
    assert outcome.operative_amount_reason is ResolutionReason.RESTRICTION_SUPERSEDES
    assert outcome.considered_candidate_index == 1
    assert outcome.source_candidate_index == 1
    assert outcome.doctrine_version == "RESTRICTION_DOCTRINE_V1"


def test_sample_3_four_failure_modes_all_escalate() -> None:
    expected = {
        "sample_3_both_candidates_unknown": RuleId.H03_AMOUNT_AMBIGUOUS,
        "sample_3_only_the_original_figure_returned": RuleId.H11_OPERATIVE_AMOUNT_UNRESOLVED,
        "sample_3_restriction_raises_the_amount": RuleId.H12_RESTRICTION_NOT_REDUCING,
        "sample_3_restricted_role_label_not_trusted": RuleId.H03_AMOUNT_AMBIGUOUS,
    }
    for name, expected_rule in expected.items():
        outcome = route(_case(name).extraction)
        assert outcome.route is Route.NEEDS_HUMAN, name
        assert outcome.decisive_rule_id is expected_rule, name


def test_the_auto_reason_quotes_the_true_minimum_across_the_gated_fields() -> None:
    outcome = route(_case("sample_1_attachment_order_auto").extraction)
    lowest = min(c.final for c in outcome.field_confidences.values())
    assert f"{lowest:.2f}" in outcome.reasons[0]


# --- rule coverage, not fixture count ------------------------------------------------------


def test_every_rule_id_fires_in_at_least_one_golden_case() -> None:
    fired: set[RuleId] = set()
    for case in GOLDEN_CASES:
        fired.update(f.rule_id for f in route(case.extraction).fired_rules)
    assert set(RuleId) - fired == set()


def test_every_route_bearing_rule_is_decisive_in_at_least_one_golden_case() -> None:
    decisive = {route(case.extraction).decisive_rule_id for case in GOLDEN_CASES}
    assert set(ROUTE_BEARING_RULE_IDS) - decisive == set()


def test_all_three_routes_are_reachable() -> None:
    assert {route(case.extraction).route for case in GOLDEN_CASES} == set(Route)


# --- the shape of the table ------------------------------------------------------------------

EXPECTED_TABLE_ORDER: tuple[RuleId, ...] = (
    # reject
    RuleId.R01_NOT_SEIZURE_ORDER,
    RuleId.R02_CASE_REFERENCE_MISSING,
    RuleId.R03_CASE_REFERENCE_MALFORMED,
    RuleId.R04_AMOUNT_MISSING,
    RuleId.R05_AMOUNT_NON_POSITIVE,
    # escalate: could we read the letter at all
    RuleId.H01_EXTRACTION_FAILED,
    RuleId.H17_DOCUMENT_TYPE_CONTRADICTION,
    RuleId.H02_DOC_TYPE_UNCERTAIN,
    # escalate: identity
    RuleId.H05_DEBTOR_NAME_PARTIAL,
    RuleId.H06_IBAN_ABSENT,
    RuleId.H07_IBAN_INVALID,
    # escalate: amount
    RuleId.H11_OPERATIVE_AMOUNT_UNRESOLVED,
    RuleId.H12_RESTRICTION_NOT_REDUCING,
    RuleId.H03_AMOUNT_AMBIGUOUS,
    RuleId.H18_MULTIPLE_FIGURES_NO_RESTRICTION_CONTEXT,
    RuleId.H04_AMOUNT_APPROXIMATE,
    RuleId.H16_AMOUNT_VALUE_UNPARSEABLE,
    RuleId.H13_AMOUNT_FORMAT_AMBIGUOUS,
    RuleId.H14_CURRENCY_NOT_EUR,
    RuleId.H15_AMOUNT_IMPLAUSIBLE,
    # escalate: evidence quality
    RuleId.H08_UNGROUNDED_EVIDENCE,
    RuleId.H09_LOW_CONFIDENCE_FIELD,
    # the automatic default
    RuleId.A00_AUTO_DEFAULT,
    # informational
    RuleId.I01_OPERATIVE_AMOUNT_RESOLVED,
    RuleId.I02_IBAN_VALIDATED,
    RuleId.I03_GROUNDED_ON_VALUE,
    RuleId.I04_INJECTION_SUSPECTED,
)


def test_the_table_is_in_the_documented_order() -> None:
    """Order is the tie-break, so it is asserted rather than assumed.

    Reject rules sit first, which makes reject dominance structural. The escalation blocks
    follow the order a clerk needs: could we read it, whose money is it, how much, how well
    evidenced. Informational rules sit last and bear no route.
    """
    assert tuple(rule.rule_id for rule in RULES) == EXPECTED_TABLE_ORDER


def test_every_rule_id_appears_in_the_table_exactly_once() -> None:
    ids = [rule.rule_id for rule in RULES]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(RuleId)


def test_routes_match_the_rule_id_prefixes() -> None:
    for rule in RULES:
        prefix = rule.rule_id.value[0]
        expected = {"R": Route.REJECT, "H": Route.NEEDS_HUMAN, "A": Route.AUTO, "I": None}[prefix]
        assert rule.route == expected, rule.rule_id


def test_the_route_enum_carries_no_severity_ordering() -> None:
    """Escalation and rejection are different kinds of outcome, not points on one scale."""
    assert not hasattr(Route, "severity")
    for member in Route:
        assert not hasattr(member, "severity")


# --- thresholds are named, referenced, and never inlined --------------------------------------

ADAPTER_SIDE_THRESHOLDS = frozenset({"EXTRACTION_MAX_RETRIES", "LLM_TEMPERATURE"})
"""Consumed by the extractor adapter, which sits outside the domain by design."""


def _threshold_names() -> list[str]:
    return [
        name
        for name in vars(thresholds)
        if name.isupper() and not name.startswith("_") and name != "CALIBRATION_PROCEDURE"
    ]


def test_every_threshold_is_referenced_by_the_code_it_governs() -> None:
    sources = {
        path: path.read_text()
        for path in SRC_ROOT.rglob("*.py")
        if path.name != "thresholds.py"
    }
    unreferenced = [
        name
        for name in _threshold_names()
        if name not in ADAPTER_SIDE_THRESHOLDS
        and not any(name in text for text in sources.values())
    ]
    assert unreferenced == []


def test_every_threshold_states_a_hypothesis_and_a_test() -> None:
    """These docstrings are the artefact to open when asked why a number sits where it does."""
    source = (DOMAIN_ROOT / "thresholds.py").read_text()
    for name in _threshold_names():
        _, _, after = source.partition(f"{name} = ")
        docstring, _, _ = after.partition('"""\n\n')
        assert "Hypothesis" in docstring, name
        assert "Test:" in docstring, name


def test_the_rule_table_contains_no_numeric_literals() -> None:
    """Every comparison the table makes must be traceable to a named threshold."""
    tree = ast.parse((DOMAIN_ROOT / "rules.py").read_text())
    literals = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ]
    assert literals == []


# --- the audit record ---------------------------------------------------------------------

FIXED_ID = "01K5QZ8T7J3V9M2X4C6B8N0P1R"
FIXED_TIME = datetime(2026, 9, 22, 14, 3, 11, 482000, tzinfo=UTC)


def _record(case: GoldenCase) -> Decision:
    return Decision.from_routing(
        decision_id=FIXED_ID,
        created_at=FIXED_TIME,
        extraction=case.extraction,
        outcome=route(case.extraction),
        extractor_provider="fake",
        raw_llm_response={"tool": "record_garnishment_extraction"},
    )


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=str)
def test_the_record_carries_the_decision_and_its_evidence(case: GoldenCase) -> None:
    decision = _record(case)
    assert decision.route is case.expected_route
    assert decision.decisive_rule_id is case.expected_decisive_rule
    assert decision.fired_rule_ids == tuple(
        f.rule_id for f in route(case.extraction).fired_rules
    )
    assert decision.letter_sha256 == sha256_of(case.extraction.letter_text)
    assert decision.doctrine_version == "RESTRICTION_DOCTRINE_V1"
    assert decision.routing_version == ROUTING_VERSION


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=str)
def test_rerouting_a_stored_record_reproduces_it_exactly(case: GoldenCase) -> None:
    """The replay guarantee, in its purest form: no persistence, just the function.

    `route()` is a pure function of `LetterExtraction` alone, so re-running it over a
    record's stored extraction must reproduce the route, the confidences, the fired rules
    and the decisive rule. The integration suite asserts the same thing through a real
    database round trip, which is what proves serialisation is lossless.
    """
    decision = _record(case)
    replayed = route(decision.extraction)
    assert replayed.route is decision.route
    assert replayed.decisive_rule_id is decision.decisive_rule_id
    assert replayed.fired_rules == decision.fired_rules
    assert replayed.field_confidences == decision.field_confidences
    assert replayed.reasons == decision.reasons
    assert replayed.operative_amount == decision.operative_amount
    assert replayed.considered_candidate_index == decision.considered_candidate_index
    assert replayed.source_candidate_index == decision.source_candidate_index


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=str)
def test_the_stored_extraction_holds_no_parsed_figure(case: GoldenCase) -> None:
    """One representation per number, and it is the one the letter used.

    Storing both a verbatim string and a parsed value would give one quantity two
    representations that could drift apart across a schema change, and replay would then
    be asserting against the wrong one.
    """
    decision = _record(case)
    for candidate in decision.extraction.amount_candidates:
        for value in vars(type(candidate)).get("__slots__", ()):
            assert not isinstance(getattr(candidate, value), Decimal), value
        assert isinstance(candidate.value_as_written, str)


def test_the_record_separates_its_input_from_its_derived_evidence() -> None:
    """Everything below the line in the record is recomputed, never fed back in."""
    case = _case("sample_3_restriction_order_auto")
    decision = _record(case)
    assert decision.extraction is case.extraction
    assert decision.operative_amount is not None
    assert str(decision.operative_amount) == "EUR 1267.00"
    assert decision.created_at == FIXED_TIME
    assert decision.id == FIXED_ID
