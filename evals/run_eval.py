"""Run every labelled letter N times and report what happened.

The report is the deliverable. This harness does **not** assert stability as a pass
condition, and that is deliberate: instability is information about the reader, and hiding
it behind a red build would be the wrong instinct in a service whose whole subject is
honest confidence. A sample that routes 3/5 the same way is a finding to write down, not a
failure to suppress.

The one outcome that is never acceptable is a **wrong `auto`** — a letter the label says a
human should see, or that should be refused, that the service cleared automatically. That
is the only condition that sets a non-zero exit code.

What the numbers mean depends on which reader answered, and the report says so at the top.
Against `FakeExtractor` the extraction is a recorded payload looked up by the SHA-256 of
the letter, so route stability is 5/5 by construction and measures nothing about a model.
It still measures something worth having: that `route()` is deterministic over a fixed
extraction, and that each letter reaches the rule the label says it should. The stability
column only becomes evidence about a model when `LLM_PROVIDER` is `anthropic` or `openai`
and that provider's key is set.

`--jsonl` writes the same runs a second time as machine-readable records, one JSON object
per (letter, run). The human report answers "what happened"; the JSONL answers "what were
the numbers behind it" — every gated field's `llm`/`grounding`/`validator`/`final`
breakdown, the fired rules, and each required field's extracted value against the value
`expectations.yaml` labels. It is what `sweep.py` reads, and what a later pass can score
per-field accuracy from without paying for the calls again.

    uv run python evals/run_eval.py                 # default: 5 runs per sample
    uv run python evals/run_eval.py --runs 3
    uv run python evals/run_eval.py --out report.txt
    uv run python evals/run_eval.py --jsonl runs.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import textwrap
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from garnishment.adapters.llm.factory import build_extractor
from garnishment.application.errors import ExtractorUnavailable
from garnishment.config import get_settings
from garnishment.domain.confidence import (
    ACCOUNT_IBAN,
    AMOUNT,
    CASE_REFERENCE,
    DEBTOR_NAME,
    gated_field_names,
)
from garnishment.domain.enums import Route, RuleId
from garnishment.domain.extraction import LetterExtraction
from garnishment.domain.routing import FiredRule, RoutingOutcome, route

EVALS_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = EVALS_DIR / "samples"
EXPECTATIONS = EVALS_DIR / "expectations.yaml"

DEFAULT_RUNS = 5

SCORED_FIELDS = (DEBTOR_NAME, CASE_REFERENCE, AMOUNT, ACCOUNT_IBAN)
"""The fields `expectations.yaml` carries an expected value for."""

THRESHOLD_GATED_RULES = frozenset({RuleId.H09_LOW_CONFIDENCE_FIELD, RuleId.A00_AUTO_DEFAULT})
"""The two route-bearing rules whose firing depends on `AUTO_MIN_FIELD_CONFIDENCE`.

Everything else in the table decides on evidence the threshold cannot move, which is what
makes the sweep in `sweep.py` an arithmetic replay rather than a re-run: if any *other*
route-bearing rule fired, that rule is decisive at every threshold.
"""

FAKE_PROVIDER_CAVEAT = (
    "Reader is FakeExtractor: every extraction is a payload recorded offline and looked\n"
    "up by the SHA-256 of the letter. Route stability below is therefore 5/5 by\n"
    "construction and is NOT evidence about a model. What it does show is that route()\n"
    "is deterministic over a fixed extraction and that each letter reaches the rule its\n"
    "label names. Set LLM_PROVIDER=anthropic or openai with that provider's key to\n"
    "make the stability column mean something."
)

LIVE_PROVIDER_NOTE = (
    "Reader is a live provider. The stability column now measures real run-to-run\n"
    "variance in the extraction. Anything below {runs}/{runs} belongs in DESIGN.md as a\n"
    "named finding with its observed variance, not in a red build."
)


@dataclass(frozen=True, slots=True)
class Expectation:
    file: str
    route: str
    decisive_rule: str
    label_source: str
    note: str
    expected_values: dict[str, str | None]
    """Per-field labels read off the letter. `None` means "no correct value exists"."""

    values_note: str = ""


@dataclass(frozen=True, slots=True)
class FieldScore:
    """One gated field's confidence breakdown, flattened for serialisation."""

    llm: float
    grounding: float
    validator: float
    final: float
    value_present: bool
    grounded_on_value_only: bool
    validator_code: str
    gated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm": self.llm,
            "grounding": self.grounding,
            "validator": self.validator,
            "final": self.final,
            "value_present": self.value_present,
            "grounded_on_value_only": self.grounded_on_value_only,
            "validator_code": self.validator_code,
            "gated": self.gated,
        }


@dataclass(slots=True)
class Observation:
    """One run of one letter."""

    route: str
    decisive_rule: str
    operative_amount: str
    role_confidences: tuple[float, ...]
    run_index: int = 0
    fired_rules: tuple[str, ...] = ()
    field_scores: dict[str, FieldScore] = field(default_factory=dict)
    gated_fields: tuple[str, ...] = ()
    observed_values: dict[str, str | None] = field(default_factory=dict)
    blocking_rule: str | None = None
    """First fired route-bearing rule that `AUTO_MIN_FIELD_CONFIDENCE` cannot move."""

    blocking_route: str | None = None
    model_id: str = ""
    prompt_version: str = ""
    provider: str = ""
    extraction_status: str = ""
    operative_amount_reason: str = ""


@dataclass(frozen=True, slots=True)
class RunError:
    """A run the reader never answered. No decision exists, so no route was observed."""

    run_index: int
    message: str


@dataclass(slots=True)
class SampleReport:
    expectation: Expectation
    observations: list[Observation] = field(default_factory=list)
    errors: list[RunError] = field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.observations) + len(self.errors)

    @property
    def route_counts(self) -> Counter[str]:
        return Counter(o.route for o in self.observations)

    @property
    def rule_counts(self) -> Counter[str]:
        return Counter(o.decisive_rule for o in self.observations)

    @property
    def route_stability(self) -> int:
        counts = self.route_counts
        return 0 if not counts else counts.most_common(1)[0][1]

    @property
    def rule_stability(self) -> int:
        counts = self.rule_counts
        return 0 if not counts else counts.most_common(1)[0][1]

    @property
    def matches_label(self) -> bool:
        return bool(self.observations) and all(
            o.route == self.expectation.route for o in self.observations
        )

    @property
    def wrong_auto_runs(self) -> int:
        """Runs that cleared a letter automatically when the label says otherwise."""
        if self.expectation.route == Route.AUTO.value:
            return 0
        return sum(1 for o in self.observations if o.route == Route.AUTO.value)

    @property
    def auto_runs(self) -> int:
        return sum(1 for o in self.observations if o.route == Route.AUTO.value)


def load_expectations(path: Path = EXPECTATIONS) -> list[Expectation]:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Expectation(
            file=str(entry["file"]),
            route=str(entry["route"]),
            decisive_rule=str(entry["decisive_rule"]),
            label_source=str(entry.get("label_source", "ours")),
            note=str(entry.get("note", "")).strip(),
            expected_values=_expected_values(entry),
            values_note=str(entry.get("expected_values_note", "")).strip(),
        )
        for entry in raw["samples"]
    ]


def _expected_values(entry: dict[str, Any]) -> dict[str, str | None]:
    """Read the per-field labels, defaulting every unlabelled field to "no value".

    A missing `expected_values` block therefore reads as "nothing is expected" rather than
    raising, so an older expectations file still loads and simply scores no fields.
    """
    raw = entry.get("expected_values") or {}
    return {name: (None if raw.get(name) is None else str(raw[name])) for name in SCORED_FIELDS}


def _normalise_value(field_name: str, value: str | None) -> str | None:
    """Put an expected and an observed value into one comparable form.

    Each field is compared the way the letter writes it, not byte for byte: an IBAN
    without its spaces is the same IBAN, a name is compared case-insensitively, and an
    amount is compared as a number so `4,250.00` and `4250.00` agree.
    """
    if value is None:
        return None
    if field_name == ACCOUNT_IBAN:
        return "".join(value.split()).upper()
    if field_name == AMOUNT:
        cleaned = value.replace(",", "").replace("EUR", "").strip()
        try:
            return str(Decimal(cleaned))
        except InvalidOperation:
            return cleaned.casefold()
    return " ".join(value.split()).casefold()


def field_comparison(expectation: Expectation, observation: Observation) -> dict[str, Any]:
    """Score every labelled field's value against what the reader produced."""
    comparison: dict[str, Any] = {}
    for name in SCORED_FIELDS:
        expected = expectation.expected_values.get(name)
        observed = observation.observed_values.get(name)
        comparison[name] = {
            "expected": expected,
            "observed": observed,
            "match": _normalise_value(name, expected) == _normalise_value(name, observed),
        }
    return comparison


def _first_blocking_rule(outcome: RoutingOutcome) -> FiredRule | None:
    for fired in outcome.fired_rules:
        if fired.route is not None and fired.rule_id not in THRESHOLD_GATED_RULES:
            return fired
    return None


def _observed_values(
    extraction: LetterExtraction, outcome: RoutingOutcome
) -> dict[str, str | None]:
    """The values a per-field score is taken against.

    `amount` is the *resolved operative* figure rather than a raw candidate, because that
    is the number the decision is about; when nothing resolved, the correct observation is
    that no amount was produced.
    """
    debtor = extraction.debtor_name.value
    case = extraction.case_reference.value
    iban = extraction.account_iban.value
    return {
        DEBTOR_NAME: None if debtor is None else debtor.raw,
        CASE_REFERENCE: None if case is None else case.raw,
        AMOUNT: None if outcome.operative_amount is None else str(outcome.operative_amount.amount),
        ACCOUNT_IBAN: None if iban is None else iban.raw,
    }


async def observe(extractor: Any, letter_text: str, run_index: int = 0) -> Observation:
    result = await extractor.extract(letter_text)
    extraction = result.extraction
    outcome = route(extraction)
    amount = "none" if outcome.operative_amount is None else str(outcome.operative_amount)
    gated = gated_field_names(extraction)
    blocking = _first_blocking_rule(outcome)
    return Observation(
        route=str(outcome.route),
        decisive_rule=str(outcome.decisive_rule_id),
        operative_amount=amount,
        role_confidences=tuple(c.role_confidence for c in extraction.amount_candidates),
        run_index=run_index,
        fired_rules=tuple(str(f.rule_id) for f in outcome.fired_rules),
        field_scores={
            name: FieldScore(
                llm=confidence.llm,
                grounding=confidence.grounding,
                validator=confidence.validator,
                final=confidence.final,
                value_present=confidence.value_present,
                grounded_on_value_only=confidence.grounded_on_value_only,
                validator_code=confidence.validator_code,
                gated=name in gated,
            )
            for name, confidence in outcome.field_confidences.items()
        },
        gated_fields=gated,
        observed_values=_observed_values(extraction, outcome),
        blocking_rule=None if blocking is None else str(blocking.rule_id),
        blocking_route=None if blocking is None or blocking.route is None else str(blocking.route),
        model_id=extraction.model_id,
        prompt_version=extraction.prompt_version,
        provider=result.provider,
        extraction_status=str(extraction.extraction_status),
        operative_amount_reason=str(outcome.operative_amount_reason),
    )


async def run_all(runs: int, concurrency: int = 1) -> tuple[list[SampleReport], str]:
    """Read every labelled letter `runs` times.

    `concurrency` defaults to 1, which reproduces the original strictly sequential order;
    a higher value only overlaps the waiting, since `route()` is pure and each run is
    recorded against its own letter and index.
    """
    settings = get_settings()
    extractor = build_extractor(settings)
    expectations = load_expectations()
    reports = [SampleReport(e) for e in expectations]
    letters = {e.file: (SAMPLES_DIR / e.file).read_text(encoding="utf-8") for e in expectations}
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(expectation: Expectation, run_index: int) -> Observation | RunError:
        async with semaphore:
            try:
                return await observe(extractor, letters[expectation.file], run_index)
            except ExtractorUnavailable as error:
                # The reader never answered. No decision exists, so there is nothing to
                # route; the run is recorded as an outage rather than as a route we did
                # not get. The exception's own text can carry the provider SDK's raw
                # error body (a 401 response can echo a partially-redacted key), and this
                # message ends up in a report file that isn't gitignored — so record the
                # class name only, never str(error).
                return RunError(run_index, f"reader unavailable: {type(error).__name__}")

    results = await asyncio.gather(
        *(one(report.expectation, i) for report in reports for i in range(runs))
    )
    for report, result in zip(
        [r for r in reports for _ in range(runs)], results, strict=True
    ):
        if isinstance(result, RunError):
            report.errors.append(result)
        else:
            report.observations.append(result)
    return reports, extractor.provider


def _format_counts(counts: Counter[str]) -> str:
    if not counts:
        return "(none)"
    return ", ".join(f"{name} x{n}" for name, n in counts.most_common())


def _format_role_confidences(report: SampleReport) -> str:
    seen: list[str] = []
    for observation in report.observations:
        rendered = (
            "(no candidates)"
            if not observation.role_confidences
            else ", ".join(f"{c:.2f}" for c in observation.role_confidences)
        )
        if rendered not in seen:
            seen.append(rendered)
    return " | ".join(seen) if seen else "(none)"


def render(reports: Sequence[SampleReport], provider: str, runs: int) -> list[str]:
    lines: list[str] = []
    lines.append("=" * 88)
    lines.append(f"Garnishment eval report — {len(reports)} letters x {runs} runs")
    lines.append("=" * 88)
    lines.append(f"provider: {provider}")
    lines.append("")
    lines.append(
        FAKE_PROVIDER_CAVEAT if provider == "fake" else LIVE_PROVIDER_NOTE.format(runs=runs)
    )
    lines.append("")

    for report in reports:
        lines.extend(_render_sample(report))

    lines.extend(_render_summary(reports, runs))
    return lines


def _render_sample(report: SampleReport) -> list[str]:
    expectation = report.expectation
    agreement = "agrees with label" if report.matches_label else "DISAGREES WITH LABEL"
    lines = [
        "-" * 88,
        f"{expectation.file}  [label source: {expectation.label_source}]",
        f"  expected route     : {expectation.route}  ({expectation.decisive_rule})",
        f"  observed routes    : {_format_counts(report.route_counts)}  -> {agreement}",
        f"  route stability    : {report.route_stability}/{len(report.observations)}",
        f"  observed decisive  : {_format_counts(report.rule_counts)}",
        f"  rule stability     : {report.rule_stability}/{len(report.observations)}",
        f"  operative amount   : {_format_counts(Counter(o.operative_amount for o in report.observations))}",  # noqa: E501
        f"  role confidences   : {_format_role_confidences(report)}",
    ]
    if report.errors:
        lines.append(f"  reader outages     : {len(report.errors)} of {report.runs}")
        lines.append(f"    first: {report.errors[0].message}")
    if report.rule_counts and expectation.decisive_rule not in report.rule_counts:
        lines.append(
            f"  NOTE: the labelled decisive rule {expectation.decisive_rule} never fired. "
            "A right route for the wrong reason is still a finding."
        )
    if expectation.note:
        lines.extend(_wrap_note(expectation.note))
    return lines


def _wrap_note(note: str) -> list[str]:
    return ["  " + line for line in textwrap.wrap(note, width=84, initial_indent="note: ")]


def _render_summary(reports: Sequence[SampleReport], runs: int) -> list[str]:
    total_runs = sum(r.runs for r in reports)
    completed = sum(len(r.observations) for r in reports)
    auto_runs = sum(r.auto_runs for r in reports)
    wrong_auto = sum(r.wrong_auto_runs for r in reports)
    unstable = [r for r in reports if r.observations and r.route_stability < len(r.observations)]
    disagreeing = [r for r in reports if not r.matches_label]

    lines = ["=" * 88, "SUMMARY", "=" * 88]
    lines.append(f"  letters            : {len(reports)}")
    lines.append(f"  runs attempted     : {total_runs}")
    lines.append(f"  runs completed     : {completed}")
    lines.append(
        f"  auto rate          : {auto_runs}/{completed}"
        + (f"  ({auto_runs / completed:.0%})" if completed else "")
    )
    lines.append(f"  wrong-auto count   : {wrong_auto}")
    lines.append("")

    if unstable:
        lines.append("  Unstable samples (record these in DESIGN.md with their variance):")
        lines.extend(
            f"    - {r.expectation.file}: {_format_counts(r.route_counts)}" for r in unstable
        )
    else:
        lines.append("  Every sample routed identically across all of its runs.")

    if disagreeing:
        lines.append("")
        lines.append("  Samples that disagree with their label (reported, not fatal):")
        lines.extend(
            f"    - {r.expectation.file}: expected {r.expectation.route}, "
            f"observed {_format_counts(r.route_counts)}"
            for r in disagreeing
        )

    lines.append("")
    if wrong_auto:
        lines.append(
            f"  FAIL: {wrong_auto} run(s) cleared a letter automatically that the label says "
            "a human should have seen. This is the one outcome the harness refuses to pass."
        )
    else:
        lines.append("  PASS: no letter was ever cleared automatically against its label.")
    lines.append("=" * 88)
    return lines


def iter_jsonl_records(
    reports: Sequence[SampleReport], provider: str
) -> Iterator[dict[str, Any]]:
    """One record per (letter, run), including runs the reader never answered."""
    for report in reports:
        expectation = report.expectation
        for observation in report.observations:
            yield {
                "sample_file": expectation.file,
                "run_index": observation.run_index,
                "label_source": expectation.label_source,
                "provider": observation.provider,
                "model_id": observation.model_id,
                "prompt_version": observation.prompt_version,
                "expected_route": expectation.route,
                "observed_route": observation.route,
                "route_matches_label": observation.route == expectation.route,
                "expected_decisive_rule": expectation.decisive_rule,
                "decisive_rule": observation.decisive_rule,
                "decisive_rule_matches_label": (
                    observation.decisive_rule == expectation.decisive_rule
                ),
                "fired_rules": list(observation.fired_rules),
                "blocking_rule": observation.blocking_rule,
                "blocking_route": observation.blocking_route,
                "extraction_status": observation.extraction_status,
                "operative_amount": observation.operative_amount,
                "operative_amount_reason": observation.operative_amount_reason,
                "role_confidences": list(observation.role_confidences),
                "gated_fields": list(observation.gated_fields),
                "field_confidences": {
                    name: score.as_dict() for name, score in observation.field_scores.items()
                },
                "fields": field_comparison(expectation, observation),
                "error": None,
            }
        for run_error in report.errors:
            yield {
                "sample_file": expectation.file,
                "run_index": run_error.run_index,
                "label_source": expectation.label_source,
                "provider": provider,
                "model_id": None,
                "prompt_version": None,
                "expected_route": expectation.route,
                "observed_route": None,
                "route_matches_label": False,
                "expected_decisive_rule": expectation.decisive_rule,
                "decisive_rule": None,
                "decisive_rule_matches_label": False,
                "fired_rules": [],
                "blocking_rule": None,
                "blocking_route": None,
                "extraction_status": None,
                "operative_amount": None,
                "operative_amount_reason": None,
                "role_confidences": [],
                "gated_fields": [],
                "field_confidences": {},
                "fields": {},
                "error": run_error.message,
            }


def write_jsonl(path: Path, reports: Sequence[SampleReport], provider: str) -> None:
    records = iter_jsonl_records(reports, provider)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def emit(lines: Iterable[str], out: Path | None) -> None:
    text = "\n".join(lines)
    print(text)
    if out is not None:
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nreport written to {out}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"how many times to read each letter (default {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the report to this file",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="also write one machine-readable JSON record per (letter, run) to this file",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="how many reader calls to have in flight at once (default 1, sequential)",
    )
    args = parser.parse_args(argv)

    reports, provider = asyncio.run(run_all(args.runs, args.concurrency))
    emit(render(reports, provider, args.runs), args.out)
    if args.jsonl is not None:
        write_jsonl(args.jsonl, reports, provider)
        print(f"jsonl written to {args.jsonl}")

    wrong_auto = sum(r.wrong_auto_runs for r in reports)
    return 1 if wrong_auto else 0


if __name__ == "__main__":
    raise SystemExit(main())
