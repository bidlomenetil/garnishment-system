"""Sweep `AUTO_MIN_FIELD_CONFIDENCE` and report what each candidate value would have cost.

DESIGN.md §5 states every threshold as a hypothesis plus the experiment that would settle
it, and `thresholds.CALIBRATION_PROCEDURE` writes that experiment down:

    Label N letters, sweep the threshold across its plausible range, and at each point
    report the count of wrong-`auto` decisions and the `needs_human` rate. Choose the
    highest threshold at which wrong-`auto` is zero, then the lowest such threshold that
    holds it at zero.

This script runs that procedure. It is a diagnostic for a design document, not a product
feature: it prints a table and exits 0 whatever the table says.

**What the simulation actually replays.** Each letter is read N times against the
configured provider, and every run records both its route and every gated field's final
confidence. `AUTO_MIN_FIELD_CONFIDENCE` has exactly one predicate in the rule table —
`H09_LOW_CONFIDENCE_FIELD`, the last route-bearing rule before `A00_AUTO_DEFAULT` — so a
run's route at a different threshold follows arithmetically from what was observed:

  * If any route-bearing rule **other than** H09 or A00 fired, that rule is decisive at
    every threshold, because none of them reads this constant. The route does not move.
  * Otherwise the run is `auto` when every gated field clears the candidate threshold and
    `needs_human` when one does not — H09 against A00, which is the choice the constant
    exists to make.

So the sweep costs N x 8 calls, not N x 8 x (number of thresholds), and it is a replay of
observed numbers rather than a re-reading of the letters.

**What it deliberately holds still.** `DOC_TYPE_AUTO_MIN_CONFIDENCE` (H02),
`DOC_TYPE_REJECT_MIN_CONFIDENCE` (R01) and `ROLE_MIN_CONFIDENCE` (amount resolution) are
separate constants that happen to share the value 0.85. This sweep moves the field floor
alone, which is the single-constant half of the joint sweep §5 asks for.

    uv run python evals/sweep.py                      # 8 runs per letter, 4 in flight
    uv run python evals/sweep.py --runs 4 --jsonl sweep.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from run_eval import Observation, SampleReport, emit, render, run_all, write_jsonl

from garnishment.domain.enums import Route
from garnishment.domain.thresholds import AUTO_MIN_FIELD_CONFIDENCE, below_floor

DEFAULT_RUNS = 8
"""Runs per letter: 64 calls over the 8-letter set.

Chosen for what it can and cannot support. It is enough to see whether a field's final
confidence sits well clear of a candidate threshold or straddles it run to run, which is
the question a sweep answers. It is nowhere near the N ~ 200 labelled letters §5 names as
the real experiment, and no conclusion here should be read as one.
"""

DEFAULT_CONCURRENCY = 4

AUTO = str(Route.AUTO.value)
NEEDS_HUMAN = str(Route.NEEDS_HUMAN.value)

CANDIDATE_THRESHOLDS = (*(round(0.50 + 0.05 * step, 2) for step in range(10)), 0.99)
"""0.50 to 0.95 in steps of 0.05, plus 0.99 — the top of the plausible range."""


@dataclass(frozen=True, slots=True)
class SweepPoint:
    threshold: float
    auto: int
    needs_human: int
    reject: int
    wrong_auto: int
    false_escalation: int
    """Runs a letter labelled `auto` would have lost to H09 at this threshold."""

    total: int

    @property
    def needs_human_rate(self) -> float:
        return 0.0 if not self.total else self.needs_human / self.total

    @property
    def auto_rate(self) -> float:
        return 0.0 if not self.total else self.auto / self.total


def simulated_route(observation: Observation, threshold: float) -> str:
    """The route this run would have reached had the field floor been `threshold`."""
    if observation.blocking_route is not None:
        return observation.blocking_route
    # `below_floor`, not `<`, so a degenerate confidence is treated the way H09 treats it.
    low = any(
        below_floor(observation.field_scores[name].final, threshold)
        for name in observation.gated_fields
    )
    return NEEDS_HUMAN if low else AUTO


def sweep(
    reports: Sequence[SampleReport], thresholds: Sequence[float] = CANDIDATE_THRESHOLDS
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for threshold in thresholds:
        auto = needs_human = reject = wrong_auto = false_escalation = total = 0
        for report in reports:
            label = report.expectation.route
            for observation in report.observations:
                total += 1
                route_at = simulated_route(observation, threshold)
                if route_at == AUTO:
                    auto += 1
                    if label != AUTO:
                        wrong_auto += 1
                elif route_at == NEEDS_HUMAN:
                    needs_human += 1
                else:
                    reject += 1
                if label == AUTO and route_at != AUTO:
                    false_escalation += 1
        points.append(
            SweepPoint(threshold, auto, needs_human, reject, wrong_auto, false_escalation, total)
        )
    return points


def _min_gated_final(observation: Observation) -> float | None:
    finals = [observation.field_scores[name].final for name in observation.gated_fields]
    return min(finals) if finals else None


def render_sweep(
    reports: Sequence[SampleReport], points: Sequence[SweepPoint], provider: str, runs: int
) -> list[str]:
    observations = sum(len(r.observations) for r in reports)
    lines = [
        "=" * 88,
        f"AUTO_MIN_FIELD_CONFIDENCE sweep — {len(reports)} letters x {runs} runs",
        "=" * 88,
        f"provider       : {provider}",
        f"observations   : {observations}",
        f"value in force : {AUTO_MIN_FIELD_CONFIDENCE:.2f}",
        "",
        "A wrong-`auto` is a run whose letter is labelled anything but `auto` that this",
        "threshold would have cleared automatically. The needs_human rate is over all",
        "observations, including the letters whose route no threshold can move.",
        "",
        f"{'threshold':>9}  {'wrong-auto':>10}  {'needs_human':>11}  {'auto':>10}  {'over-escalated':>14}",  # noqa: E501
        f"{'-' * 9}  {'-' * 10}  {'-' * 11}  {'-' * 10}  {'-' * 14}",
    ]
    for point in points:
        marker = "  <- in force" if point.threshold == AUTO_MIN_FIELD_CONFIDENCE else ""
        lines.append(
            f"{point.threshold:>9.2f}  {point.wrong_auto:>10}  "
            f"{point.needs_human_rate:>10.0%}   {point.auto_rate:>9.0%}  "
            f"{point.false_escalation:>14}{marker}"
        )
    lines.append("")
    lines.extend(_render_per_letter(reports))
    lines.extend(_render_reading(points))
    lines.append("=" * 88)
    return lines


def _render_per_letter(reports: Sequence[SampleReport]) -> list[str]:
    lines = [
        "Per letter: the lowest final confidence across that letter's gated fields, and",
        "whether any other route-bearing rule already decided the route. Where one did,",
        "this threshold is inert for that letter and its runs only pad the rate column.",
        "",
        f"  {'letter':<38} {'label':<12} {'min gated final (min/med/max)':<31} decided by",
        f"  {'-' * 38} {'-' * 12} {'-' * 31} {'-' * 24}",
    ]
    for report in reports:
        finals = [
            value
            for value in (_min_gated_final(o) for o in report.observations)
            if value is not None
        ]
        spread = (
            "(none)"
            if not finals
            else f"{min(finals):.2f} / {median(finals):.2f} / {max(finals):.2f}"
        )
        blockers = {o.blocking_rule for o in report.observations if o.blocking_rule is not None}
        decided = ", ".join(sorted(blockers)) if blockers else "the threshold (H09 vs A00)"
        lines.append(
            f"  {report.expectation.file:<38} {report.expectation.route:<12} {spread:<31} {decided}"
        )
    lines.append("")
    return lines


def _render_reading(points: Sequence[SweepPoint]) -> list[str]:
    clean = [p for p in points if p.wrong_auto == 0]
    lines = ["Reading, by the rule §5 states:"]
    if not clean:
        lines.append("  No candidate threshold held wrong-`auto` at zero on this set.")
        return lines
    cheapest = min(clean, key=lambda p: (p.needs_human_rate, p.threshold))
    lines.append(
        f"  wrong-`auto` is zero at every threshold from {min(p.threshold for p in clean):.2f} "
        f"to {max(p.threshold for p in clean):.2f}."
    )
    lines.append(
        f"  The lowest needs_human rate among those is {cheapest.needs_human_rate:.0%} at "
        f"{cheapest.threshold:.2f}."
    )
    lines.append(
        "  A flat wrong-`auto` column across the whole range means this set does not "
        "discriminate;"
    )
    lines.append(
        "  it does not mean every value is equally safe. Eight letters cannot distinguish "
        "them."
    )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--jsonl", type=Path, default=None, help="write the raw runs here")
    parser.add_argument("--out", type=Path, default=None, help="also write the table here")
    parser.add_argument(
        "--report", action="store_true", help="print the full run_eval report first"
    )
    args = parser.parse_args(argv)

    reports, provider = asyncio.run(run_all(args.runs, args.concurrency))
    if args.report:
        print("\n".join(render(reports, provider, args.runs)))
        print()
    if args.jsonl is not None:
        write_jsonl(args.jsonl, reports, provider)
        print(f"jsonl written to {args.jsonl}\n")

    emit(render_sweep(reports, sweep(reports), provider, args.runs), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
