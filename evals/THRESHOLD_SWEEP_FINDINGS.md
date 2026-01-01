# Threshold sweep: `AUTO_MIN_FIELD_CONFIDENCE`

The sweep DESIGN.md §5 and `thresholds.CALIBRATION_PROCEDURE` describe, run once, live, on
2026-09-23 against `LLM_PROVIDER=openai`, `gpt-5-nano-2025-08-07`, prompt `v1`.

Reproduce with `uv run python evals/sweep.py --runs 8 --concurrency 4`.

## N, and why

**8 letters x 8 runs = 64 observations, 0 reader outages.** Eight runs per letter is enough
to see whether a field's confidence sits clear of a candidate threshold or straddles it run
to run, which is the only question a sweep can answer. It is not the N ~ 200 labelled
letters §5 names as the real experiment. The 64 observations are eight repeats of eight
letters, so the effective sample behind any rate below is nearer eight than sixty-four.

The sweep costs 64 calls rather than 64 x 11 because `AUTO_MIN_FIELD_CONFIDENCE` has exactly
one predicate in the table — `H09`, the last route-bearing rule before `A00` — so each run's
route at another threshold follows arithmetically from its recorded field confidences.

## The table

```
threshold  wrong-auto  needs_human        auto  over-escalated
---------  ----------  -----------  ----------  --------------
     0.50           0         30%         33%               3
     0.55           0         30%         33%               3
     0.60           0         30%         33%               3
     0.65           0         30%         33%               3
     0.70           0         30%         33%               3
     0.75           0         30%         33%               3
     0.80           0         30%         33%               3
     0.85           0         30%         33%               3  <- in force
     0.90           0         30%         33%               3
     0.95           0         50%         12%              16
     0.99           0         62%          0%              24
```

`over-escalated` counts runs of an `auto`-labelled letter that the threshold sent to a human.

## Interpretation

**The wrong-`auto` column is zero everywhere, so this set does not discriminate.** Only 21 of
the 64 observations are ones the constant can move at all; in the other 43 an `R*` or an
earlier `H*` rule had already decided the route on evidence no confidence floor touches. On
those 21, the lowest gated final confidence is never beneath 0.90 — it is 0.90 ten times,
0.92 three times and 0.95 eight times. Nothing on this set sits between 0.50 and 0.90, which
is why every threshold in that span produces an identical row.

So the honest reading is that **the data is consistent with 0.85 and cannot distinguish it
from any value up to 0.90.** What the data does say is narrower and still useful: 0.85 has
about 0.05 of headroom before it starts biting, and above 0.90 it bites hard — at 0.95 the
`auto` rate falls from 33 % to 12 % and buys back no wrong-`auto`, because there was none to
buy. That is evidence against §5's "expected direction: up" for *this* constant on *this*
evidence, and it is weak evidence: eight letters, five of them labelled by us, cannot support
a claim about a production distribution.

Two observations worth carrying into the real experiment. The reader's self-reports are
coarse — 240 gated-field readings took eight distinct values, all of `{0.0, 0.5, 0.6, 0.8,
0.85, 0.9, 0.92, 0.95}` — so the floor is comparing against a near-discrete quantity, and
small moves in it are mostly no-ops. And grounding, not the self-report, is doing the
verifiable work: 216 of those 240 readings grounded exactly, 4 case-insensitively, and 20 not
at all. A sweep of the field floor is therefore measuring the weakest of the three inputs
`min()` combines.
