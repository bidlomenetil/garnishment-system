# DESIGN

A service that reads a court garnishment letter with an LLM and routes it to `auto`,
`needs_human` or `reject`. This document is the argument for how it is built; `README.md`
is the instructions for running it.

---

## Summary

This document is long; most of it is the appendix. If you read nothing else, read this.

**The structural idea (§1).** The model never decides which of two figures on a
restriction order is legally operative — it enumerates every figure and labels each one's
role, and a versioned doctrine in the domain (~20 lines, unit-tested) makes the legal
judgement. The probabilistic and deterministic halves are separated by an import test:
`domain/` cannot import anything outside the standard library, so the seam is enforced, not
just argued.

**How a letter's status moves (§8, §9).** Every routing pass evaluates all 27 rules and
records every hit; only the first *route-bearing* hit is decisive. Order is: five reject
rules (a hard claim about the letter — missing case reference, non-positive amount, and so
on), then four blocks of escalation rules (reading failure, identity, amount, evidence
quality), then one `auto` default. Our own reading failures escalate, they never reject —
that split, plus the reject rules' `extraction_status == OK` guard, is the single idea this
design is proudest of.

**Confidence (§4).** `final = min(self-report, grounding, validator)`. The self-report can
only ever lower a score. As of 2026-09-23 the self-report is a verbalised bin
(`certain`/`probable`/`uncertain`/`guess`), not a free float — the float version showed real
boundary jitter on live calls (§11.7), and a bin the model can't express between values
removes that failure mode without touching a threshold.

**Live findings (§11.7, §16).** Two real bugs were found and fixed by actually calling the
model rather than trusting the design on paper: a restriction letter's `document_is_seizure_order`
field was undescribed and the model guessed (fixed in the prompt and schema); a sole-candidate
restriction letter could resolve `auto` on a superseded figure with no role check (fixed,
§16). An independent five-lens review (code, architecture, security, LLM practice,
observability) on 2026-09-23 found 12 more real defects — three more wrong-`auto` paths, a
prompt/transport mismatch, a fence that wasn't escaped, silent failure logging — all fixed
and verified by execution, not by argument. §16 has the full account.

**What's still a known limitation, on purpose.** Eight letters is a smoke test, not the
200-letter labelled sweep §5 describes as the real answer to "why 0.85". A 64-observation
sweep against live data (§16, `evals/THRESHOLD_SWEEP_FINDINGS.md`) is consistent with 0.85
but cannot distinguish it from anything up to 0.90. Grounding proves a value is *present* in
the letter, never that it is *authoritative* — it stops the model inventing things, not a
letter that lies (§16 finding 1). Token/cost/request-id telemetry was scoped out of this
pass, explicitly, not silently (§16).

---

## 1. The one structural idea

The interesting move in this design is not the rule table. It is **refusing to let the
model make the judgement call.**

Sample 3 is a restriction order: it cites a prior attachment of EUR 4,250.00 and reduces it
to EUR 1,267.00. A schema with a single `amount` field forces the model to decide which of
those two figures is legally operative, inside a black box, and a wrong answer teaches us
nothing. So the schema does not ask. It asks the model to **enumerate every monetary figure
and label the part each one plays** — `operative`, `original`, `restricted`, `unknown` —
which is reading comprehension, the thing models are good at. The legal judgement that a
restriction supersedes the original it cites is then made by us, in about twenty lines of
tested code, under a named and versioned doctrine, and recorded as a fired rule on the
decision.

The judgement moved from the probabilistic side of the boundary to the deterministic side,
where it is testable, replayable and explainable after the fact.

A second, quieter version of the same move: grounding and validation are computed **inside
`route()`** from the stored letter text, not by the adapter. If the adapter computed
grounding and stored only its result, replaying a decision would re-read a number somebody
else had already decided. Because it is recomputed from the letter on every routing run,
the replay test genuinely re-derives the decision, and a bug in the grounding function is
caught by replay instead of being frozen into records.

---

## 2. Principles

1. **The probabilistic part ends at one boundary line.** Exactly one component
   (`ExtractorPort`) may be non-deterministic. Everything downstream is a pure function of
   its output. This is visible in the directory structure, not just in prose.
2. **Confidence is derived inside the domain, never accepted from outside.** A number the
   model reports about itself is one of three inputs, and under `min()` it can only ever
   lower a score, never raise one.
3. **Ambiguity is captured by schema, not by prompt luck.** Where a letter admits two
   readings, the schema asks for *both readings plus their roles*.
4. **Our failure is never the customer's rejection.** A malformed model response routes to
   `needs_human` under a named rule. `reject` is reserved for evidence that the *letter* is
   invalid.
5. **Every route is reconstructible from stored data alone.** The record stores exactly one
   representation of every number, and re-running routing over it reproduces the route, the
   confidences, the fired rules and the decisive rule. A test asserts this.

---

## 3. Architecture

Four layers, `Protocol` ports, one entity, no aggregates, no domain events, no CQRS.

```
domain/        PURE. stdlib only — asserted by an AST walk in tests/unit/test_domain_purity.py
application/   ports (ExtractorPort, DecisionRepositoryPort, ClockPort, IdGeneratorPort), use cases
adapters/      llm/ (schema, prompt, mapping, anthropic, fake, factory), persistence/, system.py
api/           FastAPI factory, DI wiring, DTOs, two routes
```

`src/garnishment/domain/` imports nothing outside the standard library. That restriction
turns the assignment's thesis — a clean seam between probabilistic and deterministic code —
into an enforced property rather than an import convention. It costs about four extra files
over a flat layout.

`route()` takes exactly one argument, a `LetterExtraction`, and nothing else: no clock, no
randomness, no I/O, no pre-computed confidences handed in by an adapter. A property test
asserts the signature via `inspect.signature`, so the constraint cannot erode quietly.

---

## 4. Where confidence comes from

The brief asks for this explicitly. Four options were considered.

| Option | What it contributes | Cost / limitation | Verdict |
|---|---|---|---|
| **D4a. `min(llm_self_report, grounding, validator)`** | Combines an unverifiable signal with two verifiable ones under an operator that lets the unverifiable one only ever *lower* a score. Deterministic once the extraction is in hand, which the brief requires. | The self-report is the weakest input and cannot be audited. | **Chosen.** |
| D4b. Validators + structural signals only; discard the self-report | Every input is verifiable and reproducible by hand. Genuinely attractive: it removes the one number nobody can check. | What the self-report adds is a coarse ambiguity detector for readings the validators cannot see. A name can have two substantive tokens and pass `validate_debtor_name` while the model is genuinely unsure it read it correctly; a case reference can match the regex while sitting in a context suggesting it belongs to a different matter; an amount can be well-formed while the sentence around it hedges. Validators check *form*; the self-report is the only channel carrying doubt about *reading*. Dropping it makes those cases route `auto` silently. | Rejected, narrowly. Kept because the failure direction is safe: under `min()` the self-report can only escalate, never approve. |
| D4c. Logprob-derived confidence | Model-internal rather than self-narrated. | Not exposed under forced tool-use / strict structured output on the relevant APIs, so it is unavailable in the architecture the brief asks for. Separately, the token probability of a JSON field value is not the same quantity as confidence in a reading. | Invalidated. |
| D4d. Self-consistency: N samples, or a second model, agree-or-escalate | The strongest available signal, and it measures the thing we actually care about — whether the reading is stable. The eval harness already implements its measurement half. | N× latency and spend per letter. | **Deferred — the first planned upgrade.** Promoting it is a change to one adapter, not to the boundary. |

So, concretely, for each field:

```
final_confidence = min(llm_confidence, grounding_score, validator_score)
```

**Grounding** (`domain/grounding.py`, pure, field-typed). Each field supplies its own
normaliser. The default collapses whitespace runs, strips, and applies NFKC. The **IBAN
normaliser additionally strips all whitespace and upper-cases both sides**, so a model that
returns the compact `DE44500105175407324931` still grounds against the letter's spaced
`DE44 5001 0517 5407 3249 31`. That case is load-bearing, because a present IBAN sits inside
the confidence gate, and it has a dedicated unit test.

| Condition, after the field's normaliser | Score | Recorded |
|---|---|---|
| Normalised quote occurs verbatim in the normalised letter | 1.00 | — |
| Occurs case-insensitively only | 0.95 | — |
| Quote does not occur, but the extracted **value** occurs verbatim in the letter | 0.90 | `I03_GROUNDED_ON_VALUE` |
| Neither quote nor value occurs, or both are absent | **0.00** | `H08` |

A case-only mismatch means the model retyped rather than copied: a weak paraphrase signal,
recorded but left above the auto bar. The 0.90 tier handles a real and common failure that
is not a correctness failure — the model paraphrases the surrounding quote while the value
itself is demonstrably present. Scoring that 0.00 would escalate letters whose values are
verifiably grounded; scoring it 1.00 would hide that the citation was sloppy. All three
non-zero tiers sit above the auto floor by design: grounding's job is to catch fabrication,
not to grade prose.

**Validators** (`domain/validators.py`) — each a named pure function returning
`ValidatorResult(score, code, detail)` with score ∈ {0.0, 1.0}. Binary by design: soft
signals get their own named rules rather than being blended into a number where nobody can
see them.

| Function | Passes when |
|---|---|
| `validate_case_reference` | matches `^[A-Z]{2}-\d{4}/\d{2}$` after stripping |
| `validate_iban` | compact form matches `^[A-Z]{2}\d{2}[A-Z0-9]+$`, length 15–34, **mod-97 == 1** |
| `validate_amount` | `Decimal > 0`, exactly 2 decimal places, currency EUR |
| `validate_debtor_name` | ≥ 2 substantive tokens (alphabetic, length ≥ 2, not a single letter plus a full stop) |

**What the validators do and do not prove.** Mod-97 admits roughly 1 in 97 random strings of
the right shape, so it is not a near-zero false-accept test in the abstract. The honest claim
is narrower and still strong: it catches essentially every single-character transcription
error and every adjacent-digit transposition, which are the realistic ways an IBAN goes
wrong in a typed court letter. It does not prove the account exists or belongs to the
debtor; nothing available here could. That is exactly why a *present* IBAN must also be
grounded in the letter and clear the 0.85 floor before `auto` is available — the checksum
alone must not clear a fabricated one. An *absent* IBAN is `H06` and never a reject.

---

## 5. Thresholds: hypotheses, not measurements

**No labelled evaluation set exists for this service.** Three letters with expected routes
is a smoke test. Every number below is therefore stated as a **hypothesis plus the
experiment that would settle it**, never as a measured optimum. `domain/thresholds.py`
carries both in its docstrings, and `rules.py` contains no numeric literals at all — every
comparison it makes reads a name from that module, so changing a threshold is a one-line
diff against a stated hypothesis.

**The calibration procedure**, stated once and referenced by each threshold:

> Label N letters (N ≈ 200 is the smallest set that separates a 5 % from a 15 % escalation
> rate with usable confidence) with a clerk's own route. Sweep the threshold across its
> plausible range. At each point report two numbers: the count of **wrong-`auto`** decisions,
> and the **`needs_human` rate**. Choose the highest threshold at which wrong-`auto` is zero,
> then the lowest such threshold that holds it at zero, minimising human review subject to
> that constraint. Until that experiment runs, each value below is an engineering prior
> chosen to fail toward escalation.

| Constant | Value | Hypothesis, and what would change it |
|---|---|---|
| `AUTO_MIN_FIELD_CONFIDENCE` | `0.85` | **Hypothesis:** below roughly 0.85, self-reported confidence marks readings that genuinely admit a second interpretation. **Why the number is not a claimed error rate:** self-reported confidence is not a calibrated probability, so reading 0.85 as "we tolerate 15 % wrong" misreads the quantity. The verifiable load is carried by grounding and the validators; the self-report is the coarse ambiguity detector described in D4b. **Test:** the sweep above. **Expected direction:** if it moves, it moves up, because the cost asymmetry is severe — a wrong `auto` is a regulated incident, a needless `needs_human` costs a clerk minutes. |
| `DOC_TYPE_REJECT_MIN_CONFIDENCE` | `0.90` | See the callout below. **Test:** count `R01` firings that a clerk overturns; any non-zero count raises it or removes the rule. |
| `DOC_TYPE_AUTO_MIN_CONFIDENCE` | `0.85` | "Is this even a seizure order" is the most load-bearing reading in the letter, so it gets no discount relative to any other required field. **Test:** the sweep, jointly with `AUTO_MIN_FIELD_CONFIDENCE`. |
| `ROLE_MIN_CONFIDENCE` | `0.85` | The amount design rests on role labels, so an unsure label must not silently decide which figure is operative. Same floor as a field value, for the same reason. **Test:** measure observed `role_confidence` on live samples; if the model reports uniformly high values with no discrimination, the gate is inert and the fallback in §11.5 applies instead. |
| `GROUNDING_EXACT` / `_CASE_INSENSITIVE` / `_VALUE_ONLY` / `_ABSENT` | `1.00` / `0.95` / `0.90` / `0.00` | Rationale in §4. **Test:** the sweep over the *tier assignment* rather than a single number — wrong-`auto` count partitioned by observed tier. `GROUNDING_ABSENT` is not a tunable; moving it off zero would mean accepting an ungrounded value. |
| `AUTO_MAX_AMOUNT_EUR` | `10 000 000.00` | **This is a deployment policy parameter the bank sets, not a claim about letters.** Above some figure, the cost of a wrong `auto` exceeds any clerk-time saving and the asymmetry justifying automation stops holding. We ship a placeholder; the number belongs to the bank's risk function, not to us. **Test:** not an experiment we can run. |
| `MIN_SUBSTANTIVE_NAME_TOKENS` | `2` | An account cannot be identified from a surname plus an initial without a human resolving it against the customer base. **Known limitation:** culturally naive (mononyms, particles, non-Latin scripts); the failure direction is safe, since it over-escalates and never over-approves. |
| `EXTRACTION_MAX_RETRIES` | `1` | One retry absorbs transient malformed JSON; a second mostly masks a prompt or schema bug we want to see. **Test:** count second-attempt successes in production. |
| `LLM_TEMPERATURE` | `0.0` | States the intent. **It is not sent** — see §11.4. It does not guarantee reproducibility anyway, which is why the guarantee lives downstream. |
| `RESTRICTION_DOCTRINE_V1` | string constant | Evidence base **n = 1** — see §6. |

### The R01 callout

`R01_NOT_SEIZURE_ORDER` is **the only rejection in the table that rests on a model judgement
rather than a mechanical check.** Every other reject is a fact about the letter that a person
can re-derive by hand: a missing case reference, a case reference that does not match the
format, no monetary figure anywhere, a figure that is not positive. `R01` rests on the model
having read the document correctly enough to say "this is not a court order at all".

Rejecting a genuine court order is the worst error this system can make. So
`DOC_TYPE_REJECT_MIN_CONFIDENCE = 0.90` is set deliberately **above every other floor in the
system**, making `R01` the hardest rule in the table to fire. Below 0.90, the letter escalates
under `H02` instead of being refused. If a clerk ever overturns an `R01` firing, the honest
response is to raise the number or remove the rule, not to tune around it.

---

## 6. The restriction doctrine, and its n = 1 evidence base

```python
RESTRICTION_DOCTRINE_V1 = (
    "a restriction order's restricted figure supersedes the original it cites"
)
```

**Its evidence base is exactly one letter.** It is a hypothesis about how restriction orders
work, derived from Sample 3 and nothing else. It is a named, versioned constant rather than
an `if` buried in a function precisely because of that: `doctrine_version` is recorded on
every decision, including decisions the doctrine did not decide, so a future correction is
visible in the stored data rather than inferred from git history. If the doctrine turns out
to be wrong, every record made under v1 can be found and re-examined.

Resolution (`domain/operative_amount.py`) is pure, total and never raises:

| # | Situation | `considered` | `source` | Reason |
|---|---|---|---|---|
| 1 | 0 candidates | none | none | `NO_AMOUNT` → `R04` |
| 2 | exactly 1 candidate, any role, and it does not parse | that one | none | `UNPARSEABLE` → `H16` |
| 3 | ≥ 2 candidates, a unique winner by rows 4–6, and that winner does not parse | the winner | none | `UNPARSEABLE` → `H16` |
| 4 | exactly 1 candidate, role ∈ {operative, restricted, unknown}, parses | that one | same | `SOLE_CANDIDATE` |
| 5 | exactly one candidate with role `operative` **and** `role_confidence ≥ 0.85`, parses | that one | same | `EXPLICIT_OPERATIVE` |
| 6 | `restriction_order`, exactly one `restricted` candidate **and** `role_confidence ≥ 0.85`, parses | the restricted one | same | `RESTRICTION_SUPERSEDES` |
| 7 | `restriction_order` and every candidate is labelled `original` | none | none | `RESTRICTION_UNRESOLVED` → `H11` |
| 8 | **Catch-all: anything not matched above** | none | none | `AMBIGUOUS` → `H03` |

Row 8 is a catch-all **by construction** — it is the `else` branch — so the resolver is total
for every possible candidate list. Without it, a lone `original` figure on an attachment
order (a real shape: a letter citing a prior order without restricting it) would match no row
at all. Escalation is the right default for an unclassified shape, and totality is asserted
by a property test over arbitrary candidate lists.

**Two indices, because "which figure did we mean" and "which figure did we get" are different
questions.** `considered_candidate_index` is set whenever the candidate set identifies one
figure, independently of whether that figure parsed. `source_candidate_index` is set only when
that figure also parsed into a usable `Decimal`. This matters because candidate-level
properties — approximate, ambiguous format, non-EUR currency — are facts about the *figure the
letter pointed at*, and they must stay reportable when the figure itself could not be read.
Sample 2 is exactly this case: one approximate figure written as prose, so
`considered_candidate_index = 0`, `source_candidate_index = None`, and the clerk sees both
that the amount was approximate (`H04`) and that it could not be read (`H16`). Two true
statements about one figure, neither of which should silence the other.

**Consistency check**, applied only to `RESTRICTION_SUPERSEDES`: the restricted figure must
be **strictly less than** the cited original. A restriction that raises the amount is
internally inconsistent and trips `H12`. `evals/samples/extra_04_restriction_raises_amount.txt`
exercises exactly this.

**Amount parsing** is ours, never the model's. `"4,250.00"` and `"4.250,00"` are the same
figure in different locales, and `"4.250"` is genuinely ambiguous. `parse_eur_amount` takes
the last of two separators as the decimal point, treats a lone separator followed by exactly
two digits as decimal and by exactly three as grouping (EUR has two decimal places, so three
trailing digits cannot be a fraction, but the reading is flagged `is_format_ambiguous` →
`H13`), and returns `value = None` for prose or garbage → `H16`. We never silently pick a
reading of an ambiguous figure.

---

## 7. The rule table

An ordered list of `Rule(rule_id, predicate, route, reason_template)` in `domain/rules.py`.
**Every** rule is evaluated and every fired rule is recorded, so a clerk sees all of a
letter's defects rather than only the first. The **decisive rule is the first fired
route-bearing rule in table order**; informational `I*` rules are excluded from that choice,
because "the IBAN checksum was valid" can never be the reason a letter went anywhere.

All `R*` sit first, so reject dominance is structural rather than conditional. The `H*`
blocks are ordered by **what a clerk most needs to see first**: whether we could read the
letter at all, then whose money it is, then how much, then how well-evidenced the reading was.

| # | Rule id | Predicate | Route |
|---|---|---|---|
| **Reject — evidence the letter itself is invalid** ||||
| 1 | `R01_NOT_SEIZURE_ORDER` | reading OK **and** `is_seizure_order == False` **and** confidence `>= 0.90` | reject |
| 2 | `R02_CASE_REFERENCE_MISSING` | reading OK **and** `case_reference.value is None` | reject |
| 3 | `R03_CASE_REFERENCE_MALFORMED` | reading OK **and** present **and** `validate_case_reference` fails | reject |
| 4 | `R04_AMOUNT_MISSING` | reading OK **and** `len(amount_candidates) == 0` | reject |
| 5 | `R05_AMOUNT_NON_POSITIVE` | reading OK **and** resolution yielded a non-null `Decimal` **and** it is `<= 0` | reject |
| **Escalate — could we read the letter at all?** ||||
| 6 | `H01_EXTRACTION_FAILED` | `extraction_status == MALFORMED_AFTER_RETRY` | needs_human |
| 7 | `H02_DOC_TYPE_UNCERTAIN` | (`is_seizure_order == False` and confidence `< 0.90`) or (`is_seizure_order == True` and confidence `< 0.85`) | needs_human |
| **Escalate — identity: whose money is this** ||||
| 8 | `H05_DEBTOR_NAME_PARTIAL` | `debtor_name_is_partial` **or** `validate_debtor_name` fails | needs_human |
| 9 | `H06_IBAN_ABSENT` | `account_iban.value is None` | needs_human |
| 10 | `H07_IBAN_INVALID` | present **and** `validate_iban` fails | needs_human |
| **Escalate — amount: how much** ||||
| 11 | `H11_OPERATIVE_AMOUNT_UNRESOLVED` | reason `RESTRICTION_UNRESOLVED` | needs_human |
| 12 | `H12_RESTRICTION_NOT_REDUCING` | reason `RESTRICTION_SUPERSEDES` **and** both figures parsed **and** restricted `>=` original | needs_human |
| 13 | `H03_AMOUNT_AMBIGUOUS` | reason `AMBIGUOUS` | needs_human |
| 14 | `H04_AMOUNT_APPROXIMATE` | **considered** candidate exists **and** `is_approximate` | needs_human |
| 15 | `H16_AMOUNT_VALUE_UNPARSEABLE` | reason `UNPARSEABLE` | needs_human |
| 16 | `H13_AMOUNT_FORMAT_AMBIGUOUS` | **considered** candidate parses **with** `is_format_ambiguous` | needs_human |
| 17 | `H14_CURRENCY_NOT_EUR` | **considered** candidate exists **and** its currency `!= "EUR"` | needs_human |
| 18 | `H15_AMOUNT_IMPLAUSIBLE` | resolution yielded a non-null `Decimal` **and** it is `> AUTO_MAX_AMOUNT_EUR` | needs_human |
| **Escalate — how well evidenced was the reading?** ||||
| 19 | `H08_UNGROUNDED_EVIDENCE` | any gated field **with a value** has `grounding_score == 0.0` | needs_human |
| 20 | `H09_LOW_CONFIDENCE_FIELD` | any gated field has `final_confidence < 0.85` | needs_human |
| **Automatic default** ||||
| 21 | `A00_AUTO_DEFAULT` | no `R*` and no `H*` fired | auto |
| **Informational — recorded, never decisive** ||||
| 22 | `I01_OPERATIVE_AMOUNT_RESOLVED` | resolution succeeded | — |
| 23 | `I02_IBAN_VALIDATED` | IBAN present and valid | — |
| 24 | `I03_GROUNDED_ON_VALUE` | any field grounded at the 0.90 tier | — |
| 25 | `I04_INJECTION_SUSPECTED` | letter matches the instruction-injection heuristic | — |

The **gated fields** are `debtor_name`, `case_reference` and `amount` always, plus
`account_iban` when it is present.

**On candidate-property rules versus amount-value rules.** `H04`, `H13` and `H14` describe the
*figure the letter pointed at*, so they read the **considered** candidate and fire even when
that figure never parsed. `H12`, `H15` and `R05` describe the *resolved* amount, so they carry
explicit non-null `Decimal` guards and simply do not fire when there is no number to compare.

**On injection being informational rather than route-bearing.** `I04`'s heuristic is a
substring check against a handful of phrases. It has a real false-positive rate on legal
prose and, more importantly, **it is not the control that stops an injection.** Grounding is:
an injected instruction cannot fabricate a value whose evidence appears in the letter, and
routing reads only typed fields. Recording the signal without letting a regex escalate real
court orders is the honest placement. `evals/samples/extra_05_injection_attempt.txt` is a
valid order carrying a paragraph addressed to the automated reader, and it routes `auto` with
`I04` recorded — deliberately. If telemetry ever showed genuine attempts, promoting `I04`
back to a route-bearing rule is a one-line change.

### Why `Route` has no severity ordering

`Route` is a three-member `StrEnum` with **no** ordering on it, deliberately. Escalation and
rejection are different *kinds* of outcome, not two points on one scale: `reject` says
something about the letter, `needs_human` says something about our reading of it.

A total order would invite an invariant of the form "lowering a confidence never decreases
severity", and **that invariant is false by design.** Lowering the model's confidence that a
document is *not* a seizure order moves it from `reject` to `needs_human` — a *decrease* in
any plausible severity ordering, and exactly the behaviour the `R01` callout in §5 intends,
because escalation requires less evidence than rejection.

What the property tests assert instead is the two-valued form, which is the one that actually
protects money and which is true: **if `route(x) == auto`, then lowering any single confidence
component gives `route(x') != auto`.** No silent promotion to `auto`.

---

## 8. Partial and conflicting extraction

| Situation | Handling | Rule |
|---|---|---|
| `null` required field | Confidence 0. A missing `case_reference` or a total absence of candidates is a hard reject (the brief names both). A missing `debtor_name` is `needs_human` — a name we could not read is uncertainty about *reading*, not evidence the order is invalid. | `R02` / `R04` / `H05`+`H09` |
| `null` `account_iban` | `needs_human`, never reject. The brief forbids rejecting on absence alone; we also decline to `auto`, because no automated process should pick an account from a name. | `H06` |
| IBAN present but ungrounded or low-confidence | `needs_human`. A fabricated IBAN can pass mod-97 by chance; the checksum alone must not clear it. | `H08` / `H09` |
| Two contradictory amount candidates | The resolver seeks a unique winner by role, document type and `role_confidence`. Failing that, `needs_human` with the candidate count in the reason and both candidates recorded. | `H03` / `H11` |
| A role label the model is unsure of | Resolution degrades to `AMBIGUOUS`; the reason names the low role confidence. | `H03` |
| Amount that cannot be parsed as a figure | `needs_human`, distinct from "no amount at all", which is a reject. | `H16` |
| Restriction that raises the amount | `needs_human` — internally inconsistent. | `H12` |
| Evidence quote absent from the letter | If the value itself is present, 0.90 and `I03`. If neither, confidence 0 and a distinct named rule, so the record says "unsupported" rather than merely "low confidence". | `H08` |
| Invalid JSON / schema break | Retry **once**, identically. On a second failure, build a `LetterExtraction` with `extraction_status = MALFORMED_AFTER_RETRY` and route `needs_human`. **We never reject on our own failure** — `reject` is a statement about the letter, and using it for our own malfunction would convert an outage into a stream of refused court orders and corrupt the audit trail with false invalidity findings. | `H01` |
| Provider unreachable, auth failure, or rate-limited after the client's own retries | **No decision record is created.** The endpoint returns `503`. A decision record must record a real reading; writing a `needs_human` record for an outage puts an empty work item in a clerk's queue and makes an infrastructure problem indistinguishable from an ambiguous letter. | — (transport, not a routing outcome) |

The last distinction — *the model answered badly* (record it, escalate it) versus *the model
never answered* (503, record nothing) — is the sharpest form of Principle 4.

---

## 9. How I would know the routing rules are right

Two halves, two kinds of evidence.

**The deterministic half is settled by the golden table.** `tests/fixtures/extractions.py`
carries **25 parametrised golden cases**, each asserting an exact `(route, decisive_rule_id,
fired_rule_ids)` triple. Every one of the 25 rule ids appears as *fired* in at least one case,
and every route-bearing rule appears as *decisive* in at least one case where it can be
decisive. A test fails if any rule id is uncovered, so a rule-table change cannot silently
alter behaviour.

**The probabilistic half is measured by `evals/`.** `make eval` reads 8 labelled letters — the
3 from the assignment plus 5 adversarial ones written for this harness — N times each and
reports, per letter: expected route, every observed route, route stability, decisive-rule
stability, the resolved operative amount, and the observed `role_confidence` values. Across the
set it reports the **`auto` rate** and the **wrong-`auto` count**.

The harness **always produces a report**; the report is the deliverable. It does not assert
stability as a pass condition. A sample that routes 3/5 the same way is a finding to write
down here, with its observed variance, not a failure to hide behind a red build. Instability
is information about the model. The exit code is non-zero on **one** condition only: a
**wrong `auto`** — a letter the label says a human should see, or that should be refused, that
the service cleared automatically. That is the one outcome that is never acceptable.

**Current eval findings.** With no API key available (§11.5), `make eval` runs against
`FakeExtractor`, and the report says so at the top rather than implying a signal it does not
have. All 8 letters route as labelled at 5/5 stability with the labelled decisive rule; the
`auto` rate is 15/40 runs (38 %, which is 3 of the 8 letters); the wrong-`auto` count is 0.
**Read that stability honestly: against `FakeExtractor` it is 5/5 by construction**, because
the extraction is a recorded payload looked up by the SHA-256 of the letter. What it does
establish is that `route()` is deterministic over a fixed extraction and that each letter
reaches the rule its label names. The stability column only becomes evidence about a model
when `LLM_PROVIDER=anthropic` and a key is set.

**What this does not establish.** Eight letters is a smoke test, not a validation set, and
five of the eight are labelled by us rather than by the assignment. The restriction doctrine
has an evidence base of exactly one letter. The real answer is a feedback loop: log every
`needs_human` against the clerk's eventual verdict, and re-tune against that distribution
using the sweep in §5.

**The property tests** (hypothesis, 200 examples each) cover what the golden table cannot:

1. **Determinism** — `route(x) == route(x)` for any generated extraction, including fired-rule
   order and `field_confidences`, plus an `inspect.signature` assertion that `route()` takes
   exactly one parameter.
2. **No silent promotion to `auto`** — see §7.
3. **Reject dominance** — if any `R*` predicate holds, the route is `reject` regardless of
   every other field.
4. **Totality** — `route()` never raises and always returns a route-bearing `decisive_rule_id`,
   for arbitrary inputs including all-null and empty-candidate degenerates.

**Replay** is asserted end-to-end: for every golden case, save to SQLite, reload, re-route, and
compare route, `field_confidences`, `fired_rules` and `decisive_rule_id` exactly. That is what
proves serialisation is lossless, which is the part that actually breaks.

The full suite is **500 tests**, green with no API key.

---

## 10. Determinism

1. ~~`temperature = 0.0`; `top_p` pinned in the adapter.~~ **Not implementable — see §11.4.**
2. The model id is a pinned string, never an alias like `-latest`, written into every record.
   The adapter raises at construction if it is handed an alias.
3. `PROMPT_VERSION` is a module constant written into every record; a snapshot test fails if
   the prompt changes without a bump.
4. Forced structured output (`tool_choice={"type": "tool", "name": …}`) removes free-form
   drift.
5. **The guarantee that actually holds:** routing is a pure function with no clock, no
   randomness and no I/O, over an extraction that stores each number exactly once. Identical
   extraction ⇒ identical route, asserted by a property test. LLM output may still vary run to
   run; that is exactly why the guarantee sits downstream of it rather than being asserted
   about it.
6. `evals/run_eval.py` measures the residual variance and reports it honestly.

---

## 11. Deviations from the plan, and why

The implementation plan was written before any code. Several things changed during the build,
for reasons worth recording — a reviewer comparing the code against the plan will notice each
of these immediately and should see that it was deliberate.

### 11.1 There are 25 golden cases, not the ~14 the plan budgeted

The plan sized the golden table at "~14 parametrised cases". That number cannot satisfy the
plan's own acceptance criterion 5: *every one of the 21 route-bearing rules must appear as
decisive somewhere it can be decisive, and every one of the 25 rule ids must appear as fired.*
Several rules can only be reached by extraction shapes that are mutually exclusive with one
another — a rule that needs a resolved amount cannot share a case with one that needs an
unparseable one — so the cases do not compose as densely as the estimate assumed. **25 cases**
were built instead, each asserting an exact firing set rather than a route alone. The extra
eleven are coverage the criterion required, not padding.

### 11.2 Three domain-layer judgement calls

Each is a deliberate, tested departure from a literal reading of the plan's §7.4/§7.5. All
three still satisfy every route the assignment's samples require.

**All five `R*` rules are guarded on `extraction_status == OK`.** The plan's predicates did not
carry this guard. Without it, a malformed model response — which produces a `LetterExtraction`
with a null case reference and no amount candidates — would trip `R02` and `R04` and **reject a
court order we simply failed to read**. That directly contradicts Principle 4 and ADR-006: our
own malfunction must never be reported as the letter's invalidity. With the guard, a malformed
reading escalates under `H01` and nothing else fires from the reject block. Covered by golden
case `our_own_reading_failure_never_rejects` and by the reject-dominance property test.

**Resolution row 7 is "restriction order AND every candidate is labelled `original`", not the
plan's literal "zero restricted/operative candidates".** The literal wording would capture
Sample 3's both-candidates-`unknown` failure mode — two figures, neither labelled `restricted`
nor `operative` — and route it to `H11`. But plan §7.10 requires that exact failure mode to
route to `H03_AMOUNT_AMBIGUOUS`, which is the correct answer: two figures we cannot tell apart
is ambiguity, whereas `H11` means "this letter restricts a prior order and we cannot find the
new figure". The implemented reading satisfies all four of Sample 3's required failure-mode
routes (`H03` / `H11` / `H12` / `H03`); the literal one does not.

**`H08_UNGROUNDED_EVIDENCE` fires only when a field actually has a value.** A null required
field goes through `H09` (low confidence, since `min()` of a zero component is zero), per
§7.9's own table. Firing "evidence for `debtor_name` does not appear in the letter" against a
field with no claimed value would be a false statement in an audit record: there is no evidence
to look for, because nothing was claimed. The `value_present` flag on `FieldConfidence` carries
this distinction.

### 11.3 `ExtractorPort` returns `ExtractionResult`, not a bare `LetterExtraction`

Plan §6 gives the port signature as `async def extract(self, letter_text: str) ->
LetterExtraction`. The implementation returns an `ExtractionResult(extraction,
raw_llm_response, provider, notes)` wrapper.

This is a necessary fix, not a preference. `Decision.from_routing` requires
`extractor_provider` and `raw_llm_response` as separate, non-optional arguments — the record
schema in §7.7 lists both. `LetterExtraction` is deliberately **provenance-free**: it is the
pure replay input, and which provider answered is a fact about the call rather than about the
letter, so putting them inside it would both pollute the replay input and make two records
with identical readings from different providers compare unequal. Under the literal §6
signature there was simply **no channel** for two required fields of the record. The wrapper is
the smallest thing that carries them, it lives in `application/ports.py` where the seam already
is, and the domain never sees it.

### 11.4 Sampling determinism (plan §7.8, item 1) is not implementable

The plan's first determinism lever is `temperature = 0.0` with `top_p` pinned in the adapter.
**The adapter does not send either, and cannot.** The Anthropic Messages API removed
`temperature`, `top_p` and `top_k` for the model generation in use, and the SDK dropped them
from the method signature entirely — sending one is a **call-time `TypeError`**, not a request
the model could decline. Forcing it would break every extraction.

`thresholds.LLM_TEMPERATURE` is kept as a statement of intent and as the place a future
restoration would read from. `tests/unit/test_anthropic_extractor.py` **pins the SDK's call
signature**, so if a future release re-adds sampling controls the test fails and the parameter
should be pinned here again.

What is lost is one of four variance-*reducers* upstream of the boundary. Items 2 to 5 of
§7.8's determinism story all still hold: the model id is pinned and recorded, `PROMPT_VERSION`
is recorded and snapshot-tested, output is forced through a tool schema, and — the guarantee
that actually matters — `route()` remains a pure function of a stored extraction. The
downstream guarantee was never resting on the model repeating itself; that is precisely why it
was placed downstream. The eval harness measures the residual rather than assuming it away.

`OpenAIExtractor` (ADR-007) lands in the same place for a different reason. The Responses API
still *accepts* `temperature` and `top_p`, but the GPT-5 family are reasoning models on which
neither is a meaningful control, so the adapter sends neither. What it sends instead is
`reasoning.effort`, which is a cost lever rather than a determinism lever and is configurable
to `None` so a model that rejects it stays usable. Because the omission is our choice rather
than the SDK's, `tests/unit/test_openai_extractor.py` pins the *request* rather than the
method signature.

### 11.5 No-key contingency — CLOSED 2026-09-22

**Originally:** no API key was available during the build. The `role` / `role_confidence`
schema-fidelity assumption behind the whole Sample 3 design was unverified against a real
model, and `ROLE_MIN_CONFIDENCE`'s degrade-to-`AMBIGUOUS` behaviour was untested against
anything but `FakeExtractor`.

**Closing evidence, `gpt-5-nano-2025-08-07` via `OpenAIExtractor`, `reasoning.effort=minimal`.**
The probe ran: the three assignment samples through `OpenAIExtractor` directly (raw payloads
inspected by hand), then `pytest -m live` three times, then `make eval` at 5 runs × 8 letters
(40 live calls). Full findings and the prompt fix they produced are in **§11.7**. The short
version: **role labelling works.** Across every live run, roles were correct and consistent —
`operative` / `original` / `restricted` where the letter supported one, `unknown` where it
didn't — and `role_confidence` discriminated real ambiguity from real clarity rather than
sitting at a uniform value. On Sample 3's five `make eval` runs, `role_confidence` cleared
0.85 on both candidates in four and sat at 0.60/0.60 in one; that one run correctly degraded
to `AMBIGUOUS` and reported no operative amount rather than guessing — `ROLE_MIN_CONFIDENCE`
doing exactly its job. **The named fallback (`supersedes_a_prior_figure`) is not needed** and
is not being built; the mechanism this design bet on holds up against a real model.

The residual instability that *did* turn up was in a different field — `is_seizure_order`
and its confidence — for a different, more interesting reason. See §11.7.

### 11.6 Deliverable reconciliation

**The shared brief (`../shared-brief.external.md`) was never obtained.** The deliverable set
here — a git repository, `README.md`, this `DESIGN.md`, and an automated test suite — is an
assumed superset of the likely ask, so the risk is over-delivery rather than omission. If the
brief names something not present here, it was not visible at any point during this build.

### 11.7 Live-model findings: a prompt bug found and fixed, and a threshold finding left alone

This section exists because a live probe is only worth running if a genuine surprise gets
acted on rather than filed away. One did, and it did.

**The bug: `document_is_seizure_order` was undescribed, and the model guessed.** Neither
`schema.py` nor `SYSTEM_PROMPT` ever said what "seizure order" should mean for a letter that
*restricts* a prior order rather than opening one — `document_is_seizure_order` had no
`Field(description=...)` at all. Four live calls on Sample 3, before any fix:

| Run | `is_seizure_order` | confidence | route |
|---|---|---|---|
| 1 | `false` | 0.0 | `needs_human` (`H02`) |
| 2 | `false` | 0.0 | `needs_human` (`H02`) |
| 3 | `false` | 0.0 | `needs_human` (`H02`) |
| 4 | `true` | 0.92 | `auto` (`A00`) |

Every other field — debtor name, case reference, IBAN, both amount candidates and their
roles — read identically and correctly across all four runs, at 0.95 confidence. Only this
one undescribed boolean flipped, and it flipped in a way that would have silently routed a
textbook `auto` letter to `needs_human` three times out of four. **This is exactly the kind
of thing the no-key contingency existed to catch, and it caught something the original PM-A
worry (about `role`, not about this field) never anticipated.**

**The fix.** `schema.py`'s two document fields gained
`Field(description=...)` text, and `prompt.py` gained a DOCUMENT CLASSIFICATION paragraph
stating plainly that a letter which restricts, reduces or amends a prior seizure order is
still `document_is_seizure_order = true` — judged by its operative instruction, not by
whether the word "seizure" appears. Eight follow-up live runs on Sample 3:

| Run | `is_seizure_order` | confidence | route |
|---|---|---|---|
| 1–7 | `true` | 0.75 – 0.92 | `auto` (`A00`) ×7 |
| 8 | `true` | 0.75 | `needs_human` (`H02`) |

Zero misclassifications in twelve total live runs after the fix (4 more via `pytest -m live`,
5 more via `make eval`, all `true`). The one `needs_human` result in run 8 is not a
regression: the field was read *correctly* (`true`), just with confidence that happened to
land under the 0.85 floor that run — the system escalating on an honest low self-report,
which is what it is supposed to do. This is a genuine, structural fix, not a threshold tuned
to make a sample pass: it changes what the model is told, not what confidence is required.

**The finding left alone: `is_seizure_order_confidence` has real run-to-run variance.** After
the fix, every sample's *value* for this field reads correctly and consistently. Its
*confidence* does not: across the 40-run `make eval` pass, `is_seizure_order_confidence`
was the only volatile signal on otherwise-stable letters —

- Sample 1 (the cleanest letter in the set): `auto` ×4, `needs_human` (`H02`) ×1.
- Sample 3 (post-fix): `auto` ×2, `needs_human` ×3 (`H06` once, `H03` once, `H02` once) —
  Sample 3's own amount-candidate confidences were part of this run's variance too, not only
  the doc-type field.
- Wrong-`auto` count across all 40 runs: **zero.** Every disagreement with a label moved
  toward `needs_human`, never toward a false `auto`.

**Working hypothesis, not acted on:** `reasoning.effort=minimal` (chosen for cost — see
`config.py`) may be the cause. A reasoning model at its cheapest setting is plausibly less
consistent about *how sure* it is, even when it is consistently *right*. This is a hypothesis
someone should test by re-running `make eval` with `OPENAI_REASONING_EFFORT` unset or raised
and comparing stability columns — a two-line config change and a cost trade-off, not a code
change, and one this build deliberately leaves to whoever owns that trade-off rather than
making unilaterally. **No threshold was adjusted in response to this.** `AUTO_MIN_FIELD_CONFIDENCE`
and `DOC_TYPE_AUTO_MIN_CONFIDENCE` remain exactly as §5 states them, on the same discipline
argued there: five live letters is not the 200-letter labelled sweep that would justify
moving a number, and tuning a threshold to make a single eval run look stable is precisely
the mistake §5 warns against.

**One test was loosened, deliberately, and the reasoning is worth reading if you touch
`tests/live/`.** The live test originally asserted an exact `decisive_rule_id` per sample,
which is stricter than what the plan ever specified (`§13`: "asserts the labelled routes") and
stricter than the evidence supports — Sample 2's decisive rule alone was observed to split
`H05_DEBTOR_NAME_PARTIAL` / `H02_DOC_TYPE_UNCERTAIN` roughly 3-to-2 across six runs, both
legitimately true every time, only their table-order priority moving with real confidence
variance. `tests/live/test_live_extraction.py` now asserts what the assignment actually
labelled — the route — with an explicit safe-direction allowance: a sample labelled `auto`
may read `needs_human` (escalation, safe) but never `reject`; a sample labelled
`needs_human` must read exactly that, never `auto`. The exact decisive rule is reported in
the failure message for diagnosis but is no longer a pass/fail gate, matching the philosophy
`run_eval.py` already used (§7.11): a rule mismatch is a reported finding, not a red build.

---

## 12. Options rejected, each with its reason

| Option | Reason it was not built |
|---|---|
| **LangChain / LangGraph** | There is no graph. One call, one schema, no branching, no tool loop, no handoff — and ~40 transitive dependencies to make one HTTP request. LangGraph orchestrates multi-step state machines; here the state machine is our rule table and must stay a pure function, not a framework-owned runtime. |
| **LangGraph + a vector store (pgvector / Qdrant)** | There is no retrieval problem: no corpus, no ranking, no few-shot pool needing nearest-neighbour lookup, and the whole input is one short letter that fits in the prompt. Even a future version matching a restriction against the prior order it cites would be a lookup by `case_reference`, an exact key. |
| **`instructor` / `pydantic-ai`** | They hide the retry loop, and we need that loop visible: "the model failed twice → `needs_human` under `H01`" is a decision the audit record must carry, and a library that silently re-prompts and returns success erases the evidence. (The right call in a codebase with twenty extraction schemas; the wrong one where the retry has decision semantics.) |
| **Postgres via docker-compose** | It puts a container and a failure mode between the reviewer and a green test run, to demonstrate a capability `DecisionRepositoryPort` already proves. Production is a `DATABASE_URL` change plus a JSON→JSONB column type. |
| **Full tactical DDD: aggregates, repositories per entity, domain events, CQRS** | One entity, no cross-entity invariant. It costs a day and reads as cargo cult. |
| **A second LLM provider adapter** | The port makes it a ~70-line addition; building it before a provider is chosen is speculative work. The port, not a second implementation, is what proves swappability. |
| **Self-consistency sampling for confidence** | The strongest available confidence signal and the first planned upgrade, deferred purely on cost: N× latency and spend per letter. The eval harness already implements its measurement half. |
| **In-memory persistence only** | `GET /decisions/{id}` surviving a restart *is* the audit story. Kept as the test double. |
| **Logprob-derived confidence** | Not exposed under forced tool-use on the relevant APIs, and the token probability of a JSON field value is not the same quantity as confidence in a reading. |

---

## 13. Scope

> Out of scope per the assignment brief:
> - Executing, booking, or reversing a seizure. This service decides a route and stops.
> - Idempotency. Submitting the same letter twice creates two decision records.
> - Reversal or amendment of a decision. Records are append-only; there is no `PATCH`.
> - Multi-tenancy. One bank, one dataset, no tenant scoping.
> - Authentication and authorisation. In production this sits behind the bank's gateway.
> - OCR and PDF ingestion. The API accepts plain text only.
> - Queueing and async processing. Extraction is synchronous within the request.
> - Any user interface.
>
> Excluded by our own judgement, with reasons:
> - Retrieval augmentation / vector storage — no corpus and no retrieval problem.
> - Multi-step agent orchestration — a single structured call has no graph.
> - CQRS, domain events, aggregates — one entity, no cross-entity invariants.
> - A second LLM provider adapter — the port makes it a ~70-line addition; building it before a provider is chosen is speculative.
> - A replay endpoint — replay is guaranteed by design and asserted by a test; exposing it would widen the API beyond the specified surface.
> - Self-consistency sampling for confidence — the first planned upgrade, deferred on cost.

---

## 14. Known limitations

- **The restriction doctrine has n = 1.** §6.
- **The role gate is unverified against a real model.** §11.5.
- **Name validation is culturally naive.** Mononyms, particles (`van der`) and non-Latin
  scripts are all mishandled by a two-substantive-token rule. The failure direction is safe —
  it over-escalates and never over-approves — and the production fix is to match against the
  bank's customer record rather than guessing from the string.
- **Mod-97 is not an existence proof.** §4.
- **Prompt injection is defence in depth, not a solved problem.** Four layers: the letter is
  fenced and declared untrusted, output is a forced tool schema with no free-form channel,
  grounding zeroes any value whose evidence is not in the letter, and routing reads only typed
  fields. `I04` records the signal without letting a substring heuristic refuse a real court
  order.
- **Over-escalation is a real risk and is measured rather than asserted.** The eval report
  carries the `auto` rate. A high `needs_human` rate is acceptable at launch; the sweep in §5
  lowers it once clerk feedback exists.
- **Single currency.** The `Money` value object carries a currency but only EUR passes
  validation. A non-EUR figure escalates under `H14` rather than crashing.

---

## 15. Architecture Decision Records

### ADR-001 — Bounded DDD with ports and adapters

- **Decision.** Four layers (`domain`, `application`, `adapters`, `api`), `Protocol` ports, one
  entity, no aggregates, no domain events, no CQRS. `domain/` imports stdlib only, enforced by
  a test.
- **Drivers.** The assignment is about a seam between probabilistic and deterministic code;
  explainability outranks feature count; a reviewer must find the decision logic immediately.
- **Alternatives considered.** Flat three-module layout; full tactical DDD.
- **Why chosen.** The import restriction turns the thesis into an enforced property rather than
  a convention. The cost is about four extra files.
- **Consequences.** Every adapter needs an explicit mapping step. The domain cannot use
  pydantic, so value objects are hand-written frozen dataclasses.
- **Follow-ups.** None. If the service grew a second document type, the rule table would split
  per type before any new layer appeared.

### ADR-002 — One provider SDK adapter, no LLM framework

- **Decision.** Anthropic SDK with forced tool-use behind `ExtractorPort`. No LangChain,
  LangGraph, `instructor`, `pydantic-ai`, or vector store. One adapter, not two.
- **Drivers.** The retry carries decision semantics (`H01`); the reviewer should read the whole
  probabilistic surface in one sitting; the provider is unknown.
- **Alternatives considered.** `instructor` / `pydantic-ai`; LangChain/LangGraph; vector store;
  two adapters.
- **Why chosen.** A framework would hide the one control flow that must be visible, and there
  is no graph and no retrieval problem to justify either dependency. The port, not a second
  implementation, is what proves swappability.
- **Consequences.** ~70 lines of SDK code we own, including the retry. A second provider is a
  ~70-line addition.
- **Follow-ups.** Revisited by ADR-007: the key that became available is an OpenAI one, so that
  adapter was written — and both were kept rather than one replacing the other. The estimate
  held, and nothing outside `adapters/llm/` changed, which is the claim this ADR was making.

### ADR-003 — Confidence is `min(self-report, grounding, validator)`, derived inside the domain

- **Decision.** Per-field confidence is the minimum of three components, computed by `route()`
  from the stored extraction. `field_confidences` is an output, persisted as evidence, never an
  input.
- **Drivers.** The brief requires an explicit, deterministic derivation. Replay must genuinely
  re-derive rather than re-read.
- **Alternatives considered.** Validators-only, discarding the self-report; logprob-derived;
  self-consistency sampling; computing grounding in the adapter.
- **Why chosen.** `min()` lets the one unverifiable input escalate but never approve. Computing
  inside the domain means a grounding bug is caught by the replay test instead of frozen into
  records.
- **Consequences.** `LetterExtraction` must carry `letter_text`, making records larger. The
  adapter is a pure mapper, asserted by an import test.
- **Follow-ups.** Self-consistency is the first upgrade; the eval harness already implements its
  measurement half. **The "self-report" input is now a verbalised bin, not a float — see
  ADR-008.** The `min()` operator and its reasoning are unchanged; only what the model is
  permitted to express changed.

### ADR-004 — Amount candidates with roles, resolved by a versioned doctrine

- **Decision.** The schema enumerates every monetary figure with a role and a role confidence.
  `RESTRICTION_DOCTRINE_V1` in the domain decides which is operative. `value_as_written` is
  stored verbatim and parsed at routing time; no parsed `Decimal` is persisted.
- **Drivers.** Sample 3's operative-amount subtlety must be captured by structure, not prompt
  luck. Replay requires one representation per number.
- **Alternatives considered.** A single scalar `amount` field; storing both the verbatim string
  and the parsed value; trusting `role` without a confidence gate.
- **Why chosen.** It shrinks the model's job to reading comprehension and moves the legal
  judgement into ~20 lines of tested, versioned code. Two representations of one number would
  drift and make replay assert against the wrong one.
- **Consequences.** Parsing runs on every routing call, which is free at this scale. The
  doctrine's evidence base is n = 1 and is documented as such.
- **Follow-ups.** **Closed 2026-09-22 (§11.5, §11.7).** Verified live against `gpt-5-nano`:
  role labelling and `role_confidence` both hold up across real calls, including the correct
  `AMBIGUOUS` degrade when `role_confidence` was genuinely low. The named fallback
  (`supersedes_a_prior_figure`) is not needed and was not built. A separate, unrelated
  ambiguity in `document_is_seizure_order` *was* found live and fixed (prompt and schema,
  §11.7) — the doctrine and resolver themselves were never implicated.
  **A second, more serious gap was found 2026-09-23 (§16 finding 1): the exemption that let
  a SINGLE candidate resolve regardless of role or role confidence didn't account for a
  restriction letter where the model lists only the superseded figure. Fixed — a lone
  candidate on a restriction order now requires the same role and confidence gate a
  multi-candidate letter does.**

### ADR-005 — SQLite via async SQLAlchemy behind a repository port

- **Decision.** SQLite by default through `DATABASE_URL`, async SQLAlchemy 2.0, behind
  `DecisionRepositoryPort`.
- **Drivers.** Time-to-first-green-test for the reviewer; the audit story needs durability
  across a restart.
- **Alternatives considered.** Postgres in docker-compose; in-memory only; sync SQLAlchemy in a
  threadpool.
- **Why chosen.** Zero infrastructure, and the port plus SQLAlchemy demonstrates the same swap
  capability a container would, at none of the setup cost.
- **Consequences.** JSON columns are opaque text in SQLite. Async adds some ceremony to tests.
- **Follow-ups.** Production is Postgres: a URL change plus JSON→JSONB.

### ADR-006 — Our failure never rejects; an absent answer records nothing

- **Decision.** A malformed model response after one retry yields a stored decision routed
  `needs_human` under `H01`. An unreachable provider yields `503` and writes no record.
- **Drivers.** `reject` is a statement about the letter. The audit trail must not contain
  invalidity findings we invented, nor work items containing no information.
- **Alternatives considered.** Rejecting on malformed output; writing a `needs_human` record on
  transport failure; retrying indefinitely.
- **Why chosen.** It keeps the two failure kinds distinguishable forever in the data: a bad
  reading is evidence about the letter, an outage is evidence about us.
- **Consequences.** Callers must handle 503 and retry. `EXTRACTION_MAX_RETRIES = 1` means some
  transient failures surface as 503 rather than being papered over.
- **Follow-ups.** If 503 rates are material in production, add backoff and a circuit breaker in
  the adapter; the boundary is unaffected.

### ADR-007 — Second extractor adapter on the cheapest OpenAI model

- **Decision.** Add `OpenAIExtractor` behind the unchanged `ExtractorPort`, targeting
  `gpt-5-nano-2025-08-07` via the Responses API with a strict JSON-schema text format; keep the
  Anthropic adapter.
- **Drivers.** A real key exists only for OpenAI; the cheapest capable model minimises the cost
  of the eval loop the design depends on; the PM-A probe (§11.5) needed a live model to close,
  and did close against this adapter (2026-09-22).
- **Alternatives considered.** `responses.parse` (hides the malformed/outage distinction);
  strict function calling (equivalent, less idiomatic for a schema-as-reply); Chat Completions
  (legacy surface); replacing the Anthropic adapter (loses the reference implementation and the
  port demonstration).
- **Why chosen.** Same retry and failure semantics as ADR-006 with ~90 lines; the existing
  schema is strict-compatible with zero edits; $0.0003 per letter.
- **Consequences.** One more dependency; two private-import pins to watch on SDK upgrades
  (`openai.lib._pydantic.to_strict_json_schema`, pinned by a test); sampling parameters again
  not controllable, documented in §11.4. Two extra ways for the model to answer badly — a
  `refusal` content item and a non-`completed` status — both routed to the retry-then-`H01`
  path rather than to 503, and both asserted.
- **Follow-ups.** **The live schema-fidelity probe (PM-A, §11.5) ran against this adapter on
  2026-09-22 and is closed.** Findings and the prompt fix they produced are in §11.7. Prompt
  caching (input at 10 % of standard) is a free optimisation if volume ever matters.

### ADR-008 — Self-reported confidence is a verbalised bin, not a float

- **Decision.** `document_is_seizure_order_confidence`, every `ExtractedField.confidence`, and
  both `role_confidence`/`confidence` on an amount candidate are now `Literal["certain",
  "probable", "uncertain", "guess"]` on the wire (`adapters/llm/schema.py`). Each bin maps to a
  fixed float — `certain=0.97`, `probable=0.88`, `uncertain=0.60`, `guess=0.30` — inside
  `mapping.py`, before the value ever reaches the domain.
- **Drivers.** §11.7's own live evidence: the same letter, same value, reported
  `is_seizure_order_confidence` at 0.75 one run and 0.92 the next, purely because a float
  wobbled across the 0.85 floor. A model that has no real 0.01 resolution in how sure it is
  should not be asked to produce one.
- **Alternatives considered.** Leave it a float and tune the threshold (rejected — hides the
  actual defect, which is the input's false precision, not the floor); self-consistency
  sampling (still the right next upgrade, orthogonal to this — bins reduce jitter within one
  call, self-consistency would measure jitter across calls); logprobs (unavailable under
  strict structured output for this model family, per §11.4's reasoning).
- **Why chosen.** Quantisation is honest about what the quantity actually is, and it is a
  wire-level change only — `domain/` was not touched, because `mapping.py` converts to a
  plain float before constructing `LetterExtraction`/`AmountCandidate`/`ExtractedField`. Every
  domain rule, threshold comparison and reason template is unaffected.
- **Consequences.** `probable` sits at 0.88, deliberately above the 0.85 floor rather than on
  it — landing exactly on a floating-point boundary would make "probable" a coin flip against
  the gate it's supposed to clear. The prompt digest test pins the change.
- **Follow-ups.** Verified live (2026-09-23): the real model returns a valid bin for every
  confidence field, never a float, under strict schema enforcement. Whether the bin vocabulary
  itself needs revisiting is a question for the 200-letter sweep §5 describes, same as any
  other threshold-adjacent constant.

---

## 16. Independent review, 2026-09-23: 20 findings, all addressed

A five-lens review (code quality, architecture, security/data handling, LLM-engineering
practice, logging/observability) ran against the codebase as it stood after ADR-007. Full
findings are in `.omc/research/review-2026-09-23.md`. Every finding below was reproduced by
executing the code before being accepted, and every fix shipped with a test that would have
caught the original defect.

**Three more paths to a wrong `auto`, beyond the one §11.7 already found and fixed:**

1. **A restriction letter with one listed candidate could resolve `auto` on the superseded
   figure.** The sole-candidate exemption in `resolve_operative_amount` let any single
   candidate win regardless of role or role confidence, reasoning that "with one figure the
   label decides nothing" — true for an attachment order, false for a restriction order,
   where the model omitting the second figure is a plausible failure mode. Fixed: on a
   restriction order, a lone candidate must clear `ROLE_MIN_CONFIDENCE` as `restricted` or
   `operative`, or it falls to `AMBIGUOUS`.
2. **A NaN self-report bypassed the confidence floor.** `below_floor()` exists specifically to
   treat NaN as beneath every floor; the code path finding the first low-confidence field for
   `H09` used a raw `<` comparison instead, which is `False` for any NaN comparison. Fixed,
   with a test that isolates NaN on one field so it can't pass by accident the way the
   original property test did.
3. **A contradictory `document_type`/`is_seizure_order` pair could reject a real order.** A
   restriction order read as `is_seizure_order=False` simultaneously rejected under R01 *and*
   had the restriction doctrine applied to it by the resolver — internally inconsistent, and
   rejecting on a contradiction is not evidence the letter is invalid. Fixed: R01 now requires
   `document_type == "other"`; a new rule (`H17_DOCUMENT_TYPE_CONTRADICTION`) escalates the
   contradictory pair instead.

**A fourth, related finding, not a wrong `auto` but the same class of risk:** grounding
proves a value is *present* in the letter, never that it is the *authoritative* one — an
adversarial letter with two real figures and a line coaching the model to label the wrong
one `operative` would ground at 1.00 and route on the wrong amount, because nothing checks
the *role* label against the letter. The full fix (grounding the role label itself against
an evidence-quote lexicon) is out of scope for this pass; the cheaper mitigation shipped
instead: a letter with 2+ amount candidates never routes `auto` unless it's a restriction
order (`H18_MULTIPLE_FIGURES_NO_RESTRICTION_CONTEXT`).

**Malformed extractions no longer assert false findings.** `H02` (doc-type uncertain), `H05`
(name partial) and `H06` (IBAN absent) fired on a `MALFORMED_AFTER_RETRY` extraction —
reading as claims like "no IBAN given" and "debtor name is partial" about a letter nobody
actually read. Guarded on `extraction_status == OK`, matching the guard the five reject
rules already had. `H09` still fires on a malformed extraction's zero-confidence fields,
deliberately — "confidence 0.00" is a true statement, not a fabricated one.

**The `<letter>` fence was not escaped.** A letter containing `</letter>` could close the
fence early and put attacker text outside the region the system prompt calls untrusted. The
existing test proved this happened and treated it as correct; it wasn't. Fixed: any literal
fence tag inside the letter is now replaced with a visible redaction marker before
interpolation.

**The OpenAI prompt told the model to call a tool that was never sent.** `SYSTEM_PROMPT` said
"by calling the record_garnishment_extraction tool" on both transports, but the OpenAI
adapter sends a JSON-schema text format with no `tools` parameter at all. Strict mode forced
the shape regardless, so nothing visibly broke, but the instruction was false — plausibly a
contributor to the confidence jitter ADR-008 addresses. Fixed: the shared prompt is now
transport-neutral, and each adapter appends its own one-line instruction.

**The two adapters were not the matched pair the README claimed.** OpenAI got an explicit
transport retry budget; Anthropic used the SDK default of 2, so one letter could cost up to
6 upstream calls on that path versus 2 on the other. Fixed: both clients share one retry
budget. The alias-pinning check was also inconsistent — OpenAI correctly requires a dated
snapshot, and the fix does **not** force the same requirement onto Anthropic, because
Anthropic's current generation publishes no dated snapshots at all: the bare generation id
(`claude-sonnet-5`) is the pinned form, and `-latest` is the moving alias. A shared
`model_pinning.py` module holds one rule with two vendor-appropriate predicates, documented
so the asymmetry reads as a fact about the vendors rather than an inconsistency in the code.

**Every failure path used to log nothing.** A 503 left only uvicorn's access line; four
distinct "model answered badly" causes (schema miss, refusal, truncation, unparseable JSON)
were indistinguishable in `MALFORMED_AFTER_RETRY`. Fixed: an ERROR log on the 503 path
(exception class and a truncated message only — the letter is never logged), and a WARNING
per failed extraction attempt naming the specific cause. The diagnostic payload stored on a
total failure is also no longer `{}` when every attempt was a refusal or truncation.

**The decision log line gained the keys it was missing** — `letter_sha256`, `provider`,
separate `llm_ms`/`db_ms` alongside the existing total `latency_ms` — and the logger is now
actually safe to embed in a host application: `propagate = False`, UTC ISO-8601 timestamps
instead of local time, a configurable `log_level`, and re-configuration no longer silently
no-ops when a host has already attached its own handler.

**Smaller, real fixes:** the repository's `save()` used `session.merge` (an upsert) under a
docstring that says append-only — a second save with the same id silently overwrote the
first record; now an insert that's ignored on conflict, with a test proving it (the fix was
reverted and the new test was watched fail before being restored). No generic `Exception`
handler existed, so an unhandled error returned Starlette's default plain-text 500 instead
of the RFC 9457 body every other error uses; added, logging the real detail while returning
a fixed, generic message to the client. `FakeExtractor`'s canned model id was
`"claude-sonnet-5"` — indistinguishable from a real Anthropic response in a misconfigured
deployment's logs — renamed to `"fake-claude-sonnet-5"`. The amount parser accepted
malformed thousands grouping silently (`"4,25.00"` → 425.00, no ambiguity flag); it now
requires a 3-digit width on every group but the first, or treats the figure as unreadable.
An unreachable validator branch (a 2-decimal-place check the parser already guarantees) was
removed rather than left as dead code two tests exercised directly. Grounding's value-only
tier now tolerates a case difference the same way the quote tier already did — an all-caps
but correct name used to score 0.0 and trip `H08`. The eval report's stability figures no
longer count a reader outage as instability, and an outage's logged message no longer risks
carrying provider SDK error text (which can echo partially-redacted key material on a 401)
into a report file.

**The eval harness became a real instrument, not only a report.** `evals/run_eval.py --jsonl`
emits one machine-readable record per (letter, run) with every field's full confidence
breakdown; `expectations.yaml` now carries expected field values, not only expected routes;
`evals/sweep.py` runs the labelled-sweep procedure §5 describes against live data (64
observations, one live call each) and writes its findings to
`evals/THRESHOLD_SWEEP_FINDINGS.md`. The honest result: **the data is consistent with 0.85
and cannot distinguish it from any value up to 0.90** on this small, self-labelled set — the
21 observations the constant can actually move never had a gated confidence below 0.90, so
every threshold from 0.50 to 0.90 produces an identical outcome. That is not the 200-letter
experiment §5 describes; it is a demonstration that the described experiment is now
runnable, and a first, honestly-caveated answer.

**A small reasoning-effort check, 2026-09-23.** Sample 3 at `reasoning.effort=minimal` (the
configured default, chosen for cost) was stable 5/6 live runs; at `low`, 6/6. `N=6` per arm
is too small to be more than suggestive, but it's directionally consistent with §11.7's
hypothesis that a cheaper reasoning setting costs some stability. `document_is_seizure_order`
itself was perfectly stable at both settings (12/12) — the prompt fix in §11.7 closed that
gap regardless of effort level. `OPENAI_REASONING_EFFORT` is already an environment variable;
raising it is a config change, not a code change, and is left to whoever owns that cost
trade-off in production.

**Deliberately deferred, not silently dropped:** token usage, provider request-id, and
per-attempt-count telemetry on the decision log line — the review recommended it, but
closing it cleanly means re-extending `ExtractionResult` and re-touching both LLM adapters, a
larger, separate pass. A pre-parse Content-Length check protects the 100k-character cap
before the body is fully read; a chunked request with no declared `Content-Length` still
passes through, since bounding a stream is different machinery than this pass's one-field
API warrants. Both gaps are named, not hidden.
