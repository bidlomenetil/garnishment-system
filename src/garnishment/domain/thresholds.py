"""Every numeric constant the rule table depends on, each stated as a hypothesis.

**Framing.** No labelled evaluation set exists for this service. Three letters with
expected routes is a smoke test, not a validation set. So none of the numbers below is a
measured optimum, and none is presented as one. Each is an engineering prior chosen to
fail toward escalation, paired with the experiment that would settle it.

.. data:: CALIBRATION_PROCEDURE

   The procedure, stated once and referenced by every threshold below.

``rules.py`` contains no numeric literals: every comparison it makes reads a name from
this module, so changing a threshold is a one-line change in one file and shows up in a
diff against a stated hypothesis.
"""

from __future__ import annotations

from decimal import Decimal


def below_floor(value: float, floor: float) -> bool:
    """Is a confidence beneath a floor?

    Written as ``not value >= floor`` rather than ``value < floor`` so that a degenerate
    value -- a NaN arriving from a malformed reader response -- is treated as beneath
    every floor. Both spellings agree on real numbers; they differ only on NaN, and there
    the design's rule applies: fail toward escalation.
    """
    return not value >= floor


CALIBRATION_PROCEDURE = """
Label N letters (N ~ 200 is the smallest set that separates a 5% from a 15% escalation
rate with usable confidence) with a clerk's own route. Sweep the threshold across its
plausible range. At each point report two numbers: the count of wrong-`auto` decisions,
and the `needs_human` rate. Choose the highest threshold at which wrong-`auto` is zero,
then the lowest such threshold that holds it at zero, minimising human review subject to
that constraint. Until that experiment runs, the value in force is a prior, not a finding.
""".strip()

AUTO_MIN_FIELD_CONFIDENCE = 0.85
"""Floor a field's final confidence must clear for `auto` (rule H09).

Hypothesis: below roughly 0.85, a reader's self-reported confidence marks readings that
genuinely admit a second interpretation. This is NOT a claimed error rate — a
self-reported confidence is not a calibrated probability, so reading 0.85 as "we tolerate
15% wrong" misreads the quantity. The verifiable load is carried by grounding and the
validators; the self-report is a coarse detector of doubt they cannot see, and under
`min()` it can only ever lower a score.
Test: CALIBRATION_PROCEDURE.
Expected direction if it moves: up. The cost asymmetry is severe — a wrong `auto` is a
regulated incident, a needless `needs_human` costs a clerk minutes.
"""

DOC_TYPE_REJECT_MIN_CONFIDENCE = 0.90
"""Confidence required before we reject a document as not a seizure order (rule R01).

Hypothesis: R01 is the only reject that rests on a model judgement rather than a
mechanical check, and rejecting a genuine court order is the worst error this system can
make. Setting this above every other floor makes R01 the hardest rule in the table to
fire; below it the letter escalates instead of being refused.
Test: count R01 firings a clerk later overturns. Any non-zero count raises this number or
removes the rule entirely.
"""

DOC_TYPE_AUTO_MIN_CONFIDENCE = 0.85
"""Confidence required in "this IS a seizure order" before `auto` is available (rule H02).

Hypothesis: "is this even a seizure order" is the most load-bearing reading in the letter,
so it earns no discount relative to any other required field — hence the same value as
AUTO_MIN_FIELD_CONFIDENCE, deliberately not a separate looser number.
Test: CALIBRATION_PROCEDURE, sweeping this jointly with AUTO_MIN_FIELD_CONFIDENCE.
"""

ROLE_MIN_CONFIDENCE = 0.85
"""Confidence required in a candidate's *role label* before that label may decide which
figure is operative (see `operative_amount`).

Hypothesis: the amount design rests on role labels, so a label the reader is unsure of is
exactly as dangerous as an amount it is unsure of. Same floor as a field value, for the
same reason. Below it, resolution degrades to AMBIGUOUS and a human looks. A sole
candidate is exempt *off a restriction order*, because with one figure the role label
decides nothing; on a restriction order it is not exempt, because there the lone figure
may be the superseded one and the label is the only thing that says otherwise.
Test: measure observed `role_confidence` over live samples. If the model reports uniformly
high values with no discrimination, this gate is inert and the fallback is a narrower
per-candidate boolean ("does this figure supersede a prior one?") the resolver prefers.
"""

GROUNDING_EXACT = 1.00
"""Evidence quote occurs verbatim in the letter after the field's normaliser.

Hypothesis: a verbatim match is the strongest available evidence that a value was copied
rather than invented. Grounding's job is to catch fabrication, not to grade prose, so all
three non-zero tiers sit above AUTO_MIN_FIELD_CONFIDENCE by design.
Test: CALIBRATION_PROCEDURE over the tier assignment rather than a single number —
measure wrong-`auto` count partitioned by observed tier.
"""

GROUNDING_CASE_INSENSITIVE = 0.95
"""Evidence quote occurs case-insensitively only.

Hypothesis: a case-only mismatch means the reader retyped rather than copied — a weak
paraphrase signal worth recording but not worth escalating on.
Test: as GROUNDING_EXACT.
"""

GROUNDING_VALUE_ONLY = 0.90
"""Quote absent, but the extracted value itself occurs in the letter (rule I03).

Hypothesis: this is a real and common failure that is not a correctness failure — the
reader paraphrases or mis-copies the surrounding quote while the value is demonstrably
present. Scoring it 0.00 would escalate letters whose values are verifiably grounded;
scoring it 1.00 would hide that the citation was sloppy.
Test: as GROUNDING_EXACT; the number to watch is the wrong-`auto` count among decisions
carrying I03.
"""

GROUNDING_ABSENT = 0.00
"""Neither quote nor value occurs in the letter, or both are absent (rule H08).

Hypothesis: a value whose evidence cannot be located in the source is unsupported, and
unsupported is categorically different from merely uncertain — hence zero rather than a
small number, which forces the field's final confidence to zero through `min()`.
Test: not a tunable. Moving it off zero would mean accepting an ungrounded value, which
the design does not permit.
"""

AUTO_MAX_AMOUNT_EUR = Decimal("10000000.00")
"""Ceiling above which a letter is escalated regardless of confidence (rule H15).

Hypothesis: none, and deliberately so. This is a **deployment policy parameter the bank
sets**, not a claim about letters. Above
some figure, the cost of a wrong `auto` exceeds any clerk-time saving and the asymmetry
that justifies automation stops holding. Which figure that is belongs to the bank's risk
function; the value here is a placeholder.
Test: not an experiment we can run. It is set by the risk owner, and the only engineering
obligation is that it be a named constant in the audit trail rather than a hidden literal.
"""

MIN_CANDIDATES_FOR_A_SECOND_LOOK = 2
"""Figures in a letter above which `auto` needs structural help to stay available (H18).

Hypothesis: grounding proves a *value* appears in the letter; it can never prove a *role
label* is the authoritative one. So when a letter carries more than one figure, the choice
between them rests entirely on labels an adversary who controls the letter text can steer.
A restriction order is the one case where the doctrine picks the figure structurally, on
the relationship between the two figures rather than on trust in a label; absent that, a
second figure means a clerk looks.
Known limitation: this is the smaller of two remedies. The larger one — grounding the role
label itself against the evidence quote that justifies it — is not implemented, so a
single-figure letter still routes `auto` on an unverified label.
Test: count letters carrying two or more figures in production, and the share of them a
clerk confirms were unambiguous. A high confirmation rate is evidence for implementing the
larger remedy and relaxing this rule, not for relaxing it alone.
"""

MIN_SUBSTANTIVE_NAME_TOKENS = 2
"""Substantive name tokens required before a debtor name may clear `auto` (rule H05).

Hypothesis: an account cannot be identified from a surname plus an initial without a human
resolving it against the customer base.
Known limitation: culturally naive — mononyms, particles ("van der"), and non-Latin
scripts are all mishandled. The failure direction is safe: it over-escalates and never
over-approves. The production fix is to match against the bank's customer record rather
than guessing from the string.
Test: measure the H05 firing rate against a clerk's verdict on a labelled set; any case
where a clerk confirms the name was sufficient is evidence against the token heuristic
itself, not against the number.
"""

ZERO_EUR = Decimal("0.00")
"""The boundary a resolved amount must exceed to be positive (rule R05).

Hypothesis: none. Not a tunable — this is the definition of "not positive", named here
only so that ``rules.py`` carries no numeric literals and every comparison it makes is
traceable to this module.
Test: none applies; a change here would redefine the word, not calibrate a threshold.
"""

EXTRACTION_MAX_RETRIES = 1
"""Retries allowed after a malformed reader response before escalating under H01.

Hypothesis: one retry absorbs transient malformed JSON; a second mostly masks a prompt or
schema bug we want to see rather than paper over.
Test: count second-attempt successes in production. A high rate means the prompt is
marginal and should be fixed; a near-zero rate confirms one retry is the right budget.
"""

LLM_TEMPERATURE = 0.0
"""Sampling temperature for the extraction call.

Hypothesis: temperature 0 maximises reproducibility of the boundary's *input*. It does
not guarantee it, which is precisely why the determinism guarantee this service makes
lives downstream, in `route()` being a pure function of a stored extraction.
Test: the eval harness runs each sample N times and reports observed route stability.
"""
