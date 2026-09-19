# TASK 015 — Normative Robustness Audit
## Zero-call falsification-first test of the proposed “over-responsiveness” claim

### Mission

This is a **zero-new-model-call** analysis using frozen Tasks 009–014 only.

Claude proposed elevating D1 with a normative claim:

> the router's confidence-to-coverage response is steeper than the
> loss-minimizing response, potentially such that no verification/error cost
> ratio λ rationalizes the observed responsiveness.

This claim is potentially high-value, but it is NOT yet established.

Task 015 must try to **falsify it first**.

The purpose is to determine whether D1 can support one of the following:

A. a strong normative “over-responsiveness / no-λ rationalization” claim;

B. a narrower but rigorous **calibration-shift robustness / transformation-
   invariance** claim;

C. no new normative claim beyond the existing cost/regret analysis.

Do not assume A is true.

No new model calls.

Do not change paperDirection unless this audit passes a predeclared claim gate.

---

# 0. HARD RULES

1. ZERO API/model calls.
2. Use only frozen Tasks 009–014 data.
3. Do not change any historical artifact.
4. Label:
   `POST_HOC_NORMATIVE_REANALYSIS`.
5. Derive the loss threshold algebraically before running empirical analyses.
6. Do NOT use an oracle with per-item ground-truth correctness and then call it
   an attainable Bayes policy.
7. Do NOT claim a model is irrational.
8. Do NOT claim “no λ rationalizes the policy” unless the exact meaning of
   rationalizes is defined and the statement is empirically/theoretically true.
9. Do NOT redefine λ after looking at outcomes.
10. Use the exact Task-012 λ convention:
       loss = leakage + λ * coverage
    unless a second convention is explicitly labeled separately.
11. Do NOT use λ/(1+λ) as the verification threshold under the Task-012 loss
    unless a different λ definition is introduced and derived.
12. Do NOT compare D1 to an unattainable oracle without labeling ORACLE.
13. Do NOT manufacture a theorem from empirical lower-envelope plots.
14. If Claude's proposed C2 statement is false, say so explicitly.
15. Task 015 cannot trigger new experiments.

---

# 1. INPUTS

Read:
- latest `paperDirection.txt`
- Task 012 report + cost_sweep.csv
- Task 014 report
- Task 014 rank-preserving per-item Stage-3 actions
- q1 and transformed q_delta values
- correctness labels
- coverage/leakage by delta
- existing λ grid
- any existing calibration outputs

Primary cells:
- MMLU GPT
- MMLU Claude
- code GPT
- code Claude

Task 014 delta grid:
-1.5, -0.75, 0, +0.75, +1.5

---

# 2. FIRST: DERIVE THE DECISION THEORY CORRECTLY

Let:
- Y = 1 if frozen output is incorrect, 0 if correct
- V = 1 if VERIFY_FIRST, 0 if USE_UNVERIFIED
- verification is assumed perfect for this normalized analysis
- escaped-error cost = 1
- verification cost = λ

Per-item loss:

    ℓ = Y(1 - V) + λV

Expected loss given information Z and conditional error risk

    r(Z) = P(Y=1 | Z)

is:

USE:
    r(Z)

VERIFY:
    λ

Therefore the loss-minimizing action is:

    VERIFY iff r(Z) > λ

under the Task-012 λ convention.

Write this derivation into:

`decision_theory_derivation.md`

If any prior analysis/document uses threshold λ/(1+λ) while claiming the
Task-012 convention, flag it as a notation mismatch.

Do not proceed with a normative theorem until this is clean.

---

# 3. DISTINGUISH THREE DIFFERENT “OPTIMAL” BENCHMARKS

Task 015 must never blur these.

## Benchmark A — Ground-truth oracle

Uses Y for each item at inference.

If λ < 1:
- verify exactly incorrect outputs.

If λ > 1:
- verify none.

This is unattainable and only a lower bound.

Label:
`GROUND_TRUTH_ORACLE`

Do NOT use it to claim the deployed router is Bayes-irrational.

## Benchmark B — Calibrated scalar-risk policy

Uses only q1 through a cross-fitted estimate:

    r_hat(q1) ≈ P(error | q1)

Then:
    VERIFY iff r_hat(q1) > λ

Because Task 014 q_delta is a one-to-one monotone transform of q1 (up to
recorded ties/endpoints), a delta-aware recalibration can recover the same
scalar information.

Therefore this benchmark is delta-invariant by construction.

Label:
`Q1_ONLY_CALIBRATED_BASELINE`

This is attainable in principle but does not use all question/output context.

## Benchmark C — Existing δ=0 router

Use the actual Task-014 / true-q-visible routing actions at delta=0.

This is the same black-box router under the unshifted q1 display.

For each shifted delta, compare loss to the exact same delta=0 observed policy.

Label:
`UNSHIFTED_ROUTER_BASELINE`

This measures pass-through consequences relative to the system's own
unshifted operating point.

---

# 4. THE TRANSFORMATION-INVARIANCE PROPOSITION

Formalize the following carefully.

Let:

    S_delta = T_delta(Q)

where T_delta is a known strictly monotone bijection on the interior of the
score range.

Then the information carried by S_delta is equivalent to Q:

    sigma(S_delta) = sigma(Q)

up to the known endpoint/tie caveats.

Therefore a **delta-aware Bayes policy** based on the same underlying
information and fixed cost λ has the same optimal action set for every delta.

In words:

> A known invertible reparameterization of the same scalar information does
> not change the underlying cost-sensitive decision problem.

This is a benchmark property.

CRITICAL QUALIFICATION:

The deployed LLM router is NOT told delta and sees q_delta as a confidence
probability.

Therefore violation of this invariance demonstrates **lack of robustness of the
composed routing system to confidence-level shifts**.

It does NOT by itself prove:
- irrationality;
- that the model should infer the hidden transform;
- that a naturally generated probability shift must be ignored.

Write:

`transformation_invariance_note.md`

---

# 5. FALSIFICATION TEST OF CLAUDE'S “NO λ RATIONALIZES” CLAIM

Using Task-014 observed policies only, for every task/model/delta define:

    L_delta(λ)
    =
    leakage_delta
    +
    λ * coverage_delta

For each delta, solve exactly/numerically:

> Is there any positive λ interval on which this observed delta condition
> minimizes loss among the five observed Task-014 delta policies?

Create:
- `observed_policy_rationalization_intervals.csv`

For each cell report:
- delta
- λ interval(s) where it is best observed
- whether it is Pareto dominated
- whether it lies on the lower convex envelope of coverage/leakage points

This is a critical falsification.

Decision:

If a delta condition is best observed for some λ:
- it is FALSE to say that “no λ rationalizes that operating point” under the
  observed-policy comparison.

If all/most operating points have nonempty λ intervals:
- explicitly reject the strong Claude wording.

Also test whether there exists a **single fixed λ** for which all five observed
delta policies could simultaneously be Bayes-optimal.

This is a different question.

Do not conflate:
- “each operating point is rationalizable for some λ”
with
- “one fixed λ rationalizes the entire response curve.”

---

# 6. SAME-COST PERTURBATION REGRET

For each task/model/λ:

    best_observed_loss(λ)
    =
    min_delta L_delta(λ)

For each delta:

    R_observed(delta, λ)
    =
    L_delta(λ) - best_observed_loss(λ)

Also compute:

    sensitivity_cost_spread(λ)
    =
    max_delta L_delta(λ) - min_delta L_delta(λ)

Interpretation:

> how much normalized system loss can change solely because the confidence
> level is shifted while outputs, labels, and q1 ordering stay fixed?

This is a clean systems-robustness quantity.

Create:
- `rank_preserving_loss_spread.csv`
- `figures/rank_preserving_loss_spread.png`

Summaries:
- max spread
- mean spread
- longest λ interval with spread >= 0.01
- >= 0.02
- >= 0.05
- >= 0.10

No single λ is called realistic.

---

# 7. UNSHIFTED-ROUTER PASS-THROUGH REGRET

For each nonzero delta and λ:

    R_delta0(delta, λ)
    =
    L_delta(λ) - L_delta0(λ)

This answers:

> Under a fixed downstream cost ratio, how much does a pure rank-preserving
> level shift improve or worsen realized loss relative to the same router under
> its natural q1 display?

Report:
- positive-regret λ intervals
- negative-regret λ intervals
- worst positive regret
- best improvement
- zero-crossing λ values

Do NOT call positive regret universal harm if sign flips with λ.

Create:
- `delta0_pass_through_regret.csv`
- `figures/delta0_regret.png`

---

# 8. Q1-ONLY CALIBRATED INVARIANT BASELINE

Fit a risk mapping from natural q1 to correctness:

    r_hat(q1) = P(error | q1)

Primary method:
- cross-fitted isotonic regression if sample support permits

Sensitivity:
- cross-fitted logistic calibration
- optionally beta calibration if already available in repo

Rules:
- all predictions must be out-of-fold
- no Task-014 routing actions enter the fit
- use only q1 + correctness
- preserve task/model separation
- report fold design and seeds

For every λ:

    V_hat_i(λ) = 1[r_hat_i > λ]

Evaluate:
- coverage
- leakage
- loss

Because q_delta is invertible, the calibrated q1-only policy is the same across
delta.

Compare observed Task-014 policies to this baseline.

CRITICAL:

This baseline is NOT “the Bayes-optimal router” because the LLM router sees
question/output context beyond q1.

Call it only:

    q1-only calibrated invariant baseline

Create:
- `q1_invariant_baseline.csv`
- `figures/observed_vs_invariant_baseline.png`

If it performs worse than observed LLM routing at some λ, report that honestly.

---

# 9. STATUS-QUO / UNMANIPULATED ESCAPE RATE

Compute a sharp operational descriptive statistic for delta=0 / true-q-visible:

For each task/model:

    conditional_escape_rate
    =
    P(USE_UNVERIFIED | incorrect)
    =
    leakage / error_rate

This is NOT the same as leakage over all outputs.

For GPT code, verify the reported value implied by:
- failure rate ~53.5%
- leakage ~29.7%

which should be about 55% of incorrect programs left unverified.

Compute paired/bootstrap 95% CIs.

Preferred paper language if correct:

> “In the unmanipulated q1-visible condition, X% of incorrect programs are
> left unverified.”

Do NOT call this:
- real-world status quo
- deployment failure rate
- industry defect escape rate

unless the deployment context actually matches.

Create:
- `unmanipulated_escape_rate.csv`

---

# 10. OPTIONAL SENSITIVITY INSTRUMENT — DO NOT CLAIM NOVEL METRIC

Kumaran et al. already define policy temperature and confidence/threshold scale
for confidence-driven abstention.

Therefore D1 must NOT claim to invent generic confidence-response sensitivity.

If useful, compute a domain-specific descriptive instrument:

    verification-routing sensitivity

defined as average finite-difference / average partial effect of delta on:
- coverage
- leakage

under Task 014.

Use established language:
- average partial effect
- pass-through
- sensitivity

Do NOT call the construct novel.

Its novelty, if any, is the measurement procedure:
- frozen output
- external displayed scalar
- executed verifier
- leakage consequence

Create only if it improves presentation.

---

# 11. CLAIM GATE

Before inspecting new Task-015 output, freeze:

`task015_analysis_freeze.json`

Possible outcomes:

## `STRONG_NORMATIVE_CLAIM_SUPPORTED`

Requires ALL:

1. the “no λ” statement is precisely defined and survives the observed-policy
   rationalization audit;
2. no threshold-notation error;
3. rank-preserving shifts induce material positive regret relative to an
   appropriate invariant benchmark across a broad λ interval;
4. the claim does not require a ground-truth oracle;
5. wording can avoid calling the model irrational;
6. result is nontrivial beyond the existing Task-014 cost sweep.

Only then may the paper say “over-responsive” or “no λ rationalizes” prominently.

## `NORMATIVE_RESCOPED_TO_INVARIANCE`

Use if:
- strong “no λ” wording fails or is ambiguous,
- BUT the transformation-invariance benchmark is rigorous,
- AND rank-preserving shifts produce material loss variation / pass-through
  regret.

Paper claim becomes:

> “The composed verification policy is not invariant to rank-preserving
> confidence-level shifts; the same scalar information can induce materially
> different oversight costs under fixed system objectives.”

This is likely the safest conceptual upgrade.

## `NO_NEW_NORMATIVE_UPGRADE`

Use if:
- loss spreads are small,
- invariant baseline adds little,
- or the argument reduces to Task-012/014 restatement.

Then:
- do NOT rewrite the paper around Candidate 2.
- keep current Task-014 paperDirection.

## `CONTRADICTION_OR_ERROR`

Use if:
- existing cost analysis used an inconsistent λ threshold;
- reported numbers cannot be reproduced;
- paper claims are mathematically invalid.

Then issue a correction packet.

---

# 12. PAPER INTEGRATION RULES

If `STRONG_NORMATIVE_CLAIM_SUPPORTED`:
- Candidate 2 can become a main conceptual contribution.
- update abstract/introduction/results/discussion.
- keep exact theorem/assumptions explicit.

If `NORMATIVE_RESCOPED_TO_INVARIANCE`:
- add one main-text section on calibration-shift robustness.
- do NOT use “Bayes-dominated for every λ” unless literally proven.
- use “invariance benchmark,” “pass-through regret,” and “fixed-cost
  robustness.”

If `NO_NEW_NORMATIVE_UPGRADE`:
- do not add Task 015 to headline.
- optionally keep the unmanipulated escape rate if useful.

---

# 13. REQUIRED OUTPUTS

Write to:

`to_gpt/015_normative_robustness_audit/`

Required:
- `report.md`
- `decision_theory_derivation.md`
- `transformation_invariance_note.md`
- `task015_analysis_freeze.json`
- `paper_impact_decision.md`
- `validation.md`
- `changed_files.txt`

Data:
- `observed_policy_rationalization_intervals.csv`
- `rank_preserving_loss_spread.csv`
- `delta0_pass_through_regret.csv`
- `q1_invariant_baseline.csv`
- `unmanipulated_escape_rate.csv`

Figures:
- `figures/rank_preserving_loss_spread.png`
- `figures/delta0_regret.png`
- `figures/observed_vs_invariant_baseline.png`
- `figures/task015_main_candidate.png`

Analysis:
`analysis/task015/`

---

# 14. REPORT FORMAT

`report.md` must include:

1. Plain-English bottom line
2. Correct derivation of optimal verification threshold
3. Whether λ/(1+λ) was a notation error under Task-012 convention
4. Ground-truth oracle vs q1-only baseline vs delta=0 router
5. Transformation-invariance proposition
6. Observed-policy λ rationalization intervals
7. Is Claude's exact “no λ rationalizes” statement true or false?
8. Rank-preserving loss spread across λ
9. Delta=0 pass-through regret
10. Q1-only calibrated invariant baseline
11. GPT MMLU
12. GPT code
13. Claude MMLU
14. Claude code
15. Unmanipulated conditional escape rate
16. Whether “over-responsiveness” is defensible
17. Whether “transformation non-invariance” is defensible
18. Strongest exact paper claim
19. Claims to avoid
20. Paper-impact decision
21. Whether paperDirection should be rewritten
22. Whether any new experiment is justified
23. `READY_FOR_GPT_REVIEW = YES`

---

# 15. ABSOLUTE STOP

Task 015 is zero-call.

After it:
- no new experiment
- no new benchmark
- no new model
- no new delta grid
- no mitigation search
- no theorem-shopping by changing λ definitions

If the normative claim fails:
keep the Task-014 story and write.

If it passes:
integrate it and write.

Either way:
the experimental program remains closed.
