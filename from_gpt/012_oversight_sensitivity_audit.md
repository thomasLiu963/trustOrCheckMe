# TASK 012 — Oversight Sensitivity and Error-Leakage Audit
## Zero-call reanalysis of Tasks 009 + 011

### Purpose

This is a **zero-new-model-call** reanalysis.

The goal is to test whether the strongest positive result in D1 can be reframed more usefully as a systems-level sensitivity/failure-mode result:

> A counterfactual confidence value that carries no item-discriminative information within a condition can cause large changes in how much independent verification a deployed LLM requests, while the underlying outputs and their correctness remain fixed.

Task 012 must quantify the downstream consequences of that sensitivity.

This is **post-hoc analysis of prospectively collected data**. It must never be mislabeled as a preregistered endpoint.

Do not collect any new data.

Do not modify historical Tasks 009–011.

Read:
- current `paperDirection.txt`
- Task 009 preregistration / report / item-level artifacts
- Task 010 report
- Task 011 preregistration / deviations / report / item-level artifacts
- all exact scripts needed to reproduce correctness and routing labels

Write outputs to:

`to_gpt/012_oversight_sensitivity_audit/`

Store analysis code under:

`analysis/task012/`

---

# 0. INTEGRITY RULES

1. **ZERO API / MODEL CALLS.**
2. Use only frozen Tasks 009 and 011 artifacts.
3. Do not alter any prior data or analysis.
4. Label every Task-012 result:
   `POST_HOC_SYSTEMS_REANALYSIS`.
5. Do not claim exact ranking invariance.
6. Do not claim GPT is irrational.
7. Do not claim Claude is normatively correct.
8. Do not claim optimal verification coverage equals the empirical error rate.
9. Do not claim provider guardrails cause any effect.
10. Do not equate the constant-score intervention with ordinary monotonic calibration.
11. Do not claim the experiment directly simulates additive q1 drift unless that intervention was actually run.
12. If Task 012 materially contradicts the current paper claim, report the contradiction. Do not suppress it.
13. If Task 012 is merely unhelpful, it may be omitted from the main paper, but preserve the report internally.
14. Do not edit `paperDirection.txt` unless the predeclared paper-impact gate in Section 12 passes.
15. If the gate passes, write a **new proposed** paperDirection file; do not overwrite the current source of truth.

---

# 1. CORE REFRAME TO TEST

The current paper emphasizes:

> displayed confidence can causally change verification coverage.

Task 012 asks whether the more important systems result is:

> changing confidence metadata can create large changes in verification resource use and in the number of erroneous outputs that escape independent checking, even though the underlying output population is unchanged.

Important nuance:

Within a fixed displayed-score condition, every item receives the same displayed value.

Therefore that displayed scalar contains:

- semantic information about the *claimed* correctness probability;
- **zero between-item / item-discriminative information within that condition**.

It can still interact with item features inside the adaptive LLM policy.

Use the phrase:

> “no item-discriminative information within the condition”

NOT:

> “no information whatsoever.”

---

# 2. DATASETS / MODELS

## Task 009 — MMLU-Pro

Primary:
- GPT
- Claude
- N=500 each

Conditions:
- hidden
- true_q_visible
- fixed 0.70
- fixed 0.85
- fixed 0.90
- fixed 0.95
- fixed 0.99

Need per item:
- question_id
- model
- frozen answer
- correctness
- q1
- Stage-3 condition
- Stage-3 action

## Task 011 — LiveCodeBench

Primary:
- GPT
- Claude
- N=286 hard problems each

Same Stage-3 condition family.

Need per item:
- problem_id
- model
- frozen code
- official hidden-test correctness
- q1
- Stage-3 condition
- Stage-3 action
- execution time if available

Use only confirmatory IDs.

Exclude all pilot IDs.

---

# 3. REPRODUCTION CHECK FIRST

Before new analysis, reproduce exactly:

Task 009:
- GPT 0.70→0.99 coverage shift = 50.2pp
- Claude = 35.4pp

Task 011:
- GPT = 38.8pp
- Claude = 4.9pp

Task 011 frozen-code correctness:
- GPT = 46.5% pass
- Claude = 43.0% pass

If any cannot be reproduced within rounding tolerance:

STOP with:

`TASK012_REPRODUCTION_FAILURE`

Do not continue until discrepancy is resolved.

Write:

`reproduction_check.md`

---

# 4. PRIMARY ANALYSIS A — ERROR LEAKAGE AT EACH DISPLAYED SCORE

For each:
- task
- model
- displayed score condition

compute:

1. N outputs
2. N incorrect outputs
3. N VERIFY_FIRST
4. verification coverage
5. N incorrect + VERIFY_FIRST
6. N incorrect + USE_UNVERIFIED
7. error-catch rate
8. **unverified-error rate per all outputs**
9. **unverified errors per 100 outputs**
10. **fraction of all errors left unverified**

Primary downstream quantity:

`LEAKAGE(q) = P(incorrect AND USE_UNVERIFIED | displayed q)`

Because correctness and frozen outputs are identical across counterfactual score conditions, differences in leakage across fixed-score conditions are causally attributable to the routing intervention under the experimental setup.

For each adjacent score transition and for 0.70→0.99, report:

- Δ verification coverage
- Δ unverified-error rate
- Δ unverified errors per 100 outputs
- Δ error-catch rate

Use paired question/problem bootstrap 95% CIs.

Bootstrap cluster:
- question_id for MMLU
- problem_id for code

At least 5,000 bootstrap resamples if computationally cheap; minimum 2,000.

Create:

`error_leakage_by_condition.csv`

and:

`figures/error_leakage_curve.png`

The figure should show, for each model/task:
- x = displayed confidence
- left y = verification coverage
- right or separate panel = unverified errors per 100 outputs

Do not combine scales in a visually misleading way. Prefer separate panels if necessary.

---

# 5. PRIMARY ANALYSIS B — OVERSIGHT SENSITIVITY / ROUTING GAIN

Do not force a global linear slope.

Compute finite-difference sensitivity between adjacent displayed scores.

For transition q_a → q_b:

`coverage_gain = (VERIFY_rate_b - VERIFY_rate_a) / (q_b - q_a)`

Report in an intuitive unit:

**verification-coverage percentage points per +0.10 displayed confidence**

Because higher confidence usually reduces verification, preserve the sign but also report magnitude.

Similarly:

`leakage_gain = (LEAKAGE_b - LEAKAGE_a) / (q_b - q_a)`

Report:

**additional unverified erroneous outputs per 100 total outputs per +0.10 displayed confidence**

For each task/model, report:

- adjacent local sensitivities
- 0.70→0.99 average sensitivity
- bootstrap CIs
- maximum local sensitivity
- whether sensitivity is monotonic / threshold-like / saturated

Call these:

- `coverage sensitivity`
- `error-leakage sensitivity`

Avoid the word “amplification” unless carefully defined.

Create:

`oversight_sensitivity.csv`

and:

`figures/oversight_sensitivity.png`

---

# 6. PRIMARY ANALYSIS C — NATURAL q1 OPERATING RANGE

The obvious external-validity objection is:

> fixed 0.70–0.99 confidence values are artificial.

Use the frozen q1 data to show where the model naturally operates.

For each task/model compute:

- mean q1
- median q1
- SD / IQR
- 5th/10th/25th/50th/75th/90th/95th percentiles
- fraction of q1 within [0.70, 0.99]
- fraction within each experimental interval:
  - [0.70,0.85)
  - [0.85,0.90)
  - [0.90,0.95)
  - [0.95,0.99]
- same summaries separately for correct and incorrect outputs

Compute calibration diagnostics from q1:
- Brier score
- ECE using a predeclared binning scheme
- reliability table
- mean q1 on incorrect outputs
- mean q1 on correct outputs

Do not overinterpret ECE.

Create:

`q1_distribution.csv`
`q1_calibration.csv`

Figure:

`figures/causal_curve_with_q1_distribution.png`

Recommended layout:
- top: fixed-score causal coverage curve
- bottom: q1 histogram / density / rug for correct vs incorrect outputs
- same x-axis 0–1

The visual question:

> Does the experimental intervention sweep traverse score values the model naturally reports?

Important:
Do NOT say natural q1 values prove that a constant overwrite simulates natural miscalibration.

Safe language:

> The intervention probes a numerical range that overlaps / does not overlap the model’s naturally reported confidence distribution.

---

# 7. PRIMARY ANALYSIS D — TRUE-q-VISIBLE AND HIDDEN OPERATING POINTS

For each task/model:

Compare:
- hidden
- true_q_visible
- fixed-score curve

Report:
- verification coverage
- leakage
- error catch
- precision among verified

On the fixed-score plot, add:
- hidden condition as a labeled horizontal/reference marker
- true_q_visible as another marker

Do NOT assign a single x-coordinate to hidden.

For true_q_visible, do not simply plot at mean(q1) as if Jensen/nonlinearity did not matter.

If useful:
- show it as a side marker / distinct symbol without claiming equivalence to fixed q=mean(q1).

Question:

> How does the model’s natural confidence-visible policy compare descriptively with the counterfactual fixed-score response?

Create:

`natural_vs_counterfactual.csv`

---

# 8. PRIMARY ANALYSIS E — DECISION-THEORETIC COST / REGRET SWEEP

This section introduces a principled “should” axis WITHOUT pretending that
optimal coverage equals the empirical error rate.

Define:

- cost of using one incorrect, unverified output = 1 unit
- cost of one independent verification call = λ units

For each condition/policy:

`loss(λ) = leakage_rate + λ * verification_coverage`

This is an intentionally normalized loss.

Sweep λ over a broad logarithmic grid, for example:

0.001, 0.002, 0.005,
0.01, 0.02, 0.05,
0.10, 0.20, 0.50,
1.0

Optionally use a denser log grid for figures.

For each:
- task
- model
- condition
- λ

compute normalized loss.

Then compute two regret references:

## Reference A — Best observed fixed-score condition

At each λ:

`regret_best_fixed = loss(condition) - min_fixed_score loss`

This answers:

> how costly is choosing the wrong confidence-induced operating point among the actually tested fixed-score policies?

## Reference B — Hindsight oracle lower bound

If the existing routing audit defines an oracle, compute an oracle lower-bound policy consistently.

For perfect/benchmark-ground-truth verification:
- if λ < 1, an omniscient oracle verifies exactly the incorrect outputs;
- if λ >= 1, it is no cheaper to verify an error than absorb the normalized error cost.

Use this only as an unattainable lower bound, clearly labeled ORACLE.

Do not imply deployed systems can attain it.

For code, stay within benchmark semantics:
“incorrect” means failing the official hidden suite.

Create:

`cost_sweep.csv`

Figures:

`figures/cost_regret_sweep_mmlu.png`
`figures/cost_regret_sweep_code.png`

Main questions:

1. Across what λ ranges does the confidence-induced operating point materially change expected normalized loss?
2. Does a high displayed confidence create high regret when verification is cheap relative to an escaped error?
3. Does Claude-code saturation become costly when verification is expensive?
4. Does GPT-code under-verification become costly when escaped errors are expensive?

This analysis is where the paper may obtain a real normative systems result.

Do NOT choose one λ after seeing the data and call it “the realistic cost.”

Report the full sweep.

---

# 9. SECONDARY ANALYSIS F — RESOURCE EFFICIENCY OF COVERAGE CHANGES

For every adjacent score transition:

Compute:

`additional errors caught / additional verification calls`

when moving toward lower displayed confidence / higher verification.

This is a descriptive marginal efficiency quantity.

For code, additionally compute if execution-time data exist:

- added verification CPU/wall seconds
- errors caught per 100 verifier calls
- errors caught per aggregate execution second

Do not turn wall-clock runtime into dollars unless actual pricing is documented.

Create:

`marginal_verification_efficiency.csv`

---

# 10. SECONDARY ANALYSIS G — CONSTANT-SCORE INFORMATION CHECK

Formalize the central conceptual fact.

Within each fixed displayed-score condition:

- variance of displayed score across items = 0
- mutual/discriminative information from score alone for item correctness = 0 by construction

Do not overcomplicate this with an empirical mutual-information estimator.

State analytically:

> The fixed displayed score cannot distinguish items within a condition. Any item-specific allocation differences must arise from the model’s interaction of that constant scalar with the question/output/context or from stochasticity.

This is a conceptual observation, not a new result.

Create a short section in `report.md`.

---

# 11. OPTIONAL ANALYSIS H — SIMPLE MEDIATION/DECOMPOSITION OF LEAKAGE CHANGE

Only if statistically straightforward and clearly interpretable.

Decompose changes in unverified-error rate into:

- coverage movement
- selection/allocation differences

A safe approach is matched-budget standardization:

For each pair of score conditions:
1. observe actual leakage difference;
2. estimate expected leakage difference if the lower-coverage condition simply selected a random subset from the higher-coverage VERIFY set or under a shared-prioritization ordering;
3. compare observed to that standardized reference.

If this becomes methodologically messy, SKIP.

Do not create a new headline analysis.

---

# 12. PAPER-IMPACT GATE — FREEZE BEFORE LOOKING AT TASK-012 RESULTS

Create:

`task012_analysis_freeze.json`

before computing Sections 4–9.

The decision about whether Task 012 should reframe the paper is:

## `MATERIAL_UPGRADE`

Use this only if ALL are true:

A. **Consequence**
At least one prospectively studied model shows a clearly material change in
unverified-error rate across 0.70→0.99, with bootstrap CI supporting a
nontrivial effect, AND the effect appears in at least two distinct
task/model cells (for example GPT MMLU + GPT code).

B. **Operating-range relevance**
The fixed-score sweep overlaps materially with the natural q1 distribution in
the relevant cells. As a descriptive gate:
- at least 25% of q1 values fall in [0.70,0.99], OR
- the central 50% range intersects the manipulated grid substantially.

C. **Systems interpretation**
The λ cost sweep shows that confidence-induced operating-point differences can
produce meaningful normalized-loss differences across a nontrivial interval of
cost ratios, not only at a single extreme λ.

D. **No contradiction**
The analysis does not reveal a data/reproduction problem or a consequence that
invalidates the proposed causal framing.

If A–D hold:
- classify `MATERIAL_UPGRADE`
- write a proposed revised paper direction focused on oversight sensitivity /
  confidence-metadata leverage.

## `USEFUL_SECONDARY`

Use if the consequence curves are informative but B or C is weak.

Then:
- keep current paperDirection as source of truth;
- recommend one main-text or appendix consequence figure;
- do NOT reframe the whole paper.

## `NO_MATERIAL_UPGRADE`

Use if:
- leakage changes are small,
- the natural q1 range does not overlap,
- cost consequences only appear under extreme assumptions,
- or the analysis adds little beyond the existing coverage curves.

Then:
- do not rewrite paperDirection;
- preserve Task 012 internally;
- optionally mention only if needed for completeness.

## `CONTRADICTS_CURRENT_FRAMING`

Use if:
- analysis exposes a substantive contradiction,
- reproduction fails,
- or the current paper’s claims are materially misleading.

Then:
- DO NOT “forget Task 012 exists.”
- produce an explicit correction packet.
- rewrite paperDirection to become more accurate, even if less flattering.

---

# 13. CLAIM DISCIPLINE FOR A SUCCESSFUL TASK 012

If `MATERIAL_UPGRADE`, allowed phrasing includes:

> “A counterfactual confidence value with no item-discriminative information
> within a condition can move verification coverage by tens of percentage
> points while the frozen output population and its correctness remain fixed.”

> “These coverage shifts translate into measurable changes in the number of
> erroneous outputs that escape independent checking.”

> “The routing policy exhibits high behavioral sensitivity to displayed
> confidence over a numerical range that overlaps the model’s naturally
> reported confidence.”

> “The downstream cost of this sensitivity depends on the relative cost of
> verification and escaped errors.”

> “GPT exhibits substantially higher confidence-to-oversight gain than
> Claude’s saturated hard-code policy.”

Do NOT write:

- “false confidence causes X” unless the score is explicitly defined as false;
- “miscalibration causes X” — constant overwrite is not a calibration intervention;
- “natural q1 drift would cause exactly the same slope”;
- “optimal coverage equals error rate”;
- “Claude is more rational/safe”;
- “GPT irrationally trusts confidence”;
- “guardrails explain Claude”;
- “confidence contains zero information” without “item-discriminative within condition” qualifier.

---

# 14. REQUIRED OUTPUTS

Root:

`to_gpt/012_oversight_sensitivity_audit/`

Files:

- `report.md`
- `reproduction_check.md`
- `task012_analysis_freeze.json`
- `paper_impact_decision.md`
- `changed_files.txt`
- `validation.md`

Data:

- `error_leakage_by_condition.csv`
- `oversight_sensitivity.csv`
- `q1_distribution.csv`
- `q1_calibration.csv`
- `natural_vs_counterfactual.csv`
- `cost_sweep.csv`
- `marginal_verification_efficiency.csv`

Figures:

- `figures/error_leakage_curve.png`
- `figures/oversight_sensitivity.png`
- `figures/causal_curve_with_q1_distribution.png`
- `figures/cost_regret_sweep_mmlu.png`
- `figures/cost_regret_sweep_code.png`
- `figures/task012_main_candidate.png`

Analysis code:
- `analysis/task012/reproduce.py`
- `analysis/task012/leakage.py`
- `analysis/task012/sensitivity.py`
- `analysis/task012/q1_range.py`
- `analysis/task012/cost_sweep.py`
- `analysis/task012/figures.py`
- `analysis/task012/run_all.py`

---

# 15. REPORT FORMAT

`report.md` must have:

1. Plain-English bottom line
2. Reproduction check
3. Why the constant-score intervention is item-uninformative within condition
4. Verification coverage by score
5. Unverified-error leakage by score
6. Coverage sensitivity per +0.10 displayed confidence
7. Error-leakage sensitivity per +0.10
8. Natural q1 distribution and manipulated-range overlap
9. q1 calibration diagnostics
10. Hidden / true-q-visible reference behavior
11. Cost/regret sweep
12. Marginal verification efficiency
13. GPT vs Claude
14. MMLU vs code
15. What Task 012 does NOT establish
16. Strongest systems interpretation
17. Paper-impact gate result
18. Exact recommended abstract sentence, if any
19. Exact claims to avoid
20. Whether paperDirection should be rewritten
21. `READY_FOR_GPT_REVIEW = YES`

---

# 16. CONDITIONAL PAPERDIRECTION OUTPUT

If and only if:

`paper-impact decision = MATERIAL_UPGRADE`

create:

`paperDirection_task012_proposed.txt`

This should be a **full replacement source-of-truth document**, not a patch.

New central framing should be approximately:

> confidence-metadata sensitivity / oversight leverage

rather than:

> coverage versus ranking.

The paper should emphasize:

1. fixed-output causal intervention;
2. large transfer function from displayed confidence to verification resource use;
3. downstream error leakage;
4. overlap with the natural reported-confidence range;
5. cost-dependent systems consequences;
6. model/task heterogeneity and saturation;
7. allocation as secondary/supporting analysis;
8. behavioral—not mechanistic—claims.

If decision is `USEFUL_SECONDARY` or `NO_MATERIAL_UPGRADE`:
- DO NOT generate a replacement paperDirection.
- Instead write `paperDirection_recommendation.md` describing the one or two
  Task-012 additions worth retaining.

If decision is `CONTRADICTS_CURRENT_FRAMING`:
- generate `paperDirection_correction_required.txt`.

---

# 17. STOP RULE

Task 012 is a zero-call reanalysis.

After Task 012:

- do not launch another benchmark;
- do not add models;
- do not run a q1+delta experiment merely because it would fit the new story;
- do not search for a cost ratio that makes the result dramatic;
- do not add post-hoc routers.

If Task 012 yields `MATERIAL_UPGRADE`, reframe and write.

If it yields `USEFUL_SECONDARY`, retain the current story and use the consequence
analysis as supporting evidence.

If it yields `NO_MATERIAL_UPGRADE`, keep the current story.

If it contradicts the current story, correct the story.

No result-shopping.

The next phase after Task 012 is manuscript writing.
