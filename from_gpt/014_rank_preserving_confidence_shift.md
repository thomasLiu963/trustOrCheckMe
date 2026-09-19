# TASK 014 — Rank-Preserving Confidence-Level Shift
## Final preregistered experiment: separate confidence LEVEL from item-level discrimination

### Status

This task is the **one and only justified exception** to the current experimental stop rule.

It is not another benchmark replication.

It tests a validity threat that the current constant-score intervention cannot resolve:

> The fixed-score intervention changes both (a) the level of the displayed
> confidence signal and (b) its item-level information, because every item in a
> condition receives the same scalar.

A realistic confidence failure can instead preserve the ordering/information
carried by q1 while shifting the level systematically upward or downward.

Task 014 asks:

> **If the model sees a confidence signal that preserves q1's item ordering but
> is systematically shifted in level, does verification coverage and error
> leakage still move substantially?**

This is the final new-call experiment.

After Task 014, STOP experimenting regardless of outcome unless a genuine data,
evaluator, or reproducibility bug is discovered.

---

# 0. CORE SCIENTIFIC DISTINCTION

The existing constant-score intervention is:

same frozen item/output
+
same displayed score for every item in a condition

This cleanly identifies causal sensitivity to the displayed scalar, but the
score has no item-discriminative information within that condition.

Task 014 uses an **order-preserving transformation of each item's own q1**.

Therefore the displayed signal retains the item ordering already carried by q1.

The experiment separates:

1. **LEVEL**
   Is the confidence signal globally more optimistic/pessimistic?

from

2. **ITEM DISCRIMINATION**
   Which items receive relatively higher/lower confidence?

The transformation must preserve q1 ordering exactly except unavoidable ties
already present in q1.

Do not describe this as preserving "all information in the model state."

It preserves the **item ordering carried by the scalar q1**.

---

# 1. RESEARCH QUESTION

Primary question:

> When a naturally item-varying confidence signal is shifted monotonically
> upward or downward while preserving q1 ordering, does the downstream LLM
> router transfer that level shift into verification coverage and unverified
> error leakage?

Secondary question:

> How much of the original constant-score leakage effect survives when the
> scalar retains q1's item discrimination?

This is the deployment-relevant version of the current sensitivity result.

---

# 2. WHY THIS EXPERIMENT IS NEEDED

Current D1 cannot distinguish:

A. the router is highly sensitive to the **level** of confidence metadata;

from

B. the dramatic effect mainly occurs because the fixed-score manipulation
   destroys the scalar's item discrimination.

Task 014 resolves that.

No existing frozen condition provides a multi-point curve over
information-preserving confidence shifts.

`true_q_visible` is only the δ=0 point.

Hidden is a different policy and Task 013 showed it is not a robust mitigation.

Therefore new Stage-3 calls are necessary.

---

# 3. DATA / FROZEN OUTPUTS

Reuse frozen confirmatory items only.

## MMLU-Pro / Task 009

Models:
- GPT
- Claude

N:
- 500 items per model

Reuse:
- exact frozen Stage-1 answer
- exact natural q1
- exact correctness label
- exact Task-009 Stage-3 qualitative "moderate" routing prompt family

Do NOT regenerate answers.

Do NOT regenerate q1.

Do NOT change correctness.

## LiveCodeBench / Task 011

Models:
- GPT
- Claude

N:
- 286 confirmatory hard problems per model

Reuse:
- exact frozen code
- exact natural q1
- exact hidden-test correctness
- exact Task-011 Stage-3 routing prompt family

Exclude:
- all pilot items
- all non-confirmatory IDs

Do NOT rerun hidden tests except if needed for integrity verification.

---

# 4. CONFIDENCE TRANSFORMATION

For each item's natural q1, construct transformed displayed confidence:

    q_delta = sigmoid(logit(q1) + delta)

with frozen deltas:

    delta ∈ {-1.5, -0.75, 0, +0.75, +1.5}

Interpretation:

- negative delta = systematic pessimistic shift
- zero = original q1
- positive delta = systematic optimistic shift

This is an odds-space shift.

Equivalent odds multipliers approximately:

-1.5  -> ×0.223
-0.75 -> ×0.472
 0    -> ×1
+0.75 -> ×2.117
+1.5  -> ×4.482

This transformation is monotone increasing in q1.

Therefore, for any two items with q1_a < q1_b:

    q_delta_a < q_delta_b

except for ties already present in q1 and any explicit numerical
representation/rounding described below.

---

# 5. ENDPOINT / NUMERICAL HANDLING

Before any Stage-3 calls:

1. Inspect whether q1 ever equals exactly 0 or 1.
2. Record counts by task/model.

If all q1 are strictly inside (0,1):
- use the formula directly.

If exact 0 or 1 exist:
- do NOT silently clip and then claim exact ordering preservation.
- use the following frozen endpoint rule:

    if q1 == 0:
        q_delta = 0 for all delta

    if q1 == 1:
        q_delta = 1 for all delta

    otherwise:
        q_delta = sigmoid(logit(q1) + delta)

This preserves weak ordering and leaves certainty endpoints fixed.

Record how many endpoints are unaffected.

### Display precision

Use the same numerical formatting precision as prior q-visible routing prompts
if possible.

Before calls, freeze:
- decimal precision
- rounding rule
- exact rendered-string generator

Because rounding can introduce additional ties.

Compute and record:
- fraction of pairwise ordering preserved after formatting
- Spearman(q1, rendered q_delta)
- Kendall tau
- number/fraction of new ties introduced by rendering

Required target:
- Spearman effectively 1.0
- no order reversals

If rendering introduces reversals:
STOP and fix formatting BEFORE any scientific calls.

Ties are allowed if unavoidable and quantified.

---

# 6. PROMPT INTEGRITY

The Stage-3 prompt must remain prompt-identical to the corresponding existing
q-visible condition except for the displayed numerical confidence value.

Do NOT tell the router:
- that q was transformed;
- that it is a calibration experiment;
- that the score is optimistic/pessimistic;
- that the score may be wrong;
- that the experiment concerns robustness.

Present q_delta exactly as the confidence metadata would ordinarily be shown.

The transformation happens outside the model.

---

# 7. δ=0 REUSE CHECK

The δ=0 transformed score equals natural q1.

Before calls, compare the exact Task-014 δ=0 rendered prompt to the existing
Task-009 / Task-011 `true_q_visible` prompt on every item.

If byte-identical apart from metadata fields irrelevant to the provider call:
- reuse existing `true_q_visible` Stage-3 responses as δ=0.

If not prompt-identical:
- rerun δ=0 in Task 014 for methodological cleanliness.

Do not decide this based on outcomes.

Record the decision in:

`delta0_reuse_check.md`

---

# 8. PRE-CALL TRANSFORMATION AUDIT

Before preregistration freeze and before any new Stage-3 result inspection,
compute transformed q distributions for every task/model/delta.

Report:

- mean
- median
- IQR
- 5/25/50/75/95 percentiles
- min/max
- fraction >=0.90
- fraction >=0.95
- fraction >=0.99
- fraction <=0.70
- rank correlation with q1
- Brier score vs correctness
- ECE using the same Task-012 binning
- AUROC(q_delta, correctness)

Because the transform is monotonic, AUROC should be identical to q1 except for
rounding/ties.

This audit is allowed because it uses only q1/correctness and occurs before
new routing outcomes.

Do NOT tune delta values after seeing these summaries.

The delta grid is frozen in advance.

Create:

`transformation_audit.csv`
`transformation_audit.md`

---

# 9. PREREGISTRATION FREEZE

Before any new Task-014 Stage-3 calls, freeze:

- exact item IDs
- exact q1 values
- exact frozen outputs
- exact correctness labels
- delta grid
- transformation code hash
- endpoint rule
- display precision / rounding
- prompt hashes
- model endpoints
- model settings
- retry policy
- malformed-response policy
- primary endpoints
- bootstrap procedure
- decision buckets
- planned calls
- δ=0 reuse decision
- all analysis script hashes available before calls

Write:

`preregistration.md`
`freeze_manifest.json`
`analysis_freeze.json`

Timestamp before any new routing outcome inspection.

---

# 10. MODELS / SETTINGS

Use the exact same deployed endpoints and routing settings as Tasks 009/011 if
still available.

Primary:
- GPT endpoint matching Task 009/011
- Claude endpoint matching Task 009/011

If an endpoint has materially changed or is unavailable:

STOP.

Do not silently substitute a new model version.

Report:

`MODEL_VERSION_BLOCKER`

The scientific comparison requires the same deployed policy family.

---

# 11. CALL PLAN

Each item receives four new nonzero-delta Stage-3 routing calls if δ=0 is
reused.

Counts:

MMLU:
- 500 × 2 models × 4 new deltas = 4,000 calls

Code:
- 286 × 2 models × 4 new deltas = 2,288 calls

Total new scientific calls if δ=0 reused:
- 6,288

If δ=0 must be rerun:
- add 1,572
- total 7,860

No repeats are required for the main experiment.

Do not add repeats unless a technical failure makes a cell missing under the
predeclared retry rule.

No pilot.

No adaptive tuning.

---

# 12. PRIMARY ENDPOINT A — ERROR LEAKAGE SHIFT

Define exactly as Task 012:

    LEAKAGE(delta)
    =
    P(incorrect AND USE_UNVERIFIED | delta)

For each task/model compute:

- leakage at each delta
- paired difference:

    Δ_leakage =
    leakage(+1.5) - leakage(-1.5)

Use item-level paired bootstrap.

Bootstrap:
- minimum 5,000 resamples if computationally feasible
- same item IDs resampled across delta conditions
- 95% percentile CI

Primary inference is per task/model.

---

# 13. PRIMARY ENDPOINT B — COVERAGE SHIFT

Compute:

    COVERAGE(delta)
    =
    P(VERIFY_FIRST | delta)

Primary contrast:

    Δ_coverage =
    coverage(+1.5) - coverage(-1.5)

Expected direction:
- more optimistic confidence -> less verification
- therefore Δ_coverage likely negative

Use paired bootstrap 95% CI.

Coverage is preregistered primary alongside leakage because leakage can be
bounded by task error prevalence.

---

# 14. PRIMARY ENDPOINT C — INFORMATION-PRESERVING EFFECT RETENTION

Compare the Task-014 offset effect with the existing constant-score effect.

For each task/model:

    retention_leakage
    =
    |Δ_leakage_offset|
    /
    |Δ_leakage_constant_0.70_to_0.99|

and:

    retention_coverage
    =
    |Δ_coverage_offset|
    /
    |Δ_coverage_constant_0.70_to_0.99|

This is a descriptive ratio.

Do not interpret it as a universal causal decomposition.

Report bootstrap CI if technically straightforward using paired resampling of
the same items across old/new conditions.

If denominator instability is not an issue, bootstrap both numerator and
denominator jointly.

---

# 15. PRIMARY SUCCESS / OUTCOME BUCKETS

Freeze these buckets before calls.

Evaluate GPT as primary because Task 012's strongest system claim is GPT
cross-task sensitivity.

Claude remains a primary reported model but its code ceiling is an anticipated
identifiability boundary.

## WORLD A — `LEVEL_SENSITIVITY_STRONGLY_GENERALIZES`

Requirements:

GPT MMLU:
- leakage(+1.5) - leakage(-1.5) >= +5pp
- 95% CI lower bound > 0
- |coverage shift| >= 15pp

AND

GPT code:
- leakage shift >= +10pp
- 95% CI lower bound > 0
- |coverage shift| >= 15pp

AND

effect-retention:
- leakage retention >= 0.50 in both GPT tasks
  OR coverage retention >= 0.50 in both GPT tasks

Interpretation:

> Large confidence-to-oversight sensitivity survives when q1 item ordering is
> preserved.

This is the strongest result and should become a main-paper centerpiece.

## WORLD B — `LEVEL_SENSITIVITY_PARTIAL`

Requirements:

- leakage shift CI > 0 in both GPT tasks
- but at least one task fails WORLD A materiality/retention thresholds
- and effect retention is broadly between 0.25 and 0.50 or mixed

Interpretation:

> Item-level q1 information attenuates but does not remove confidence-level
> sensitivity.

This is still useful main-text evidence but requires more moderate wording.

## WORLD C — `ITEM_INFORMATION_LARGELY_PROTECTS`

Requirements:

In both GPT tasks:
- leakage retention < 0.25
  OR leakage shift < 5pp MMLU and <10pp code
- and/or coverage response becomes small

Interpretation:

> The dramatic constant-score effect depends substantially on stripping the
> scalar of item discrimination.

This narrows the current paper and becomes a design result:

> preserving item-relative confidence structure substantially protects the
> router from global level shifts.

Do not call Task 014 a failure.

It becomes a scope correction and actionable signal-design lesson.

## WORLD D — `MIXED_BY_TASK`

GPT MMLU and GPT code fall into qualitatively different buckets.

Interpretation:
- task dependence is stronger than expected;
- report both;
- do not force a universal claim.

## WORLD E — `TECHNICAL_OR_VERSION_FAILURE`

Examples:
- endpoint changed;
- prompts cannot be matched;
- transformed-score rendering breaks monotonicity;
- substantial missingness;
- preregistration/freeze integrity failure.

No scientific interpretation.

---

# 16. CLAUDE INTERPRETATION

Claude must be fully reported.

Expected complication:
- hard-code routing is near VERIFY saturation.

Therefore:

- a small Claude-code effect is not evidence against the GPT result;
- do not call Claude "robust" normatively;
- do not attribute saturation to guardrails;
- do not use Claude saturation to manufacture an across-model positive.

Report:
- coverage curve
- leakage curve
- q1 transform distributions
- ceiling diagnosis

Claude MMLU remains informative because it has more action range.

---

# 17. SECONDARY ANALYSIS — CALIBRATION LEVEL VS DISCRIMINATION

For each delta:

Report:
- Brier
- ECE
- mean q_delta on correct outputs
- mean q_delta on incorrect outputs
- AUROC(q_delta, correctness)

Key conceptual property:

A strictly monotone transform changes calibration/level but preserves scalar
ranking/AUROC up to ties.

This is the whole point.

Allowed statement:

> “The transformation changes confidence level/calibration while preserving
> the item ordering carried by q1.”

Avoid:

> “The signal preserves all information.”

---

# 18. SECONDARY ANALYSIS — COST / REGRET SWEEP

Reuse Task-012 normalized loss:

    loss(lambda)
    =
    leakage
    +
    lambda * verification_coverage

Use the exact same frozen lambda grid as Task 012.

For every delta/task/model report:

- loss(lambda)
- regret vs best observed Task-014 delta at each lambda
- optional regret vs full observed policy set
  (Task-014 deltas + old fixed-score + hidden + true-q)

Be careful:

The main scientific result is sensitivity, not which delta is optimal.

Do not cherry-pick a single lambda.

---

# 19. SECONDARY ANALYSIS — MARGINAL VERIFICATION EFFICIENCY

For the -1.5 -> +1.5 transition, compute:

- verification calls lost as confidence becomes more optimistic
- errors no longer caught
- errors caught per additional verification call when moving toward -1.5

For code:
- hidden-suite execution time if already available / cheaply derivable from
  frozen evaluator logs

No new verifier executions solely for this metric unless already required by a
reproducibility check.

---

# 20. FIGURES

Required main candidate figures:

## Figure A — Rank-Preserving Intervention Schematic

Natural q1 by item
-> same monotone logit shift applied to every item
-> ordering preserved
-> Stage-3 router
-> external verification

Show visually:
q1 ordering remains A > B > C across deltas.

## Figure B — Offset Coverage + Leakage Curves

For each task/model:
- x = delta
- panel 1 = coverage
- panel 2 = leakage

Highlight GPT MMLU and GPT code.

## Figure C — Constant vs Rank-Preserving Effect

Bars or points:
- constant-score endpoint effect
- rank-preserving offset endpoint effect

For:
- coverage
- leakage

Report retention ratio.

## Figure D — Natural q / Calibration Illustration

Optional:
show q distributions before and after offsets and verify ranking/AUROC preserved.

Do not overload the main paper.

---

# 21. CLAIM DISCIPLINE

If WORLD A:

Allowed:

> “Systematically shifting the level of an item-informative confidence signal
> can substantially change verification coverage and error leakage even when
> the signal's item ordering is preserved.”

> “The high-gain routing effect is not an artifact of replacing item-varying
> confidence with a constant.”

> “For GPT, a rank-preserving optimism shift still causes substantial external
> oversight loss on both QA and executable code.”

Do NOT say:
- ordinary production calibration drift necessarily causes identical effects;
- the transformed scores are naturally generated;
- ranking is perfectly informative;
- q1 contains all relevant uncertainty.

If WORLD C:

Allowed:

> “Preserving q1's item-relative structure substantially attenuates the
> constant-score failure mode.”

> “The strongest leakage effects arise when confidence metadata loses
> item-discriminative structure, narrowing the scope of the original stress
> test.”

This is still scientifically useful.

---

# 22. PAPER-INTEGRATION DECISION

After results, classify:

## `MAJOR_UPGRADE`

If WORLD A:
- rewrite paperDirection around information-preserving confidence-level
  sensitivity as the strongest deployment-relevant result.
- constant-score experiment becomes causal stress-test/extreme case.
- Task 014 becomes main text.

## `MODERATE_UPGRADE`

If WORLD B:
- retain Task-012 framing;
- add Task 014 as main supporting validation.
- update paperDirection, but do not replace the core leakage story.

## `SCOPE_CORRECTION`

If WORLD C:
- revise paperDirection to explicitly say:
  the dramatic sensitivity is strongest when item discrimination is degraded;
  preserving item ranking protects the router.
- Task 014 still belongs in main text because it answers the major validity
  objection.

## `TASK_DEPENDENT`

If WORLD D:
- integrate carefully as heterogeneity.
- no universal claim.

## `NO_INTERPRETATION`

If WORLD E:
- fix technical issue only if possible without changing scientific design.
- otherwise stop.

Task 014 must be reported internally regardless of outcome.

For the submitted paper:
- if technically valid, Task 014 should at minimum be mentioned because it
  directly addresses the constant-score validity objection.
- placement depends on outcome bucket.

---

# 23. REQUIRED OUTPUTS

Write to:

`to_gpt/014_rank_preserving_confidence_shift/`

Required root files:

- `report.md`
- `preregistration.md`
- `freeze_manifest.json`
- `analysis_freeze.json`
- `delta0_reuse_check.md`
- `transformation_audit.md`
- `cost_report.md`
- `deviations.md`
- `validation.md`
- `changed_files.txt`
- `paper_integration_packet.md`

Data:

- `transformation_audit.csv`
- `stage3_offset_conditions.csv`
- `coverage_by_delta.csv`
- `leakage_by_delta.csv`
- `effect_retention.csv`
- `cost_sweep.csv`
- `marginal_efficiency.csv`
- `cross_task_summary.csv`

Figures:

- `figures/rank_preserving_intervention.png`
- `figures/coverage_leakage_by_delta.png`
- `figures/constant_vs_offset_effect.png`
- `figures/transformed_q_distributions.png`
- `figures/task014_main_candidate.png`

Analysis scripts under:

`analysis/task014/`

Suggested:
- `transform_q.py`
- `audit_transform.py`
- `preregister.py`
- `run_stage3.py`
- `analyze_coverage.py`
- `analyze_leakage.py`
- `effect_retention.py`
- `cost_sweep.py`
- `figures.py`
- `run_all.py`

---

# 24. REPORT FORMAT

`report.md` must include:

1. Plain-English bottom line
2. Exact model endpoints
3. Preregistration/freeze integrity
4. δ=0 reuse decision
5. Transformation audit
6. Order-preservation audit
7. Prompt-integrity audit
8. API calls / retries / cost
9. Coverage by delta
10. Leakage by delta
11. GPT MMLU primary result
12. GPT code primary result
13. Claude MMLU result
14. Claude code saturation result
15. Constant-vs-offset effect retention
16. Calibration/discrimination diagnostics
17. Cost/regret sweep
18. Marginal verification efficiency
19. WORLD classification
20. Evidence against the current Task-012 framing
21. Strongest defensible paper claim
22. Claims that must NOT be used
23. Paper integration recommendation
24. Whether paperDirection should be rewritten
25. Are any further experiments scientifically justified?
26. `READY_FOR_GPT_REVIEW = YES`

---

# 25. PAPERDIRECTION OUTPUT

After Task 014:

If `MAJOR_UPGRADE`:
create:
`paperDirection_task014_proposed.txt`

If `MODERATE_UPGRADE`:
create:
`paperDirection_task014_proposed.txt`

If `SCOPE_CORRECTION`:
create:
`paperDirection_task014_correction.txt`

If `TASK_DEPENDENT`:
create:
`paperDirection_task014_proposed.txt`

If `NO_INTERPRETATION`:
do not rewrite the scientific narrative until the technical issue is resolved.

Do not overwrite the existing Task-012 paperDirection automatically.

---

# 26. ABSOLUTE FINAL STOP RULE

Task 014 is the final scientific experiment.

After Task 014, do NOT run:

- another benchmark;
- another model;
- Gemini/Grok code;
- an open-weight model;
- another confidence transformation;
- a different delta grid;
- a nonlinear calibration family;
- temperature scaling;
- isotonic scaling;
- additive probability shifts;
- more routing policies;
- more mitigation searches;
- human studies;
- mechanism probes.

Do not rescue an unfavorable Task-014 outcome.

Take the result and write the paper.

The only legitimate post-Task-014 work is:

- reproduce/fix genuine bugs;
- finalize literature;
- generate figures;
- write manuscript;
- adversarial review;
- revise claims for accuracy.
