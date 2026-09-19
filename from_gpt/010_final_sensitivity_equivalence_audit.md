# TASK 010 — Final Zero-Call Sensitivity, Equivalence, and Routing Audit

## Purpose

Task 009 prospectively supported the central MMLU-Pro result:

- counterfactual displayed confidence produces large verification-coverage shifts;
- the preregistered condition-sensitive reprioritization model provided essentially no held-out predictive improvement over the shared-ranking model.

Before freezing the exact wording of the paper's equivalence / ranking-stability claim, run one final **zero-API-call** audit on the existing Task-009 data.

This task does NOT search for a new story.

It has five purposes:

1. verify that the Task-009 H2 analysis had enough sensitivity to detect scientifically meaningful reranking if it had existed;
2. test whether the observed binary-action AUROC drift is quantitatively predicted by the shared-ranking / moving-threshold model;
3. estimate latent cross-condition ranking stability while accounting properly for the fact that each repeat propensity is based on only three Bernoulli generations;
4. test two specific post-hoc but plausible mismatch-driven reranking alternatives;
5. finish the already-planned matched-budget routing analysis and determine whether it deserves a figure or only a paragraph.

All analyses are on already-generated data. **0 API calls.**

Save this exact task as:

`from_gpt/010_final_sensitivity_equivalence_audit.md`

Write outputs to:

`to_gpt/010_final_sensitivity_equivalence_audit/`

Do not modify `paperDirection.txt` during this task.

---

# NON-NEGOTIABLE RULES

1. **0 API calls.**
2. Read-only use of existing Tasks 007–009 data.
3. Do not alter historical databases.
4. Do not redefine Task-009 preregistered hypotheses or pretend Task-010 was preregistered.
5. Explicitly label:
   - confirmatory Task-009 evidence;
   - Task-010 sensitivity analyses;
   - Task-010 post-hoc robustness analyses.
6. All bootstrap/resampling units are question IDs.
7. Do not infer hidden neural states.
8. Do not call cross-model difficulty a deployable production feature.
9. Do not search many feature combinations or model classes for a favorable result.
10. The paper claim must become *more precise* after this task, not simply stronger.
11. If the sensitivity audit shows Task-009 could only detect very large reranking, weaken the paper wording accordingly.
12. If any analysis contradicts the current narrative, report it prominently.
13. Do not launch any new benchmark or model calls.

---

# READ FIRST

Read:

- current `paperDirection.txt`
- Task 007 report + analysis outputs
- Task 008 report + analysis scripts
- Task 009:
  - `report.md`
  - `preregistration.md`
  - `freeze_manifest.json`
  - `stage3_results.csv`
  - `repeat_results.csv`
  - `shared_ranking_model_results.csv`
  - `condition_sensitive_model_results.csv`
  - `condition_auc.csv`
  - `risk_coverage.csv`
  - `matched_budget_routing.csv`
  - exact Task-009 analysis scripts
- Task 009 deviations log

Do not trust prose summaries when exact values can be read from the underlying CSVs/scripts.

---

# PART 0 — CLAIM WORDING MUST BE PREDECLARED BEFORE THE SIMULATION

Before running the positive-control simulation, create:

`claim_wording_preregistered.md`

Write exactly three conditional paper-ready wording templates.

The templates must be frozen before inspecting simulation results.

Use these logical categories:

## TIGHT DETECTION FLOOR

Use if the existing Task-009 design has >=80% power to detect reranking that would induce only a small scientifically meaningful disturbance to the ranking, as defined below.

Template should be approximately:

> "Displayed confidence strongly shifts verification coverage, while any score-dependent reranking large enough to exceed [DETECTION FLOOR] is inconsistent with the observed data under our sensitivity model."

## MODERATE DETECTION FLOOR

Use if the design reliably detects moderate reranking but not small reranking.

Template approximately:

> "Displayed confidence strongly shifts verification coverage. We find no evidence for score-dependent reranking of moderate or larger magnitude; smaller changes remain below the study's detection floor."

## WEAK DETECTION FLOOR

Use if the design only reliably detects large reranking.

Template approximately:

> "Displayed confidence strongly shifts verification coverage, while our data do not support large score-dependent reranking. The study is not sufficiently sensitive to rule out smaller ranking changes."

Do not change these templates after the simulation except to insert the measured numerical floor / human-readable effect size.

---

# PART 1 — LOAD-BEARING POSITIVE CONTROL / SENSITIVITY ANALYSIS

## Goal

Task 009 found approximately zero held-out gain from the richer condition-sensitive model.

We must answer:

> Would the exact Task-009 analysis pipeline have detected scientifically meaningful reranking if it had actually been present?

A null is only persuasive if the analysis had adequate sensitivity.

## 1A. Reconstruct the exact Task-009 pipeline

Use the same:

- sample size;
- score conditions;
- question grouping;
- cross-validation;
- regularization;
- shared-ranking model;
- condition-sensitive alternative;
- decision metric;
- 0.01 held-out log-loss equivalence boundary;
- bootstrap/resampling logic.

Do not create a more powerful detector for the simulation than was used on the real data.

The positive control must evaluate the actual inferential pipeline supporting the paper.

## 1B. Simulation base

For GPT and Claude separately:

1. fit the Task-009 shared-ranking data-generating model;
2. use its fitted condition intercepts and item structure as the baseline;
3. inject known score-dependent reranking;
4. generate synthetic Stage-3 actions with the same number of items/conditions;
5. rerun the exact Task-009 shared-vs-condition-sensitive comparison.

Include a true-zero injection condition to check calibration / false-positive behavior.

Use enough Monte Carlo replications for stable power estimates:
- target >=1000 simulations per key magnitude if computationally reasonable;
- minimum 500 if runtime is substantial.

Record simulation seed(s).

## 1C. Injection family 1 — structured score × difficulty reranking

Inject a condition-dependent interaction between displayed score and the existing evaluation-side difficulty variable.

The injection grid must include:

- 0
- 0.25× development anchor
- 0.5× development anchor
- 1.0× development anchor
- 1.5× development anchor
- 2.0× development anchor

The **development anchor** must be extracted from the Task-007/008 development analysis, not invented.

Prefer an interpretable anchor such as:
- the score×difficulty slope / slope change previously estimated in Task 007/008;
- if multiple plausible development estimates exist, document the choice and include the main alternative as a secondary anchor.

Also create a **between-model anchor** if it can be defined cleanly:
- e.g. the magnitude of the GPT-vs-Claude difference in difficulty slope / prioritization observed in development data.

Do not force a between-model anchor if there is no scientifically coherent common scale.

For every injection magnitude, report:
- power to reject/pass beyond the Task-009 equivalence boundary;
- mean and distribution of richer-model held-out log-loss gain;
- induced pairwise rank-reversal rate;
- induced latent rank correlation across score conditions;
- any easier-to-interpret change in error-catch/ranking quality.

## 1D. Injection family 2 — generic condition-specific item reranking

Inject score-dependent item deviations that do NOT rely on the difficulty proxy.

Use a simple, prespecified family such as:

`u_item,condition = u_item + delta_condition,item`

with condition-specific perturbations scaled to produce controlled ranking disruption.

Calibrate magnitudes in terms of human-readable ranking changes, not arbitrary coefficient units.

Target a grid approximately corresponding to:
- near-zero disruption
- latent rank correlation ~0.95
- ~0.90
- ~0.80
- ~0.70
- ~0.60

or, if more stable:
- pairwise ordering reversal rates around 2%, 5%, 10%, 20%, 30%.

Document the exact construction.

For each magnitude, report the same power/recovery outputs as 1C.

## 1E. Detection floor

For each model and injection family, estimate the smallest reranking magnitude for which the Task-009 pipeline detects the departure with >=80% probability.

Also report 90% power if easy.

The detection floor must be expressed in at least TWO forms:

1. model/statistical scale:
   - interaction coefficient / perturbation SD / held-out loss change;

2. human-readable ranking scale:
   - latent rank correlation;
   - pairwise reversal rate;
   - or another intuitive ordering-change metric.

The final report must state plainly:

> "The Task-009 analysis could reliably detect reranking of approximately X magnitude or larger under this simulated alternative family."

If the floor is weak, say so.

## 1F. False-positive calibration

At zero injected reranking, report how often the analysis incorrectly indicates material reranking / fails the invariance criterion.

If the false-positive behavior is poor, flag it.

---

# PART 2 — CAN THE SHARED-RANKING MODEL EXPLAIN THE H4 AUROC DRIFT?

## Motivation

Task 009 binary-action AUROCs changed across score conditions:

GPT decreased as coverage became very low.
Claude increased as coverage moved away from near-saturation.

For a binary decision variable:

`AUROC = 0.5 + (TPR - FPR)/2`

Therefore binary-action AUROC can change mechanically as a threshold moves even if the underlying ranking is fixed.

Do not merely assert this. Test it quantitatively.

## 2A. Predicted AUROC under shared ranking

Using the fitted Task-009 shared-ranking model:

For each primary model and visible score condition:
- compute predicted VERIFY probabilities;
- threshold/sample according to the observed condition operating point as appropriate;
- calculate the binary-action AUROC expected under the shared-ranking model;
- use posterior/simulation/bootstrap uncertainty.

Create:

`observed_vs_shared_predicted_auc.csv`

with:
- model
- score
- observed coverage
- observed binary AUROC
- predicted shared-ranking AUROC
- prediction interval
- residual observed - predicted

## 2B. Overlay figure

Create a publication-quality figure:

`figures/observed_vs_shared_predicted_auc.png`

Show observed and shared-model-predicted AUROC across score conditions for GPT and Claude.

Key question:

> Are the opposite-direction GPT/Claude AUROC trends quantitatively compatible with moving along different portions of one shared latent ranking?

## 2C. Residual assessment

Report:
- RMSE / MAE of predicted vs observed AUROC;
- bootstrap intervals;
- whether any condition materially exceeds shared-model predictions.

If the shared model explains the drift well:
state that the apparent AUROC movement is expected from operating-point movement of a binary action.

If not:
do NOT hand-wave it away; treat it as evidence of residual condition-dependent ranking behavior.

---

# PART 3 — LATENT RANK STABILITY WITH THREE BERNOULLI REPEATS

## Motivation

Raw repeat-based Spearman correlations are attenuated because each item's empirical propensity is estimated from only 3 Bernoulli decisions and therefore takes only {0, 1/3, 2/3, 1}.

Do NOT use a naive Spearman-Brown correction as the primary analysis.

## 3A. Hierarchical/binomial latent-propensity model

For GPT and Claude separately, fit a model to the repeat subset that estimates latent item verification propensities for each visible-score condition while accounting for binomial sampling noise.

A reasonable family:
- hierarchical logistic/binomial model;
- item-level latent effects;
- condition effects;
- correlated item deviations across conditions if needed.

The goal is to estimate:

> correlation between latent item verification propensities across score conditions.

Do not overparameterize beyond what N=100 × 3 repeats can support.

Use regularization / partial pooling.

## 3B. Adjacent-condition latent correlations

Estimate latent rank/propensity association for:

- 0.70 vs 0.85
- 0.85 vs 0.90
- 0.90 vs 0.95
- 0.95 vs 0.99

Report:
- posterior/bootstrapped latent correlation estimate;
- interval;
- raw empirical Spearman next to it;
- expected raw 3-repeat Spearman implied by the fitted latent model.

If direct latent Spearman is difficult, report Pearson correlation on the latent logit-propensity scale plus simulation-derived rank correlation.

## 3C. Posterior predictive attenuation check

Simulate 3 Bernoulli generations/item/condition from the fitted latent model.

Ask:

> If the latent relationship were as estimated, what raw Spearman distribution would we expect with only 3 draws?

This shows how much of raw 0.477 etc. can plausibly be measurement noise.

## 3D. Claim consequence

Return one of:

- `LATENT_STABILITY_STRONG`
- `LATENT_STABILITY_MODERATE`
- `LATENT_STABILITY_WEAK_OR_MIXED`
- `INSUFFICIENT_REPEAT_INFORMATION`

Do not choose the label based on desired narrative.

---

# PART 4 — TARGETED POST-HOC MISMATCH-RERANKING ROBUSTNESS

## Status

This section is explicitly:

`POST_HOC_ROBUSTNESS — NOT PREREGISTERED TASK-009 CONFIRMATION`

Its purpose is to test a plausible specific rival that a reviewer could raise.

Do NOT merge these results into the preregistered H2 result.

## Rival mechanism

Perhaps the model detects implausibility when:

- displayed confidence differs substantially from its own fresh q1;
- displayed confidence is very high on an empirically difficult item.

That could create localized score-dependent reranking even if a generic interaction test is null.

## 4A. Prespecified targeted predictors

Test ONLY these targeted variables unless a coding necessity requires equivalent parameterization:

1. signed confidence mismatch:
   `displayed_q - q1`

2. absolute confidence mismatch:
   `abs(displayed_q - q1)`

3. logit mismatch if numerically stable:
   `abs(logit(displayed_q) - logit(q1_clipped))`

4. displayed score × leave-one-target-out difficulty

Do not test dozens of transformations.

If signed and absolute mismatch are collinear/redundant, fit them in separate predeclared models.

## 4B. Held-out comparison

Ask whether adding each targeted mechanism to the shared-ranking model materially improves question-grouped held-out log loss.

Report:
- Δ held-out log loss
- question-bootstrap CI
- coefficient/effect direction
- whether improvement reaches the same 0.01 materiality scale used in Task 009

## 4C. Interpretation

A null result:
- strengthens robustness against this particular alternative;
- does NOT become "confirmatory" evidence.

A positive result:
- may identify a specific limited form of reranking;
- must be reported honestly;
- may require narrowing the final ranking-invariance wording.

---

# PART 5 — MATCHED-BUDGET ROUTING ANALYSIS

## Goal

Finish the preplanned practical consequence analysis on Task-009 data.

Question:

> At the SAME verification budget, what fraction of actual target-model errors does each routing signal catch?

This determines whether the routing result deserves a main figure or only a paragraph.

## 5A. Baselines

Use only the predeclared/simple baselines:

1. random verification
2. raw q1
3. cross-fitted calibrated q1, if already specified/implemented in Task 009
4. one hidden qualitative verification judgment
5. q1 + one hidden judgment
6. hindsight oracle

Evaluation-only optional oracle:
- leave-one-target-out cross-model difficulty

Do not add many new router variants.

## 5B. Budgets

Evaluate common budgets where support permits:

- 10%
- 20%
- 30%
- 40%
- 50%

Use fractional inclusion / random tie-breaking at cutoffs in the same principled way used historically.

## 5C. Metrics

Primary:
- fraction of actual errors caught

Secondary:
- precision among verified
- residual error rate among unverified

Use question bootstrap.

## 5D. Figure decision

Create:
`figures/matched_budget_routing.png`

Then classify:

- `MATERIAL_ROUTING_GAIN`
- `MODEST_ROUTING_GAIN`
- `NO_MATERIAL_ROUTING_GAIN`

Predefine a descriptive scale before inspecting final tables:
- material: >=10 percentage-point error-catch gain over the strongest realistic baseline at 20–40% budget, replicated in both GPT and Claude
- modest: 3–10pp or inconsistent across models
- no material: <3pp / CIs broadly overlap

This scale is descriptive, not a preregistered hypothesis.

Do not search for a winner.

---

# PART 6 — INTEGRATED CLAIM AUDIT

Create:

`final_claim_audit.md`

Answer these questions directly.

## A. Positive-control sensitivity

- What magnitude of reranking could Task 009 reliably detect?
- Express in intuitive ranking terms.
- Which prewritten equivalence wording template is now justified?

## B. AUROC drift

- Is GPT's decreasing and Claude's increasing binary-action AUROC quantitatively predicted by the shared-ranking model?
- Or is there unexplained residual condition dependence?

## C. Latent rank stability

- After accounting for 3-draw binomial noise, how stable are latent item propensities?
- Does the raw 0.477 GPT correlation remain concerning?

## D. Targeted mismatch rivals

- Do q1 mismatch or score×difficulty materially improve held-out prediction?
- Label all results post-hoc.

## E. Routing consequence

- Is there a meaningful routing gain at matched budget?
- Main figure or paragraph?

## F. Final recommended wording

Give exactly:

1. one maximally strong but defensible headline sentence;
2. one abstract-level sentence;
3. one conservative reviewer-proof sentence;
4. a list of forbidden stronger phrasings.

---

# PART 7 — REQUIRED OUTPUTS

Create:

`to_gpt/010_final_sensitivity_equivalence_audit/`

with:

- `report.md`
- `validation.md`
- `claim_wording_preregistered.md`
- `positive_control_design.md`
- `positive_control_results.csv`
- `positive_control_power_curve.csv`
- `positive_control_detection_floor.md`
- `observed_vs_shared_predicted_auc.csv`
- `latent_rank_stability.csv`
- `latent_rank_posterior_predictive.csv`
- `mismatch_robustness.csv`
- `matched_budget_routing.csv`
- `final_claim_audit.md`
- `changed_files.txt`
- exact analysis scripts under `analysis/task010/`
- figures/
  - `positive_control_power.png`
  - `observed_vs_shared_predicted_auc.png`
  - `latent_rank_stability.png`
  - `matched_budget_routing.png`

---

# PART 8 — ROOT REPORT FORMAT

`report.md` must contain:

1. Plain-English bottom line
2. Validation / zero-call confirmation
3. Positive-control simulation design
4. Detection floor and power
5. Which predeclared equivalence wording is justified
6. Shared-ranking prediction of AUROC drift
7. Hierarchical latent rank-stability result
8. Targeted mismatch robustness
9. Matched-budget routing
10. Any evidence against the current narrative
11. Final strongest defensible claim
12. Exact wording that should be removed from `paperDirection.txt`, if any
13. Whether MMLU analysis should now stop
14. `READY_FOR_GPT_REVIEW = YES`

---

# FINAL STOP RULE

After Task 010:

- do NOT run more MMLU-Pro analyses unless Task 010 discovers a concrete statistical error;
- do NOT invent new post-hoc interactions;
- do NOT start another score grid;
- do NOT modify the central empirical story merely to improve aesthetics.

Task 010 exists to bound and stress-test the claim.

After it, the project moves to:
1. literature/methods positioning;
2. one final task-generalization experiment chosen once;
3. paper writing.
