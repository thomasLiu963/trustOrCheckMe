# TASK 005C — Hidden-Action Residual After Difficulty Control

## Purpose

Task 005 found that historical hidden verification behavior still predicts correctness after two independently elicited confidence reports (`q1` and `q2`).

Task 005B then showed that Claude's *fixed-score manipulated* VERIFY/USE signal is largely explained by observable item difficulty estimated from the other models.

The immediate question is therefore:

> Does the historical hidden verification signal still add correctness information after controlling for BOTH repeated confidence (`q1`, `q2`) AND observable item difficulty?

This is a zero-cost exploratory analysis. It is critical because it tells us whether the current q2 result reflects:
- something about the verification judgment beyond confidence and difficulty,
- or mostly a difficulty-sensitive routing heuristic.

Make **0 API calls**.

Save this exact task as:

`from_gpt/005c_hidden_residual_after_difficulty.md`

Write outputs to:

`to_gpt/005c_hidden_residual_after_difficulty/`

Do not modify:
- `paperDirection.txt`
- historical V2 data
- Study 1 data
- Task 005 q2 data
- Task 005B data
- Task 006 outputs/data if Task 006 is running concurrently

---

# Models

Analyze both:
- GPT
- Claude

Use all 500 canonical historical V2-B questions.

---

# Core variables

For each target model/question:

## Confidence features
- `q1`: historical Stage-2 verbal confidence
- `q2`: independent Task-005 re-elicited confidence

## Hidden verification feature
Construct the same question-level historical hidden-action summary used in Task 005:
- fraction of historical HIDDEN Stage-3 cells choosing VERIFY across the pre-existing L/authority cells

Also, where feasible, include a **single canonical hidden-action** sensitivity analysis so the multi-elicitation aggregate is not mistaken for a one-call deployment signal.

## Leakage-safe observable difficulty

For target GPT:
- difficulty = correctness of the OTHER three historical models:
  - Claude
  - Gemini
  - Grok

For target Claude:
- difficulty = correctness of:
  - GPT
  - Gemini
  - Grok

Primary difficulty variable:
- count/fraction of the other 3 models that answered correctly

Also include observable pre-existing item metadata where available:
- MMLU-Pro category
- question/stem length
- choice count
- other deterministic item metadata already used in Task 005B if clearly pre-outcome

Do NOT use the target model's own correctness inside its difficulty predictor.

Do NOT use any Study-1 manipulated action as a difficulty feature.

---

# Validation

Before analysis confirm:
- exactly 500 q1 rows/model
- exactly 500 q2 rows/model
- exactly 500 correctness labels/model
- other-model difficulty available for all/most questions
- no target-model correctness leaks into its difficulty feature
- historical hidden summary matches Task 005 construction
- question ID is the unit

If joins fail materially, STOP and report.

---

# Primary grouped-CV analysis

Use 5-fold cross-validation grouped by question ID.

Use regularized logistic regression or another simple predeclared classifier already available in the repo.

For each model compare these nested feature sets:

1. `q1`
2. `q1 + q2`
3. `q1 + q2 + difficulty`
4. `q1 + q2 + difficulty + hidden_action_summary`

Optional secondary:
5. `difficulty` only
6. `q1 + q2 + hidden_action_summary` (to replicate Task 005)
7. `q1 + q2 + difficulty + single_hidden_action`

All preprocessing / regularization / calibration must be fit only on training folds.

Report:
- AUROC
- log loss
- Brier score if straightforward
- question-bootstrap 95% CIs

Most important contrasts:

### Contrast A
`q1 + q2 + hidden` vs `q1 + q2`
(replication of Task 005)

### Contrast B — PRIMARY
`q1 + q2 + difficulty + hidden`
vs
`q1 + q2 + difficulty`

This asks whether hidden verification adds information after both repeated confidence and observable difficulty.

Do not rely only on coefficients/p-values.

---

# Matched-budget routing analysis

For each model, using properly out-of-fold scores, compare error-catching at verification budgets:

- 10%
- 20%
- 30%
- 40%
- 50%

Routers:

1. q1
2. q1+q2 predictor
3. q1+q2+difficulty predictor
4. q1+q2+difficulty+hidden predictor

Primary practical question:

> At the same number of checks, does adding hidden verification behavior still catch more actual errors after confidence and difficulty are already available?

Report:
- fraction of actual errors caught
- raw error counts
- pairwise difference vs `q1+q2+difficulty`
- question-bootstrap 95% CIs

Treat the hidden aggregate as a multi-elicitation diagnostic / upper bound, not automatically a fair one-call production router.

If feasible, repeat with the single canonical hidden action.

---

# Difficulty interpretation

Also report descriptively:

- how strongly other-model difficulty predicts target-model correctness
- how strongly it predicts hidden VERIFY propensity
- correlation/association between hidden action and difficulty

This will tell us whether hidden verification is basically a difficulty detector.

---

# Interpretation buckets

Assign one per model:

## HIDDEN_ADDS_BEYOND_CONFIDENCE_AND_DIFFICULTY
Adding hidden verification after q1+q2+difficulty gives a material, reproducible improvement in predictive performance and/or matched-budget error catching.

## HIDDEN_PARTIALLY_ADDS
A positive residual remains, but substantially smaller than before difficulty adjustment or uncertain across metrics.

## DIFFICULTY_LARGELY_EXPLAINS_HIDDEN_RESIDUAL
Hidden verification adds little/no useful signal once q1+q2+difficulty are included.

## AMBIGUOUS
Data/estimation uncertainty prevents a useful conclusion.

Important:
- None of these establishes an internal hidden-state mechanism.
- If difficulty explains the signal, that is still scientifically useful: the verification decision may be acting as a cheap item-difficulty estimator.
- If hidden remains useful beyond difficulty, that strengthens the deeper D1 story.

---

# Required outputs

Create:

`to_gpt/005c_hidden_residual_after_difficulty/`

with:

- `report.md`
- `validation.md`
- `question_level_features.csv`
- `grouped_cv_nested_models.csv`
- `bootstrap_contrasts.csv`
- `matched_budget_routing.csv`
- `difficulty_associations.csv`
- `single_hidden_sensitivity.csv` if feasible
- `changed_files.txt`
- exact analysis scripts

`report.md` must contain:

1. Plain-English bottom line
2. Validation / leakage audit
3. Replication of Task-005 q2 result
4. Strength of observable difficulty
5. PRIMARY: does hidden action add after q1+q2+difficulty?
6. Matched-budget routing after difficulty control
7. GPT interpretation bucket
8. Claude interpretation bucket
9. What this DOES establish
10. What it DOES NOT establish
11. Recommendation only — do not launch paid experiments
12. `READY_FOR_GPT_REVIEW = YES`

Make 0 API calls.
