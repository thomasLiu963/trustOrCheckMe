# TASK 008 — Zero-Cost Ranking-Invariance Audit + Prospective Protocol Decision Packet

## Purpose

Claude's adversarial review raised a potentially stronger final-paper hypothesis:

> The displayed confidence scalar may act primarily as a coverage / verification-rate control knob, while leaving the model's item ordering / discrimination largely invariant.

This could be a cleaner final story than the current operating-point/prioritization framing IF it is actually supported.

Do not accept this conclusion from four hand-calculated ROC points. Task 008 must rigorously test the hypothesis on all existing exploratory data before we freeze a prospective protocol.

This task makes **0 API calls**.

Save this exact instruction to:
`from_gpt/008_ranking_invariance_audit.md`

Write outputs to:
`to_gpt/008_ranking_invariance_audit/`

## Non-negotiable rules

1. 0 API calls.
2. Do not alter any historical or experimental database.
3. Do not modify `paperDirection.txt`.
4. Read-only analysis of Tasks 003–007.
5. Do not infer hidden-state mechanisms.
6. Do not call empirical cross-model difficulty a production feature.
7. Do not declare invariance merely because VERIFY sets are nested.
8. Do not declare reshaping merely because probability-scale hard/easy gaps change.
9. All uncertainty/resampling uses question ID as the unit.
10. Explicitly distinguish coverage change, ranking/discrimination change, difficulty-slope change, and saturation/link-function artifacts.
11. The purpose is to choose the FINAL prospective hypothesis, not defend the current narrative.

## Read first

Read:
- current `paperDirection.txt` read-only
- Task 007 report
- Task 007 Lane C outputs/scripts
- Task 006 merged qualitative data
- Task 005B / 005C reports
- Task 004 stability report
- Task 003 Study-1 report
- Task 007 `paper_direction_decision_packet.md`

Primary audit uses the full N=100 GPT/Claude qualitative grid:
- 2 stakes families
- hidden + displayed 0.70 / 0.90 / 0.99
- one binary action per item/condition

Gemini/Grok N=20 are descriptive only.

# Competing hypotheses

## H0 — Pure coverage / threshold shift

Displayed confidence mainly shifts a condition-level threshold/intercept while approximately preserving one latent item ordering.

Predictions:
- VERIFY rate changes strongly with displayed score;
- VERIFY sets are approximately nested as score rises;
- action-vs-correctness discrimination remains similar;
- difficulty slope is approximately stable on the logit scale;
- condition intercepts + shared item propensity explain the data well;
- condition-sensitive reordering adds little out-of-sample value.

## H1 — Prioritization reshaping

Displayed confidence changes not only the verification rate but which items are prioritized.

Predictions may include:
- substantial non-nesting/reversals;
- material discrimination changes;
- material score × difficulty interaction on the latent scale;
- condition-sensitive item effects improve predictive fit;
- ROC/risk-coverage points are incompatible with one stable ranking.

Neither hypothesis is assumed true.

# Lane A — ROC / risk-coverage compatibility

For every model × stakes × condition report:
- VERIFY rate
- FPR = P(VERIFY | target answer correct)
- TPR = P(VERIFY | target answer wrong)
- errors caught / total errors

Create ROC-space and risk/coverage plots.

For each ordered visible-score pair (0.70→0.90, 0.90→0.99, 0.70→0.99), count:
- V→V
- V→U
- U→U
- U→V

Report reversal/non-nesting rate and question-bootstrap CI.

Nesting supports but does NOT prove invariance.

Fit a shared-ranking model such as:

`logit P(VERIFY_ij) = alpha_condition_j + u_item_i`

where:
- alpha_condition = operating-point shift
- u_item = shared item verification propensity

Use an appropriate hierarchical/Rasch-style/random-item-intercept formulation.

Compare against a richer condition-sensitive alternative that permits prioritization to change, using out-of-sample or penalized comparison rather than in-sample fit alone.

Report whether the richer model gives a MATERIAL improvement.

# Lane B — Latent-scale difficulty interaction

Use leakage-safe leave-one-target-out empirical difficulty:

GPT: Claude/Gemini/Grok correctness
Claude: GPT/Gemini/Grok correctness

Primary: other-models-correct count 0–3.
Secondary covariates: category, question length, choice count.

For visible conditions fit, separately by model and stakes:

`logit P(VERIFY) = alpha_score + beta*difficulty + gamma(score × difficulty)`

Account for repeated observations by question with GEE/mixed model/cluster-robust approach plus question bootstrap.

Report:
- beta at reference score
- interaction coefficients
- implied difficulty slopes at 0.70 / 0.90 / 0.99
- 95% question-bootstrap intervals
- no-interaction vs interaction held-out/CV log loss

Primary question:
Does the score × difficulty interaction materially differ from zero on the LOGIT scale?

Do not call tiny effects "reshaping."

# Lane C — Discrimination / ranking-quality invariance

For each model × stakes × visible score:
- action-vs-wrongness AUROC
- bootstrap CI

State clearly that AUROC from a single binary action is limited.

For adjacent score pairs bootstrap:
- ΔAUROC
- discrimination differences

Do not compare raw errors caught across very different coverage and call that ranking degradation.

Using all visible conditions, test whether observed (FPR,TPR) points are materially inconsistent with a fitted monotone common ROC/risk-coverage relationship beyond bootstrap uncertainty.

If too few points make this formal test weak, state that.

# Lane D — Cross-condition latent item prediction

Because qualitative Task 007 has only one generation per item/condition, do not pretend to know exact per-condition propensities.

Instead:

1. Estimate shared latent item propensity from some conditions.
2. Test how well it predicts actions in a held-out score condition.
3. Compare:
   - condition intercept only
   - difficulty only
   - shared item propensity + condition intercept
   - richer condition-sensitive model

Perform leave-one-condition-out prediction where feasible.

Key question:
Does a single item propensity learned under some score conditions predict which items will be verified under another score condition?

Report out-of-condition log loss / ranking metrics and whether richer condition-sensitive models materially improve.

If not stable, return `INSUFFICIENT_FOR_LATENT_RANK_TEST`.

# Lane E — Reinterpret original workshop result

Using historical numeric Study 1 / Checkpoint A data, test:

> Is the visible-confidence error-catching degradation primarily explained by moving to a different coverage / operating point rather than by deterioration in ranking quality?

Distinguish:
- raw observed-coverage differences
- matched-budget differences
- calibration effects
- ranking effects

Write a concise self-correction paragraph suitable for a future paper if supported.

Do not force it.

# Lane F — Prospective design prep

0 calls. Do NOT freeze yet.

Prepare design matrices under both possible final worlds:

## World I — invariance
Displayed confidence strongly changes coverage while ranking/discrimination stays approximately invariant.

## World R — reshaping
Displayed confidence changes both coverage and prioritization.

Prepare exact call counts for:

Primary models:
- GPT
- Claude

Secondary:
- Gemini
- Grok on smaller subset

Fresh N:
- 300
- 500
- 700

Candidate primary conditions:

P1:
- hidden
- true-q visible
- 0.70
- 0.85
- 0.95
- 0.99

P2:
- hidden
- true-q visible
- 0.70
- 0.85
- 0.90
- 0.95
- 0.99

Use ONE stakes family in the main prospective study unless Task 008 finds a compelling reason otherwise.

Prepare repeat-subset options:
- 100 items × 3 total generations/cell
- 150 items × 3 total generations/cell

Fresh Stage 1 + q1 are mandatory.
q2 is not mandatory unless Task 008 finds a reason to retain it.

# Lane G — Final narrative decision packet

Create:
`to_gpt/008_ranking_invariance_audit/final_narrative_decision_packet.md`

Ignore "how we got here." Focus on the strongest FINAL paper.

Provide three worlds:

## WORLD 1 — Ranking invariance
If score changes coverage but ranking/discrimination is essentially invariant.

Give:
- strongest honest narrative
- exact novelty claim
- max 4 contribution bullets
- Figures 1–4
- practical implication
- closest prior-work distinction
- prospective result needed to make it bulletproof
- likely second-task requirement

## WORLD 2 — Ranking reshaping
If score materially changes prioritization at non-saturated operating points.

Same fields.

## WORLD 3 — Mixed/model-specific
If GPT/Claude do not share a simple law.

Same fields, without automatically treating heterogeneity as a contribution.

State which world existing evidence favors:
- STRONGLY_FAVORS
- MODERATELY_FAVORS
- WEAKLY_FAVORS
- AMBIGUOUS

# Decision bucket

Assign exactly one:

- `INVARIANCE_SUPPORTED_EXPLORATORY`
- `RESHAPING_SUPPORTED_EXPLORATORY`
- `MIXED_OR_MODEL_SPECIFIC`
- `AMBIGUOUS_NEEDS_REPEATS`

Do not choose based on publishability.

# Required outputs

Create:
`to_gpt/008_ranking_invariance_audit/`

with:
- `report.md`
- `validation.md`
- `roc_condition_points.csv`
- `risk_coverage_condition_points.csv`
- `nesting_reversals.csv`
- `shared_ranking_model_results.csv`
- `condition_sensitive_model_results.csv`
- `difficulty_logit_interactions.csv`
- `condition_auc.csv`
- `common_curve_analysis.csv`
- `latent_item_prediction.csv`
- `workshop_reinterpretation.md`
- `prospective_design_matrix.csv`
- `prospective_power_notes.md`
- `final_narrative_decision_packet.md`
- figures/
- exact analysis scripts
- `changed_files.txt`

# Root report must include

1. Plain-English bottom line
2. Validation
3. ROC/risk-coverage geometry
4. Nesting/reversal results
5. Shared-ranking vs condition-sensitive model
6. Logit-scale score×difficulty interaction
7. Condition-specific discrimination quality
8. Cross-condition latent-item prediction
9. Whether Claude "sharpening" survives latent-scale correction
10. Reinterpretation of original workshop error-catching result
11. Current decision bucket
12. Final-paper narrative favored by current evidence
13. What prospective study must prove/falsify
14. Recommended protocol shape, but DO NOT execute
15. `READY_FOR_GPT_REVIEW = YES`

# Scientific cautions

- Four ROC points that look concave are not proof of an invariant ranking.
- Nested binary action sets are evidence for thresholding, not a full test.
- Probability-scale hard/easy gaps can change under a constant logit slope.
- A null interaction is not automatically proof of invariance; report precision.
- A positive interaction at saturated cells may be a boundary artifact.
- Rank invariance is most compelling if:
  1. coverage moves strongly,
  2. reversals stay low,
  3. a shared latent item score predicts held-out conditions,
  4. condition-sensitive models add little,
  5. discrimination stays similar,
  6. prospective repeated generations later confirm rank stability.

# Final project vision

We do not care about preserving the current exploratory narrative. We care about the strongest FINAL paper that survives adversarial scrutiny.

If ranking invariance is real, the final paper can be:

> Counterfactual confidence feedback is a powerful control over how much an LLM verifies, but not over which answers it prioritizes; it moves self-verification along a largely fixed risk-coverage curve.

If ranking reshaping is real, that may be even more interesting:

> Counterfactual confidence feedback changes not only the amount of oversight but the model's prioritization of which outputs deserve scrutiny.

Task 008's job is to determine which final-paper hypothesis deserves prospective confirmation.
