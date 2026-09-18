# TASK 005 — Parallel Diagnostic Sprint After Study 1

## Purpose

Study 1 is complete as an exploratory causal result. The next goal is **not** to scale blindly or write the paper. The goal is to resolve the three highest-value uncertainties that determine what the D1 paper actually is:

1. Does any reproducible item-specific verification behavior remain useful for identifying errors when displayed score is held fixed?
2. Is the extra correctness information in the historical hidden verification action mostly explained by a second independent confidence assessment?
3. Is GPT's near-deterministic score threshold specific to the current low/no-reasoning inference mode?

In parallel, prepare—but **do not run**—the qualitative-stakes experiment needed to address the "you handed GPT the arithmetic" objection.

This task is intentionally structured as parallel lanes to save wall-clock time while preserving scientific discipline.

---

# NON-NEGOTIABLE RULES

1. **Do not modify `paperDirection.txt`.**
2. **Do not modify historical V2 data.**
3. Do not overwrite Task 003 or Task 004 data.
4. Do not run a fresh confirmatory study.
5. Do not add Grok, Gemini, open-weight models, new tasks, provenance controls, interpretability, reliability warnings, or C-variation in this task.
6. Do not run the qualitative-stakes experiment yet. Only prepare it offline for GPT review.
7. Do not call any current exploratory result "confirmatory."
8. Do not infer hidden-state information loss/suppression from behavioral saturation.
9. Do not call GPT irrational for following the supplied probability/cost rule.
10. Treat question identity—not repeated conditions or repeated generations—as the fundamental statistical unit whenever correctness is involved.
11. Save this exact task instruction to:
   `from_gpt/005_parallel_diagnostic_sprint.md`
12. Append meaningful implementation/scientific decisions to `docs/D1_DECISION_LOG.md`, but do not rewrite historical entries.

---

# PARALLELIZATION PLAN

Run four logically separate lanes.

- **Lane A — zero-cost existing-data analysis:** 0 API calls.
- **Lane B — independent second-confidence (`q2`) re-elicitation:** 1,000 scientific calls.
- **Lane C — GPT reasoning-mode mini-grid:** up to 800 scientific calls, only if a supported nonzero reasoning setting is already available in the current adapter/provider implementation.
- **Lane D — qualitative-stakes experiment PREPARATION ONLY:** 0 API calls.

Lanes A and D can run immediately in parallel.

Lanes B and C may run concurrently **only if** provider-specific rate limiting and the repository architecture make this safe. They MUST use separate result directories, separate databases/manifests, and separate scientific call caps. Do not have multiple processes write to the same SQLite database.

If concurrent GPT traffic would create rate-limit noise, prefer:
- run Claude q2 while GPT reasoning runs;
- then run GPT q2;
rather than forcing maximum concurrency.

Scientific cleanliness is more important than shaving a few minutes.

---

# INPUTS TO READ FIRST

Read at minimum:

- `paperDirection.txt` — READ ONLY; do not edit.
- `docs/D1_MASTER_RESEARCH_PLAN.md`
- `docs/D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md`
- `docs/D1_DECISION_LOG.md`
- `to_gpt/003_study1_primary_results/report.md`
- `to_gpt/004_study1_stability_results/report.md`
- Study 1 prompt audit / call-plan artifacts
- Checkpoint A artifacts and analysis code
- historical V2 configuration/data
- current model adapter/config code

Before any paid call, print a short preflight summary showing:
- exact historical V2 dataset path;
- exact Study-1 database path;
- exact model IDs/endpoints;
- exact scientific call cap for Lane B;
- whether Lane C is supported and, if so, the exact reasoning parameter/value;
- confirmation that `paperDirection.txt` will not be changed.

---

# LANE A — ZERO-COST EXISTING-DATA ANALYSIS

## Goal

Separate three concepts without overclaiming:

### A1. Score dominance
How strongly does changing the displayed score move the final VERIFY/USE action?

### A2. Item sensitivity at fixed score
When every question is shown the same displayed score, does the model still treat different questions differently in a reproducible way?

### A3. Useful item sensitivity
Where fixed-score action variation exists, does it preferentially send actually-wrong answers to verification?

This is a PILOT SIGNAL CHECK on the existing 100-question Study-1 sample, not a publication-grade evidence-retention estimate.

## A.1 Data validation

Confirm:
- exactly 100 Study-1 primary question IDs;
- GPT and Claude primary cells complete;
- displayed-score values match the frozen Study-1 design;
- correctness joins cleanly to the frozen Stage-1 answer;
- the 20-question repeated subset has the expected three observations per repeated cell;
- no historical V2 mutation.

Report any mismatch and STOP Lane A if correctness or question-ID joins are not reliable.

## A.2 Score-dominance summaries

Reproduce the transparent paired effects for GPT and Claude, separately for L=10 and L=20:
- verification rate by manipulated displayed score;
- near-threshold contrast;
- extreme low-vs-high contrast;
- paired VERIFY→USE, USE→VERIFY, same-action counts;
- question-bootstrap 95% CIs (5,000 resamples).

Use the existing Study-1 analysis logic where correct.

Do not invent a new "score dominance index" unless it materially simplifies the reporting. The percentage-point contrasts remain the primary interpretable quantities.

## A.3 Fixed-score correctness tables

For every `model × L × manipulated displayed score`, create the raw 2×2 table:
- wrong + VERIFY
- wrong + USE
- correct + VERIFY
- correct + USE

From each table compute:
- `P(VERIFY | wrong)` = error-catch rate;
- `P(VERIFY | correct)` = correct-check rate;
- difference `P(VERIFY | wrong) - P(VERIFY | correct)`;
- odds ratio with a sparse-cell-safe interval if feasible;
- number of wrong answers and correct answers.

The question is:

> When the displayed score is identical across questions, are wrong answers still more likely to be verified?

### Saturation rule

If a fixed-score cell is effectively pinned to one action such that useful discrimination cannot be estimated, label it explicitly:

`ACTION_SATURATED__ITEM_DISCRIMINATION_NOT_IDENTIFIABLE`

Do **not** translate a constant binary action into:
- "zero internal evidence,"
- "evidence destroyed,"
- "evidence suppressed,"
- or similar mechanistic language.

For GPT, this warning is expected to apply to many manipulated-score cells.

## A.4 Hidden-action continuity

For each model/L/manipulated displayed score, compare the manipulated action to the HIDDEN-confidence action for the same question.

Produce transition counts:

| hidden action | fixed-score VERIFY | fixed-score USE |

Also report:
- raw agreement;
- agreement separately for wrong and correct answers;
- descriptive Cohen's kappa only where both actions have adequate support.

Interpret this only as whether the same questions continue to be routed similarly. It is not proof of hidden-state continuity.

## A.5 Repeat-noise decomposition

Use the frozen 20-question Task-004 subset.

For every repeated `model × L × condition`:
- all-three agreement;
- 2–1 split;
- pairwise disagreement;
- per-question 3-generation VERIFY propensity: 0, 1/3, 2/3, 1.

Then ask whether any apparent fixed-score wrong-vs-correct routing difference seen in the original generation is still directionally present when using the 3-generation mean.

Do not treat 60 generations as 60 independent questions.

## A.6 Incremental error-prediction diagnostic

Exploratory only.

Using grouped cross-validation by QUESTION ID, compare:
- baseline: correctness predicted from displayed score;
- augmented: correctness predicted from displayed score + action;
- optionally displayed score × action interaction if stable.

All rows for one question must stay in the same fold.

Report:
- cross-validated log loss;
- AUROC;
- incremental change from adding action;
- bootstrap uncertainty by resampling questions.

If separation/sparsity makes a model unstable, say so rather than forcing a coefficient.

Conditional mutual information may be reported as a SECONDARY descriptive summary if easy, but it is not required and must not be treated as a solution to GPT's saturation problem.

## A.7 Existing-data matched-budget routing baseline

Using the historical 500-question V2 data, reproduce/extend the existing Checkpoint-A routing comparison with no new API calls.

At minimum compare, where feasible:
- raw verbal confidence ranking;
- cross-fitted calibrated confidence ranking;
- natural hidden verification judgment.

Primary intuitive metric:

> At the same verification budget, what fraction of actual wrong answers is caught?

Produce error-catch-versus-budget curves or a clean discrete approximation.

Important:
- calibration must be cross-fitted; no test-label leakage;
- tied confidence values must be handled explicitly;
- repeated L/authority cells are not independent questions;
- do not give a method credit merely for verifying more answers.

This is a baseline for later q2/combined routers, not the final routing result.

## A.8 Power / precision simulation

Use current empirical error rates and plausible pilot effect sizes to simulate candidate future sample sizes:

`N = 100, 200, 400, 600, 800, 1000`

The simulation should answer:

> How precisely could a prospective study estimate a fixed-score wrong-vs-correct verification difference or matched-budget error-catch advantage of the magnitude suggested by the pilot?

Use question as the simulated unit.

Do not declare "600–1000 is required" simply because an earlier critique proposed it.

Output the relationship between N and:
- expected number of wrong answers;
- CI width / precision;
- power for a few clearly labeled effect-size scenarios if defensible.

---

# LANE B — INDEPENDENT SECOND-CONFIDENCE (`q2`) RE-ELICITATION

## Scientific question

Checkpoint A found that the hidden verification action can predict actual correctness beyond the original verbal confidence `q1`.

The cheapest alternative explanation is:

> Stage 3 simply gives the model another look at the question; the hidden action is acting like a second noisy confidence assessment.

Test that directly.

## B.1 Sample

Use ALL 500 canonical historical V2-B questions.

Models:
- current GPT-5.6 Sol endpoint used for Study 1;
- current Claude Sonnet 5 endpoint used for Study 1.

Total scientific calls:

`500 questions × 2 models = 1,000 calls`

Do not select a favorable subset.

## B.2 Prompt

Reuse the original Stage-2 confidence-elicitation prompt as closely/exactly as possible.

For each question:
- present the same question and same frozen historical Stage-1 answer;
- ask for a numerical probability the frozen answer is correct;
- DO NOT reveal `q1`;
- DO NOT reveal correctness;
- DO NOT reveal historical hidden action;
- DO NOT reveal Study-1 manipulated scores;
- DO NOT mention the current hypothesis.

This is an independent re-elicitation of confidence from the current endpoint, not a request to revise `q1`.

Record the exact prompt bytes/template and inference settings.

## B.3 Storage

Use a NEW database/result path, e.g.:

`results/study2_q2_diagnostic/`

Do not write into Study-1 SQLite.

For every q2 row store at minimum:
- question_id;
- model;
- frozen answer;
- q1 historical reported confidence;
- q2 newly elicited confidence;
- correctness;
- raw output;
- parse status / repair attempts;
- exact endpoint;
- inference settings;
- timestamp;
- prompt hash / config hash.

## B.4 Analysis

Primary question:

> After knowing q1 and q2, does the historical hidden verification judgment still add useful information about actual correctness?

Because correctness is a question-level target, avoid pseudo-replicating the same question across many historical Stage-3 cells as if they were independent.

Create a question-level hidden-action summary, for example the fraction of historical HIDDEN cells that chose VERIFY across L/authority, while also preserving per-cell secondary analyses.

Compare grouped-cross-validated error prediction for:
1. `q1`
2. `q1 + q2`
3. `q1 + hidden_action_summary`
4. `q1 + q2 + hidden_action_summary`

Report:
- AUROC;
- log loss;
- calibration where meaningful;
- question-bootstrap uncertainty;
- matched-budget error-catch for the resulting ranking scores if feasible.

Secondary:
- q1 vs q2 correlation;
- q2 calibration/discrimination;
- whether q2 changes especially on questions the hidden policy tended to VERIFY.

Do not claim a hidden mechanism from this experiment.

## B.5 Interpretation buckets

Do NOT auto-scale after seeing the result.

### B-PASS-DEEPER-RESIDUAL
Hidden action still adds a material, reproducible correctness signal after q1+q2.

Meaning:
- "just a second noisy confidence sample" is not sufficient.

### B-PASS-SECOND-READ
q2 largely absorbs the hidden-action advantage.

Meaning:
- the hidden result is well explained as repeated assessment / ensembling;
- this is still useful because q1+q2 becomes a candidate practical router.

### B-AMBIGUOUS
Sample/measurement noise prevents a useful conclusion.

STOP after reporting. Do not add q3 or other ad hoc measurements.

---

# LANE C — GPT REASONING-MODE MINI-GRID

## Scientific question

Does a higher reasoning setting change GPT's extreme score-dominance behavior?

Do not assume the direction.

Possible outcomes include:
- reasoning weakens the score cliff because the model re-evaluates the answer;
- reasoning sharpens the score cliff because it executes the expected-cost calculation even more faithfully;
- reasoning changes hidden/true-visible behavior without changing the cliff.

All are informative.

## C.1 Precondition

Inspect the existing GPT adapter/provider implementation.

Proceed ONLY if there is already a clearly supported nonzero reasoning setting for the exact current GPT endpoint.

- Do not guess a parameter name.
- Do not cycle through undocumented settings.
- Do not switch to a different model family to manufacture a "reasoning" comparison.

If no supported setting is available:
- mark Lane C `BLOCKED_UNSUPPORTED_REASONING_SETTING`;
- make 0 paid Lane-C calls;
- report what is missing.

A tiny technical smoke test of at most 4 calls is allowed solely to verify an already-supported setting.

## C.2 Sample and conditions

Reuse the exact 100 Study-1 frozen questions and frozen answers.

GPT only.

For each question run:

### L = 10
- hidden
- true-confidence-visible
- manipulated 0.89
- manipulated 0.91

### L = 20
- hidden
- true-confidence-visible
- manipulated 0.94
- manipulated 0.96

Total:

`100 × 8 = 800 scientific calls`

Use the SAME Stage-3 wording/templates as Study 1. The intended experimental difference from the existing GPT Study-1 cells is the reasoning setting.

Use a separate database/result directory:

`results/study1_reasoning_diagnostic/`

## C.3 Analysis

Compare reasoning-mode cells against the exact existing low/no-reasoning Study-1 cells on the same questions.

Report:
- near-threshold verification rates;
- paired near-threshold action-change rate;
- hidden verification rate;
- true-visible verification rate;
- wrong-vs-correct verification rates within each condition;
- question-paired bootstrap CIs.

Do not call reasoning beneficial or harmful merely because it changes verification frequency.

The key question is how it changes:
- score responsiveness;
- item-sensitive routing;
- and error-catching.

If the reasoning endpoint is materially version-shifted relative to the Study-1 endpoint, flag this prominently.

---

# LANE D — QUALITATIVE-STAKES EXPERIMENT: PREP ONLY

## Goal

Prepare the next experiment needed to test whether GPT's spectacular Study-1 cliff is only the result of explicit numerical threshold arithmetic.

NO API CALLS IN THIS LANE.

## D.1 Draft two matched qualitative-stakes framings

Create two candidate prompt families with:
- no numerical `L`;
- no numerical `C`;
- no expected-value formula;
- no mathematically implied exact confidence cutoff;
- verification still described as useful but resource-consuming;
- an unchecked wrong answer described as costly;
- one moderate-stakes version;
- one stronger-stakes version.

Within a prompt family, manipulated-score conditions must differ ONLY in the displayed confidence number.

Suggested exploratory score grid to evaluate in the dry run:
- hidden
- 0.70
- 0.90
- 0.99

Do not run them.

## D.2 Prompt audit

For each candidate family produce:
- exact prompt templates;
- byte/line diff showing what changes between score conditions;
- explanation of possible wording confounds;
- why moderate vs stronger stakes differ;
- proposed output schema;
- proposed parser;
- estimated call plan for `100 questions × 2 models × 2 stakes × 4 conditions = 1,600 calls`.

Do not decide the final prompt silently.

The purpose is to let GPT review/freeze the prompt before any data are collected.

---

# OUTPUT STRUCTURE

Create:

`to_gpt/005_parallel_diagnostic_sprint/`

with at least:

## Root
- `report.md`
- `changed_files.txt`
- `run_manifest.json`
- `cost_summary.md`
- `parallel_execution_log.md`

## Lane A
- `lane_A_existing_data/validation.md`
- `lane_A_existing_data/score_dominance.csv`
- `lane_A_existing_data/fixed_score_correctness_contingency.csv`
- `lane_A_existing_data/useful_item_sensitivity.csv`
- `lane_A_existing_data/hidden_action_continuity.csv`
- `lane_A_existing_data/repeat_noise_decomposition.csv`
- `lane_A_existing_data/grouped_cv_incremental_error_prediction.csv`
- `lane_A_existing_data/existing_routing_baseline.csv`
- `lane_A_existing_data/power_precision_simulation.csv`
- `lane_A_existing_data/analysis_notes.md`
- `lane_A_existing_data/figures/`
- exact analysis scripts

## Lane B
- `lane_B_q2/report.md`
- `lane_B_q2/validation.md`
- `lane_B_q2/q2_results.csv`
- `lane_B_q2/q2_vs_q1.csv`
- `lane_B_q2/incremental_hidden_action_after_q2.csv`
- `lane_B_q2/matched_budget_q2_routing.csv`
- `lane_B_q2/call_manifest.csv`
- `lane_B_q2/failures.csv`
- `lane_B_q2/cost_summary.md`
- exact prompt templates and analysis scripts

## Lane C
- `lane_C_reasoning/report.md`
- `lane_C_reasoning/validation.md`
- `lane_C_reasoning/results.csv`
- `lane_C_reasoning/paired_comparison_to_study1.csv`
- `lane_C_reasoning/call_manifest.csv`
- `lane_C_reasoning/failures.csv`
- `lane_C_reasoning/cost_summary.md`
- exact config/prompt templates and analysis scripts
- OR, if blocked, `lane_C_reasoning/BLOCKED.md`

## Lane D
- `lane_D_qualitative_prep/prompt_candidate_A.md`
- `lane_D_qualitative_prep/prompt_candidate_B.md`
- `lane_D_qualitative_prep/prompt_audit.md`
- `lane_D_qualitative_prep/dry_run_call_plan.md`
- NO results database because no calls are allowed

---

# REQUIRED FINAL REPORT STRUCTURE

`to_gpt/005_parallel_diagnostic_sprint/report.md` must contain:

1. **Executive plain-English bottom line**
2. **Exact API calls made**
   - Lane A = 0
   - Lane B target = 1,000
   - Lane C target = 800 if supported
   - Lane D = 0
3. **Lane A: Does fixed-score item variation exist, and is any of it useful?**
4. **Lane A: What is structurally unidentifiable because of GPT saturation?**
5. **Lane A: Existing matched-budget routing baseline**
6. **Lane A: Power/precision implications**
7. **Lane B: Does q2 explain the historical hidden-action residual?**
8. **Lane C: Does higher reasoning change GPT score dominance or error targeting?**
9. **Lane D: Qualitative-stakes prompt candidates for GPT review**
10. **What the combined evidence DOES establish**
11. **What it still DOES NOT establish**
12. **Which deeper D1 story is currently supported**
    - deeper residual beyond q2
    - repeated-assessment story
    - arithmetic-specific score execution
    - reasoning-mode moderation
    - ambiguous
13. **Recommended next scientific action**
    - recommendation only; DO NOT execute it
14. **Exact cost/runtime and retry accounting**
15. `READY_FOR_GPT_REVIEW = YES`

---

# STOP RULES

STOP the affected lane immediately if:
- correctness/question joins are unreliable;
- a prompt differs from the intended historical frozen content in a way that changes the scientific construct;
- q2 accidentally exposes q1/correctness/hidden action;
- Lane C reasoning setting is unsupported or requires switching model family;
- provider failures or parse failures become strongly condition-dependent;
- call caps would be exceeded;
- any step would require modifying historical V2 data or `paperDirection.txt`.

Do not compensate for a failed lane by inventing a replacement experiment.

---

# FINAL REMINDER

The goal of Task 005 is not to "prove the paper."

It is to cheaply determine which of these worlds we are in:

1. **There is useful item-specific routing information beyond one or even two confidence reports.**
2. **The hidden-action advantage is mostly a second-read/repeated-assessment effect.**
3. **GPT's score dominance is largely an explicit-arithmetic phenomenon.**
4. **Reasoning mode materially changes the phenomenon.**
5. **The current data are too weak/ambiguous and a different confirmation target is needed.**

Do not scale beyond Task 005 until GPT reviews the outputs.
