# TASK 009 — Prospective Confirmation of Confidence-as-Coverage Control

## Purpose

Task 008 ended the exploratory phase with:
- decision bucket: `INVARIANCE_SUPPORTED_EXPLORATORY`
- favored world: ranking invariance, moderately favored

The next step is NOT more mining of the old 100 questions.

This task is the first prospective confirmatory study.

Central claim to test:

> On the exact same frozen answer, counterfactual displayed confidence strongly changes how often the model requests independent verification, while the relative ordering of which items it prioritizes for verification remains largely stable.

This task must be able to falsify that claim. If the data instead show material score-dependent reprioritization, report that honestly.

Save this exact task as:
`from_gpt/009_prospective_ranking_invariance_confirmation.md`

Write outputs to:
`to_gpt/009_prospective_ranking_invariance_confirmation/`

# NON-NEGOTIABLE RULES

1. This is PROSPECTIVE CONFIRMATION.
2. Freeze question IDs, model endpoints, prompts, score conditions, primary metrics, analysis configs, retry policy BEFORE inspecting scientific results.
3. Do not modify `paperDirection.txt`.
4. Do not use any historical/development 500 questions.
5. Use the unseen MMLU-Pro pool prepared in Task 006/007.
6. Same fresh questions across all models.
7. Fresh Stage-1 answers and fresh q1 confidence are mandatory.
8. No q2.
9. One stakes family only: use the exact Task-006/007 MODERATE qualitative wording.
10. No explicit numerical L/C or expected-value formula.
11. Primary inference is GPT + Claude.
12. Gemini + Grok are secondary only.
13. No result-driven prompt or score-grid changes.
14. Question ID is the resampling unit.
15. No hidden-state mechanism claims.
16. Cross-model difficulty is evaluation-only, never deployable.
17. Do not inspect partial condition results until all primary GPT/Claude cells complete, except engineering validation.
18. Technical retries follow a deterministic predeclared rule.
19. If an endpoint is unavailable, mark BLOCKED; do not silently substitute another model.

# PART 0 — PREREGISTRATION / FREEZE

Before scientific calls, create:
- `preregistration.md`
- `freeze_manifest.json`

Include:
- hypotheses
- exact sample IDs + SHA256
- exact model endpoints
- exact prompts + hashes
- exact conditions
- exact primary and secondary metrics
- exact retry/exclusion rules
- saturation flags
- bootstrap seeds
- planned call counts
- confirmation/falsification criteria

After writing, do not alter except with a timestamped deviation note for unavoidable engineering issues.

# PART 1 — SAMPLE

Primary:
- N=500 unseen MMLU-Pro test questions
- deterministic stratified sampling from the prepared unseen pool
- exclude all V2/Tasks 003–008 IDs
- seed `20260918`
- freeze IDs, category counts, source revision, SHA256

Secondary:
- deterministic stratified N=200 subset of those same 500 for Gemini/Grok
- freeze before seeing GPT/Claude results

# PART 2 — MODELS

Primary:
- exact GPT endpoint/settings used in Task 007
- exact Claude endpoint/settings used in Task 007

Secondary:
- exact Gemini endpoint used in Task 007-B
- exact Grok endpoint used in Task 007-B

Read exact strings from repo config; do not infer if config differs.

# PART 3 — FRESH STAGE 1 + q1

For every selected model/question:

Stage 1:
- generate fresh answer
- freeze answer
- score correctness against benchmark gold

q1:
- separate call
- fresh numerical confidence that the frozen answer is correct
- use current repo q1 format/config
- do not reveal correctness or future conditions

Store raw and parsed outputs plus metadata.

No q2.

# PART 4 — STAGE 3 CONDITIONS

Use exactly ONE qualitative stakes family:
- MODERATE wording from Task 006/007

Each fresh frozen answer gets 7 conditions:

1. hidden
2. true_q_visible
3. displayed_0.70
4. displayed_0.85
5. displayed_0.90
6. displayed_0.95
7. displayed_0.99

Definitions:
- hidden: no confidence sentence
- true_q_visible: show fresh q1
- manipulated: show exactly the stated fixed score

Frozen Stage-1 answer is identical across all 7 conditions.

Only valid actions:
- USE_UNVERIFIED
- VERIFY_FIRST

The external verifier performs the check; the target model only decides whether to REQUEST it.

# PART 5 — REPEAT SUBSET

Freeze deterministic N=100 subset of the primary 500 using seed `20260919`.

For GPT and Claude only:
- run each Stage-3 condition 3 total times/item
- original primary generation counts as repeat 1
- add exactly 2 extra generations
- do not repeat Stage 1 or q1

Purpose:
- estimate per-item verification propensity
- rank correlations across conditions
- rerun-noise baseline
- reversal rates relative to stochasticity

# PART 6 — CALL COUNTS

GPT + Claude:
- Stage1+q1: 500 × 2 × 2 = 2,000
- Stage3: 500 × 2 × 7 = 7,000
- repeat extras: 100 × 2 × 7 × 2 = 2,800
- primary total = 11,800

Gemini + Grok:
- Stage1+q1: 200 × 2 × 2 = 800
- Stage3: 200 × 2 × 7 = 2,800
- secondary total = 3,600

Grand planned scientific calls = 15,400
Retries reported separately.

# PART 7 — PRIMARY HYPOTHESES

## H1 — Coverage control

Primary effect:
`VERIFY_rate(0.70) - VERIFY_rate(0.99)`

Report paired effect + question-bootstrap 95% CI.

Substantive replication floors:
- GPT >= 20 percentage points
- Claude >= 15 percentage points

These define a large effect, not a publication gate.

## H2 — Shared-ranking invariance

Compare:
A. shared-ranking model:
`condition-specific intercept + shared item propensity`

B. condition-sensitive model:
allows score-dependent reprioritization

Use grouped/question-wise cross-validation.

Primary metric:
held-out log-loss improvement from richer model.

Predeclared invariance criterion:
- richer model improves held-out log loss by < 0.01
- question-bootstrap 95% upper bound also < 0.01
- evaluate separately for GPT and Claude

If unstable, report INCONCLUSIVE rather than changing the threshold.

## H3 — Repeat-based rank stability

On N=100 repeats, estimate per-item verification propensity from 3 generations.

For adjacent visible-score pairs:
- .70 vs .85
- .85 vs .90
- .90 vs .95
- .95 vs .99

Report:
- Spearman rank correlation
- bootstrap CI
- pairwise ordering agreement
- reversals beyond ordinary rerun noise

Do not reduce invariance to one arbitrary cutoff.

## H4 — Discrimination stability

For each visible score:
- AUROC of repeat-averaged verification propensity for target wrongness
- bootstrap CI
- pairwise ΔAUROC

Also report full-N binary-action AUROC as coarse secondary evidence.

# PART 8 — SECONDARY ANALYSES

A. Nesting/reversals
- V→V, V→U, U→U, U→V
- compare reversal rate to same-prompt rerun disagreement

B. Risk/coverage curves
- x = verification coverage
- y = errors caught / residual risk

C. Hidden and true-q bridge
- contextual/historical bridge only
- do not force them into the same threshold family

D. Difficulty robustness
- leave-one-target-out other-model correctness
- evaluation-only
- test score×difficulty interaction
- not central

E. Error-catching
- VERIFY rate
- fraction of target errors verified
- fraction of correct answers verified
- do not call raw catch changes ranking changes when coverage differs

# PART 9 — MATCHED-BUDGET ROUTING

Consequence analysis, not headline.

Compare at common budgets where support permits:
- 10%
- 20%
- 30%
- 40%
- 50%

Baselines:
1. random
2. raw q1
3. cross-fitted calibrated q1
4. one hidden qualitative verification judgment
5. q1 + one hidden judgment
6. hindsight oracle

Optional labeled oracles:
- cross-model difficulty
- historical multi-cell aggregate

Metric:
“At the same verification budget, what fraction of actual errors is caught?”

Do not post-hoc search many routers for a winner.

# PART 10 — DECISION LOGIC

Classify:

`PROSPECTIVE_INVARIANCE_SUPPORTED`
if:
- coverage response large
- shared-ranking equivalence-style test passes
- repeat rank stability strong
- discrimination stable across non-saturated conditions
- richer condition-sensitive model adds no material held-out value

`PROSPECTIVE_RESHAPING_SUPPORTED`
if:
- coverage changes
- richer model materially/reproducibly beats shared ranking
- repeats show score-dependent reordering beyond rerun noise
- discrimination changes materially at non-saturated operating points

`MIXED_BY_MODEL`
if GPT and Claude genuinely diverge under the same tests

`INCONCLUSIVE`
if saturation/low error counts/uncertainty prevent distinction

Do not redefine after seeing results.

# PART 11 — GEMINI / GROK SECONDARY

Run same fresh Stage1/q1 + 7 Stage3 conditions on frozen N=200 subset.

No repeats required.

Report:
- coverage response
- nesting/reversals
- action-vs-error discrimination
- compatibility with primary pattern

Do not let secondary oddities redefine the GPT/Claude primary hypothesis.

# PART 12 — OUTPUTS

Create:
`to_gpt/009_prospective_ranking_invariance_confirmation/`

Root:
- report.md
- preregistration.md
- freeze_manifest.json
- validation.md
- cost_summary.md
- changed_files.txt
- deviations.md

Results:
- sample_500.csv
- secondary_200.csv
- repeat_100.csv
- stage1_q1_results.csv
- stage3_results.csv
- repeat_results.csv
- coverage_response.csv
- shared_ranking_model_results.csv
- condition_sensitive_model_results.csv
- rank_stability.csv
- condition_auc.csv
- nesting_reversals.csv
- risk_coverage.csv
- matched_budget_routing.csv
- secondary_models_summary.csv

Figures:
- figures/coverage_response.png
- figures/rank_stability.png
- figures/risk_coverage.png
- figures/matched_budget_error_catch.png

Analysis scripts:
- analysis/task009/

# PART 13 — ROOT REPORT

Include:
1. Plain-English bottom line
2. Preregistration integrity
3. Exact calls/failures/retries/cost
4. Coverage result
5. Shared-ranking vs condition-sensitive result
6. Repeat rank stability
7. Discrimination stability
8. Nesting/reversals
9. Risk-coverage interpretation
10. Hidden/true-q bridge
11. Matched-budget routing
12. Gemini/Grok secondary replication
13. Decision bucket
14. Which final-paper world won
15. Narrative implication
16. Whether to run a second task
17. `READY_FOR_GPT_REVIEW = YES`

# PART 14 — AFTER TASK 009

Do NOT launch a second benchmark inside this task.

If prospective invariance is supported:
- recommend a second non-MCQ task as the next high-value generalization study

If reshaping is supported:
- design the second task around that result

If mixed/inconclusive:
- stop and explain why before spending more

# FINAL REMINDER

The prospective question is simple:

> For the exact same frozen answer, when we counterfactually change displayed confidence, does the model mostly change HOW MANY answers it sends for independent verification, or does it change WHICH answers it prioritizes?

Current exploratory expectation: mostly HOW MANY.

Task 009 must give fresh data a real chance to say otherwise.
