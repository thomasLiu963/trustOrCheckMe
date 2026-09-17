TASK ID: 003_run_study1_primary
STATUS: EXPLORATORY
PAID API CALLS AUTHORIZED: YES — PRIMARY STUDY-1 PILOT ONLY

Before doing anything:

1. Read:
   - docs/D1_MASTER_RESEARCH_PLAN.md
   - docs/D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md
   - docs/GPT_CURSOR_WORKFLOW.md
   - docs/D1_DECISION_LOG.md
   - from_gpt/001_build_study1_pilot.md
   - from_gpt/002_smoke_test_study1.md
   - to_gpt/001_study1_pilot_build/report.md
   - to_gpt/002_study1_smoke_test/report.md

2. Save this exact task as:
   from_gpt/003_run_study1_primary.md

3. This task authorizes ONLY the 2,800 primary Study-1 cells.

DO NOT run the 1,120 repeat-extra calls.

DO NOT run optional micro-controls.

DO NOT automatically move to later studies.

==================================================
PURPOSE
==================================================

Run the complete 100-question PRIMARY exploratory causal-confidence pilot.

Scientific question:

Holding the question and frozen answer fixed, does changing only the confidence number displayed to the model change whether it chooses:

VERIFY_FIRST

versus

USE_UNVERIFIED?

This is exploratory evidence used to decide whether a later fresh confirmatory study is justified.

==================================================
1. USE THE FROZEN TASK-001 DESIGN EXACTLY
==================================================

Use the already-frozen:

100 Study-1 primary question IDs

Models:
- openai_gpt56_sol
- anthropic_sonnet5

Authority:
- AI-system only

Costs:
- C = 1
- L = 10
- L = 20

L=10 manipulated confidence grid:
- 0.80
- 0.88
- 0.89
- 0.91
- 0.99

L=20 manipulated confidence grid:
- 0.90
- 0.93
- 0.94
- 0.96
- 0.99

Conditions per question × model × L:
1. hidden
2. true_confidence_visible
3–7. five manipulated-confidence conditions

Total intended primary cells:

100 × 2 × 2 × 7 = 2,800

Do not change any scientific design choice.

==================================================
2. IMPORTANT: SMOKE-TEST QUESTION
==================================================

The Task-002 smoke-test question was:

mmlu_pro:test:7552

It is already part of the frozen 100-question primary sample.

Its 12 smoke-test cells should NOT create duplicate experimental observations.

Use the existing valid Study-1 records for any exact primary cells already completed during Task 002.

The primary Study-1 dataset should ultimately contain exactly ONE valid primary observation per intended request key.

Therefore:

- reuse already-successful matching smoke-test cells;
- do not rerun successful identical cells merely because they occurred during Task 002;
- run all remaining primary cells.

Report exactly how many Task-002 cells were reused and how many NEW paid primary calls were required.

Expected:
2,800 total primary cells represented,
with approximately 12 already present from smoke testing and approximately 2,788 new calls.

If request-key structure means the smoke records cannot legitimately count as the identical planned primary cells, STOP and report this before proceeding rather than inventing a merge rule.

==================================================
3. PRE-FLIGHT
==================================================

Before any new paid call:

Validate:

- frozen 100-ID hash matches Task 001;
- model roster matches;
- exact 2,800 intended primary request keys;
- no duplicate intended cells;
- correct confidence grid by L;
- reported_confidence matches historical V2;
- frozen answers match historical V2;
- Study-1 output path is NOT historical v2.sqlite3;
- historical V2 hash/state is unchanged;
- existing Task-002 successful cells are identified correctly;
- no repeat-extra cells are included;
- no provenance/control cells are included.

Save:

to_gpt/003_study1_primary_results/preflight_manifest.json

If pre-flight fails:
MAKE NO NEW API CALLS.

==================================================
4. CALL AUTHORIZATION AND CAPS
==================================================

Authorized scientific dataset size:

2,800 primary cells total.

Authorized NEW scientific cell calls:

at most the number needed to complete those 2,800 unique primary cells after reusing valid Task-002 cells.

Do not initiate any scientific request outside the frozen primary grid.

Set an explicit provider-request-attempt cap with reasonable retry headroom.

Preferred:

maximum provider attempts =
NEW primary cells needed
+ 10% retry/repair headroom

but never use unused headroom to create extra scientific cells.

Retry headroom is ONLY for:
- provider/transport failures;
- standardized parse repair.

If retry/repair failures become unexpectedly common (e.g. >2% of cells or visibly condition-dependent):
pause execution and report rather than blindly consuming the entire cap.

==================================================
5. EXECUTION
==================================================

Use the exact Study-1 infrastructure audited in Tasks 001 and 002.

Do not change:

- prompt wording;
- probability formatting;
- model endpoints;
- reasoning settings;
- output schema;
- confidence grids;
- question IDs;
- authority;
- costs;
- parser;
- scientific conditions.

Record for every request:

- question_id
- model
- exact provider model metadata
- L
- C
- condition
- reported_confidence
- displayed_confidence
- frozen_answer
- Stage-1 correctness
- prompt_hash
- request key
- attempt numbers
- raw response
- parsed action
- parse status
- latency
- tokens
- estimated cost
- timestamp

==================================================
6. POST-RUN DATA VALIDATION
==================================================

Before scientific analysis, verify:

A. Exactly 2,800 unique primary cells are represented.

B. Expected per-model count:
1,400 each.

C. Expected per-L count:
1,400 each.

D. Expected per-condition structure:
100 questions × 2 models × 2 L for each of seven relevant condition slots.

E. No unintended models/conditions.

F. No duplicate successful primary request keys.

G. reported_confidence is unchanged across conditions for the same question/model.

H. frozen_answer is unchanged across conditions.

I. displayed_confidence:
- null in hidden;
- equals historical confidence in true-visible;
- equals assigned grid value in manipulated cells.

J. historical V2 remains unchanged.

K. every valid action is:
VERIFY_FIRST
or
USE_UNVERIFIED.

Do NOT start scientific analysis if data validation fails materially.

==================================================
7. PRIMARY SCIENTIFIC ANALYSIS
==================================================

This task SHOULD analyze the 2,800 primary cells after validation.

Do NOT wait for repeats before computing the primary exploratory result.

However, clearly state that repeated-generation stability has not yet been tested.

--------------------------------------------------
7A. VERIFICATION-RESPONSE CURVES
--------------------------------------------------

For each model separately and each L separately:

Compute verification rate at every manipulated displayed-confidence value.

Plot:

x-axis:
displayed confidence

y-axis:
fraction choosing VERIFY_FIRST

Produce four main curves:

GPT, L=10
GPT, L=20
Claude, L=10
Claude, L=20

Report exact rates and question-level bootstrap 95% intervals.

Main descriptive expectation:

lower displayed confidence
→ more verification.

Do not force monotonicity in the analysis.

--------------------------------------------------
7B. PRIMARY NEAR-THRESHOLD CONTRASTS
--------------------------------------------------

These are the most important local causal contrasts.

For L=10:

0.89 vs 0.91

Compute:

P(VERIFY | displayed 0.89)
-
P(VERIFY | displayed 0.91)

For L=20:

0.94 vs 0.96

Compute:

P(VERIFY | displayed 0.94)
-
P(VERIFY | displayed 0.96)

Do this separately for GPT and Claude.

Because every question appears in both conditions, use paired question-level analysis.

Report:

- verification rates in each condition;
- percentage-point difference;
- paired bootstrap 95% CI;
- number/fraction:
  VERIFY → USE
  USE → VERIFY
  no change.

Do not treat the two observations as independent samples.

--------------------------------------------------
7C. EXTREME-SCORE CONTRASTS
--------------------------------------------------

For L=10:

0.80 vs 0.99

For L=20:

0.90 vs 0.99

Again compute separately by model:

- verification-rate difference;
- paired bootstrap CI;
- within-question flip direction.

This establishes whether displayed confidence has a large overall effect even if the exact threshold behavior is imperfect.

--------------------------------------------------
7D. WITHIN-QUESTION MONOTONICITY
--------------------------------------------------

For every question/model/L, order the five manipulated confidence conditions from low to high.

Classify trajectories.

At minimum report:

- perfectly non-increasing verification trajectory;
- contains at least one reverse USE → VERIFY transition as confidence increases;
- number of action transitions;
- whether behavior resembles a single threshold.

Because the output is binary and there is only one generation per cell, describe this cautiously.

--------------------------------------------------
7E. THRESHOLD LOCATION
--------------------------------------------------

Explore whether observed action changes cluster near the mathematically supplied threshold:

L=10 threshold = 0.90
L=20 threshold = 0.95

Do not assume a sharp threshold.

Report how closely aggregate verification curves resemble:

VERIFY below threshold
USE above threshold.

Also compare the empirical response to the mechanical confidence rule.

This is descriptive/exploratory.

--------------------------------------------------
7F. HIDDEN AND TRUE-VISIBLE REFERENCES
--------------------------------------------------

For each model/L report:

- hidden verification rate;
- true-visible verification rate;
- manipulated-score curve.

Ask descriptively:

Where does the actual historical reported-confidence condition fall relative to the manipulated curve?

Do NOT compare different displayed values across different questions naively.

The true-visible condition contains question-specific historical confidence, unlike the constant manipulated values.

Treat it as a reference, not another point on a constant-score dose-response curve.

--------------------------------------------------
7G. CURRENT-VS-HISTORICAL SANITY CHECK
--------------------------------------------------

Compare current hidden/true-visible aggregate behavior on these 100 frozen questions to the corresponding historical V2 behavior for the exact same questions/model/authority/L where possible.

This is NOT a new hypothesis.

It checks whether endpoint/model drift has changed the basic historical phenomenon.

Report:

- historical verification rate;
- current verification rate;
- difference;
- agreement at question level if useful.

Do not pretend stochastic disagreement means model identity changed.

--------------------------------------------------
7H. CORRECTNESS-CONDITIONED EXPLORATION
--------------------------------------------------

Because Stage-1 correctness is already frozen, exploratory analysis may report whether responsiveness to manipulated confidence differs between:

- originally correct answers;
- originally wrong answers.

This is SECONDARY.

Do not select or exclude questions based on correctness.

Do not let this replace the primary all-question causal effect.

Given only 100 questions, especially per model, label these estimates exploratory and potentially noisy.

==================================================
8. BOOTSTRAP
==================================================

Use paired question-level bootstrap.

Recommended:

5,000 resamples
fixed seed = 20260917

Resample question IDs.

Preserve all within-question conditions together.

Compute nominal percentile 95% intervals.

Do not call exploratory intervals confirmatory significance tests.

==================================================
9. GO / AMBIGUOUS / FAIL ASSESSMENT
==================================================

Apply the Experimental Protocol’s Study-1 gate conservatively.

Assess:

1. Does verification materially change across displayed confidence?
2. Is the direction generally:
   lower confidence → more verification?
3. Does it occur within the same frozen questions?
4. Is it broadly distributed rather than caused by a few questions?
5. Does at least GPT show a substantial effect?
6. Is Claude compatible or scientifically informative as a contrast?
7. Does the effect look large enough to justify repeats and prospective confirmation?

Remember:

The protocol’s ~10–15 percentage-point low-vs-high contrast is a practical pilot heuristic, NOT a confirmatory significance cutoff.

VERY IMPORTANT:

We have NOT yet run repeated sampling.

Therefore Task 003 may conclude:

STRONG_BEHAVIORAL_SIGNAL_PENDING_STABILITY

but it may NOT claim the effect exceeds ordinary repeated-generation noise.

That question belongs to the withheld 1,120 repeat-extra experiment if GPT later authorizes it.

Possible recommendation labels:

A. FAIL / rethink
B. AMBIGUOUS — inspect before further calls
C. PASS PRIMARY — run stability repeats next
D. PASS PRIMARY — but design issue requires correction before repeats

Cursor may recommend one.

Cursor may NOT launch the next task.

==================================================
10. COST / RUNTIME ACCOUNTING
==================================================

Report:

- new scientific calls;
- reused Task-002 successful cells;
- provider request attempts;
- retries;
- repairs;
- input tokens;
- output tokens;
- GPT cost;
- Claude cost;
- total cost;
- wall-clock runtime;
- summed provider latency if available.

==================================================
11. REQUIRED OUTPUT FOLDER
==================================================

Create:

to_gpt/003_study1_primary_results/

Required:

1. report.md
2. changed_files.txt
3. run_manifest.json
4. preflight_manifest.json
5. data_validation.md
6. primary_results.csv
7. verification_rates.csv
8. threshold_contrasts.csv
9. trajectory_results.csv
10. historical_comparison.csv
11. provider_attempts.csv
12. cost_summary.md
13. analysis_method.md
14. figures/
15. analysis script(s)
16. raw/results references or manifest sufficient to locate every raw response

==================================================
12. FIGURES
==================================================

At minimum create:

figures/gpt_L10_response_curve.png
figures/gpt_L20_response_curve.png
figures/claude_L10_response_curve.png
figures/claude_L20_response_curve.png

Also useful:

figures/near_threshold_effects.png
figures/extreme_score_effects.png
figures/trajectory_monotonicity.png
figures/current_vs_historical.png

Do not create decorative figures that do not answer a scientific question.

==================================================
13. REPORT.MD STRUCTURE
==================================================

Use exactly:

# Study 1 Primary Causal Pilot

## 1. Data validation

## 2. Bottom line in plain English

Maximum roughly 10 bullets.

Directly answer:

- Did changing ONLY displayed confidence change verification?
- Roughly how large was the effect for GPT?
- Roughly how large was the effect for Claude?
- Did behavior respond around the mathematical threshold?
- Was the dose-response broadly monotonic?
- Did GPT and Claude behave similarly or differently?
- Did current hidden/true-visible behavior resemble historical V2?
- Is the result broadly distributed across questions?
- What remains unknown because repeats have not been run?
- Should GPT authorize the 1,120-call stability subset?

## 3. GPT results

## 4. Claude results

## 5. Near-threshold causal effects

## 6. Extreme-score causal effects

## 7. Within-question trajectories

## 8. Current vs historical sanity check

## 9. Secondary correctness-conditioned analysis

## 10. What this DOES establish

Be conservative.

## 11. What this DOES NOT establish

Explicitly include:
- not confirmatory;
- does not establish “own” provenance matters;
- does not establish real-world generalization;
- does not establish a hidden-state mechanism;
- does not establish the effect exceeds ordinary rerun noise until repeats are run.

## 12. Study-1 gate recommendation

Choose:
A / B / C / D

Explain why.

## 13. Exact numbers GPT should know

Compact table.

## 14. Cost and runtime

==================================================
14. CLAIM DISCIPLINE
==================================================

Allowed if supported:

“Changing only displayed confidence causally changed verification behavior in this exploratory pilot.”

Allowed if supported:

“Verification decreased as displayed confidence increased.”

Allowed if supported:

“Behavior changed sharply around the supplied decision threshold.”

NOT allowed from Task 003:

“Self-confidence uniquely causes this.”
(provenance not tested)

“Showing confidence suppresses internal uncertainty.”
(no mechanism evidence)

“The model loses information.”
(not established by this manipulation)

“This generalizes to agents.”
(not tested)

“This is confirmatory evidence.”
(exploratory reused question set)

“This effect exceeds stochastic rerun noise.”
(repeats not yet run)

==================================================
15. DO NOT RUN
==================================================

Do NOT run:

- the 1,120 repeats;
- provenance control;
- qualitative-stakes control;
- contradiction control;
- additional questions;
- additional models;
- another task;
- interpretability;
- Deep Research.

==================================================
16. RUN MANIFEST
==================================================

run_manifest.json must include at minimum:

task_id: 003_run_study1_primary
status
git_commit
paid_calls_authorized: true
primary_dataset_target: 2800
existing_smoke_cells_reused
new_scientific_calls
provider_attempts_total
retry_attempts
repair_attempts
successful_unique_primary_cells
failed_primary_cells
models
L_values
confidence_grids
selected_question_hash
api_cost_usd
wall_clock_seconds
historical_artifacts_modified
data_validation_passed
gate_recommendation
ready_for_gpt_review: true

==================================================
FINAL RULE
==================================================

After completing the 2,800-cell primary dataset and analysis:

STOP.

Do not run repeats.

Do not run controls.

Do not run Deep Research.

Print:

READY_FOR_GPT_REVIEW = YES

and the path:

to_gpt/003_study1_primary_results/report.md
