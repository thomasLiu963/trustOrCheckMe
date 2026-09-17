TASK ID: 002_smoke_test_study1
STATUS: DEVELOPMENT
PAID API CALLS AUTHORIZED: YES — TINY SMOKE TEST ONLY

Before doing anything:

1. Read:
   - docs/D1_MASTER_RESEARCH_PLAN.md
   - docs/D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md
   - docs/GPT_CURSOR_WORKFLOW.md
   - docs/D1_DECISION_LOG.md
   - from_gpt/001_build_study1_pilot.md
   - to_gpt/001_study1_pilot_build/report.md
   - to_gpt/001_study1_pilot_build/prompt_audit.md
   - to_gpt/001_study1_pilot_build/study1_call_plan_summary.md

2. Save this exact task as:
   from_gpt/002_smoke_test_study1.md

3. Do NOT infer authorization for the full Study-1 pilot.

==================================================
PURPOSE
==================================================

Run a tiny REAL end-to-end API smoke test of the Study-1 pipeline.

This task tests ENGINEERING only:

- live GPT provider path;
- live Claude provider path;
- rendered prompts;
- strict action parsing;
- retry/repair machinery;
- reported_confidence vs displayed_confidence storage;
- Study-1 SQLite writes;
- request keys;
- call-cap enforcement;
- logging and cost accounting.

It is NOT intended to estimate scientific effects.

Do not interpret these 12 calls as evidence for or against the hypothesis.

==================================================
1. QUESTION SELECTION
==================================================

Use exactly ONE question.

Choose:

the FIRST question ID in the already-frozen ordered
data/development/study1_ids.json

Do not select based on:
- correctness;
- confidence;
- historical Stage-3 behavior;
- how interesting the question looks.

Record the selected ID in the smoke-test report.

Reuse the existing historical frozen answer and historical reported confidence for each model.

Do NOT regenerate Stage 1 or Stage 2.

==================================================
2. MODEL ROSTER
==================================================

Use exactly:

- openai_gpt56_sol
- anthropic_sonnet5

No Gemini.
No Grok.
No open models.

Record exact provider-returned model/version metadata.

Do not silently substitute another endpoint.

If a configured endpoint is unavailable or materially changed:
STOP that provider and report it rather than choosing a replacement.

==================================================
3. EXACT 12 SCIENTIFIC CELLS AUTHORIZED
==================================================

Run exactly the following Study-1 cells for the selected question.

For GPT:

L = 10:
1. hidden
2. true_confidence_visible
3. manipulated displayed_confidence = 0.89
4. manipulated displayed_confidence = 0.91

L = 20:
5. manipulated displayed_confidence = 0.94
6. manipulated displayed_confidence = 0.96

For Claude:

L = 10:
7. hidden
8. true_confidence_visible
9. manipulated displayed_confidence = 0.89
10. manipulated displayed_confidence = 0.91

L = 20:
11. manipulated displayed_confidence = 0.94
12. manipulated displayed_confidence = 0.96

Total planned experimental cells:

12

Do not add:
- other confidence values;
- other questions;
- repeats;
- provenance controls;
- contradiction controls;
- qualitative-stakes controls.

==================================================
4. CALL CAPS
==================================================

AUTHORIZED SCIENTIFIC CELLS:

12 maximum.

The runner must refuse to initiate a 13th scientific cell.

Because provider transport failures or standardized parse repair can create additional raw network requests:

TOTAL PROVIDER REQUEST ATTEMPT CAP:

20 maximum

including:
- retries;
- repair attempts.

If completing the 12 cells would require exceeding 20 total provider request attempts:

STOP and return the partial smoke test.

Do not relax the cap.

==================================================
5. PROMPT / SCIENTIFIC INVARIANCE
==================================================

Use exactly the Study-1 implementation audited in Task 001.

Do not change:
- prompt wording;
- confidence formatting;
- authority framing;
- action schema;
- model settings;
- L or C;
- frozen answer;
- reported confidence;
- confidence grids.

For manipulated pairs:

0.89 vs 0.91 at L=10
and
0.94 vs 0.96 at L=20

the rendered prompt should differ only in the displayed confidence value.

==================================================
6. PRE-FLIGHT CHECK BEFORE ANY PAID CALL
==================================================

Before the first network request:

A. Render all 12 intended requests.

B. Validate:
- one frozen question only;
- two models only;
- exactly 12 cells;
- correct L values;
- correct conditions;
- correct displayed confidences;
- correct historical reported confidence;
- correct frozen answer;
- Study-1 write path, NOT v2.sqlite3;
- no duplicate request keys;
- paid call cap = 12 scientific cells;
- provider-attempt cap = 20.

C. Save the exact pre-flight manifest.

If ANY validation fails:

MAKE ZERO API CALLS.

Return failure to GPT.

==================================================
7. EXECUTION
==================================================

If and only if pre-flight passes:

run the 12 authorized cells.

Use the already-implemented provider adapters and Study-1 runner.

Preserve:
- raw responses;
- parsed actions;
- attempt number;
- parse status;
- provider status;
- latency;
- input/output token counts;
- estimated cost;
- model/version metadata.

Do not rerun a successful valid cell.

Transport failures:
follow the fixed retry policy subject to the global 20-attempt cap.

Parse failures:
use only the already-defined standardized repair mechanism, subject to the cap.

==================================================
8. POST-RUN VALIDATION
==================================================

After execution, verify:

1. Historical V2 SQLite is unchanged.
2. Study-1 records went only to the new Study-1 checkpoint.
3. reported_confidence equals historical Stage-2 confidence.
4. displayed_confidence is:
   - null in hidden;
   - historical reported confidence in true-visible;
   - assigned fake number in manipulated cells.
5. frozen answer matches historical Stage 1.
6. all valid parsed actions are exactly:
   VERIFY_FIRST
   or
   USE_UNVERIFIED.
7. no unintended scientific cells were run.
8. scientific-cell count <= 12.
9. total provider request attempts <= 20.

==================================================
9. DO NOT SCIENTIFICALLY INTERPRET THE SMOKE TEST
==================================================

The output may incidentally show action changes across fake confidence values.

Do NOT conclude:
- confidence causally controls verification;
- the hypothesis passed;
- GPT followed the fake number;
- Claude behaved differently;
- the project should scale.

n = 1 question is not a scientific test.

Cursor may plainly report the raw action observed in each of the 12 cells so GPT can inspect that the pipeline worked.

No statistical tests.

No paper framing.

==================================================
10. RETURN FOLDER
==================================================

Create:

to_gpt/002_study1_smoke_test/

Required files:

1. report.md
2. changed_files.txt
3. run_manifest.json
4. preflight_manifest.json
5. smoke_test_cells.csv
6. provider_attempts.csv
7. raw_responses/
8. rendered_prompts/
9. prompt_diffs/
10. sqlite_validation.md
11. cost_summary.md

==================================================
11. REPORT CONTENT
==================================================

report.md must answer:

- What question ID was used?
- What were GPT and Claude's historical reported confidences?
- Did pre-flight validation pass?
- Were exactly 12 scientific cells attempted?
- How many completed successfully?
- How many raw provider request attempts occurred?
- Were any retries needed?
- Were any parse repairs needed?
- Did every valid response parse into the two-action schema?
- Were reported_confidence and displayed_confidence stored correctly?
- Was historical V2 untouched?
- What exact provider model/version metadata was returned?
- What was total measured/estimated API cost?
- What was wall-clock runtime?
- Did the call caps work?
- Is the pipeline technically ready for the full 2,800-call primary pilot?
- Is there any engineering issue GPT must resolve first?

Include a compact raw table:

model
L
condition
reported_confidence
displayed_confidence
parsed_action
attempts
latency
tokens
cost

This table is for engineering inspection only.

==================================================
12. RUN MANIFEST
==================================================

run_manifest.json must include at minimum:

task_id: 002_smoke_test_study1
status
git_commit
paid_calls_authorized: true
scientific_cell_cap: 12
provider_attempt_cap: 20
selected_question_id
models
L_values
planned_cells
successful_cells
failed_cells
provider_attempts_total
retry_attempts
repair_attempts
api_cost_usd
historical_artifacts_modified
study1_checkpoint_path
provider_model_metadata
ready_for_task_003

==================================================
13. FINAL GATE
==================================================

At the end print exactly one:

READY_FOR_TASK_003 = YES

or

READY_FOR_TASK_003 = NO

YES means only:

“The engineering pipeline appears technically ready for GPT to consider authorizing the full 2,800-call primary pilot.”

YES does NOT authorize Task 003 automatically.

Then STOP.
