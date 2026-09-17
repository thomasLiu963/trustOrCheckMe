TASK ID: 001_build_study1_pilot
STATUS: DEVELOPMENT
PAID API CALLS AUTHORIZED: NO
NETWORK MODEL CALLS AUTHORIZED: NO

Before doing anything else:

1. Read:
   - docs/D1_MASTER_RESEARCH_PLAN.md
   - docs/D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md
   - docs/GPT_CURSOR_WORKFLOW.md
   - docs/D1_DECISION_LOG.md
   - to_gpt/000_workflow_setup/report.md
   - to_gpt/000_workflow_setup/repository_readiness.md

2. Save a copy of this exact task specification as:
   from_gpt/001_build_study1_pilot.md

3. Obey the GPT-Cursor workflow contract.

==================================================
PURPOSE
==================================================

Build and fully audit the infrastructure for the small Study-1 causal confidence pilot.

DO NOT run the real pilot.

DO NOT make any GPT, Claude, Gemini, Grok, OpenAI, Anthropic, or other paid/network model calls.

This task ends with a LOCAL / OFFLINE dry run and rendered-prompt audit.

The next GPT task, after manual review, will separately authorize a tiny real API smoke test.

==================================================
SCIENTIFIC QUESTION
==================================================

Study 1 asks:

If the question and frozen answer stay fixed, but we deliberately change only the confidence number displayed during the later verification decision, does the model change whether it chooses:

VERIFY_FIRST

versus

USE_UNVERIFIED?

This is exploratory.

Do not build later studies.
Do not implement interpretability.
Do not add datasets/models/controls beyond what is specified here.

==================================================
1. HISTORICAL INPUTS — READ ONLY
==================================================

Reuse, READ ONLY:

results/v2/raw/v2.sqlite3
data/samples/mmlu_pro_v2b.jsonl
data/manifests/mmlu_pro_v2b_manifest.json
src/v2_prompts.py
src/model_adapters.py
src/checkpointing.py
src/v2_datasets.py
config/models.yaml

Never write Study-1 records into:

results/v2/raw/v2.sqlite3

Never modify historical V2 schemas/configs/prompts to accommodate Study 1.

Checkpoint A is also immutable:

to_gpt/checkpoint_A/

==================================================
2. SCIENTIFIC WORDING DECISION — NOW FROZEN FOR PILOT
==================================================

Cursor previously identified an ambiguity:

Historical V2 visible wording uses approximately:

“The AI previously estimated a {p} probability that this frozen answer is correct.”

The draft protocol includes an example using:

“Your previously reported probability ...”

For STUDY 1, use the HISTORICAL V2 AI-AUTHORITY WORDING as the base.

Reason:
We want continuity with the historical visibility effect.

Therefore:

- clone/adapt the historical AI-authority Stage-3 template;
- do not independently rewrite or “improve” its prose;
- for manipulated-confidence conditions, the intended experimental change relative to the corresponding visible template is ONLY the displayed confidence value;
- preserve question text, frozen answer, costs, instructions, authority framing, and output schema.

Document the exact historical source lines/function used.

==================================================
3. STUDY-1 MODEL ROSTER
==================================================

Prepare infrastructure for exactly:

- GPT-5.6 Sol
- Claude Sonnet 5

Use the existing repository model IDs/endpoints currently configured for those historical names.

Do not make network calls in this task.

Do not add Gemini, Grok, or other models.

At future runtime, provider-returned model/version metadata must be recorded so model drift can be documented.

==================================================
4. AUTHORITY CONDITION
==================================================

Use ONLY:

AI-system verification authority.

Do not implement/run the human-authority factorial condition for Study 1.

==================================================
5. QUESTION SAMPLE
==================================================

Select and freeze exactly 100 question IDs from the canonical historical 500-question V2-B sample.

Selection MUST NOT use:
- correctness;
- historical confidence;
- historical Stage-3 actions;
- category performance;
- whether a question looks interesting.

Use the protocol rule:

- canonical V2-B question universe
- deterministic seed: 20260917
- sample 100 unique IDs without replacement

Store:

data/development/study1_ids.json

The file must include:
- ordered selected IDs;
- selection seed;
- source sample path;
- source sample hash if available;
- timestamp;
- hash of the final ordered 100-ID list.

Also deterministically choose the 20-question repeated-sampling subset NOW, before any Study-1 outcomes exist.

Use seed:

20260918

Store the repeat subset separately inside the same file or as:

data/development/study1_repeat_ids.json

Hash it.

Do NOT make any API calls after selecting IDs.

==================================================
6. FROZEN ANSWER / REPORTED CONFIDENCE
==================================================

For each selected (question, model):

read from historical V2:

- frozen Stage-1 answer;
- Stage-1 correctness;
- historical Stage-2 confidence.

Do not regenerate the answer.
Do not regenerate confidence.

In Study-1 terminology:

reported_confidence =
the model’s real historical Stage-2 confidence.

displayed_confidence =
the number actually placed into the Study-1 Stage-3 prompt.

These MUST be separate database/schema fields.

Never overwrite or relabel the historical V2 value.

==================================================
7. ERROR COSTS
==================================================

Study 1 uses exactly:

C = 1

L = 10
L = 20

No other L values.

==================================================
8. DISPLAYED-CONFIDENCE GRID — NOW FROZEN FOR PILOT
==================================================

For L = 10 use exactly:

0.80
0.88
0.89
0.91
0.99

The key local threshold contrast is:

0.89 vs 0.91

because the mechanical threshold is 0.90.

For L = 20 use exactly:

0.90
0.93
0.94
0.96
0.99

The key local threshold contrast is:

0.94 vs 0.96

because the mechanical threshold is 0.95.

This task specification resolves the draft protocol’s 0.949/0.951 vs simpler-format ambiguity in favor of the simpler natural-looking grid.

Do NOT change this grid.

==================================================
9. SEVEN STAGE-3 CONDITIONS PER QUESTION × MODEL × L
==================================================

For each selected question, model, and L prepare exactly seven primary conditions:

1. hidden
2. true_confidence_visible
3. manipulated_1
4. manipulated_2
5. manipulated_3
6. manipulated_4
7. manipulated_5

Where:

HIDDEN:
No numerical confidence displayed.

TRUE_CONFIDENCE_VISIBLE:
displayed_confidence = reported_confidence.

MANIPULATED:
displayed_confidence equals the applicable frozen grid value.

The original reported_confidence remains recorded separately in every row.

==================================================
10. PRIMARY AND REPEAT CALL PLAN
==================================================

The intended PRIMARY plan must contain exactly:

100 questions
× 2 models
× 2 L values
× 7 conditions

= 2,800 planned primary Stage-3 calls.

The intended REPEAT plan must contain:

20 questions
× 2 models
× 2 L values
× 7 conditions
× 2 EXTRA repetitions

= 1,120 additional planned calls.

Total possible later pilot:

3,920 calls.

IMPORTANT:

Task 001 executes NONE of these calls.

Produce the call plans/manifests offline so GPT can verify all counts before authorizing anything.

==================================================
11. NEW STUDY-1 IMPLEMENTATION
==================================================

Create new Study-1 infrastructure rather than modifying frozen V2 implementation.

Preferred structure from Task 000 readiness audit:

config/experiment_study1.yaml
src/study1_prompts.py
src/study1_runner.py
src/study1_schemas.py
data/development/study1_ids.json
data/development/study1_repeat_ids.json
prompts/study1/
results/study1_causal_pilot/
analysis/study1/

Exact internal filenames can differ only for ordinary engineering reasons.

Do not make material scientific design changes.

==================================================
12. STUDY-1 SCHEMA
==================================================

Create a new Study-1 record/schema.

Do NOT relax or rewrite the historical V2 VerificationDecisionRecord.

At minimum, Study-1 records should preserve:

study_id
pilot_or_confirmatory
question_id
model_id
model_endpoint
api_timestamp
L
C
reported_confidence
displayed_confidence
display_condition
display_source_condition
frozen_answer
stage1_correct
prompt_hash
attempt_number
raw_response
parsed_action
parse_status
latency_ms
input_tokens
output_tokens
estimated_cost_usd

Use null/absent displayed_confidence for the hidden condition in a principled way.

Also preserve enough metadata to reproduce:
- model settings;
- prompt version;
- code commit;
- selected sample hash.

==================================================
13. OUTPUT ACTION SCHEMA
==================================================

Keep exactly the historical two-action decision:

VERIFY_FIRST
USE_UNVERIFIED

Preserve the strict JSON/structured-output approach already implemented successfully.

Do not add free-form explanation requirements.

==================================================
14. RETRIES / REPAIR
==================================================

Reuse the historical principles:

Transport/provider failure:
- bounded retries;
- retain failure logs.

Parse failure:
- preserve raw response;
- at most the standardized fixed repair mechanism;
- never write response-specific repair wording.

No actual network retries occur in Task 001 because no network calls are authorized.

==================================================
15. PAID-CALL SAFETY
==================================================

Study-1 code must default to NO PAID CALLS.

Reuse or strengthen the existing allow_paid safeguards.

The Study-1 runner must make it impossible to accidentally run thousands of calls merely by invoking the default command.

Implement explicit phase/cap controls.

Preferred behavior:

- dry-run is default;
- paid execution requires an explicit authorization flag;
- caller must specify a maximum call cap;
- runner refuses to exceed that cap;
- primary and repeats can be separately invoked;
- optional controls are separate phases and cannot be implicitly included.

Examples of phases conceptually:

primary
repeats

Do NOT implement later scientific studies as phases.

No paid call is authorized in Task 001.

==================================================
16. OFFLINE VALIDATION / DRY RUN
==================================================

Run the entire Study-1 planning pipeline OFFLINE.

It should:

- load the 100 frozen question IDs;
- load historical frozen answers/confidences for both models;
- generate every intended Study-1 request;
- compute stable request keys;
- render prompts;
- validate schemas;
- validate condition counts;
- validate no historical files would be overwritten;
- validate all primary rows are unique;
- validate all repeat rows are unique;
- validate the correct confidence grids attach to the correct L.

Expected plan counts:

PRIMARY = 2,800
REPEAT EXTRA = 1,120
TOTAL POSSIBLE = 3,920

If counts differ:
STOP and report the error.

==================================================
17. RENDERED-PROMPT AUDIT
==================================================

This is one of the most important outputs.

Create:

prompts/study1/audit/

Render representative complete prompts for BOTH GPT and Claude for:

- hidden, L=10
- true-visible, L=10
- each of the five manipulated values at L=10
- hidden, L=20
- true-visible, L=20
- each of the five manipulated values at L=20

Use at least 2 representative question IDs if practical:
- one where historical reported confidence is high;
- one where historical reported confidence is lower.

But DO NOT choose those examples by correctness or historical Stage-3 outcome.
If selecting by confidence for display purposes, state clearly that this is only prompt-audit presentation and does not alter the experimental sample.

Also create machine-generated diffs demonstrating that, within a given L and question:

manipulated-confidence prompts differ from one another ONLY in the displayed numeric confidence value.

If another difference appears:
STOP and report it.

For hidden vs visible, naturally the confidence sentence differs; document that separately.

==================================================
18. LOCAL TESTS
==================================================

Add Study-1 unit/integration tests sufficient to establish:

- deterministic question selection;
- correct hashes;
- correct grid by L;
- historical confidence loaded correctly;
- reported_confidence never changes when displayed_confidence changes;
- hidden condition contains no explicit confidence number;
- true-visible shows exactly reported_confidence;
- manipulated conditions show exactly assigned displayed_confidence;
- frozen answer is unchanged;
- action schema remains binary;
- primary call plan = 2,800;
- repeat-extra call plan = 1,120;
- no Study-1 writer points to v2.sqlite3;
- paid calls remain disabled by default;
- cap enforcement works.

Run the relevant tests locally.

No external/model API tests.

==================================================
19. DO NOT DO THESE THINGS
==================================================

DO NOT:

- make any model API call;
- run the 2,800 primary conditions;
- run repeats;
- run provenance controls;
- run qualitative-stakes controls;
- run contradiction controls;
- add open models;
- add datasets;
- perform interpretability;
- run Deep Research;
- modify Master Plan scientific content;
- modify Experimental Protocol scientific content;
- modify Checkpoint A;
- modify historical V2 data/code in place;
- interpret Study-1 results, because there are no Study-1 model results yet.

==================================================
20. RETURN ARTIFACTS
==================================================

Create:

to_gpt/001_study1_pilot_build/

Required:

1. report.md
2. changed_files.txt
3. run_manifest.json
4. study1_call_plan_summary.md
5. study1_primary_call_plan.csv
6. study1_repeat_call_plan.csv
7. selected_question_ids.json
8. selected_repeat_ids.json
9. schema_summary.md
10. prompt_audit.md
11. test_results.txt
12. config_snapshot.yaml
13. rendered_prompts/
14. prompt_diffs/

report.md must answer plainly:

- Was Study-1 infrastructure built successfully?
- Were exactly 100 primary question IDs frozen?
- Were exactly 20 repeat-subset IDs frozen?
- What are their hashes?
- Does the primary plan contain exactly 2,800 calls?
- Does the repeat-extra plan contain exactly 1,120?
- Are reported_confidence and displayed_confidence stored separately?
- Is historical V2 untouched?
- Are manipulated prompts identical except for the confidence number?
- Are hidden prompts free of the explicit confidence number?
- Is true-visible exactly using historical reported confidence?
- Did all local tests pass?
- Were ANY network/API calls made?
- Is the repository ready for a tiny real smoke test?

At the end print one of:

READY_FOR_TASK_002 = YES

or

READY_FOR_TASK_002 = NO

If NO:
state exact blocker(s).

==================================================
21. RUN MANIFEST REQUIREMENTS
==================================================

run_manifest.json must explicitly include:

task_id: 001_build_study1_pilot
status
git commit
api_calls_made
paid_calls_authorized: false
network_calls_authorized: false
historical_artifacts_modified
masterplan_modified
protocol_modified
selected_question_count
selected_question_hash
repeat_question_count
repeat_question_hash
planned_primary_calls
planned_repeat_extra_calls
planned_total_calls
models
L_values
confidence_grids
authority_condition
all_local_tests_passed
ready_for_task_002

==================================================
FINAL RULE
==================================================

When this task is complete:

STOP.

Do not run a smoke test.
Do not run Study 1.
Do not infer that READY_FOR_TASK_002 authorizes anything automatically.

GPT must inspect the returned artifacts first.
