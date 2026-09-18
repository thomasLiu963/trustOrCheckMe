# TASK 007 — Full Qualitative Expansion, Four-Model Pilot, and Paper-Direction Decision Packet

## Purpose

We are now close to revising `paperDirection.txt`, but one key scientific question must be answered first.

Task 006 established on 20 frozen questions that displayed confidence still strongly changes verification **without numerical L/C arithmetic**. GPT also moved off its 0%/100% corners, making item-sensitive routing measurable.

Task 005B/005C clarified that much of the apparent “extra error signal” in natural verification is actually explained by **observable item difficulty**. Therefore the paper should not assume a mysterious hidden correctness signal.

The next question is cleaner:

> When a displayed confidence score changes verification, does it merely change the overall checking rate while preserving item-sensitive/difficulty-sensitive prioritization, or does it change which questions receive scrutiny?

Task 007 should answer that on the full 100-question exploratory set, while simultaneously accelerating model generalization and the later prospective study.

This is still **exploratory development**, not the final confirmatory study.

---

# NON-NEGOTIABLE RULES

1. **Do not modify `paperDirection.txt`.**
2. Save this exact task to:
   `from_gpt/007_full_qualitative_four_model_decision_packet.md`
3. Do not modify historical V2, Study 1, Tasks 004/005/005B/005C/006 data.
4. Reuse the **exact frozen qualitative prompts from Task 006**. No wording changes.
5. Reuse all existing Task-006 GPT/Claude cells. Do **not** rerun the original 20 questions for GPT/Claude.
6. New paid scientific call cap:
   - Lane A: 1,280
   - Lane B: up to 320
   - Total maximum: **1,600**
7. Do not auto-scale to a fresh confirmatory sample.
8. Do not run q2 on Gemini/Grok.
9. Do not run interpretability/probes/activation steering.
10. Do not invent a reasoning setting.
11. Do not infer internal-state suppression from behavior.
12. Do not call cross-model difficulty a deployable production signal; it is an evaluation-side empirical item-difficulty proxy.
13. Question identity is the resampling/statistical unit.
14. Parallel workers must use separate databases/result paths.
15. No result-driven prompt changes, score-grid changes, model substitutions, or question substitutions.

---

# PARALLEL EXECUTION PLAN

Run these lanes as independently as possible:

## Lane A — COMPLETE GPT/CLAUDE QUALITATIVE GRID
Paid: **1,280 new calls**

Use the remaining 80 of the original frozen Study-1 100 questions.

## Lane B — GEMINI/GROK QUALITATIVE PILOT
Paid: **up to 320 new calls**

Use the same frozen 20-question pilot subset and exact Task-006 prompts.

Lane A and Lane B may run concurrently if provider/rate-limit architecture permits. Use separate databases.

## Lane C — SCORE vs ITEM-DIFFICULTY / PRIORITIZATION ANALYSIS
0 calls.

This starts after enough Lane-A data exist and then runs on the complete 100-question GPT/Claude qualitative grid. It may also report descriptive 20-question Gemini/Grok results.

## Lane D — PROSPECTIVE FOUR-MODEL STUDY PREP
0 calls.

Prepare—not execute—the likely fresh four-model confirmatory infrastructure.

## Lane E — PAPER-DIRECTION DECISION PACKET
0 calls.

After A–D finish, create a structured evidence ledger and proposed revision outline for GPT review. Do **not** edit `paperDirection.txt`.

---

# READ FIRST

At minimum read:

- `paperDirection.txt` — READ ONLY
- `docs/D1_MASTER_RESEARCH_PLAN.md`
- `docs/D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md`
- `docs/D1_DECISION_LOG.md`
- Task 003 report
- Task 004 report
- Task 005 report
- Task 005B report
- Task 005C report
- Task 006 report
- Task 006 prompt-freeze and prompt-diff artifacts
- historical V2 data/config
- Study-1 100 frozen IDs
- current adapters for GPT, Claude, Gemini, Grok

Before paid calls, print a preflight summary with:
- Task-006 prompt hashes
- Task-006 20 IDs/hash
- full Study-1 100 IDs/hash
- exact 80 remaining IDs for Lane A
- exact endpoint/model IDs for all attempted models
- Lane-A cap 1,280
- Lane-B cap 320
- total cap 1,600
- confirmation Task-006 existing GPT/Claude cells will be reused
- confirmation `paperDirection.txt` remains untouched

---

# LANE A — COMPLETE GPT/CLAUDE QUALITATIVE GRID

## A.1 Sample

Use the exact 100-question Study-1 primary set from Task 001/003.

Expected full-100 hash:

`badd6938e5ded12c9dd62733426e1db26d9843bb6a2321a4e4c9eb7e3547fe94`

Task 006 already ran the frozen 20-question subset with hash:

`45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38`

Compute the exact 80-question complement.

If hashes/counts do not reconcile, STOP Lane A.

---

## A.2 Models

Exactly the same Task-006 endpoints/settings:

- GPT: `gpt-5.6-sol`, `reasoning.effort=none`
- Claude: `claude-sonnet-5`, thinking disabled

Do not change settings.

---

## A.3 Conditions

Reuse exactly the Task-006 frozen qualitative design:

### Stakes families
- moderate
- stronger

### Score conditions
- hidden
- displayed 0.70
- displayed 0.90
- displayed 0.99

The rendered prompt text and numeric formatting must be byte-compatible with Task 006.

Do not rewrite any phrase.

---

## A.4 New call count

80 remaining questions
× 2 models
× 2 stakes families
× 4 score conditions
= **1,280 new scientific calls**

After merge with Task 006:
- total qualitative GPT/Claude cells should be:
  `100 × 2 × 2 × 4 = 1,600`

Do not rerun the original 320 Task-006 cells.

Store new calls separately, then create a merged read-only analysis view/table.

---

## A.5 Primary replication summaries

For each model × stakes family:

Report verification rates for:
- hidden
- 0.70
- 0.90
- 0.99

Report paired score effects:
- 0.70 → 0.90
- 0.90 → 0.99
- 0.70 → 0.99

Report stronger-minus-moderate stakes effects at:
- hidden
- 0.70
- 0.90
- 0.99

Question-bootstrap 95% CIs, 5,000 resamples.

Also report whether the original 20-question pilot direction replicated on the remaining 80 questions.

This is still exploratory because the full 100 set was used during project development.

---

# LANE B — GEMINI/GROK QUALITATIVE PILOT

## Scientific purpose

The original V2 had four models. Before the later fresh prospective study, cheaply learn whether the arithmetic-free displayed-score phenomenon is obviously GPT/Claude-specific or appears across the other two historical model families.

This is **exploratory model-selection/generalization**, not confirmation.

---

## B.1 Preconditions

For each of Gemini and Grok separately:

Proceed only if:
- current adapter is available;
- exact configured endpoint exists;
- historical frozen Stage-1 answer exists for all 20 IDs;
- exact Task-006 prompt can be rendered;
- structured action parser works.

If a model fails preflight, mark that model BLOCKED and continue the other model.

Do not substitute another endpoint.

---

## B.2 Sample

Use exactly the Task-006 frozen 20 IDs.

For each available model:

20 questions
× 2 stakes families
× 4 score conditions
= **160 calls/model**

Both models total = **320 calls**.

One generation per cell.

No repeats.

---

## B.3 Prompt

Use the exact frozen Task-006 prompt bodies and confidence formatting.

The only model-specific scientific content should be:
- that model's frozen historical Stage-1 answer.

Do not insert its historical q1 unless the condition explicitly displays the manipulated qualitative-study value.

---

## B.4 Analysis

For each model:
- verification rate by stakes × score
- 0.70→0.90, 0.90→0.99, 0.70→0.99 paired changes
- stronger-minus-moderate effects
- saturation diagnostic
- raw wrong-vs-correct routing where identifiable
- descriptive empirical-difficulty sensitivity using the other three historical models

n=20: no paper-ready claims.

Classify:
- `QUALITATIVE_SCORE_RESPONSE_STRONG`
- `QUALITATIVE_SCORE_RESPONSE_MODERATE`
- `QUALITATIVE_SCORE_RESPONSE_WEAK_OR_NULL`
- `BLOCKED`

---

# LANE C — SCORE CONTROL vs ITEM-DIFFICULTY / PRIORITIZATION

## Core question

Do not assume “crowding out.”

Test competing possibilities:

### Possibility 1 — Global operating-point shift
Displayed confidence mostly changes **how many** items are verified, while the model continues to prioritize the harder/riskier-looking items.

### Possibility 2 — Attenuated item sensitivity
As displayed confidence rises, the final verification action becomes less dependent on empirical item difficulty.

### Possibility 3 — Enhanced prioritization
Higher displayed confidence reduces total checks but concentrates the remaining checks more strongly on difficult items.

The data should decide.

---

## C.1 Difficulty proxy

For each target model/question, define leakage-safe empirical item difficulty from the correctness of the OTHER THREE historical V2 models.

For GPT:
- Claude + Gemini + Grok correctness

For Claude:
- GPT + Gemini + Grok correctness

Primary simple variable:
- `other_models_correct ∈ {0,1,2,3}`
- lower = empirically harder

Also retain:
- category
- question length
- choice count

Do not use target-model correctness inside the target-model difficulty proxy.

Call this an **empirical cross-model difficulty proxy**, not “true difficulty.”

---

## C.2 Difficulty sensitivity at each condition

For each model × stakes family × score condition (including hidden), report:

- verification rate by `other_models_correct` bin
- mean `other_models_correct` among VERIFY vs USE
- logistic association of VERIFY with difficulty
- AUROC of difficulty for discriminating VERIFY vs USE where defined
- question-bootstrap uncertainty

Primary intuitive quantity:

> How much more likely is a hard item to be verified than an easy item under this score condition?

Define hard/easy transparently, e.g.:
- hard: 0–1 of 3 other models correct
- medium: 2/3
- easy: 3/3

If bins are sparse, merge only by a predeclared rule and report counts.

---

## C.3 Score × difficulty interaction

For each model and stakes family, fit an exploratory repeated-measures model:

`VERIFY ~ displayed_score_condition + difficulty + displayed_score_condition × difficulty`

Use the visible conditions 0.70/0.90/0.99.

Use a method that accounts for repeated observations by question:
- clustered bootstrap by question and/or
- GEE / mixed model / cluster-robust logistic regression

Do not treat rows as independent.

The key estimand is whether the effect of difficulty on VERIFY changes across displayed scores.

Report:
- interaction estimates
- uncertainty
- marginal verification-vs-difficulty curves by score

If model fitting is unstable, rely on stratified paired summaries rather than forcing coefficients.

---

## C.4 Hidden vs displayed comparison

For each stakes family compare the empirical difficulty sensitivity of:
- hidden
- 0.70
- 0.90
- 0.99

Do not claim hidden and visible are byte-identical interventions; hidden lacks the confidence sentence.

This is a policy-composition comparison, not a pure single-token causal contrast.

---

## C.5 Switch-set analysis — IMPORTANT

Because score changes overall verification rate, do not infer “lost difficulty sensitivity” merely from smaller raw percentage-point gaps.

For paired score transitions:

- 0.70 → 0.90
- 0.90 → 0.99

classify each question:
- VERIFY at both
- switches VERIFY → USE
- USE at both
- USE → VERIFY

For the questions that were VERIFY at the lower score, compare empirical difficulty of:
- those that remain VERIFY
- those that switch to USE

Key interpretation:

> If the model preferentially keeps the hardest questions in VERIFY as confidence rises, score conditioning may be changing the verification budget while preserving or sharpening item prioritization.

This analysis is central. Do not skip it.

Report target correctness descriptively too, but difficulty is the main predeclared explanatory axis.

---

## C.6 Error-catching consequence

Within each model/stakes/score condition report:
- overall verification rate
- fraction of actual target-model errors verified
- fraction of correct answers verified

Then compare conditions at similar realized verification rates where any naturally occur.

Do not claim a score is a “better router” merely because it verifies more.

For the final paper, matched-budget routing will need a fresh study; here this is exploratory consequence analysis only.

---

## C.7 Decision bucket

For GPT and Claude separately, classify:

### SCORE_SHIFTS_BUDGET_PRESERVES_DIFFICULTY_PRIORITY
Score strongly changes verification rate, but hard items remain preferentially selected; switch-set analysis shows difficult items retained.

### SCORE_ATTENUATES_DIFFICULTY_SENSITIVITY
Score strongly changes verification and empirical difficulty has materially less influence at higher displayed scores, beyond simple saturation concerns.

### SCORE_SHARPENS_DIFFICULTY_PRIORITY
Higher score reduces total checking but remaining verification becomes more concentrated on empirically difficult items.

### MIXED_OR_MODEL_SPECIFIC
No single pattern across stakes/scores.

### UNIDENTIFIABLE
Saturation/sparsity prevents meaningful analysis.

Do not force a preferred paper narrative.

---

# LANE D — PROSPECTIVE FOUR-MODEL STUDY PREP

0 API calls.

Use the unseen 11,532-item pool prepared in Task 006.

Prepare infrastructure for a likely fresh study involving:
- GPT
- Claude
- Gemini
- Grok

Do **not** freeze final sample size or conditions yet.

---

## D.1 Proposed common fresh pipeline

Prepare a pipeline that can, after GPT review:

1. generate fresh Stage-1 answers for all selected models on the same unseen questions;
2. score correctness from benchmark labels;
3. elicit q1;
4. optionally elicit q2 only if retained as a baseline;
5. run hidden qualitative verification;
6. run selected manipulated displayed-score conditions;
7. compute leave-one-model-out empirical difficulty for evaluation;
8. compute matched-budget routing.

No calls now.

---

## D.2 Candidate efficient condition sets

Prepare cost/cell-count estimates for at least:

### Design D1 — lean
- one stakes family
- hidden
- 0.70
- 0.90
- 0.99

### Design D2 — visibility bridge
- one stakes family
- hidden
- true-q visible
- 0.70
- 0.90
- 0.99

### Design D3 — stakes replication
- moderate + stronger
- hidden + 0.70 + 0.90 + 0.99

For N=200/400/600 and 4 models, give exact expected Stage-3 cell counts.

Include Stage-1/q1 costs separately.

Do not select a winner automatically.

---

## D.3 Routing baselines to prepare

Prepare analysis scaffolding for fair deployment-oriented baselines:

- raw q1
- cross-fitted calibrated q1
- q1+q2 if q2 is retained
- one hidden qualitative verification judgment
- simple combined q1 + one hidden judgment

Keep multi-cell historical hidden aggregate out of the fair primary comparison; it can remain an oracle/upper-bound diagnostic.

---

# LANE E — PAPER-DIRECTION DECISION PACKET

## Purpose

We want to revise `paperDirection.txt` soon after Task 007, then send the revised direction to Claude for adversarial critique.

Do NOT edit `paperDirection.txt` in this task.

Create:

`to_gpt/007_full_qualitative_four_model_decision_packet/paper_direction_decision_packet.md`

It must make revision fast and evidence-grounded.

---

## E.1 Evidence ledger

Create a table with one row per important candidate claim:

Columns:
- claim
- current evidence
- strongest supporting task/result
- status:
  - `SUPPORTED_EXPLORATORY`
  - `SUPPORTED_STRONGLY_BUT_NOT_CONFIRMATORY`
  - `WEAK`
  - `FALSIFIED_OR_LARGELY_EXPLAINED`
  - `UNTESTED`
- allowed wording
- wording to avoid

Must include at least:

1. Displayed confidence changes verification.
2. GPT numerical threshold cliff.
3. Score effect survives without explicit L/C arithmetic.
4. Verification contains information beyond q1.
5. Verification contains information beyond q1+q2.
6. Verification contains information beyond q1+q2+difficulty.
7. Claude fixed-score wrong-vs-correct gap.
8. Difficulty explanation of Claude gap.
9. Natural/hidden verification tracks empirical item difficulty.
10. Score conditioning attenuates/preserves/sharpens difficulty sensitivity — fill from Task 007.
11. Routing gain from a single hidden judgment.
12. Routing gain from multi-elicitation hidden aggregate.
13. Cross-model generality.
14. Hidden-state mechanism.
15. Practical matched-budget payoff.

---

## E.2 Candidate paper spine(s)

Based strictly on Task-007 outcomes, draft up to **three** possible paper spines ranked only by evidentiary fit, NOT by hype:

### Spine A
If score conditioning attenuates item sensitivity.

### Spine B
If score conditioning mostly shifts the verification budget while preserving difficulty prioritization.

### Spine C
If effects are strongly model-specific/mixed.

For each spine write:
- one-paragraph layman explanation
- central research question
- 3–5 contribution bullets
- likely Figure 1–4 sequence
- key reviewer attack
- what fresh confirmatory study must establish
- exact claims that must NOT be made

Do not score/rank political content; irrelevant here.

---

## E.3 Proposed edits to `paperDirection.txt`

Create a **diff-style recommendation only**, not an edit.

Sections:
- KEEP
- REWRITE
- DELETE/DEMOTE
- ADD

Quote section headings or short phrases from `paperDirection.txt` where necessary, but do not rewrite the actual file.

This will be reviewed by GPT before any bookkeeping update.

---

# OUTPUT STRUCTURE

Create:

`to_gpt/007_full_qualitative_four_model_decision_packet/`

Root:
- `report.md`
- `run_manifest.json`
- `changed_files.txt`
- `cost_summary.md`
- `parallel_execution_log.md`
- `paper_direction_decision_packet.md`

Lane A:
- `lane_A_full_qualitative/validation.md`
- `lane_A_full_qualitative/results_new80.csv`
- `lane_A_full_qualitative/results_merged100.csv`
- `lane_A_full_qualitative/score_response.csv`
- `lane_A_full_qualitative/stakes_effects.csv`
- `lane_A_full_qualitative/pilot_vs_remaining80.csv`
- `lane_A_full_qualitative/call_manifest.csv`
- `lane_A_full_qualitative/failures.csv`

Lane B:
- `lane_B_gemini_grok/validation.md`
- `lane_B_gemini_grok/results.csv`
- `lane_B_gemini_grok/score_response.csv`
- `lane_B_gemini_grok/saturation.csv`
- `lane_B_gemini_grok/difficulty_descriptive.csv`
- `lane_B_gemini_grok/call_manifest.csv`
- `lane_B_gemini_grok/failures.csv`

Lane C:
- `lane_C_score_vs_difficulty/difficulty_features.csv`
- `lane_C_score_vs_difficulty/difficulty_by_condition.csv`
- `lane_C_score_vs_difficulty/interaction_models.csv`
- `lane_C_score_vs_difficulty/marginal_curves.csv`
- `lane_C_score_vs_difficulty/switch_set_analysis.csv`
- `lane_C_score_vs_difficulty/error_catching_by_condition.csv`
- `lane_C_score_vs_difficulty/model_decision_buckets.md`
- figures/
- exact analysis scripts

Lane D:
- `lane_D_prospective_prep/four_model_pipeline.md`
- `lane_D_prospective_prep/design_cost_matrix.csv`
- `lane_D_prospective_prep/routing_analysis_plan.md`
- configs/scaffolding only, no calls

Lane E:
- root `paper_direction_decision_packet.md`

---

# ROOT REPORT REQUIRED CONTENT

`report.md` must include:

1. Plain-English executive bottom line
2. Exact paid scientific calls
3. GPT full-100 qualitative replication
4. Claude full-100 qualitative replication
5. Gemini/Grok 20-question qualitative pilot
6. Does score conditioning preserve, attenuate, or sharpen empirical difficulty prioritization?
7. Switch-set result
8. Error-catching consequences
9. Model-specific decision buckets
10. Prospective four-model study readiness
11. Best-supported current paper spine
12. Which prior candidate story was weakened/killed
13. What is now ready to go into `paperDirection.txt`
14. What still needs fresh confirmation
15. Exact cost/runtime/retries
16. `READY_FOR_GPT_REVIEW = YES`

---

# GO / STOP LOGIC

## GO toward paperDirection revision
If Task 007 yields a coherent empirical pattern that can be stated without hidden-state claims—for example:
- score responsiveness robustly replicates on N=100;
- difficulty prioritization is clearly preserved, attenuated, sharpened, or model-specific;
- four-model pilot does not reveal an obvious fatal contradiction.

Then recommend revising `paperDirection.txt` immediately after GPT review.

## HOLD revision
If:
- full-100 qualitative effect collapses relative to the 20-question pilot;
- interaction/difficulty pattern is too unstable to describe honestly;
- major data validation issue appears.

Then recommend resolving that issue first.

Do not make this decision by paper aesthetics; make it from evidence.

---

# FINAL REMINDER

We are not trying to force the old “hidden rich uncertainty gets suppressed” story.

The likely final paper must follow the data.

The current candidate scientific framing is:

> A displayed uncertainty score can causally reshape selective oversight even without explicit arithmetic. Natural verification behavior strongly tracks empirical item difficulty. The key question is whether score conditioning merely changes the checking budget while preserving that item-sensitive prioritization, or changes which information controls scrutiny.

Task 007 should tell us which version is true.

After Task 007, the bookkeeping goal is to revise `paperDirection.txt` promptly, then send that revised direction plus the evidence ledger to Claude for an adversarial narrative/novelty critique before launching the fresh confirmatory study.
