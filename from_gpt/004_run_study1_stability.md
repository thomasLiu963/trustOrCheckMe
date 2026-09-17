TASK ID: 004_run_study1_stability
STATUS: EXPLORATORY
PAID API CALLS AUTHORIZED: YES — FROZEN REPEAT-EXTRA SUBSET ONLY

Before doing anything:

1. Read all current D1 source-of-truth documents and Tasks 001–003.
2. Read:
   to_gpt/003_study1_primary_results/report.md
3. Save this exact task as:
   from_gpt/004_run_study1_stability.md

This task authorizes ONLY the already-frozen 1,120 repeat-extra calls.

Do not run new scientific conditions.
Do not run provenance controls.
Do not run qualitative-stakes controls.
Do not run contradiction controls.
Do not expand the sample.

==================================================
PURPOSE
==================================================

Test whether the causal displayed-confidence effects observed in Task 003
are larger and more systematic than ordinary repeated-generation variation.

The frozen repeat design was selected BEFORE Task-003 outcomes were observed.

It consists of:

20 frozen question IDs
× 2 models
× 2 L values
× 7 conditions
× 2 additional repetitions

= exactly 1,120 repeat-extra scientific calls.

Combined with each cell's original primary observation, these yield
three total observations per repeated cell.

==================================================
0. ZERO-COST PARSE-REPAIR AUDIT BEFORE PAID CALLS
==================================================

Before making any new API call, audit Task-003 provider/parse behavior.

Task 003 reports that 52 distinct cells had a parse-repair attempt and one
Claude cell required a same-key rerun after truncated JSON.

Produce counts of parse-repair occurrence by:

- model;
- L;
- condition;
- displayed-confidence value.

Also report:

- whether repairs are concentrated in a particular model;
- whether repairs are concentrated in any scientific condition;
- whether repaired cells' final parsed action distribution differs obviously
  from non-repaired cells;
- whether any repair behavior could plausibly confound a confidence condition.

This is an engineering diagnostic, not a new hypothesis test.

If repair attempts are strongly condition-dependent in a way that could
distort Study-1 results:

STOP BEFORE PAID CALLS and return to GPT.

Otherwise proceed.

==================================================
1. FROZEN REPEAT SAMPLE
==================================================

Use exactly the 20 IDs already frozen under seed 20260918.

Verify their stored hash matches Task 001.

Do not reselect questions.

Do not substitute questions.

==================================================
2. EXACT REPEAT DESIGN
==================================================

For every frozen repeat question:

Models:
- openai_gpt56_sol
- anthropic_sonnet5

Authority:
- AI-system only

L:
- 10
- 20

Conditions:
- hidden
- true_confidence_visible
- all five already-frozen manipulated confidence conditions

Run exactly TWO additional independent generations per cell.

Do not rerun the original primary observation.

After completion, each repeated cell should have:

original primary observation
+ repeat 1
+ repeat 2

= three observations total.

==================================================
3. CALL LIMIT
==================================================

Authorized new scientific calls:

1,120 maximum.

No unused capacity may be applied to any other experiment.

Use reasonable retry/repair headroom for provider failures only.

If failures or repairs become unexpectedly frequent or condition-dependent,
pause rather than silently expanding retries.

==================================================
4. SCIENTIFIC INVARIANCE
==================================================

Do not change:

- prompts;
- formatting;
- endpoints;
- model settings;
- confidence values;
- question IDs;
- frozen answers;
- reported confidence;
- authority framing;
- costs;
- parser;
- action schema.

The only intended difference between repetitions is independent model
generation.

==================================================
5. DATA VALIDATION
==================================================

After execution verify:

- exactly 1,120 new repeat-extra scientific cells;
- no new primary conditions;
- exactly 20 repeat IDs;
- each original repeated cell now has three total observations;
- reported_confidence remains fixed;
- displayed_confidence remains correct;
- frozen answer remains fixed;
- historical V2 remains untouched;
- actions remain exactly VERIFY_FIRST / USE_UNVERIFIED.

==================================================
6. PRIMARY STABILITY ANALYSIS
==================================================

The central question is:

Does changing displayed confidence produce more systematic action change
than simply rerunning the same prompt?

For every model × L × condition:

report:

- fraction where all 3 generations agree;
- fraction with a 2–1 split;
- action entropy or equivalent simple disagreement measure;
- pairwise rerun disagreement.

==================================================
7. RE-ESTIMATE RESPONSE CURVES USING REPEATS
==================================================

For the 20-question repeated subset, estimate verification probability using
all three generations per cell.

Produce response curves separately for:

- GPT L=10
- GPT L=20
- Claude L=10
- Claude L=20

Compare qualitatively to Task-003 primary curves.

Do not pretend the 60 observations per condition are 60 independent questions.

Question remains the unit of resampling.

==================================================
8. THRESHOLD CONTRAST STABILITY
==================================================

Recompute the key contrasts on the repeated subset:

GPT and Claude:

L=10:
0.89 vs 0.91

L=20:
0.94 vs 0.96

Also extreme contrasts:

L=10:
0.80 vs 0.99

L=20:
0.90 vs 0.99

For each report:

- original primary-only contrast on these SAME 20 questions;
- repeat-aggregated contrast;
- bootstrap uncertainty by question;
- whether direction is consistent across the three generations.

==================================================
9. MANIPULATION EFFECT VS RERUN NOISE
==================================================

This is the most important analysis.

Quantify:

A. Typical action-change probability caused by repeating an IDENTICAL prompt.

versus

B. Action-change probability associated with changing displayed confidence
across the key threshold contrast.

Do this separately for GPT and Claude.

Present in plain language, e.g.:

"Repeating the same prompt changed GPT's decision X% of the time, whereas
crossing the displayed-confidence threshold changed it Y% of the time."

Use paired/question-aware analysis.

Do not inflate sample size by treating repetitions as independent questions.

==================================================
10. TRAJECTORY STABILITY
==================================================

Assess whether the qualitative manipulated-confidence trajectory survives
across repetitions.

For GPT especially:

does the near-step around 0.90 / 0.95 reproduce?

For Claude:

does the more gradual downward relationship reproduce?

Report exceptions rather than hiding them.

==================================================
11. STUDY-1 FINAL GATE
==================================================

After repeats, assess Study 1 as one of:

A. FAIL
B. AMBIGUOUS
C. PASS CAUSAL BEHAVIORAL PHENOMENON
D. PASS BUT ENGINEERING/DESIGN ISSUE REQUIRES RESOLUTION

A PASS means only:

Changing displayed confidence robustly and causally changes verification
behavior in this controlled exploratory setup, beyond ordinary repeated-prompt
variation.

It does NOT establish:

- self-provenance;
- practical agent generalization;
- a hidden-state mechanism;
- suppression of internal uncertainty;
- novelty relative to all prior literature.

==================================================
12. COST AND RUNTIME
==================================================

Report:

- new scientific calls;
- provider attempts;
- retries;
- repairs;
- input/output tokens;
- GPT cost;
- Claude cost;
- total cost;
- wall-clock runtime.

==================================================
13. OUTPUT FOLDER
==================================================

Create:

to_gpt/004_study1_stability_results/

Required:

1. report.md
2. changed_files.txt
3. run_manifest.json
4. preflight_parse_repair_audit.md
5. data_validation.md
6. repeat_results.csv
7. stability_summary.csv
8. threshold_stability.csv
9. rerun_noise_vs_manipulation.csv
10. cost_summary.md
11. analysis_method.md
12. figures/
13. analysis scripts

==================================================
14. REPORT STRUCTURE
==================================================

# Study 1 Repeated-Sampling Stability

## 1. Preflight parse-repair audit

## 2. Data validation

## 3. Bottom line in plain English

Directly answer:

- How often does an identical prompt naturally flip its decision?
- Does GPT's huge displayed-confidence effect survive reruns?
- Does Claude's smaller effect survive reruns?
- Is the manipulation effect larger than ordinary rerun noise?
- Does GPT still look like a near-threshold policy?
- Does Claude still look gradual?
- Were there meaningful condition-dependent parse issues?
- Does Study 1 pass its final exploratory gate?

## 4. GPT stability

## 5. Claude stability

## 6. Threshold contrasts

## 7. Manipulation effect versus rerun noise

## 8. Response-curve stability

## 9. What Study 1 now DOES establish

## 10. What Study 1 still DOES NOT establish

## 11. Final Study-1 gate

## 12. Exact numbers GPT should know

## 13. Cost and runtime

==================================================
15. AFTER COMPLETION
==================================================

STOP.

Do not run micro-controls.

Do not run Deep Research.

Do not design the confirmatory study.

Print:

READY_FOR_GPT_REVIEW = YES

and return:

to_gpt/004_study1_stability_results/report.md
