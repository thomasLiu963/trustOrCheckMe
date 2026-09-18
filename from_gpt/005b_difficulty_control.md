# TASK 005B — Zero-Cost Difficulty-Control Audit

## Purpose

Task 005 found an exploratory Claude result: at fixed displayed confidence values, Claude still verifies wrong answers more often than correct answers. Before treating this as evidence that Claude's verification judgment contains correctness-relevant information beyond the displayed score, test the simpler explanation:

> Claude may simply be checking questions that are objectively harder, and harder questions are more often wrong.

This task is analysis only. Make 0 API calls. Do not modify paperDirection.txt, historical V2, Study 1, or Task 005 data.

Save this exact instruction to:
from_gpt/005b_difficulty_control.md

Create outputs under:
to_gpt/005b_difficulty_control/

## Inputs

Read:
- to_gpt/005_parallel_diagnostic_sprint/report.md
- Task 005 Lane A analysis outputs/scripts
- historical V2-B 500-question data for all four historical models: GPT, Claude, Gemini, Grok
- Study 1 100-question IDs and Claude manipulated-score actions
- Stage-1 correctness for all models
- question metadata available in the canonical dataset

Do not make any paid calls.

## Core scientific question

Within a fixed displayed confidence value, does Claude's VERIFY/USE action still predict Claude's actual correctness after controlling for observable question difficulty?

This is exploratory. The goal is to determine whether "difficulty" is a sufficient explanation of the Task-005 Claude signal.

## Difficulty features

Use only pre-existing, leakage-safe item features.

1. Other-model empirical difficulty
   - For each question, estimate difficulty from the correctness of the OTHER three historical models.
   - When predicting Claude correctness/action, do NOT include Claude's own correctness in the difficulty estimate.
   - Use fraction/count of GPT/Gemini/Grok correct.

2. Historical four-model difficulty, descriptive only
   - May be reported descriptively, but do NOT use as the primary adjusted predictor because it leaks Claude correctness.

3. Category / subject
   - Use canonical MMLU-Pro category if available.

4. Choice count
   - Include only if it actually varies. If constant, report and drop it.

5. Question length
   - Character count and/or token count of question stem.

6. Answer / option text length
   - Frozen answer text length if meaningful.
   - If outputs are only option letters, omit and explain.

Do not add post-treatment features derived from Study-1 actions or displayed confidence.

## Analyses

### A. Reproduce the raw fixed-score result

For Claude only, every L × manipulated displayed score report:
- n wrong / n correct
- P(VERIFY | wrong)
- P(VERIFY | correct)
- raw difference
- question-bootstrap 95% CI

This should match Task 005.

### B. Does difficulty predict Claude VERIFY?

At each fixed displayed score, compare observable difficulty between VERIFY and USE questions.

Report:
- mean/median other-model difficulty
- category composition
- question-length summaries

Descriptive only.

### C. Adjusted correctness model

Primary exploratory model should predict Claude correctness/error from:
- displayed-score stratum (or analyze within stratum)
- Claude VERIFY/USE action
- other-model empirical difficulty
- category
- question length

Because each question repeats across displayed-score conditions, do not treat repeated rows as independent questions.

Compare grouped-CV models:
1. difficulty only
2. difficulty + Claude action

Group all rows from the same question into the same fold.

Report:
- AUROC
- log loss
- incremental change from adding action
- question-bootstrap uncertainty

If separation/small cells make a model unstable, report it.

### D. Matched / stratified sanity check

Stratify questions by other-model difficulty (0/3, 1/3, 2/3, 3/3 other models correct; merge sparse bins if necessary).

Within difficulty strata and fixed displayed score, compare wrong-vs-correct VERIFY tendency where estimable.

A carefully implemented nearest-neighbor match on difficulty + category + question length is also acceptable as a secondary check.

Do not force estimates in sparse strata.

### E. Repeat-subset robustness

On the frozen 20-question, 3-generation subset:
- use mean VERIFY propensity per question
- check whether the adjusted direction remains consistent after difficulty control
- treat as qualitative robustness only because n=20 is small

## Interpretation buckets

Conclude with exactly one:

DIFFICULTY_DOES_NOT_ABSORB_SIGNAL
Claude action still adds a material correctness signal after observable difficulty controls.

DIFFICULTY_PARTIALLY_ABSORBS_SIGNAL
Claude action still adds some signal, but meaningfully less than before adjustment.

DIFFICULTY_LARGELY_EXPLAINS_SIGNAL
Claude action adds little/no correctness information after difficulty controls.

TOO_UNDERPOWERED_TO_TELL
The 100-question pilot is too sparse for a reliable adjusted conclusion.

Important:
- Even DIFFICULTY_DOES_NOT_ABSORB_SIGNAL does NOT prove an internal hidden-state mechanism.
- Even DIFFICULTY_LARGELY_EXPLAINS_SIGNAL does NOT make the routing signal useless; difficulty itself may be a practical routing cue.
- This task is exploratory and cannot replace fresh prospective confirmation.

## Required outputs

Create:
to_gpt/005b_difficulty_control/

with:
- report.md
- raw_fixed_score_results.csv
- difficulty_features.csv
- difficulty_vs_action.csv
- adjusted_grouped_cv.csv
- stratified_or_matched_analysis.csv
- repeat_subset_check.csv
- changed_files.txt
- exact analysis script(s)

report.md should contain:
1. Plain-English bottom line
2. Validation
3. Raw Task-005 replication
4. How strongly observable difficulty predicts Claude action
5. Whether Claude action adds correctness information after difficulty control
6. Repeat-subset robustness
7. What this DOES establish
8. What it DOES NOT establish
9. Interpretation bucket
10. Recommendation only — do not launch new experiments
11. READY_FOR_GPT_REVIEW = YES

Make 0 API calls.
