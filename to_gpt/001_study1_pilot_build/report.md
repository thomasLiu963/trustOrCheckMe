# Task 001 report — Build Study 1 pilot infrastructure

Date: 2026-09-17  
Task: `001_build_study1_pilot`  
Status: DEVELOPMENT  
Paid/network model calls authorized: **NO**  
API calls made: **0**

---

## 1. What Cursor did

Read the D1 sources of truth, workflow contract, decision log, and Task 000 readiness report. Saved this task as `from_gpt/001_build_study1_pilot.md`.

Built new Study 1 infrastructure without modifying historical V2 data, V2 schemas, V2 prompts, Checkpoint A, the Master Plan, or the Experimental Protocol scientific text.

Froze:

- 100 primary question IDs from V2-B using seed `20260917`
- 20 repeat-subset IDs from those 100 using seed `20260918`

Loaded historical Stage-1 answers and Stage-2 `probability_correct` values read-only from `results/v2/raw/v2.sqlite3` (URI `mode=ro`). Mapped that field into Study 1 as `reported_confidence`. Built a 2,800-row primary call plan and a 1,120-row repeat-extra plan. Rendered prompt audits for two presentation IDs. Ran local tests.

Study 1 Stage-3 wording is the historical V2 AI-authority primary template from `src/v2_prompts.py:build_verification_prompt`.

## 2. What files changed

New Study 1 modules, config, frozen IDs, call plans, prompt audits, tests, this return folder, and an appended decision-log entry. `src/cli.py` gained Study 1 subcommands only (`study1-prepare`, `study1-plan`, `study1-run`). See `changed_files.txt`.

## 3. What Cursor did **not** do

- No GPT, Claude, Gemini, Grok, OpenAI, Anthropic, or other model/network calls
- Did not run the 2,800 primary conditions or 1,120 repeats
- Did not run a smoke test
- Did not implement provenance / qualitative-stakes / contradiction controls
- Did not add models or datasets
- Did not run interpretability or Deep Research
- Did not modify Master Plan or Experimental Protocol scientific content
- Did not modify Checkpoint A or historical V2 data/code in place (`src/v2_prompts.py` was imported, not edited)
- Did not interpret Study 1 behavioral results; there are none yet

## 4. Answers required by the task

**Was Study-1 infrastructure built successfully?**  
Yes.

**Were exactly 100 primary question IDs frozen?**  
Yes.

**Were exactly 20 repeat-subset IDs frozen?**  
Yes.

**What are their hashes?**  
Primary ID list SHA-256: `badd6938e5ded12c9dd62733426e1db26d9843bb6a2321a4e4c9eb7e3547fe94`  
Repeat ID list SHA-256: `45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38`  
Hash method: SHA-256 of `"\n".join(ordered_ids) + "\n"`.

**Does the primary plan contain exactly 2,800 calls?**  
Yes.

**Does the repeat-extra plan contain exactly 1,120?**  
Yes. Total possible later pilot: 3,920.

**Are reported_confidence and displayed_confidence stored separately?**  
Yes. Hidden rows store `displayed_confidence = null`. Historical V2 still uses `probability_correct` unchanged.

**Is historical V2 untouched?**  
Yes. `results/v2/raw/v2.sqlite3` mtime remains 2026-09-05. Historical checkpoint was opened read-only. Study 1 write path is `results/study1_causal_pilot/study1.sqlite3`.

**Are manipulated prompts identical except for the confidence number?**  
Yes. The builder stops if any other line changes. Example diffs are in `prompt_diffs/`.

**Are hidden prompts free of the explicit confidence number?**  
Yes. They contain no “The AI previously estimated a …” sentence.

**Is true-visible exactly using historical reported confidence?**  
Yes. Example: GPT on `mmlu_pro:test:10539` shows `0.999`, that item’s historical reported value.

**Did all local tests pass?**  
Yes. 69 passed, 0 failed (`test_results.txt`).

**Were ANY network/API calls made?**  
No. Count = 0.

**Is the repository ready for a tiny real smoke test?**  
The infrastructure is ready for GPT to *specify* a tiny smoke test. This task does not authorize that test.

## 5. Frozen scientific choices taken from this numbered task

- Authority: AI-system only
- Models: `openai_gpt56_sol` (`gpt-5.6-sol`), `anthropic_sonnet5` (`claude-sonnet-5`)
- Wording: historical V2 AI-authority visible/hidden template
- L=10 grid: 0.80, 0.88, 0.89, 0.91, 0.99
- L=20 grid: 0.90, 0.93, 0.94, 0.96, 0.99
- Displayed numbers are formatted with historical `.12g`, so 0.80 appears as `0.8` and 0.90 as `0.9`

## 6. Ambiguities / notes for GPT

1. Prompt-audit IDs `mmlu_pro:test:10539` (GPT q=0.999) and `mmlu_pro:test:774` (GPT q=0.03) are presentation-only. They were chosen from the frozen 100 using GPT reported confidence, not correctness or Stage-3 outcomes.
2. Shared `config/models.yaml` still lists Gemini and Grok for historical V2. Study 1 config and runner refuse to use them.
3. `study1-run --yes` still refuses execution until a later numbered task removes that Task-001 hard stop, even if `--max-calls` is supplied.

## 7. What GPT should inspect before Task 002

- Rendered prompts and manipulated diffs in this folder
- Frozen ID lists and hashes
- Call-plan CSVs (2,800 / 1,120)
- Schema split of `reported_confidence` vs `displayed_confidence`
- Paid-call gates (`dry-run` default; cap required; no implicit phases beyond `primary` / `repeats`)

If those look correct, Task 002 should explicitly authorize any smoke test with model roster and a small `max_calls` cap. Do not treat this report as that authorization.

READY_FOR_TASK_002 = YES
