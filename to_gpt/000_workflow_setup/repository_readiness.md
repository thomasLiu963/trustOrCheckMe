# Repository readiness for Study 1

Audit date: 2026-09-17  
Scope: inspect historical components needed to **build** the Study 1 causal-pilot pipeline.  
No files in this audit were modified. No model/API calls were made.

This is a readiness map for Task 001 (build and audit infrastructure). It is **not** authorization to implement the confidence grid, select the 100 questions, change Stage-3 prompts, or run any smoke test / paid calls.

---

## 1. Historical data that exists and should be reused read-only

| Component | Path | What is there | Reuse |
|---|---|---|---|
| V2 raw checkpoint | `results/v2/raw/v2.sqlite3` (194 MB; local, gitignored) | SQLite `requests` / `attempts` / `manifests`. Counts: 2,000 Stage-1 answers, 2,000 Stage-2 confidence records, 38,400 Stage-3 verification rows (32,000 primary `v2_owner_match_v1` + 6,400 paraphrase). All listed stages are `success`. | **Read-only source** of frozen answers and historical reported confidence. Never write Study 1 rows into this file. |
| V2-B question list | `data/samples/mmlu_pro_v2b.jsonl` | 500 MMLU-Pro items; first ID `mmlu_pro:test:79`. | **Read-only sample universe** for the future deterministic 100-ID draw. Do not rewrite. |
| V2-B manifest | `data/manifests/mmlu_pro_v2b_manifest.json` | Created 2026-09-04; category counts; dataset revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`. | Reuse for ID/order provenance. |
| V2-A / robustness samples | `data/samples/mmlu_pro_v2a.jsonl`, `data/samples/mmlu_pro_v2_robustness.jsonl` | Historical subsets. | Not the Study 1 sample source. Protocol says draw from V2-B. |
| V1 pilot sqlite | `results/raw/pilot.sqlite3` | Historical V1 checkpoint. | Not needed for Study 1 primary path. |
| V2 analysis pointer | `paper_outputs/v2/analysis_manifest.json` | Records checkpoint path, 32,000 decisions, seed `20260904`. | Evidence pointer; do not overwrite. |
| V2 configs | `config/experiment_v2.yaml`, `config/models.yaml` | Frozen V2 grid, costs, `C = 1`, owners, visibility, checkpoint path, GPT/Claude/Gemini/Grok endpoints. | Copy ideas into a **new** Study 1 config. Do not edit these files to run Study 1. |
| V2 run manifests inside sqlite | `manifests.manifest_json` | Includes model configs, git commit, sample hashes, token usage. | Provenance for historical endpoints. |

Checkpoint A independently validated: no missing primary cells, no duplicate primary cells, frozen answer and `probability_correct` consistent across Stage 1/2/3.

---

## 2. Historical Stage-3 prompt / template code

| Path | Role |
|---|---|
| **`src/v2_prompts.py`** | Canonical V2 Stage-3 builder. `build_verification_prompt(...)` is the matched owner × visibility template. Visible branch inserts historical `probability_correct` as: “The AI previously estimated a {p} probability that this frozen answer is correct.” Hidden branch inserts no number. Also contains `parse_verification_response` and `build_verification_repair_prompt` (one standardized JSON repair). |
| `src/prompts.py` | **V1** Stage-3 / trust prompt (`STAGE_3_TEMPLATE`, `build_stage_3_prompt`). Different wording and `RELY`/`VERIFY` schema. Do **not** use this as the Study 1 template. |
| `results/v2/prompt_diffs/example_0{1-5}.diff` | Historical prompt-inspection diffs. |
| `protocol_packet/src/v2_prompts.py` | Snapshot copy of V2 prompts. Historical; do not treat as the live implementation. |

Study 1 should start from `src/v2_prompts.py` AI-authority visible/hidden templates, then create **new** Study 1 template files so historical V2 wording remains frozen.

---

## 3. Provider / API wrappers used for GPT and Claude

| Path | Role |
|---|---|
| **`src/model_adapters.py`** | Live adapters. `OpenAIAdapter` (`gpt-5.6-sol`, Responses API, `reasoning.effort = none`, strict JSON schema). `AnthropicAdapter` (`claude-sonnet-5`, Messages API, thinking disabled, JSON schema). Also Gemini and xAI, which Study 1 should not call. |
| `create_adapter(...)` | Factory frozen to those exact API model IDs. |
| `ModelAdapter.generate(..., allow_paid=False)` | Refuses paid calls unless `allow_paid=True`. Dry-run path is `prepare_request()`. |
| `config/models.yaml` | Pricing, token caps (GPT/Claude `max_output_tokens: 64`), API styles. |
| CLI paid gate | `src/cli.py` passes `allow_paid=bool(args.yes)`. `src/v2_runner.py` raises unless dry-run or `allow_paid`. |

Reusable for Study 1 GPT and Claude **if** Study 1 keeps the same frozen endpoints. Do not silently change reasoning mode. Gemini/Grok adapters exist but are out of the Study 1 roster.

---

## 4. Parsing / structured-output logic

| Path | Role |
|---|---|
| `src/model_adapters.py` → `output_schema("verification")` | Provider JSON schema: `action` ∈ `{USE_UNVERIFIED, VERIFY_FIRST}`. |
| `src/v2_prompts.py` → `parse_verification_response` | Pydantic `VerificationPayload.model_validate_json`. |
| `src/schemas.py` → `VerificationPayload`, `VerificationAction` | Strict two-action payload. |
| `src/v2_runner.py` `_execute` | Transport retries + at most one repair via `build_verification_repair_prompt`; records `attempt_number` / parse errors in `attempts`. |
| `src/v2_scoring.py` | Historical scoring of `USE_UNVERIFIED` / `VERIFY_FIRST`; confidence-threshold policy uses `probability_correct`. |

This stack is reusable. Study 1 should keep the same two-action JSON schema across all displayed-confidence conditions.

---

## 5. Checkpoint A outputs

All present under `to_gpt/checkpoint_A/` (do not overwrite):

- `checkpoint_A_report.md`
- `analyze_checkpoint_A.py`
- `data_validation.json`
- `run_metadata.json`
- `checkpoint_A_cell_results.csv`
- `checkpoint_A_model_summary.csv`
- `checkpoint_A_pooled_catch_summaries.csv`
- `checkpoint_A_calibration_results.csv`
- `checkpoint_A_residual_results.csv`
- `checkpoint_A_residual_bins.csv`
- `part4_descriptive_correlation.json`
- `figures/`

These are analysis artifacts of frozen V2-B. They are evidence for why Study 1 exists. They are not Study 1 implementation.

---

## 6. Relevant V2 configs, manifests, paper artifacts

| Path | Role |
|---|---|
| `config/experiment_v2.yaml` | Frozen V2 experiment; checkpoint `results/v2/raw/v2.sqlite3`. |
| `config/models.yaml` | Four-family roster; Study 1 needs only GPT + Claude. |
| `EXPERIMENT_V2.md`, `protocol_packet/EXPERIMENT_V2.md` | Original frozen V2 protocol. Immutable historical record. |
| `paper_outputs/v2/` | Workshop analysis tables, figures, calibration splits. Immutable. |
| `paper/` including `paper/85_Trust_Me_or_Check_Me_Self_R.pdf` | Workshop paper artifacts. Immutable. |
| `src/checkpointing.py` | Deterministic request keys + SQLite store. Reuse the **class** against a **new** sqlite path. |
| `src/v2_datasets.py` | Load frozen V2-B JSONL. |
| `src/cli.py` | `inspect-v2-prompts`, `plan-v2`, `run-v2-decisions` (historical V2 only). |
| `tests/test_v2.py`, `tests/test_model_adapters.py`, `tests/test_runner.py` | Historical tests; useful patterns, not Study 1 coverage. |

---

## 7. What currently makes Stage-3 calls

Live paid Stage-3 path:

1. CLI: `trust-or-check-me run-v2-decisions --yes` (or robustness equivalent) in `src/cli.py`.
2. Runner: `V2VerificationRunner.run(...)` in `src/v2_runner.py`.
3. Prompt: `build_verification_prompt` in `src/v2_prompts.py`.
4. Network: `adapter.generate(stage="verification", allow_paid=True)` in `src/model_adapters.py`.
5. Storage: `CheckpointStore` writes into `results/v2/raw/v2.sqlite3`.

V1 path `run-trust` / `src/runner.py` is a different experiment and should not be used for Study 1.

**Do not invoke `run-v2-decisions --yes` for Study 1.** That command writes historical V2 cells.

---

## 8. What is reusable vs what is missing for Study 1

### Reusable as-is (read-only or copy-from)

- Frozen V2-B answers and confidence in `results/v2/raw/v2.sqlite3`
- 500-ID list in `data/samples/mmlu_pro_v2b.jsonl`
- V2 Stage-3 prompt components in `src/v2_prompts.py`
- GPT and Claude adapters, JSON schema, parser, repair prompt, retry policy
- Checkpoint store implementation
- `allow_paid` / dry-run safeguards
- Checkpoint A findings (scientific motivation only)

### Missing — expected; Task 001 should create these as **new** files

These absences do **not** block starting Task 001. They **are** Task 001’s job. They were not created in this bootstrap.

- Separate `reported_confidence` and `displayed_confidence` fields (historical records only have `probability_correct`)
- Study 1 config (100 IDs, 2 models, `L ∈ {10, 20}`, 7 Stage-3 conditions, seeds `20260917` / `20260918`)
- Deterministic 100-question ID list and hash (must be frozen **before** any new API call)
- Study 1 prompt templates that can swap only `displayed_confidence`
- Study 1 runner / CLI that writes to a **new** sqlite/path (`results/study1_causal_pilot/` or equivalent)
- Study 1 schema that is not `VerificationDecisionRecord(experiment_version="v2")`
- Rendered prompt examples for manual audit
- Call-cap enforcement for 2,800 primary / 1,120 repeat / optional smoke-test limits

### Must not be used as the Study 1 home

- `src/prompts.py` V1 Stage-3
- `src/runner.py` V1 trust stage
- Writing into `results/v2/raw/v2.sqlite3`
- Editing `src/schemas.py` `VerificationDecisionRecord` in place (it is frozen to `experiment_version="v2"` and the V2 cost grid)
- Gemini / Grok adapters for the primary pilot roster

---

## 9. Where a clean Study 1 implementation should probably live

Recommended new tree, matching Experimental Protocol §66, without touching historical artifacts:

```
config/experiment_study1.yaml          # new config; do not edit experiment_v2.yaml
src/study1_prompts.py                  # clone/adapt v2_prompts.py; keep V2 file frozen
src/study1_runner.py                   # new runner; new sqlite path; new request keys
src/study1_schemas.py                  # reported_confidence vs displayed_confidence
data/development/study1_ids.json       # selected 100 IDs + hash (Task 001, not now)
prompts/study1/                        # rendered templates for GPT audit
results/study1_causal_pilot/           # new checkpoint; never v2.sqlite3
to_gpt/001_study1_pilot_build/         # return folder after Task 001
```

Ordinary engineering choice: exact filenames inside that tree may differ. Scientific choices (grid, models, wording, sample rule) must come from the protocol / a numbered task, not from this audit.

---

## 10. Implementation constraints Task 001 will have to respect

1. Historical confidence field is `probability_correct`, not `reported_confidence`. New records must keep both original and displayed values; do not rename-in-place inside V2 rows.
2. `VerificationDecisionRecord` currently forbids any experiment version other than `"v2"` and any error cost outside `{2, 5, 10, 20}`. Study 1 needs a new record type, not a silent relaxation of the V2 schema.
3. Visible V2 wording is “The AI previously estimated a … probability …”, while the protocol’s example is “Your previously reported probability …”. That is a scientific wording choice. Cursor must not independently “improve” it.
4. `results/` is gitignored. The 194 MB sqlite exists locally and is required. It is not in git.
5. Paid-call guards already exist and must remain on until a numbered task explicitly authorizes calls.

---

## 11. Readiness verdict for Task 001

Historical components required to **start building** Study 1 infrastructure are present locally.

Nothing in this audit authorizes implementing the grid, selecting IDs, changing prompts, or making API calls.
