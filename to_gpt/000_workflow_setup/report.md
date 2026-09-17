# Task 000 report — GPT ↔ Cursor workflow setup

Date: 2026-09-17  
Task: bootstrap the D1 communication/repository workflow and audit Study 1 readiness.  
This is **not** Study 1.

---

## 1. What Cursor did

Located the current D1 sources of truth, already present under non-canonical names:

- `masterplan.txt` — D1 Master Research Plan **v1.1**, post-Checkpoint-A / pre-causal-pilot, dated 2026-09-17
- `experimentalProtocol.txt` — D1 Experimental Protocol **v1.0-DRAFT**, dated 2026-09-17

Read both documents. Placed byte-identical canonical copies at:

- `docs/D1_MASTER_RESEARCH_PLAN.md`
- `docs/D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md`

Left the originals in place. SHA-256 of each pair matches.

Created the GPT ↔ Cursor transport directories and contract:

- `from_gpt/` (new) and `from_gpt/README.md`
- `to_gpt/` already existed with `checkpoint_A/`; added `to_gpt/README.md` only
- `docs/GPT_CURSOR_WORKFLOW.md`
- `docs/D1_DECISION_LOG.md` with the 2026-09-17 entry specified for this bootstrap

Inspected the repository for historical Study 1 inputs without modifying them. Wrote the readiness audit and this return folder.

## 2. What files changed

New files only. See `changed_files.txt`.

Nothing under `results/`, `data/samples/`, `src/` experiment code, `paper/`, `paper_outputs/`, or `to_gpt/checkpoint_A/` was modified.

## 3. What Cursor did **not** do

- Did not run the Study 1 causal pilot.
- Did not implement the new confidence grid.
- Did not select the 100 questions.
- Did not change Stage-3 prompts.
- Did not make a smoke-test API call.
- Did not make the 2,800 primary calls or the 1,120 repeat calls.
- Did not modify historical V2 results, Checkpoint A, or the scientific contents of the Master Plan / Experimental Protocol.
- Did not execute, invent, or queue Task 001.

## 4. Whether any API calls were made

No.

## 5. Exact number of API calls

0

## 6. Ambiguities / errors

No blocking errors. Read-only inspection found several implementation facts that Task 001 must not silently “fix”:

1. Historical V2 stores the original score as `probability_correct`. There are no `reported_confidence` / `displayed_confidence` fields yet.
2. V2 `VerificationDecisionRecord` is frozen to `experiment_version="v2"` and the original cost grid. Study 1 needs new records/paths, not an in-place schema change.
3. Historical visible wording (“The AI previously estimated a … probability …”) differs from the protocol’s example (“Your previously reported probability …”). That is a scientific wording choice for GPT, not an engineering default.
4. `results/v2/raw/v2.sqlite3` exists locally (194 MB) but is gitignored. Study 1 construction depends on this local file remaining present.

## 7. What GPT should inspect before authorizing Task 001

- This workflow contract: `docs/GPT_CURSOR_WORKFLOW.md`
- Decision log entry: `docs/D1_DECISION_LOG.md`
- Canonical plan/protocol copies vs originals (byte-identical; originals retained)
- `repository_readiness.md` in this folder, especially the recommended new Study 1 paths and the “do not write into `v2.sqlite3`” constraint
- That `to_gpt/checkpoint_A/` was left untouched

If those look correct, the first numbered Cursor task should be a **BUILD / DRY RUN** specification for Study 1 infrastructure only, with any smoke-test call cap written explicitly if GPT wants one. This bootstrap does not authorize that task.

READY_FOR_TASK_001 = YES
