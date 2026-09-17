# GPT ↔ Cursor Research Workflow

**Project:** D1 — Confidence as a Control Signal for LLM Verification  
**Document version:** 1.0  
**Date:** 2026-09-17  
**Status:** Binding engineering contract for D1 work executed in Cursor.

This document governs how GPT, Cursor, and the human researcher collaborate. It does not replace the scientific sources of truth. It exists to make accidental scientific or operational overreach difficult.

The human researcher is the transport layer. GPT never writes directly into this repository. Cursor never uploads to GPT. Files move by hand.

---

## Intended loop

1. GPT writes a numbered task specification.
2. The researcher places that file in `from_gpt/`.
3. Cursor executes **only that numbered task**.
4. Cursor writes a matching folder in `to_gpt/`.
5. The researcher uploads the returned report (and any requested artifacts) to GPT.
6. GPT reviews the result and only then authors the next numbered task.

Example:

```
from_gpt/001_build_study1_pilot.md
    → to_gpt/001_study1_pilot_build/

from_gpt/002_run_study1_pilot.md
    → to_gpt/002_study1_pilot_results/
```

Task numbering is sequential. A later implied step is not authorized merely because it is obvious.

---

## A. Source-of-truth priority

For D1 scientific decisions, use this order:

1. **The newest explicit numbered `from_gpt/` task** controls the current engineering action.
2. **`docs/D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md`** (and later dated addenda) controls experimental details.
3. **`docs/D1_MASTER_RESEARCH_PLAN.md`** controls high-level scientific strategy.
4. **Historical code and data** are evidence and implementation resources. They are not permission to change the current study.

Byte-identical originals of the current plan and protocol also exist at `masterplan.txt` and `experimentalProtocol.txt`. Those files must not be deleted. Do not treat them as a second, independently editable scientific source. If a later numbered task updates the canonical `docs/` copies, follow the `docs/` copies.

If instructions conflict:

- **STOP.**
- Document the conflict in the current `to_gpt/` report.
- Do **not** silently resolve a scientific conflict.

Conflicts include, but are not limited to: sample size, model roster, displayed-confidence grid, stakes, prompt wording, primary metrics, question selection, and whether a paid run is authorized.

---

## B. Task execution rule

Cursor must execute **only** the specific numbered task currently supplied.

Do not automatically continue to:

- the next study;
- a larger sample;
- optional controls;
- confirmatory data collection;
- mechanistic interpretability;
- additional models;
- additional datasets.

A successful task ends by returning artifacts to the specified `to_gpt/` folder.

Do not execute future implied tasks because they appear in the Master Plan, Experimental Protocol, or a “next steps” list. Those documents describe a program. Only a numbered `from_gpt/` task authorizes the current action.

This bootstrap / Task 000 setup does **not** authorize Study 1 implementation or any Study 1 API calls.

---

## C. Paid / API call safety

Do not make paid API or model calls unless the **current numbered task** explicitly says they are authorized.

If authorized, obey exactly:

- the model roster;
- the maximum call count;
- the sample;
- the experimental grid.

If implementation appears likely to exceed the authorized call cap:

- **STOP before making the excess calls**;
- report why in `to_gpt/`.

A task that says **BUILD**, **PREPARE**, **IMPLEMENT**, **VALIDATE**, or **DRY RUN** does **not** authorize a full paid experiment.

A tiny smoke test is allowed only if that numbered task explicitly authorizes it, including model roster and maximum call count.

Existing code already requires `allow_paid=True` / CLI `--yes` for non-dry-run generation. That safeguard does not replace this contract. Cursor must not pass those flags unless the current numbered task explicitly authorizes paid calls.

---

## D. Historical immutability

Never overwrite:

- original V2 raw data (`results/v2/raw/v2.sqlite3` and associated V2 records);
- original frozen protocol (`EXPERIMENT_V2.md`, `protocol_packet/EXPERIMENT_V2.md`);
- workshop paper artifacts (`paper/`, `paper_outputs/v2/`, `paper_outputs/v2a/`);
- Checkpoint A artifacts (`to_gpt/checkpoint_A/`).

New experiments must use new paths, fields, and tables.

In particular:

```
reported_confidence  = original model-reported confidence
displayed_confidence = value actually shown in the new experimental prompt
```

Never overwrite one with the other.

Historical V2 currently stores the original score as `probability_correct`. Any Study 1 implementation must preserve that historical field unchanged in V2 records. New Study 1 records must keep original and displayed values in separate fields.

Do not modify the scientific contents of:

- `docs/D1_MASTER_RESEARCH_PLAN.md`
- `docs/D1_EXPERIMENTAL_PROTOCOL_v1.0_DRAFT.md`
- `masterplan.txt`
- `experimentalProtocol.txt`

unless a numbered task explicitly authorizes a documentation change. Even then, do not rewrite history; append or add a dated addendum.

---

## E. Scientific choice rule

Cursor may make ordinary software-engineering choices when necessary (file layout inside a new Study 1 tree, logging, tests, dry-run plumbing, names of new non-historical tables).

Cursor must **not** independently make material scientific choices such as:

- changing sample size;
- changing model roster;
- changing confidence values;
- changing stakes levels;
- dropping or adding experimental conditions;
- choosing questions based on outcomes;
- changing primary metrics;
- changing prompts in a scientifically meaningful way.

If a scientific choice is underspecified:

- report the ambiguity to GPT in the `to_gpt/` report;
- do **not** guess.

Examples of ambiguities that must be returned rather than resolved:

- whether Study 1 should change historical visible-confidence wording;
- whether to use the three-decimal `L = 20` grid or the documented fallback;
- how to map historical `probability_correct` into new `reported_confidence` / `displayed_confidence` fields, beyond preserving both values separately.

---

## F. Output rule

Every numbered task must produce a matching folder under `to_gpt/`.

Every returned folder should contain at minimum:

- `report.md`
- `changed_files.txt`
- `run_manifest.json`

When relevant also include:

- configs;
- rendered prompts;
- selected IDs;
- validation output;
- logs;
- result CSVs;
- figures;
- scripts.

`report.md` must explain in plain English:

1. what Cursor did;
2. what files changed;
3. what it did **not** do;
4. whether any API calls were made;
5. exact number of API calls if any;
6. any ambiguities or errors;
7. anything GPT should inspect before authorizing the next step.

Do not mix results from different tasks in one folder.

---

## G. No automatic scientific interpretation

Cursor may calculate requested statistics and summarize results.

Cursor should not:

- rewrite the paper narrative;
- declare the hypothesis proven;
- invent novelty claims;
- decide to scale an experiment;
- decide to begin the next study.

Those decisions happen after GPT reviews the returned report.

---

## Current program status (as of 2026-09-17)

| Artifact | Status |
|---|---|
| Historical V2-B | HISTORICAL-FROZEN |
| Checkpoint A | EXPLORATORY (complete) |
| Study 1 causal pilot | EXPLORATORY (not built, not run) |
| Study 1 micro-controls | CONDITIONAL / EXPLORATORY |
| Studies 2–6 | CONDITIONAL |

Current next engineering task after this workflow is reviewed: **build and audit Study 1 pilot infrastructure only**. Do not launch the full pilot until a later numbered task explicitly authorizes it.
