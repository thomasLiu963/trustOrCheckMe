# `to_gpt/`

This folder contains Cursor outputs intended to be manually uploaded to GPT.

Every numbered `from_gpt/` task should produce its own numbered subfolder here. Do not mix results from different tasks.

## Naming

Match the originating task number. Use a slug that describes the returned artifact, not a new task:

```
from_gpt/001_build_study1_pilot.md  →  to_gpt/001_study1_pilot_build/
from_gpt/002_run_study1_pilot.md    →  to_gpt/002_study1_pilot_results/
```

Bootstrap / workflow-setup output lives in `000_workflow_setup/`.

## Required files in each task folder

At minimum:

- `report.md`
- `changed_files.txt`
- `run_manifest.json`

Add configs, rendered prompts, selected IDs, validation output, logs, result CSVs, figures, and scripts when they are part of the task.

`report.md` is the primary file to upload to GPT.

## Historical contents

`checkpoint_A/` is a completed historical/exploratory artifact. Do not overwrite it. New analyses must use new folders.

## Immutability

Do not use this folder as a scratch area that overwrites earlier task outputs. If a task must be redone, create a clearly named new folder or a dated revision inside the task folder without deleting the original report GPT already reviewed.
