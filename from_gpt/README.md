# `from_gpt/`

This folder holds numbered task specifications written by GPT for Cursor.

The human researcher copies each GPT-authored task file here. Cursor should not invent, renumber, or skip tasks.

## Naming

Use a zero-padded sequential number plus a short slug:

```
001_build_study1_pilot.md
002_run_study1_pilot.md
```

Task numbering is sequential. The existence of a later implied experiment in the Master Plan or Experimental Protocol does not authorize that experiment.

## Execution rule

Cursor should execute **only** the requested numbered task.

A BUILD / PREPARE / IMPLEMENT / VALIDATE / DRY RUN task does not authorize a full paid experiment.

After finishing the requested task, Cursor must place results in the `to_gpt/` folder named by that task. Do not continue to the next study, a larger sample, optional controls, or additional models unless the current file explicitly says to.

If this folder contains multiple numbered files, execute only the task the researcher asked for in the current conversation. Do not drain the queue.

## Conflict rule

If the numbered task conflicts with the Experimental Protocol or Master Plan, stop and document the conflict in the matching `to_gpt/` report. Do not silently resolve scientific conflicts.
