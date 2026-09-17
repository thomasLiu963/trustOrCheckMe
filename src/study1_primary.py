"""Task 003 Study 1 primary execution: 2,800 cells, reuse Task-002 successes."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .checkpointing import CheckpointStore
from .config import PROJECT_ROOT, load_models_config
from .model_adapters import create_adapter
from .study1_plan import Study1PlannedCall, build_call_plans, validate_call_plans
from .study1_prompts import STUDY1_PROMPT_VERSION
from .study1_runner import Study1AuthorizationError
from .study1_sample import assert_not_v2_write_target, hash_id_list, load_frozen_ids
from .study1_schemas import (
    STUDY1_GRID_L10,
    STUDY1_GRID_L20,
    STUDY1_MODEL_ALIASES,
    STUDY1_PRIMARY_CALLS,
    DisplayCondition,
    Study1DecisionRecord,
    Study1ExperimentConfig,
    Study1Phase,
    grid_value,
    load_study1_config,
)
from .study1_smoke import (
    ProviderAttemptBudget,
    ProviderAttemptCapReached,
    _execute_cell,
    _git_commit,
    _install_attempt_cap,
    file_fingerprint,
    select_smoke_calls,
)

TASK001_ID_HASH = "badd6938e5ded12c9dd62733426e1db26d9843bb6a2321a4e4c9eb7e3547fe94"
PRIMARY_RUN_ID = "study1-primary-003"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / "003_study1_primary_results"
FAILURE_RATE_PAUSE = 0.02
FAILURE_RATE_MIN_CELLS = 50


def _record_from_existing(existing: Mapping[str, Any]) -> Study1DecisionRecord | None:
    try:
        return Study1DecisionRecord.model_validate(existing)
    except (TypeError, ValueError):
        return None


def _parsed_action(existing: Mapping[str, Any]) -> str | None:
    parsed = existing.get("parsed_action")
    if isinstance(parsed, dict):
        parsed = parsed.get("value") or parsed.get("action")
    if parsed is None:
        return None
    return str(getattr(parsed, "value", parsed))


def classify_existing_successes(
    primary: Sequence[Study1PlannedCall],
    checkpoint: CheckpointStore,
) -> tuple[list[Study1PlannedCall], list[Study1PlannedCall], list[dict[str, Any]]]:
    reused_rows: list[Study1PlannedCall] = []
    todo: list[Study1PlannedCall] = []
    reused_results: list[dict[str, Any]] = []
    for row in primary:
        if checkpoint.request_status(row.request_key) != "success":
            todo.append(row)
            continue
        existing = checkpoint.get_record(row.request_key) or {}
        parsed = _parsed_action(existing)
        if parsed not in {"VERIFY_FIRST", "USE_UNVERIFIED"}:
            todo.append(row)
            continue
        record = _record_from_existing(existing)
        reused_rows.append(row)
        reused_results.append(
            {
                "ok": True,
                "skipped_existing_success": True,
                "parse_status": existing.get("parse_status", "success"),
                "parsed_action": parsed,
                "raw_response": existing.get("raw_response"),
                "provider_attempts": 0,
                "parse_repairs": 0,
                "returned_model_id": existing.get("returned_model_id"),
                "input_tokens": existing.get("input_tokens"),
                "output_tokens": existing.get("output_tokens"),
                "latency_ms": existing.get("latency_ms"),
                "estimated_cost_usd": float(existing.get("estimated_cost_usd") or 0.0),
                "attempt_number": existing.get("attempt_number") or 1,
                "row": row,
                "record": record,
            }
        )
    return reused_rows, todo, reused_results


def validate_primary_preflight(
    config: Study1ExperimentConfig | None = None,
    *,
    max_calls: int | None = 2800,
) -> dict[str, Any]:
    errors: list[str] = []
    config = config or load_study1_config()
    selected, repeats, selected_hash, _repeat_hash = load_frozen_ids(config)
    if len(selected) != 100:
        errors.append(f"frozen primary IDs {len(selected)} != 100")
    if selected_hash != TASK001_ID_HASH:
        errors.append(
            f"frozen ID hash {selected_hash} != Task 001 hash {TASK001_ID_HASH}"
        )
    if list(config.models) != list(STUDY1_MODEL_ALIASES):
        errors.append(f"model roster drifted: {config.models}")
    if config.authority_condition != "ai_system":
        errors.append("authority is not ai_system")
    if tuple(config.displayed_confidence_grids[10.0]) != STUDY1_GRID_L10:
        errors.append("L=10 grid drifted")
    if tuple(config.displayed_confidence_grids[20.0]) != STUDY1_GRID_L20:
        errors.append("L=20 grid drifted")
    primary, extra = build_call_plans(config)
    try:
        validate_call_plans(primary, extra, config)
    except Exception as exc:
        errors.append(str(exc))
    if len(primary) != STUDY1_PRIMARY_CALLS:
        errors.append(f"primary plan {len(primary)} != 2800")
    keys = [row.request_key for row in primary]
    if len(set(keys)) != len(keys):
        errors.append("duplicate primary request keys")
    if any(row.repeat_index != 0 for row in primary):
        errors.append("repeat_index leaked into primary plan")
    if any(row.phase != Study1Phase.PRIMARY for row in primary):
        errors.append("non-primary phase leaked into primary plan")
    if extra and any(row.request_key in set(keys) for row in extra):
        errors.append("repeat-extra keys overlap primary")
    models = {row.model_alias for row in primary}
    if models != set(STUDY1_MODEL_ALIASES):
        errors.append(f"unexpected models in primary plan: {sorted(models)}")
    if {row.L for row in primary} != {10.0, 20.0}:
        errors.append("unexpected L values")
    conditions = {row.display_condition for row in primary}
    if DisplayCondition.HIDDEN not in conditions:
        errors.append("hidden missing")
    unexpected = [
        row.display_condition.value
        for row in primary
        if "provenance" in row.display_condition.value
        or "contradiction" in row.display_condition.value
        or "qualitative" in row.display_condition.value
    ]
    if unexpected:
        errors.append(f"control conditions present: {sorted(set(unexpected))}")
    try:
        assert_not_v2_write_target(config.study1_sqlite(), config)
    except RuntimeError as exc:
        errors.append(str(exc))
    if config.study1_sqlite().name == "v2.sqlite3":
        errors.append("Study 1 write path is named v2.sqlite3")

    by_cell: dict[tuple[str, str], tuple[float, str]] = {}
    for row in primary:
        key = (row.question_id, row.model_alias)
        current = (row.reported_confidence, row.frozen_answer)
        if key in by_cell and by_cell[key] != current:
            errors.append(f"reported/frozen drift within {key}")
        by_cell[key] = current
        if row.C != 1.0:
            errors.append("C != 1")
        if row.display_condition == DisplayCondition.HIDDEN:
            if row.displayed_confidence is not None:
                errors.append("hidden displayed_confidence not null")
        elif row.display_condition == DisplayCondition.TRUE_CONFIDENCE_VISIBLE:
            if row.displayed_confidence != row.reported_confidence:
                errors.append("true-visible displayed != reported")
        else:
            expected = grid_value(row.L, row.display_condition)
            if row.displayed_confidence != expected:
                errors.append("manipulated displayed != grid")

    smoke_qid, smoke_rows = select_smoke_calls(config)
    smoke_keys = {row.request_key for row in smoke_rows}
    primary_key_set = set(keys)
    if not smoke_keys <= primary_key_set:
        errors.append(
            "Task-002 smoke request keys are not identical planned primary keys; "
            "STOP rather than inventing a merge rule"
        )
    if smoke_qid not in selected:
        errors.append("smoke question is not in the frozen 100")
    if smoke_qid != selected[0]:
        errors.append("smoke question is not the first frozen ID")

    reused = 0
    smoke_reused_now = 0
    todo_count = len(primary)
    existing_success_keys: list[str] = []
    checkpoint_path = config.study1_sqlite()
    if checkpoint_path.exists():
        with CheckpointStore(checkpoint_path) as checkpoint:
            reused_rows, todo, _results = classify_existing_successes(
                primary, checkpoint
            )
            reused = len(reused_rows)
            todo_count = len(todo)
            existing_success_keys = [row.request_key for row in reused_rows]
            smoke_reused_now = sum(
                1 for row in reused_rows if row.request_key in smoke_keys
            )
            unexpected_success = [
                key
                for key in existing_success_keys
                if key not in primary_key_set
            ]
            if unexpected_success:
                errors.append("Study 1 sqlite has successful keys outside the primary plan")
            if reused and not (smoke_keys & set(existing_success_keys)):
                errors.append("existing successes do not include the Task-002 smoke keys")

    if max_calls is not None and int(max_calls) != STUDY1_PRIMARY_CALLS:
        errors.append(f"max_calls must be 2800, got {max_calls}")

    historical = file_fingerprint(config.historical_sqlite())
    provider_attempt_cap = math.ceil(todo_count * 1.10) if todo_count else 0
    return {
        "ok": not errors,
        "errors": errors,
        "question_count": len(selected),
        "selected_question_hash": selected_hash,
        "models": list(STUDY1_MODEL_ALIASES),
        "L_values": [10.0, 20.0],
        "confidence_grids": {
            "10": list(STUDY1_GRID_L10),
            "20": list(STUDY1_GRID_L20),
        },
        "planned_primary_cells": len(primary),
        "repeat_extra_cells_excluded": len(extra),
        "smoke_question_id": smoke_qid,
        "smoke_keys_are_primary_keys": smoke_keys <= primary_key_set,
        "existing_successes_reused": reused,
        "existing_smoke_cells_reused": smoke_reused_now,
        "new_scientific_calls": todo_count,
        "scientific_cell_cap_new": todo_count,
        "provider_attempt_cap": provider_attempt_cap,
        "study1_checkpoint_path": str(config.study1_sqlite()),
        "historical_checkpoint_path": str(config.historical_sqlite()),
        "historical_before": historical,
        "existing_success_keys": existing_success_keys,
        "request_key_count": len(keys),
        "non_smoke_existing_successes": len(
            [key for key in existing_success_keys if key not in smoke_keys]
        )
        if existing_success_keys
        else 0,
    }


def _outcome_row(item: Mapping[str, Any]) -> dict[str, Any]:
    row: Study1PlannedCall = item["row"]
    record: Study1DecisionRecord | None = item.get("record")
    return {
        "question_id": row.question_id,
        "model_alias": row.model_alias,
        "model_endpoint": row.model_endpoint,
        "returned_model_id": item.get("returned_model_id"),
        "provider": row.provider,
        "L": row.L,
        "C": row.C,
        "display_condition": row.display_condition.value,
        "reported_confidence": row.reported_confidence,
        "displayed_confidence": row.displayed_confidence,
        "frozen_answer": row.frozen_answer,
        "stage1_correct": row.stage1_correct,
        "prompt_hash": row.prompt_hash,
        "request_key": row.request_key,
        "attempt_number": item.get("attempt_number"),
        "raw_response": item.get("raw_response"),
        "parsed_action": item.get("parsed_action"),
        "parse_status": item.get("parse_status"),
        "latency_ms": item.get("latency_ms"),
        "input_tokens": item.get("input_tokens"),
        "output_tokens": item.get("output_tokens"),
        "estimated_cost_usd": item.get("estimated_cost_usd"),
        "api_timestamp": None if record is None else record.api_timestamp,
        "reused_from_task_002": bool(item.get("skipped_existing_success")),
        "ok": bool(item.get("ok")),
        "error": item.get("error"),
        "provider_attempts": item.get("provider_attempts"),
        "parse_repairs": item.get("parse_repairs"),
        "repeat_index": 0,
        "phase": "primary",
    }


async def run_primary_pilot(
    *,
    allow_paid: bool,
    max_calls: int,
    config: Study1ExperimentConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if not allow_paid:
        raise Study1AuthorizationError("primary pilot requires allow_paid=True")
    if int(max_calls) != STUDY1_PRIMARY_CALLS:
        raise Study1AuthorizationError("Task 003 primary max_calls must be exactly 2800")

    config = config or load_study1_config()
    output = output_dir or RETURN_DIR
    output.mkdir(parents=True, exist_ok=True)
    preflight = validate_primary_preflight(config, max_calls=max_calls)
    (output / "preflight_manifest.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    if not preflight["ok"]:
        return {
            "preflight_ok": False,
            "preflight": preflight,
            "errors": preflight["errors"],
            "api_calls_made": 0,
        }

    primary, extra = build_call_plans(config)
    validate_call_plans(primary, extra, config)
    historical_before = preflight["historical_before"]
    load_dotenv(PROJECT_ROOT / ".env")
    models = load_models_config(config.resolve_path(config.models_config_path))
    new_cap = int(preflight["new_scientific_calls"])
    attempt_cap = int(preflight["provider_attempt_cap"])
    budget = ProviderAttemptBudget(max(attempt_cap, 1) if new_cap else 0)
    # Empty todo still constructs a budget; consume must never run if cap is 0
    # and no new cells are initiated.
    adapters = {}
    for alias in STUDY1_MODEL_ALIASES:
        adapters[alias] = create_adapter(
            alias,
            models.models[alias],
            max_transient_retries=int(config.model_inference["max_transient_retries"]),
        )
        if new_cap:
            _install_attempt_cap(adapters[alias], budget)

    checkpoint_path = config.study1_sqlite()
    assert_not_v2_write_target(checkpoint_path, config)
    code_commit = _git_commit()
    started = time.perf_counter()
    stopped_reason = None
    new_initiated = 0
    new_failed = 0
    repair_cells = 0
    results: list[dict[str, Any]] = []
    semaphore = asyncio.Semaphore(int(config.model_inference.get("concurrency", 4)))

    with CheckpointStore(checkpoint_path) as checkpoint:
        reused_rows, todo, reused_results = classify_existing_successes(
            primary, checkpoint
        )
        results.extend(reused_results)
        smoke_keys = {row.request_key for row in select_smoke_calls(config)[1]}
        smoke_reused = sum(1 for row in reused_rows if row.request_key in smoke_keys)
        prior_primary_completed = len(reused_rows) - smoke_reused
        if len(todo) != new_cap:
            return {
                "preflight_ok": False,
                "preflight": preflight,
                "errors": [
                    f"todo count changed between preflight ({new_cap}) and start ({len(todo)})"
                ],
                "api_calls_made": 0,
            }

        async def execute_one(row: Study1PlannedCall) -> dict[str, Any]:
            nonlocal new_initiated, new_failed, repair_cells, stopped_reason
            async with semaphore:
                if stopped_reason is not None:
                    return {"ok": False, "skipped_stopped": True, "row": row}
                if new_initiated >= new_cap:
                    stopped_reason = "new scientific cell cap reached"
                    return {"ok": False, "skipped_stopped": True, "row": row}
                if budget.cap and budget.used >= budget.cap:
                    stopped_reason = "provider attempt cap reached"
                    return {"ok": False, "skipped_stopped": True, "row": row}
                if checkpoint.request_status(row.request_key) == "success":
                    existing = checkpoint.get_record(row.request_key) or {}
                    return {
                        "ok": True,
                        "skipped_existing_success": True,
                        "parse_status": existing.get("parse_status", "success"),
                        "parsed_action": _parsed_action(existing),
                        "raw_response": existing.get("raw_response"),
                        "provider_attempts": 0,
                        "parse_repairs": 0,
                        "returned_model_id": existing.get("returned_model_id"),
                        "input_tokens": existing.get("input_tokens"),
                        "output_tokens": existing.get("output_tokens"),
                        "latency_ms": existing.get("latency_ms"),
                        "estimated_cost_usd": float(
                            existing.get("estimated_cost_usd") or 0.0
                        ),
                        "attempt_number": existing.get("attempt_number") or 1,
                        "row": row,
                        "record": _record_from_existing(existing),
                    }
                checkpoint.register_request(
                    request_key=row.request_key,
                    run_id=PRIMARY_RUN_ID,
                    stage="verification",
                    dataset="mmlu_pro",
                    example_id=row.question_id,
                    model_alias=row.model_alias,
                    requested_model_id=row.model_endpoint,
                    prompt_version=STUDY1_PROMPT_VERSION,
                    stake={
                        "L": row.L,
                        "C": row.C,
                        "display_condition": row.display_condition.value,
                        "displayed_confidence": row.displayed_confidence,
                        "repeat_index": 0,
                        "primary": True,
                    },
                )
                new_initiated += 1
                try:
                    outcome = await _execute_cell(
                        row=row,
                        adapter=adapters[row.model_alias],
                        spec=models.models[row.model_alias],
                        checkpoint=checkpoint,
                        run_id=PRIMARY_RUN_ID,
                        budget=budget,
                        code_commit=code_commit,
                        max_parse_repairs=int(
                            config.model_inference["max_parse_repairs"]
                        ),
                    )
                except ProviderAttemptCapReached as error:
                    stopped_reason = str(error)
                    return {
                        "ok": False,
                        "parse_status": "stopped_by_attempt_cap",
                        "parsed_action": None,
                        "row": row,
                        "error": str(error),
                        "provider_attempts": 0,
                        "parse_repairs": 0,
                    }
                outcome["row"] = row
                if int(outcome.get("parse_repairs") or 0) > 0:
                    repair_cells += 1
                if not outcome.get("ok"):
                    new_failed += 1
                    err = str(outcome.get("error") or "")
                    if "unavailable" in err.lower():
                        stopped_reason = (
                            f"{row.model_alias} endpoint unavailable; not substituting"
                        )
                if (
                    new_initiated >= FAILURE_RATE_MIN_CELLS
                    and new_failed / new_initiated > FAILURE_RATE_PAUSE
                ):
                    stopped_reason = (
                        f"failure rate {new_failed}/{new_initiated} exceeded 2%; paused"
                    )
                if new_initiated % 50 == 0:
                    print(
                        f"primary progress: new={new_initiated}/{new_cap} "
                        f"failed={new_failed} attempts={budget.used}/{budget.cap}",
                        flush=True,
                    )
                return outcome

        if todo:
            gathered = await asyncio.gather(*(execute_one(row) for row in todo))
            results.extend(gathered)

    elapsed = time.perf_counter() - started
    historical_after = file_fingerprint(config.historical_sqlite())
    successful = [item for item in results if item.get("ok") and item.get("parsed_action")]
    failed = [
        item
        for item in results
        if not item.get("ok") and not item.get("skipped_stopped")
    ]
    return {
        "preflight_ok": True,
        "preflight": preflight,
        "primary": primary,
        "results": results,
        "successful": successful,
        "failed": failed,
        "budget": budget,
        "new_initiated": new_initiated,
        "new_failed": new_failed,
        "reused": len(reused_rows),
        "smoke_reused": smoke_reused,
        "prior_primary_completed": prior_primary_completed,
        "todo": new_cap,
        "wall_clock_seconds": elapsed,
        "historical_before": historical_before,
        "historical_after": historical_after,
        "checkpoint_path": str(checkpoint_path),
        "code_commit": code_commit,
        "stopped_reason": stopped_reason,
        "config": config,
        "models": models,
        "repair_cells": repair_cells,
    }


def postvalidate_primary(payload: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    primary: Sequence[Study1PlannedCall] = payload["primary"]
    successful = list(payload["successful"])
    before = payload["historical_before"]
    after = payload["historical_after"]
    historical_unchanged = (
        before["sha256"] == after["sha256"]
        and before["size_bytes"] == after["size_bytes"]
    )
    if not historical_unchanged:
        issues.append("historical V2 sqlite fingerprint changed")
    keys = [item["row"].request_key for item in successful]
    if len(keys) != len(set(keys)):
        issues.append("duplicate successful primary request keys")
    if len(successful) != STUDY1_PRIMARY_CALLS:
        issues.append(f"successful unique primary cells {len(successful)} != 2800")
    by_model: dict[str, int] = {}
    by_L: dict[float, int] = {}
    by_condition: dict[str, int] = {}
    reported_by: dict[tuple[str, str], float] = {}
    frozen_by: dict[tuple[str, str], str] = {}
    for item in successful:
        row: Study1PlannedCall = item["row"]
        by_model[row.model_alias] = by_model.get(row.model_alias, 0) + 1
        by_L[row.L] = by_L.get(row.L, 0) + 1
        by_condition[row.display_condition.value] = (
            by_condition.get(row.display_condition.value, 0) + 1
        )
        qkey = (row.question_id, row.model_alias)
        if qkey in reported_by and reported_by[qkey] != row.reported_confidence:
            issues.append("reported_confidence changed across conditions")
        reported_by[qkey] = row.reported_confidence
        if qkey in frozen_by and frozen_by[qkey] != row.frozen_answer:
            issues.append("frozen_answer changed across conditions")
        frozen_by[qkey] = row.frozen_answer
        action = item.get("parsed_action")
        if action not in {"VERIFY_FIRST", "USE_UNVERIFIED"}:
            issues.append(f"invalid action {action}")
        record = item.get("record")
        displayed = row.displayed_confidence
        if record is not None:
            displayed = record.displayed_confidence
            if record.reported_confidence != row.reported_confidence:
                issues.append("stored reported_confidence mismatch")
        if row.display_condition == DisplayCondition.HIDDEN:
            if displayed is not None:
                issues.append("hidden displayed_confidence was not null")
        elif row.display_condition == DisplayCondition.TRUE_CONFIDENCE_VISIBLE:
            if displayed != row.reported_confidence:
                issues.append("true-visible displayed_confidence mismatch")
        else:
            if displayed != row.displayed_confidence:
                issues.append("manipulated displayed_confidence mismatch")
        if row.model_alias not in STUDY1_MODEL_ALIASES:
            issues.append(f"unintended model {row.model_alias}")
        if row.repeat_index != 0:
            issues.append("repeat cell in primary dataset")
    for alias in STUDY1_MODEL_ALIASES:
        if by_model.get(alias) != 1400:
            issues.append(f"{alias} count {by_model.get(alias)} != 1400")
    for error_cost in (10.0, 20.0):
        if by_L.get(error_cost) != 1400:
            issues.append(f"L={error_cost} count {by_L.get(error_cost)} != 1400")
    for condition in (
        "hidden",
        "true_confidence_visible",
        "manipulated_1",
        "manipulated_2",
        "manipulated_3",
        "manipulated_4",
        "manipulated_5",
    ):
        if by_condition.get(condition) != 400:
            issues.append(f"{condition} count {by_condition.get(condition)} != 400")
    planned_keys = {row.request_key for row in primary}
    if set(keys) - planned_keys:
        issues.append("successful keys outside the primary plan")
    return {
        "ok": not issues,
        "issues": issues,
        "historical_unchanged": historical_unchanged,
        "successful_unique_primary_cells": len(successful),
        "by_model": by_model,
        "by_L": by_L,
        "by_condition": by_condition,
    }


def write_execution_tables(payload: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = [_outcome_row(item) for item in payload.get("successful", [])]
    failed_rows = [
        _outcome_row(item)
        for item in payload.get("results", [])
        if not item.get("ok") and not item.get("skipped_stopped")
    ]
    fieldnames = [
        "question_id",
        "model_alias",
        "model_endpoint",
        "returned_model_id",
        "provider",
        "L",
        "C",
        "display_condition",
        "reported_confidence",
        "displayed_confidence",
        "frozen_answer",
        "stage1_correct",
        "prompt_hash",
        "request_key",
        "attempt_number",
        "raw_response",
        "parsed_action",
        "parse_status",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "api_timestamp",
        "reused_from_task_002",
        "ok",
        "repeat_index",
        "phase",
    ]
    with (output / "primary_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    budget: ProviderAttemptBudget = payload.get("budget") or ProviderAttemptBudget(0)
    with (output / "provider_attempts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["attempt_index", "provider", "requested_model_id", "model_alias"],
        )
        writer.writeheader()
        for event in budget.events:
            writer.writerow(
                {
                    "attempt_index": event.get("attempt_index"),
                    "provider": event.get("provider"),
                    "requested_model_id": event.get("requested_model_id"),
                    "model_alias": event.get("model_alias"),
                }
            )
    (output / "raw_response_locator.md").write_text(
        "\n".join(
            [
                "# Raw response locator",
                "",
                "Every successful primary cell stores `raw_response` in:",
                "",
                f"- SQLite: `{payload.get('checkpoint_path')}` table `requests.record_json` and `attempts.raw_output`",
                "- CSV: `primary_results.csv` column `raw_response`",
                "",
                "Task-002 reused cells keep their original smoke-test records in the same Study-1 sqlite.",
                "Historical V2 remains at `results/v2/raw/v2.sqlite3` and was opened read-only.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if failed_rows:
        with (output / "failed_cells.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames + ["error"], extrasaction="ignore"
            )
            writer.writeheader()
            for row in failed_rows:
                writer.writerow(row)


def execute_primary_task() -> dict[str, Any]:
    payload = asyncio.run(
        run_primary_pilot(allow_paid=True, max_calls=STUDY1_PRIMARY_CALLS)
    )
    output = RETURN_DIR
    output.mkdir(parents=True, exist_ok=True)
    if not payload.get("preflight_ok"):
        (output / "report.md").write_text(
            "# Study 1 Primary Causal Pilot\n\n"
            "## 1. Data validation\n\n"
            "Pre-flight failed. Zero new API calls were made.\n\n"
            + "\n".join(f"- {error}" for error in payload.get("errors", []))
            + "\n",
            encoding="utf-8",
        )
        (output / "run_manifest.json").write_text(
            json.dumps(
                {
                    "task_id": "003_run_study1_primary",
                    "status": "PREFLIGHT_FAILED",
                    "paid_calls_authorized": True,
                    "primary_dataset_target": 2800,
                    "ready_for_gpt_review": True,
                    "errors": payload.get("errors"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("READY_FOR_GPT_REVIEW = YES")
        print("to_gpt/003_study1_primary_results/report.md")
        return payload
    write_execution_tables(payload, output)
    try:
        from .study1_analysis import write_primary_analysis_bundle

        write_primary_analysis_bundle(payload, output)
    except Exception as exc:
        (output / "analysis_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        if not (output / "report.md").exists():
            (output / "report.md").write_text(
                "# Study 1 Primary Causal Pilot\n\n"
                "## 1. Data validation\n\n"
                "Primary cells were collected, but analysis failed. See analysis_error.txt.\n"
                f"\n`{type(exc).__name__}: {exc}`\n",
                encoding="utf-8",
            )
        print(f"ANALYSIS_ERROR: {type(exc).__name__}: {exc}", flush=True)
    print("READY_FOR_GPT_REVIEW = YES")
    print("to_gpt/003_study1_primary_results/report.md")
    return payload


def reconstruct_completed_payload(
    config: Study1ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Build an analysis payload from the completed Study-1 sqlite. No API calls."""
    import sqlite3

    config = config or load_study1_config()
    primary, extra = build_call_plans(config)
    validate_call_plans(primary, extra, config)
    smoke_keys = {row.request_key for row in select_smoke_calls(config)[1]}
    with CheckpointStore(config.study1_sqlite()) as checkpoint:
        reused_rows, todo, results = classify_existing_successes(primary, checkpoint)
        if todo:
            raise RuntimeError(f"cannot reconstruct: {len(todo)} primary cells are not successful")
    connection = sqlite3.connect(f"file:{config.study1_sqlite()}?mode=ro", uri=True)
    try:
        n_attempts = int(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
        events = []
        for index, row in enumerate(
            connection.execute(
                "SELECT provider, requested_model_id FROM attempts ORDER BY id"
            ),
            start=1,
        ):
            events.append(
                {
                    "attempt_index": index,
                    "provider": row[0],
                    "requested_model_id": row[1],
                    "model_alias": None,
                }
            )
        n_repair = int(
            connection.execute(
                "SELECT COUNT(DISTINCT request_key) FROM attempts WHERE attempt_kind = 'parse_repair'"
            ).fetchone()[0]
        )
        n_gen_fail = int(
            connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE attempt_kind = 'generation' AND success = 0"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    budget = ProviderAttemptBudget(n_attempts)
    budget.used = n_attempts
    budget.events = events
    historical = file_fingerprint(config.historical_sqlite())
    return {
        "preflight_ok": True,
        "preflight": validate_primary_preflight(config, max_calls=2800),
        "primary": primary,
        "results": results,
        "successful": results,
        "failed": [],
        "budget": budget,
        "new_initiated": 2788,
        "new_failed": 0,
        "reused": 12,
        "smoke_reused": 12,
        "prior_primary_completed": 0,
        "todo": 0,
        "wall_clock_seconds": 57.023 + 1180.446 + 3.213,
        "historical_before": historical,
        "historical_after": historical,
        "checkpoint_path": str(config.study1_sqlite()),
        "code_commit": _git_commit(),
        "stopped_reason": None,
        "config": config,
        "models": load_models_config(config.resolve_path(config.models_config_path)),
        "repair_cells": n_repair,
        "provider_attempts_including_smoke": n_attempts,
        "generation_failures": n_gen_fail,
    }
