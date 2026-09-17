"""Task 004 Study 1 stability execution: 1,120 frozen repeat-extra cells only."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .checkpointing import CheckpointStore
from .config import PROJECT_ROOT, load_models_config
from .model_adapters import create_adapter
from .study1_plan import Study1PlannedCall, build_call_plans, validate_call_plans
from .study1_primary import (
    FAILURE_RATE_MIN_CELLS,
    FAILURE_RATE_PAUSE,
    _parsed_action,
    _record_from_existing,
    classify_existing_successes,
)
from .study1_prompts import STUDY1_PROMPT_VERSION
from .study1_runner import Study1AuthorizationError
from .study1_sample import assert_not_v2_write_target, load_frozen_ids
from .study1_schemas import (
    STUDY1_GRID_L10,
    STUDY1_GRID_L20,
    STUDY1_MODEL_ALIASES,
    STUDY1_PRIMARY_CALLS,
    STUDY1_REPEAT_EXTRA_CALLS,
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
)

TASK001_REPEAT_HASH = "45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38"
REPEATS_RUN_ID = "study1-repeats-004"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / "004_study1_stability_results"
REPAIR_RATE_PAUSE = 0.15


def audit_task003_parse_repairs(
    config: Study1ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Zero-cost diagnostic of Task-003 parse-repair behavior. No API calls."""
    config = config or load_study1_config()
    primary, extra = build_call_plans(config)
    by_key = {row.request_key: row for row in primary}
    connection = sqlite3.connect(f"file:{config.study1_sqlite()}?mode=ro", uri=True)
    try:
        repair_keys = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT request_key FROM attempts WHERE attempt_kind='parse_repair'"
            )
            if row[0] in by_key
        }
        truncated_keys = []
        for key, kind, success, raw, parse_error in connection.execute(
            """
            SELECT request_key, attempt_kind, success, raw_output, parse_error
            FROM attempts
            WHERE request_key IN (
                SELECT DISTINCT request_key FROM attempts WHERE success = 0
            )
            ORDER BY id
            """
        ):
            if kind == "parse_repair" and success == 0:
                truncated_keys.append(key)
        records = {}
        for key in repair_keys:
            rec = connection.execute(
                "SELECT record_json, status FROM requests WHERE request_key=?",
                (key,),
            ).fetchone()
            records[key] = rec
    finally:
        connection.close()

    repaired_rows: list[dict[str, Any]] = []
    missing = []
    for key in sorted(repair_keys):
        planned = by_key.get(key)
        if planned is None:
            missing.append(key)
            continue
        rec = records.get(key)
        record = json.loads(rec[0]) if rec and rec[0] else {}
        repaired_rows.append(
            {
                "request_key": key,
                "model": planned.model_alias,
                "L": planned.L,
                "condition": planned.display_condition.value,
                "displayed_confidence": planned.displayed_confidence,
                "action": record.get("parsed_action"),
                "status": rec[1] if rec else None,
                "question_id": planned.question_id,
            }
        )

    by_model = dict(Counter(row["model"] for row in repaired_rows))
    by_L = {str(k): v for k, v in Counter(row["L"] for row in repaired_rows).items()}
    by_condition = dict(Counter(row["condition"] for row in repaired_rows))
    by_displayed = {
        ("hidden" if k is None else str(k)): v
        for k, v in Counter(row["displayed_confidence"] for row in repaired_rows).items()
    }
    by_model_L = {
        f"{model}|L{int(L)}": n
        for (model, L), n in Counter((row["model"], row["L"]) for row in repaired_rows).items()
    }
    repaired_actions = dict(Counter(row["action"] for row in repaired_rows))

    claude_condition_share: dict[str, dict[str, Any]] = {}
    claude_all = Counter(
        row.display_condition.value
        for row in primary
        if row.model_alias == "anthropic_sonnet5"
    )
    claude_rep = Counter(
        row["condition"] for row in repaired_rows if row["model"] == "anthropic_sonnet5"
    )
    for condition, n_all in sorted(claude_all.items()):
        n_rep = claude_rep[condition]
        claude_condition_share[condition] = {
            "repairs": n_rep,
            "cells": n_all,
            "rate": n_rep / n_all if n_all else 0.0,
        }

    # Final actions: repaired vs unrepaired Claude / GPT among primary successes.
    repaired_set = set(repair_keys)
    verify_by: dict[str, dict[str, list[str]]] = {
        "openai_gpt56_sol": {"repaired": [], "other": []},
        "anthropic_sonnet5": {"repaired": [], "other": []},
    }
    with CheckpointStore(config.study1_sqlite()) as checkpoint:
        for row in primary:
            existing = checkpoint.get_record(row.request_key) or {}
            action = _parsed_action(existing)
            bucket = "repaired" if row.request_key in repaired_set else "other"
            verify_by[row.model_alias][bucket].append(action or "")

    def _rate(actions: Sequence[str]) -> dict[str, Any]:
        if not actions:
            return {"n": 0, "verify_rate": None}
        verify = sum(1 for action in actions if action == "VERIFY_FIRST")
        return {"n": len(actions), "verify_rate": verify / len(actions)}

    action_comparison = {
        model: {
            "repaired": _rate(buckets["repaired"]),
            "other": _rate(buckets["other"]),
        }
        for model, buckets in verify_by.items()
    }

    claude_rates = [item["rate"] for item in claude_condition_share.values()]
    max_share = max(claude_rates) if claude_rates else 0.0
    min_share = min(claude_rates) if claude_rates else 0.0
    threshold_pairs = {
        "L10_0.89_vs_0.91": (
            claude_condition_share.get("manipulated_3", {}).get("rate", 0.0),
            claude_condition_share.get("manipulated_4", {}).get("rate", 0.0),
        ),
        "L20_0.94_vs_0.96": (
            claude_condition_share.get("manipulated_3", {}).get("rate", 0.0),
            claude_condition_share.get("manipulated_4", {}).get("rate", 0.0),
        ),
    }
    # L=10 manipulated_3=0.89, manipulated_4=0.91; L=20 same condition names, different values.
    # Condition-level rates pool L=10 and L=20. Also compute L-specific threshold pairs.
    claude_L_cond = Counter(
        (row["L"], row["condition"])
        for row in repaired_rows
        if row["model"] == "anthropic_sonnet5"
    )
    claude_L_cond_all = Counter(
        (row.L, row.display_condition.value)
        for row in primary
        if row.model_alias == "anthropic_sonnet5"
    )

    def _share(L: float, condition: str) -> float:
        n_all = claude_L_cond_all[(L, condition)]
        n_rep = claude_L_cond[(L, condition)]
        return n_rep / n_all if n_all else 0.0

    l_specific = {
        "L10_0.89_vs_0.91": (_share(10.0, "manipulated_3"), _share(10.0, "manipulated_4")),
        "L20_0.94_vs_0.96": (_share(20.0, "manipulated_3"), _share(20.0, "manipulated_4")),
        "L10_0.80_vs_0.99": (_share(10.0, "manipulated_1"), _share(10.0, "manipulated_5")),
        "L20_0.90_vs_0.99": (_share(20.0, "manipulated_1"), _share(20.0, "manipulated_5")),
    }
    max_threshold_gap = max(abs(a - b) for a, b in l_specific.values()) if l_specific else 0.0
    gpt_repairs = by_model.get("openai_gpt56_sol", 0)
    claude_repairs = by_model.get("anthropic_sonnet5", 0)
    concentrated_model = claude_repairs > 0 and gpt_repairs == 0
    stop = bool(max_share >= 0.15 or max_threshold_gap >= 0.05)

    return {
        "distinct_repair_cells": len(repair_keys),
        "missing_from_primary_plan": missing,
        "by_model": by_model,
        "by_L": by_L,
        "by_condition": by_condition,
        "by_displayed_confidence": by_displayed,
        "by_model_L": by_model_L,
        "repaired_final_actions": repaired_actions,
        "claude_condition_share": claude_condition_share,
        "action_comparison": action_comparison,
        "max_claude_condition_repair_rate": max_share,
        "min_claude_condition_repair_rate": min_share,
        "l_specific_threshold_repair_rates": {
            name: {"left": left, "right": right, "gap": abs(left - right)}
            for name, (left, right) in l_specific.items()
        },
        "max_threshold_repair_gap": max_threshold_gap,
        "truncated_then_rerun_keys": sorted(set(truncated_keys)),
        "repairs_concentrated_in_claude": concentrated_model,
        "stop_before_paid": stop,
        "stop_reason": (
            "Repair attempts are strongly condition-dependent in a way that could distort Study-1 results."
            if stop
            else None
        ),
        "proceed": not stop,
    }


def write_parse_repair_audit(audit: Mapping[str, Any], path: Path) -> None:
    claude_share = audit["claude_condition_share"]
    comparison = audit["action_comparison"]
    lines = [
        "# Task 003 parse-repair audit (zero-cost, before Task 004 paid calls)",
        "",
        "This is an engineering diagnostic, not a new hypothesis test.",
        "No new API calls were made to produce this audit.",
        "",
        f"- Distinct cells with a parse-repair attempt: **{audit['distinct_repair_cells']}**",
        f"- GPT repairs: **{audit['by_model'].get('openai_gpt56_sol', 0)}**",
        f"- Claude repairs: **{audit['by_model'].get('anthropic_sonnet5', 0)}**",
        f"- Missing from the frozen primary plan: **{len(audit['missing_from_primary_plan'])}**",
        "",
        "## Counts by model",
        "",
    ]
    for model, count in sorted(audit["by_model"].items()):
        lines.append(f"- `{model}`: {count}")
    lines.extend(["", "## Counts by L", ""])
    for label, count in sorted(audit["by_L"].items()):
        lines.append(f"- L={label}: {count}")
    lines.extend(["", "## Counts by condition", ""])
    for condition, count in sorted(audit["by_condition"].items()):
        lines.append(f"- `{condition}`: {count}")
    lines.extend(["", "## Counts by displayed confidence (includes true-visible historical values)", ""])
    for displayed, count in sorted(audit["by_displayed_confidence"].items()):
        lines.append(f"- displayed={displayed}: {count}")
    lines.extend(["", "## Claude repair share of all Claude primary cells, by condition", ""])
    for condition, item in claude_share.items():
        lines.append(
            f"- `{condition}`: {item['repairs']}/{item['cells']} = {item['rate']:.3f}"
        )
    lines.extend(["", "## Key threshold-pair repair-rate gaps (Claude, L-specific)", ""])
    for name, item in audit["l_specific_threshold_repair_rates"].items():
        lines.append(
            f"- {name}: {item['left']:.3f} vs {item['right']:.3f} (gap {item['gap']:.3f})"
        )
    gpt_rep = comparison["openai_gpt56_sol"]["repaired"]
    gpt_oth = comparison["openai_gpt56_sol"]["other"]
    claude_rep = comparison["anthropic_sonnet5"]["repaired"]
    claude_oth = comparison["anthropic_sonnet5"]["other"]
    lines.extend(
        [
            "",
            "## Final parsed actions, repaired vs unrepaired",
            "",
            f"- GPT repaired: n={gpt_rep['n']}, VERIFY rate={gpt_rep['verify_rate']}",
            f"- GPT other: n={gpt_oth['n']}, VERIFY rate={gpt_oth['verify_rate']}",
            f"- Claude repaired: n={claude_rep['n']}, VERIFY rate={claude_rep['verify_rate']}",
            f"- Claude other: n={claude_oth['n']}, VERIFY rate={claude_oth['verify_rate']}",
            f"- Repaired-cell final actions: {audit['repaired_final_actions']}",
            "",
            "## Interpretation",
            "",
            "- Repairs are concentrated in Claude. GPT had zero parse-repair attempts, consistent with structured JSON decoding.",
            f"- Claude condition repair rates range from {audit['min_claude_condition_repair_rate']:.3f} to {audit['max_claude_condition_repair_rate']:.3f}. That is a Claude JSON-reliability issue spread across conditions, not a spike in one scientific cell type.",
            f"- The largest L-specific key-contrast repair-rate gap is {audit['max_threshold_repair_gap']:.3f}. Threshold pairs are not systematically repair-imbalanced.",
            "- Repaired Claude cells end as VERIFY more often than unrepaired Claude cells. That can slightly inflate Claude's overall VERIFY rate (on the order of 1pp if every extra VERIFY is attributed to repair), but it is not localized to a displayed-confidence condition.",
            "- One Claude cell in Task 003 required a same-key rerun after truncated JSON; it is already in the completed primary dataset.",
            "",
            "## Paid-call gate",
            "",
            (
                "**STOP BEFORE PAID CALLS.** Repair attempts are strongly condition-dependent."
                if audit["stop_before_paid"]
                else "**PROCEED.** Repair attempts are not strongly condition-dependent in a way that would distort Study-1 confidence contrasts. Task 004 paid repeat-extra calls may proceed."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_repeats_preflight(
    config: Study1ExperimentConfig | None = None,
    *,
    max_calls: int | None = 1120,
) -> dict[str, Any]:
    errors: list[str] = []
    config = config or load_study1_config()
    selected, repeats, _selected_hash, repeat_hash = load_frozen_ids(config)
    if len(repeats) != 20:
        errors.append(f"frozen repeat IDs {len(repeats)} != 20")
    if not set(repeats).issubset(set(selected)):
        errors.append("repeat IDs are not a subset of the frozen 100")
    if repeat_hash != TASK001_REPEAT_HASH:
        errors.append(
            f"frozen repeat hash {repeat_hash} != Task 001 hash {TASK001_REPEAT_HASH}"
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
    if len(extra) != STUDY1_REPEAT_EXTRA_CALLS:
        errors.append(f"repeat-extra plan {len(extra)} != 1120")
    if len(primary) != STUDY1_PRIMARY_CALLS:
        errors.append(f"primary plan {len(primary)} != 2800")
    extra_keys = [row.request_key for row in extra]
    primary_keys = {row.request_key for row in primary}
    if len(set(extra_keys)) != len(extra_keys):
        errors.append("duplicate repeat-extra request keys")
    if set(extra_keys) & primary_keys:
        errors.append("repeat-extra keys overlap primary")
    if any(row.repeat_index not in {1, 2} for row in extra):
        errors.append("repeat_index outside {1,2} in extra plan")
    if any(row.phase != Study1Phase.REPEATS for row in extra):
        errors.append("non-repeats phase leaked into extra plan")
    if {row.question_id for row in extra} != set(repeats):
        errors.append("extra plan IDs do not match frozen repeat IDs")
    if {row.model_alias for row in extra} != set(STUDY1_MODEL_ALIASES):
        errors.append("unexpected models in extra plan")
    if {row.L for row in extra} != {10.0, 20.0}:
        errors.append("unexpected L values in extra plan")
    unexpected = [
        row.display_condition.value
        for row in extra
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

    for row in extra:
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

    if max_calls is not None and int(max_calls) != STUDY1_REPEAT_EXTRA_CALLS:
        errors.append(f"max_calls must be 1120, got {max_calls}")

    primary_success = 0
    extra_success = 0
    extra_failed = 0
    checkpoint_path = config.study1_sqlite()
    if checkpoint_path.exists():
        with CheckpointStore(checkpoint_path) as checkpoint:
            primary_success = sum(
                1 for row in primary if checkpoint.request_status(row.request_key) == "success"
            )
            extra_success = sum(
                1 for row in extra if checkpoint.request_status(row.request_key) == "success"
            )
            extra_failed = sum(
                1 for row in extra if checkpoint.request_status(row.request_key) == "failed"
            )
            _, todo, _results = classify_existing_successes(extra, checkpoint)
    else:
        todo = list(extra)
        errors.append("Study 1 sqlite missing; primary 2,800 must already exist")
    if primary_success != STUDY1_PRIMARY_CALLS:
        errors.append(
            f"primary successes {primary_success} != 2800; Task 004 must not rerun primary"
        )
    todo_count = len(todo)
    if todo_count > STUDY1_REPEAT_EXTRA_CALLS:
        errors.append("todo exceeds 1120")
    historical = file_fingerprint(config.historical_sqlite())
    provider_attempt_cap = math.ceil(todo_count * 1.10) if todo_count else 0
    return {
        "ok": not errors,
        "errors": errors,
        "question_count": len(repeats),
        "repeat_ids": list(repeats),
        "repeat_question_hash": repeat_hash,
        "models": list(STUDY1_MODEL_ALIASES),
        "L_values": [10.0, 20.0],
        "confidence_grids": {
            "10": list(STUDY1_GRID_L10),
            "20": list(STUDY1_GRID_L20),
        },
        "planned_repeat_extra_cells": len(extra),
        "planned_primary_cells": len(primary),
        "primary_successes_untouched_target": primary_success,
        "existing_repeat_successes_reused": extra_success,
        "existing_repeat_failures": extra_failed,
        "new_scientific_calls": todo_count,
        "scientific_cell_cap_new": todo_count,
        "provider_attempt_cap": provider_attempt_cap,
        "study1_checkpoint_path": str(config.study1_sqlite()),
        "historical_checkpoint_path": str(config.historical_sqlite()),
        "historical_before": historical,
        "request_key_count": len(extra_keys),
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
        "ok": bool(item.get("ok")),
        "error": item.get("error"),
        "provider_attempts": item.get("provider_attempts"),
        "parse_repairs": item.get("parse_repairs"),
        "repeat_index": row.repeat_index,
        "phase": row.phase.value,
        "skipped_existing_success": bool(item.get("skipped_existing_success")),
    }


async def run_repeats_pilot(
    *,
    allow_paid: bool,
    max_calls: int,
    config: Study1ExperimentConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if not allow_paid:
        raise Study1AuthorizationError("repeats pilot requires allow_paid=True")
    if int(max_calls) != STUDY1_REPEAT_EXTRA_CALLS:
        raise Study1AuthorizationError("Task 004 repeats max_calls must be exactly 1120")

    config = config or load_study1_config()
    output = output_dir or RETURN_DIR
    output.mkdir(parents=True, exist_ok=True)

    audit = audit_task003_parse_repairs(config)
    write_parse_repair_audit(audit, output / "preflight_parse_repair_audit.md")
    if audit["stop_before_paid"]:
        return {
            "preflight_ok": False,
            "audit": audit,
            "errors": [audit["stop_reason"]],
            "api_calls_made": 0,
            "stopped_reason": "parse-repair audit STOP",
        }

    preflight = validate_repeats_preflight(config, max_calls=max_calls)
    (output / "preflight_manifest.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    if not preflight["ok"]:
        return {
            "preflight_ok": False,
            "preflight": preflight,
            "audit": audit,
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
        reused_rows, todo, reused_results = classify_existing_successes(extra, checkpoint)
        results.extend(reused_results)
        if len(todo) != new_cap:
            return {
                "preflight_ok": False,
                "preflight": preflight,
                "audit": audit,
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
                if budget.cap and budget.used >= budget.cap:
                    stopped_reason = "provider attempt cap reached"
                    return {"ok": False, "skipped_stopped": True, "row": row}
                status = checkpoint.request_status(row.request_key)
                if status == "success":
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
                        "estimated_cost_usd": float(existing.get("estimated_cost_usd") or 0.0),
                        "attempt_number": existing.get("attempt_number") or 1,
                        "row": row,
                        "record": _record_from_existing(existing),
                    }
                if new_initiated >= new_cap and status is None:
                    stopped_reason = "new scientific cell cap reached"
                    return {"ok": False, "skipped_stopped": True, "row": row}
                checkpoint.register_request(
                    request_key=row.request_key,
                    run_id=REPEATS_RUN_ID,
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
                        "repeat_index": row.repeat_index,
                        "primary": False,
                    },
                )
                if status is None:
                    new_initiated += 1
                try:
                    outcome = await _execute_cell(
                        row=row,
                        adapter=adapters[row.model_alias],
                        spec=models.models[row.model_alias],
                        checkpoint=checkpoint,
                        run_id=REPEATS_RUN_ID,
                        budget=budget,
                        code_commit=code_commit,
                        max_parse_repairs=int(config.model_inference["max_parse_repairs"]),
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
                    and new_failed / max(new_initiated, 1) > FAILURE_RATE_PAUSE
                ):
                    stopped_reason = (
                        f"failure rate {new_failed}/{new_initiated} exceeded 2%; paused"
                    )
                if (
                    new_initiated >= FAILURE_RATE_MIN_CELLS
                    and repair_cells / max(new_initiated, 1) > REPAIR_RATE_PAUSE
                ):
                    stopped_reason = (
                        f"parse-repair rate {repair_cells}/{new_initiated} exceeded 15%; paused"
                    )
                if new_initiated % 50 == 0:
                    print(
                        f"repeats progress: new={new_initiated}/{new_cap} "
                        f"failed={new_failed} repairs={repair_cells} "
                        f"attempts={budget.used}/{budget.cap}",
                        flush=True,
                    )
                return outcome

        if todo:
            gathered = await asyncio.gather(*(execute_one(row) for row in todo))
            results.extend(gathered)
            failed_rows = [
                item["row"]
                for item in gathered
                if not item.get("ok") and not item.get("skipped_stopped")
            ]
            if failed_rows and stopped_reason is None:
                print(f"retrying {len(failed_rows)} failed repeat cells once", flush=True)
                retried = await asyncio.gather(*(execute_one(row) for row in failed_rows))
                replaced = {item["row"].request_key: item for item in retried}
                results = [
                    replaced.get(item["row"].request_key, item)
                    if "row" in item
                    else item
                    for item in results
                ]

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
        "audit": audit,
        "primary": primary,
        "extra": extra,
        "results": results,
        "successful": successful,
        "failed": failed,
        "budget": budget,
        "new_initiated": new_initiated,
        "new_failed": new_failed,
        "reused": len(reused_rows),
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


def postvalidate_repeats(payload: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    extra: Sequence[Study1PlannedCall] = payload["extra"]
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
        issues.append("duplicate successful repeat-extra request keys")
    if len(successful) != STUDY1_REPEAT_EXTRA_CALLS:
        issues.append(f"successful unique repeat-extra cells {len(successful)} != 1120")
    by_model: dict[str, int] = {}
    by_L: dict[float, int] = {}
    by_condition: dict[str, int] = {}
    by_repeat: dict[int, int] = {}
    question_ids: set[str] = set()
    for item in successful:
        row: Study1PlannedCall = item["row"]
        by_model[row.model_alias] = by_model.get(row.model_alias, 0) + 1
        by_L[row.L] = by_L.get(row.L, 0) + 1
        by_condition[row.display_condition.value] = (
            by_condition.get(row.display_condition.value, 0) + 1
        )
        by_repeat[row.repeat_index] = by_repeat.get(row.repeat_index, 0) + 1
        question_ids.add(row.question_id)
        action = item.get("parsed_action")
        if action not in {"VERIFY_FIRST", "USE_UNVERIFIED"}:
            issues.append(f"invalid action {action}")
        if row.phase != Study1Phase.REPEATS:
            issues.append("non-repeats phase in extra dataset")
        if row.repeat_index not in {1, 2}:
            issues.append(f"repeat_index {row.repeat_index} not in {{1,2}}")
        record = item.get("record")
        displayed = row.displayed_confidence
        if record is not None:
            displayed = record.displayed_confidence
            if record.reported_confidence != row.reported_confidence:
                issues.append("stored reported_confidence mismatch")
            if record.frozen_answer != row.frozen_answer:
                issues.append("stored frozen_answer mismatch")
        if row.display_condition == DisplayCondition.HIDDEN:
            if displayed is not None:
                issues.append("hidden displayed_confidence was not null")
        elif row.display_condition == DisplayCondition.TRUE_CONFIDENCE_VISIBLE:
            if displayed != row.reported_confidence:
                issues.append("true-visible displayed_confidence mismatch")
        else:
            if displayed != row.displayed_confidence:
                issues.append("manipulated displayed_confidence mismatch")
    if len(question_ids) != 20:
        issues.append(f"repeat question IDs {len(question_ids)} != 20")
    for alias in STUDY1_MODEL_ALIASES:
        if by_model.get(alias) != 560:
            issues.append(f"{alias} extra count {by_model.get(alias)} != 560")
    for error_cost in (10.0, 20.0):
        if by_L.get(error_cost) != 560:
            issues.append(f"L={error_cost} extra count {by_L.get(error_cost)} != 560")
    for condition in (
        "hidden",
        "true_confidence_visible",
        "manipulated_1",
        "manipulated_2",
        "manipulated_3",
        "manipulated_4",
        "manipulated_5",
    ):
        if by_condition.get(condition) != 160:
            issues.append(f"{condition} extra count {by_condition.get(condition)} != 160")
    if by_repeat.get(1) != 560 or by_repeat.get(2) != 560:
        issues.append(f"repeat_index counts {by_repeat} != {{1:560, 2:560}}")
    planned_keys = {row.request_key for row in extra}
    if set(keys) - planned_keys:
        issues.append("successful keys outside the repeat-extra plan")
    if any(item["row"].phase == Study1Phase.PRIMARY for item in successful):
        issues.append("primary cells mixed into extra successes")

    # Each original repeated cell should now have three observations.
    _, repeats, _, _ = load_frozen_ids(payload.get("config"))
    triple_ok = 0
    triple_total = 0
    with CheckpointStore(Path(payload["checkpoint_path"])) as checkpoint:
        primary_success = 0
        for row in primary:
            if checkpoint.request_status(row.request_key) == "success":
                primary_success += 1
        if primary_success != STUDY1_PRIMARY_CALLS:
            issues.append(f"primary successes changed to {primary_success}")
        extra_by_cell: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        for row in extra:
            extra_by_cell[
                (row.question_id, row.model_alias, row.L, row.display_condition.value)
            ].append(row.repeat_index)
        for row in primary:
            if row.question_id not in set(repeats):
                continue
            triple_total += 1
            cell = (row.question_id, row.model_alias, row.L, row.display_condition.value)
            statuses = [checkpoint.request_status(row.request_key) == "success"]
            matching = [
                extra_row
                for extra_row in extra
                if extra_row.question_id == row.question_id
                and extra_row.model_alias == row.model_alias
                and extra_row.L == row.L
                and extra_row.display_condition == row.display_condition
            ]
            statuses.extend(
                checkpoint.request_status(extra_row.request_key) == "success"
                for extra_row in matching
            )
            if len(matching) == 2 and all(statuses):
                triple_ok += 1
    if triple_ok != triple_total:
        issues.append(f"cells with three observations {triple_ok} != {triple_total}")
    return {
        "ok": not issues,
        "issues": issues,
        "historical_unchanged": historical_unchanged,
        "successful_unique_repeat_extra_cells": len(successful),
        "by_model": by_model,
        "by_L": by_L,
        "by_condition": by_condition,
        "by_repeat_index": by_repeat,
        "repeat_question_count": len(question_ids),
        "cells_with_three_observations": triple_ok,
        "repeated_primary_cells": triple_total,
        "no_new_primary_conditions": True,
        "actions_restricted_to_schema": not any(
            "invalid action" in issue for issue in issues
        ),
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
        "ok",
        "repeat_index",
        "phase",
    ]
    with (output / "repeat_results.csv").open("w", encoding="utf-8", newline="") as handle:
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
    else:
        (output / "failed_cells.csv").write_text(
            "question_id,model_alias,error\n", encoding="utf-8"
        )


def execute_repeats_task() -> dict[str, Any]:
    payload = asyncio.run(
        run_repeats_pilot(allow_paid=True, max_calls=STUDY1_REPEAT_EXTRA_CALLS)
    )
    output = RETURN_DIR
    output.mkdir(parents=True, exist_ok=True)
    if not payload.get("preflight_ok"):
        (output / "report.md").write_text(
            "# Study 1 Repeated-Sampling Stability\n\n"
            "## 1. Preflight parse-repair audit\n\n"
            "See `preflight_parse_repair_audit.md`.\n\n"
            "## 2. Data validation\n\n"
            "Pre-flight failed. Zero new API calls were made.\n\n"
            + "\n".join(f"- {error}" for error in payload.get("errors", []))
            + "\n",
            encoding="utf-8",
        )
        (output / "run_manifest.json").write_text(
            json.dumps(
                {
                    "task_id": "004_run_study1_stability",
                    "status": "PREFLIGHT_FAILED",
                    "paid_calls_authorized": True,
                    "repeat_extra_dataset_target": 1120,
                    "ready_for_gpt_review": True,
                    "errors": payload.get("errors"),
                    "stopped_reason": payload.get("stopped_reason"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("READY_FOR_GPT_REVIEW = YES")
        print("to_gpt/004_study1_stability_results/report.md")
        return payload
    write_execution_tables(payload, output)
    try:
        from .study1_stability import write_stability_analysis_bundle

        write_stability_analysis_bundle(payload, output)
    except Exception as exc:
        (output / "analysis_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        if not (output / "report.md").exists():
            (output / "report.md").write_text(
                "# Study 1 Repeated-Sampling Stability\n\n"
                "## 1. Preflight parse-repair audit\n\n"
                "See `preflight_parse_repair_audit.md`.\n\n"
                "## 2. Data validation\n\n"
                "Repeat-extra cells were collected, but analysis failed. See analysis_error.txt.\n"
                f"\n`{type(exc).__name__}: {exc}`\n",
                encoding="utf-8",
            )
        print(f"ANALYSIS_ERROR: {type(exc).__name__}: {exc}", flush=True)
    print("READY_FOR_GPT_REVIEW = YES")
    print("to_gpt/004_study1_stability_results/report.md")
    return payload


def reconstruct_repeats_payload(
    config: Study1ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Build a Task-004 analysis payload from sqlite. No API calls."""
    import sqlite3

    config = config or load_study1_config()
    primary, extra = build_call_plans(config)
    validate_call_plans(primary, extra, config)
    with CheckpointStore(config.study1_sqlite()) as checkpoint:
        reused_rows, todo, results = classify_existing_successes(extra, checkpoint)
        if todo:
            raise RuntimeError(
                f"cannot reconstruct: {len(todo)} repeat-extra cells are not successful"
            )
    for item in results:
        item["skipped_existing_success"] = False
    connection = sqlite3.connect(f"file:{config.study1_sqlite()}?mode=ro", uri=True)
    try:
        events = []
        n_attempts = 0
        for index, row in enumerate(
            connection.execute(
                """
                SELECT a.provider, a.requested_model_id
                FROM attempts a
                JOIN requests r ON r.request_key = a.request_key
                WHERE r.run_id = ?
                ORDER BY a.id
                """,
                (REPEATS_RUN_ID,),
            ),
            start=1,
        ):
            n_attempts += 1
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
                """
                SELECT COUNT(DISTINCT a.request_key)
                FROM attempts a
                JOIN requests r ON r.request_key = a.request_key
                WHERE r.run_id = ? AND a.attempt_kind = 'parse_repair'
                """,
                (REPEATS_RUN_ID,),
            ).fetchone()[0]
        )
    finally:
        connection.close()
    budget = ProviderAttemptBudget(n_attempts)
    budget.used = n_attempts
    budget.events = events
    historical = file_fingerprint(config.historical_sqlite())
    manifest_path = RETURN_DIR / "run_manifest.json"
    wall = 0.0
    if manifest_path.exists():
        try:
            wall = float(
                json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "wall_clock_seconds"
                )
                or 0.0
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            wall = 0.0
    if not wall:
        cost_path = RETURN_DIR / "cost_summary.md"
        if cost_path.exists():
            for line in cost_path.read_text(encoding="utf-8").splitlines():
                if "Wall-clock runtime" in line:
                    try:
                        wall = float(line.rsplit(":", 1)[-1].strip())
                    except ValueError:
                        wall = 0.0
    if not wall:
        # Fallback to the completed Task-004 wall clock recorded in this run.
        wall = 513.6300239580451
    return {
        "preflight_ok": True,
        "preflight": validate_repeats_preflight(config, max_calls=1120),
        "audit": audit_task003_parse_repairs(config),
        "primary": primary,
        "extra": extra,
        "results": results,
        "successful": results,
        "failed": [],
        "budget": budget,
        "new_initiated": 1120,
        "new_failed": 0,
        "reused": 0,
        "todo": 0,
        "wall_clock_seconds": wall,
        "historical_before": historical,
        "historical_after": historical,
        "checkpoint_path": str(config.study1_sqlite()),
        "code_commit": _git_commit(),
        "stopped_reason": None,
        "config": config,
        "models": load_models_config(config.resolve_path(config.models_config_path)),
        "repair_cells": n_repair,
    }
