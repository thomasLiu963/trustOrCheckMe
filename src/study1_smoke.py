"""Task 002 Study 1 smoke test: 12 scientific cells, 20 provider attempts max."""

from __future__ import annotations

import asyncio
import csv
import difflib
import hashlib
import json
import sqlite3
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .checkpointing import CheckpointStore
from .config import PROJECT_ROOT, load_models_config
from .model_adapters import create_adapter
from .study1_plan import (
    Study1PlannedCall,
    _strip_confidence_number,
    build_call_plans,
    validate_call_plans,
)
from .study1_prompts import (
    STUDY1_PROMPT_FAMILY,
    STUDY1_PROMPT_VERSION,
    build_study1_repair_prompt,
    parse_study1_verification_response,
)
from .study1_runner import Study1AuthorizationError
from .study1_sample import assert_not_v2_write_target, load_frozen_ids
from .study1_schemas import (
    DisplayCondition,
    Study1DecisionRecord,
    Study1ExperimentConfig,
    Study1Phase,
    load_study1_config,
)

SMOKE_SCIENTIFIC_CELL_CAP = 12
SMOKE_PROVIDER_ATTEMPT_CAP = 20
SMOKE_RUN_ID = "study1-smoke-002"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / "002_study1_smoke_test"

# displayed token: None = hidden, "reported" = true-visible, float = manipulated grid value
SMOKE_SPECS: tuple[tuple[str, float, DisplayCondition, float | str | None], ...] = (
    ("openai_gpt56_sol", 10.0, DisplayCondition.HIDDEN, None),
    ("openai_gpt56_sol", 10.0, DisplayCondition.TRUE_CONFIDENCE_VISIBLE, "reported"),
    ("openai_gpt56_sol", 10.0, DisplayCondition.MANIPULATED_3, 0.89),
    ("openai_gpt56_sol", 10.0, DisplayCondition.MANIPULATED_4, 0.91),
    ("openai_gpt56_sol", 20.0, DisplayCondition.MANIPULATED_3, 0.94),
    ("openai_gpt56_sol", 20.0, DisplayCondition.MANIPULATED_4, 0.96),
    ("anthropic_sonnet5", 10.0, DisplayCondition.HIDDEN, None),
    ("anthropic_sonnet5", 10.0, DisplayCondition.TRUE_CONFIDENCE_VISIBLE, "reported"),
    ("anthropic_sonnet5", 10.0, DisplayCondition.MANIPULATED_3, 0.89),
    ("anthropic_sonnet5", 10.0, DisplayCondition.MANIPULATED_4, 0.91),
    ("anthropic_sonnet5", 20.0, DisplayCondition.MANIPULATED_3, 0.94),
    ("anthropic_sonnet5", 20.0, DisplayCondition.MANIPULATED_4, 0.96),
)


class ProviderAttemptCapReached(Study1AuthorizationError):
    """Raised when the smoke-test provider-attempt cap would be exceeded."""


class ProviderAttemptBudget:
    def __init__(self, cap: int = SMOKE_PROVIDER_ATTEMPT_CAP) -> None:
        self.cap = int(cap)
        self.used = 0
        self.events: list[dict[str, Any]] = []

    def consume(self, **meta: Any) -> None:
        if self.used >= self.cap:
            raise ProviderAttemptCapReached(
                f"provider attempt cap {self.cap} already reached; STOP"
            )
        self.used += 1
        event = {"attempt_index": self.used, **meta}
        self.events.append(event)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
            cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def smoke_question_id(config: Study1ExperimentConfig | None = None) -> str:
    selected, _repeats, _selected_hash, _repeat_hash = load_frozen_ids(config)
    return selected[0]


def select_smoke_calls(
    config: Study1ExperimentConfig | None = None,
) -> tuple[str, list[Study1PlannedCall]]:
    config = config or load_study1_config()
    question_id = smoke_question_id(config)
    primary, extra = build_call_plans(config)
    validate_call_plans(primary, extra, config)
    selected: list[Study1PlannedCall] = []
    for alias, error_cost, condition, displayed in SMOKE_SPECS:
        matches = [
            row
            for row in primary
            if row.question_id == question_id
            and row.model_alias == alias
            and row.L == error_cost
            and row.display_condition == condition
            and row.repeat_index == 0
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected 1 planned cell for {alias} L={error_cost} "
                f"{condition.value}; found {len(matches)}"
            )
        row = matches[0]
        if displayed == "reported":
            if row.displayed_confidence != row.reported_confidence:
                raise RuntimeError("true-visible displayed_confidence != reported")
        elif displayed is None:
            if row.displayed_confidence is not None:
                raise RuntimeError("hidden cell stored a displayed_confidence")
        elif row.displayed_confidence != displayed:
            raise RuntimeError(
                f"displayed_confidence {row.displayed_confidence} != {displayed}"
            )
        selected.append(row)
    if len(selected) != SMOKE_SCIENTIFIC_CELL_CAP:
        raise RuntimeError(f"smoke selection produced {len(selected)} cells")
    if extra and any(row.question_id == question_id and row.repeat_index != 0 for row in selected):
        raise RuntimeError("smoke selection included repeats")
    return question_id, selected


def _response_value(response: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(response, Mapping) and name in response:
            value = response[name]
        else:
            value = getattr(response, name, None)
        if value is not None:
            return value
    return default


def estimate_cost_usd(spec: Any, input_tokens: int | None, output_tokens: int | None) -> float:
    pricing = spec.pricing_per_million_tokens
    return (
        int(input_tokens or 0) * pricing.input
        + int(output_tokens or 0) * pricing.output
    ) / 1_000_000


def validate_preflight(
    *,
    question_id: str,
    calls: Sequence[Study1PlannedCall],
    config: Study1ExperimentConfig,
    scientific_cell_cap: int,
    provider_attempt_cap: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if scientific_cell_cap != SMOKE_SCIENTIFIC_CELL_CAP:
        errors.append(f"scientific_cell_cap must be 12, got {scientific_cell_cap}")
    if provider_attempt_cap != SMOKE_PROVIDER_ATTEMPT_CAP:
        errors.append(f"provider_attempt_cap must be 20, got {provider_attempt_cap}")
    if len(calls) != 12:
        errors.append(f"planned cells {len(calls)} != 12")
    question_ids = {row.question_id for row in calls}
    if question_ids != {question_id}:
        errors.append(f"question IDs drifted: {sorted(question_ids)}")
    models = [row.model_alias for row in calls]
    if set(models) != {"openai_gpt56_sol", "anthropic_sonnet5"}:
        errors.append(f"unexpected models: {sorted(set(models))}")
    if any(alias not in {"openai_gpt56_sol", "anthropic_sonnet5"} for alias in models):
        errors.append("Gemini/Grok/other model leaked into smoke cells")
    if any(row.repeat_index != 0 for row in calls):
        errors.append("repeat cells are not authorized")
    keys = [row.request_key for row in calls]
    if len(set(keys)) != len(keys):
        errors.append("duplicate request keys")
    try:
        assert_not_v2_write_target(config.study1_sqlite(), config)
    except RuntimeError as exc:
        errors.append(str(exc))
    if config.study1_sqlite().resolve() == config.historical_sqlite().resolve():
        errors.append("Study 1 write path equals historical V2 sqlite")
    if config.study1_sqlite().name == "v2.sqlite3":
        errors.append("Study 1 write path is named v2.sqlite3")

    expected = [
        (alias, error_cost, condition.value, displayed)
        for alias, error_cost, condition, displayed in SMOKE_SPECS
    ]
    observed = []
    for row in calls:
        displayed: float | str | None
        if row.display_condition == DisplayCondition.TRUE_CONFIDENCE_VISIBLE:
            displayed = "reported"
        elif row.display_condition == DisplayCondition.HIDDEN:
            displayed = None
        else:
            displayed = row.displayed_confidence
        observed.append(
            (row.model_alias, row.L, row.display_condition.value, displayed)
        )
    if observed != expected:
        errors.append(f"cell roster mismatch: {observed} vs {expected}")

    for left, right in (
        (2, 3),
        (4, 5),
        (8, 9),
        (10, 11),
    ):
        a, b = calls[left], calls[right]
        if _strip_confidence_number(a.prompt) != _strip_confidence_number(b.prompt):
            errors.append(
                f"manipulated pair {a.display_condition.value}/"
                f"{b.display_condition.value} differs by more than the number"
            )

    for row in calls:
        if row.C != 1.0:
            errors.append(f"C must be 1, got {row.C}")
        if row.L not in {10.0, 20.0}:
            errors.append(f"unexpected L {row.L}")
        if not row.frozen_answer:
            errors.append("missing frozen Stage-1 answer")
        if row.reported_confidence is None:
            errors.append("missing historical reported_confidence")
        if "The AI SYSTEM controls" not in row.prompt:
            errors.append("AI-authority wording missing")
        if "HUMAN USER" in row.prompt:
            errors.append("human-authority wording leaked")
        if '{"action":"VERIFY_FIRST"}' not in row.prompt:
            errors.append("binary action schema missing")
        if row.display_condition == DisplayCondition.HIDDEN:
            if "The AI previously estimated a " in row.prompt:
                errors.append("hidden prompt contains confidence sentence")
        else:
            if "The AI previously estimated a " not in row.prompt:
                errors.append("visible prompt missing confidence sentence")

    return {
        "ok": not errors,
        "errors": errors,
        "question_id": question_id,
        "n_cells": len(calls),
        "models": ["openai_gpt56_sol", "anthropic_sonnet5"],
        "scientific_cell_cap": scientific_cell_cap,
        "provider_attempt_cap": provider_attempt_cap,
        "study1_checkpoint_path": str(config.study1_sqlite()),
        "historical_checkpoint_path": str(config.historical_sqlite()),
        "request_keys": keys,
        "cells": [
            {
                "model_alias": row.model_alias,
                "model_endpoint": row.model_endpoint,
                "L": row.L,
                "C": row.C,
                "display_condition": row.display_condition.value,
                "reported_confidence": row.reported_confidence,
                "displayed_confidence": row.displayed_confidence,
                "frozen_answer": row.frozen_answer,
                "stage1_correct": row.stage1_correct,
                "request_key": row.request_key,
                "prompt_hash": row.prompt_hash,
            }
            for row in calls
        ],
    }


def _install_attempt_cap(adapter: Any, budget: ProviderAttemptBudget) -> None:
    original_send = adapter._send

    async def capped_send(payload: Mapping[str, Any]) -> Any:
        budget.consume(
            provider=adapter.provider,
            requested_model_id=adapter.api_model,
            model_alias=adapter.model_alias,
        )
        return await original_send(payload)

    adapter._send = capped_send  # type: ignore[method-assign]


def _write_prompt_artifacts(calls: Sequence[Study1PlannedCall], output: Path) -> None:
    rendered = output / "rendered_prompts"
    diffs = output / "prompt_diffs"
    rendered.mkdir(parents=True, exist_ok=True)
    diffs.mkdir(parents=True, exist_ok=True)
    for row in calls:
        name = f"{row.model_alias}__L{int(row.L)}__{row.display_condition.value}.txt"
        (rendered / name).write_text(row.prompt + "\n", encoding="utf-8")
    pairs = [
        (calls[2], calls[3], "gpt_L10_0.89_vs_0.91.diff"),
        (calls[4], calls[5], "gpt_L20_0.94_vs_0.96.diff"),
        (calls[8], calls[9], "claude_L10_0.89_vs_0.91.diff"),
        (calls[10], calls[11], "claude_L20_0.94_vs_0.96.diff"),
    ]
    for left, right, filename in pairs:
        text = "\n".join(
            difflib.unified_diff(
                left.prompt.splitlines(),
                right.prompt.splitlines(),
                fromfile=left.display_condition.value,
                tofile=right.display_condition.value,
                lineterm="",
            )
        )
        (diffs / filename).write_text(text + "\n", encoding="utf-8")


async def _execute_cell(
    *,
    row: Study1PlannedCall,
    adapter: Any,
    spec: Any,
    checkpoint: CheckpointStore,
    run_id: str,
    budget: ProviderAttemptBudget,
    code_commit: str | None,
    max_parse_repairs: int,
) -> dict[str, Any]:
    attempts_before = budget.used
    prompt = row.prompt
    last_raw = ""
    parse_repairs = 0
    record_kwargs = dict(
        phase=Study1Phase.PRIMARY,
        run_id=run_id,
        request_key=row.request_key,
        question_id=row.question_id,
        model_id=row.model_alias,
        model_alias=row.model_alias,
        model_endpoint=row.model_endpoint,
        provider=row.provider,
        L=row.L,
        C=row.C,
        reported_confidence=row.reported_confidence,
        displayed_confidence=row.displayed_confidence,
        display_condition=row.display_condition,
        display_source_condition=row.display_source_condition,
        frozen_answer=row.frozen_answer,
        stage1_correct=row.stage1_correct,
        prompt_version=STUDY1_PROMPT_VERSION,
        prompt_family=STUDY1_PROMPT_FAMILY,
        prompt_hash=row.prompt_hash,
        selected_sample_hash=row.selected_sample_hash,
        code_commit=code_commit,
        repeat_index=0,
        historical_answer_request_key=row.historical_answer_request_key,
        historical_confidence_request_key=row.historical_confidence_request_key,
        model_settings={
            "requested_model_id": spec.api_model,
            "provider": spec.provider,
            "api_style": spec.api_style,
            "max_output_tokens": spec.max_output_tokens,
        },
    )
    for parse_attempt in range(max_parse_repairs + 1):
        attempt_kind = "generation" if parse_attempt == 0 else "parse_repair"
        if parse_attempt:
            parse_repairs += 1
            prompt = build_study1_repair_prompt(row.prompt, last_raw)
        started = time.perf_counter()
        try:
            response = await adapter.generate(
                stage="verification",
                prompt=prompt,
                request_key=row.request_key,
                allow_paid=True,
                attempt_kind=attempt_kind,
            )
        except Exception as error:
            cause: BaseException | None = error
            seen: set[int] = set()
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                if isinstance(cause, ProviderAttemptCapReached):
                    checkpoint.record_attempt(
                        request_key=row.request_key,
                        attempt_kind=attempt_kind,
                        success=False,
                        provider=adapter.provider,
                        requested_model_id=adapter.api_model,
                        latency_seconds=time.perf_counter() - started,
                        error=cause,
                    )
                    checkpoint.mark_failed(row.request_key, cause)
                    raise cause
                cause = cause.__cause__ or getattr(cause, "__context__", None)
            checkpoint.record_attempt(
                request_key=row.request_key,
                attempt_kind=attempt_kind,
                success=False,
                provider=adapter.provider,
                requested_model_id=adapter.api_model,
                latency_seconds=time.perf_counter() - started,
                error=error,
                sanitized_payload=adapter.prepare_request(
                    stage="verification", prompt=prompt
                ).sanitized_payload(),
            )
            checkpoint.mark_failed(row.request_key, error)
            return {
                "ok": False,
                "parse_status": "failed",
                "parsed_action": None,
                "raw_response": None,
                "error": str(error),
                "provider_attempts": budget.used - attempts_before,
                "parse_repairs": parse_repairs,
                "returned_model_id": None,
                "input_tokens": None,
                "output_tokens": None,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "estimated_cost_usd": 0.0,
                "attempt_number": 1,
            }

        raw = str(_response_value(response, "raw_response", default="") or "")
        last_raw = raw
        refusal = bool(_response_value(response, "refused", "refusal", default=False))
        returned_model_id = _response_value(
            response, "provider_model_id", "returned_model_id"
        )
        input_tokens = _response_value(response, "input_tokens")
        output_tokens = _response_value(response, "output_tokens")
        latency_seconds = float(_response_value(response, "latency_seconds") or 0.0)
        try:
            if refusal:
                raise ValueError("Provider returned a refusal")
            payload = parse_study1_verification_response(raw)
        except (TypeError, ValueError) as error:
            checkpoint.record_attempt(
                request_key=row.request_key,
                attempt_kind=attempt_kind,
                success=False,
                provider=adapter.provider,
                requested_model_id=adapter.api_model,
                returned_model_id=returned_model_id,
                latency_seconds=latency_seconds,
                raw_output=raw,
                parse_error=str(error),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=_response_value(response, "total_tokens"),
                finish_reason=_response_value(response, "finish_reason"),
                refusal=refusal,
                sanitized_payload=adapter.prepare_request(
                    stage="verification", prompt=prompt
                ).sanitized_payload(),
            )
            if parse_attempt < max_parse_repairs:
                continue
            failed = Study1DecisionRecord(
                **record_kwargs,
                returned_model_id=returned_model_id,
                api_timestamp=datetime.now(UTC).isoformat(),
                attempt_number=parse_attempt + 1,
                raw_response=raw,
                parsed_action=None,
                parse_status="parse_failed" if not refusal else "refused",
                latency_ms=latency_seconds * 1000,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimate_cost_usd(spec, input_tokens, output_tokens),
            )
            checkpoint.mark_failed(row.request_key, error)
            return {
                "ok": False,
                "parse_status": failed.parse_status,
                "parsed_action": None,
                "raw_response": raw,
                "error": str(error),
                "provider_attempts": budget.used - attempts_before,
                "parse_repairs": parse_repairs,
                "returned_model_id": returned_model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_seconds * 1000,
                "estimated_cost_usd": failed.estimated_cost_usd,
                "attempt_number": parse_attempt + 1,
                "record": failed,
            }

        attempt_number = checkpoint.record_attempt(
            request_key=row.request_key,
            attempt_kind=attempt_kind,
            success=True,
            provider=adapter.provider,
            requested_model_id=adapter.api_model,
            returned_model_id=returned_model_id,
            latency_seconds=latency_seconds,
            raw_output=raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=_response_value(response, "total_tokens"),
            finish_reason=_response_value(response, "finish_reason"),
            refusal=False,
            sanitized_payload=adapter.prepare_request(
                stage="verification", prompt=prompt
            ).sanitized_payload(),
        )
        record = Study1DecisionRecord(
            **record_kwargs,
            returned_model_id=returned_model_id,
            api_timestamp=datetime.now(UTC).isoformat(),
            attempt_number=int(attempt_number),
            raw_response=raw,
            parsed_action=payload.action,
            parse_status="success",
            latency_ms=latency_seconds * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimate_cost_usd(spec, input_tokens, output_tokens),
        )
        checkpoint.mark_success(row.request_key, record)
        return {
            "ok": True,
            "parse_status": "success",
            "parsed_action": payload.action.value,
            "raw_response": raw,
            "error": None,
            "provider_attempts": budget.used - attempts_before,
            "parse_repairs": parse_repairs,
            "returned_model_id": returned_model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_seconds * 1000,
            "estimated_cost_usd": record.estimated_cost_usd,
            "attempt_number": int(attempt_number),
            "record": record,
        }
    raise RuntimeError("parse-repair loop ended without a result")


def _sqlite_has_request(path: Path, request_key: str) -> bool:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT 1 FROM requests WHERE request_key = ? LIMIT 1",
            (request_key,),
        ).fetchone()
    except sqlite3.Error:
        return False
    finally:
        connection.close()
    return row is not None


async def run_smoke_test(
    *,
    allow_paid: bool,
    scientific_cell_cap: int = SMOKE_SCIENTIFIC_CELL_CAP,
    provider_attempt_cap: int = SMOKE_PROVIDER_ATTEMPT_CAP,
    config: Study1ExperimentConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if not allow_paid:
        raise Study1AuthorizationError("smoke test requires allow_paid=True")
    if scientific_cell_cap != SMOKE_SCIENTIFIC_CELL_CAP:
        raise Study1AuthorizationError("Task 002 scientific cell cap must be exactly 12")
    if provider_attempt_cap != SMOKE_PROVIDER_ATTEMPT_CAP:
        raise Study1AuthorizationError("Task 002 provider attempt cap must be exactly 20")

    config = config or load_study1_config()
    output = output_dir or RETURN_DIR
    output.mkdir(parents=True, exist_ok=True)
    question_id, calls = select_smoke_calls(config)
    preflight = validate_preflight(
        question_id=question_id,
        calls=calls,
        config=config,
        scientific_cell_cap=scientific_cell_cap,
        provider_attempt_cap=provider_attempt_cap,
    )
    (output / "preflight_manifest.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    _write_prompt_artifacts(calls, output)
    if not preflight["ok"]:
        return {
            "preflight_ok": False,
            "preflight": preflight,
            "api_calls_made": 0,
            "provider_attempts_total": 0,
            "errors": preflight["errors"],
            "question_id": question_id,
            "calls": calls,
        }

    historical_before = file_fingerprint(config.historical_sqlite())
    load_dotenv(PROJECT_ROOT / ".env")
    models = load_models_config(config.resolve_path(config.models_config_path))
    budget = ProviderAttemptBudget(provider_attempt_cap)
    adapters = {}
    for alias in ("openai_gpt56_sol", "anthropic_sonnet5"):
        adapters[alias] = create_adapter(
            alias,
            models.models[alias],
            max_transient_retries=int(config.model_inference["max_transient_retries"]),
        )
        _install_attempt_cap(adapters[alias], budget)

    checkpoint_path = config.study1_sqlite()
    assert_not_v2_write_target(checkpoint_path, config)
    code_commit = _git_commit()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    stopped_reason = None
    scientific_cells_initiated = 0
    with CheckpointStore(checkpoint_path) as checkpoint:
        for row in calls:
            if scientific_cells_initiated >= scientific_cell_cap:
                stopped_reason = "scientific cell cap reached before initiating another cell"
                break
            if budget.used >= budget.cap:
                stopped_reason = "provider attempt cap reached before initiating another cell"
                break
            status = checkpoint.request_status(row.request_key)
            if status == "success":
                existing = checkpoint.get_record(row.request_key) or {}
                parsed = existing.get("parsed_action")
                if isinstance(parsed, dict):
                    parsed = parsed.get("value") or parsed.get("action")
                record = None
                try:
                    record = Study1DecisionRecord.model_validate(existing)
                except (TypeError, ValueError):
                    record = None
                results.append(
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
                        "estimated_cost_usd": existing.get("estimated_cost_usd") or 0.0,
                        "attempt_number": existing.get("attempt_number") or 1,
                        "row": row,
                        "record": record,
                    }
                )
                continue
            checkpoint.register_request(
                request_key=row.request_key,
                run_id=SMOKE_RUN_ID,
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
                    "smoke_test": True,
                },
            )
            scientific_cells_initiated += 1
            try:
                outcome = await _execute_cell(
                    row=row,
                    adapter=adapters[row.model_alias],
                    spec=models.models[row.model_alias],
                    checkpoint=checkpoint,
                    run_id=SMOKE_RUN_ID,
                    budget=budget,
                    code_commit=code_commit,
                    max_parse_repairs=int(config.model_inference["max_parse_repairs"]),
                )
            except ProviderAttemptCapReached as error:
                stopped_reason = str(error)
                results.append(
                    {
                        "ok": False,
                        "parse_status": "stopped_by_attempt_cap",
                        "parsed_action": None,
                        "raw_response": None,
                        "provider_attempts": 0,
                        "parse_repairs": 0,
                        "returned_model_id": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "latency_ms": None,
                        "estimated_cost_usd": 0.0,
                        "attempt_number": 1,
                        "row": row,
                        "error": str(error),
                    }
                )
                break
            outcome["row"] = row
            results.append(outcome)
            if not outcome["ok"] and isinstance(outcome.get("error"), str) and "unavailable" in outcome["error"].lower():
                stopped_reason = (
                    f"{row.model_alias} endpoint unavailable; not substituting another model"
                )
                break

    elapsed = time.perf_counter() - started
    historical_after = file_fingerprint(config.historical_sqlite())
    return {
        "preflight_ok": True,
        "preflight": preflight,
        "question_id": question_id,
        "calls": calls,
        "results": results,
        "budget": budget,
        "scientific_cells_initiated": scientific_cells_initiated,
        "wall_clock_seconds": elapsed,
        "historical_before": historical_before,
        "historical_after": historical_after,
        "checkpoint_path": str(checkpoint_path),
        "code_commit": code_commit,
        "stopped_reason": stopped_reason,
        "config": config,
        "models": models,
    }


def postvalidate_smoke(payload: Mapping[str, Any]) -> dict[str, Any]:
    config: Study1ExperimentConfig = payload["config"]
    calls: Sequence[Study1PlannedCall] = payload["calls"]
    results: Sequence[dict[str, Any]] = payload["results"]
    budget: ProviderAttemptBudget = payload["budget"]
    issues: list[str] = []
    before = payload["historical_before"]
    after = payload["historical_after"]
    historical_unchanged = (
        before["sha256"] == after["sha256"]
        and before["size_bytes"] == after["size_bytes"]
    )
    if not historical_unchanged:
        issues.append("historical V2 sqlite fingerprint changed")
    if payload["scientific_cells_initiated"] > SMOKE_SCIENTIFIC_CELL_CAP:
        issues.append("more than 12 scientific cells were initiated")
    if budget.used > SMOKE_PROVIDER_ATTEMPT_CAP:
        issues.append("provider attempts exceeded 20")
    if len(results) > 12:
        issues.append("more than 12 result rows")
    leaked = []
    historical = config.historical_sqlite()
    for row in calls[: payload["scientific_cells_initiated"]]:
        if _sqlite_has_request(historical, row.request_key):
            leaked.append(row.request_key)
    if leaked:
        issues.append("Study 1 request keys found in historical V2 sqlite")
    for item in results:
        row: Study1PlannedCall = item["row"]
        record = item.get("record")
        if record is None:
            continue
        if record.reported_confidence != row.reported_confidence:
            issues.append("reported_confidence drifted from historical Stage-2")
        if row.display_condition == DisplayCondition.HIDDEN:
            if record.displayed_confidence is not None:
                issues.append("hidden displayed_confidence was not null")
        elif row.display_condition == DisplayCondition.TRUE_CONFIDENCE_VISIBLE:
            if record.displayed_confidence != row.reported_confidence:
                issues.append("true-visible displayed_confidence != reported")
        else:
            if record.displayed_confidence != row.displayed_confidence:
                issues.append("manipulated displayed_confidence mismatch")
        if record.frozen_answer != row.frozen_answer:
            issues.append("frozen answer drifted from historical Stage-1")
        if record.parse_status == "success" and record.parsed_action is not None:
            if record.parsed_action.value not in {"VERIFY_FIRST", "USE_UNVERIFIED"}:
                issues.append(f"invalid parsed action {record.parsed_action}")
    return {
        "ok": not issues,
        "issues": issues,
        "historical_unchanged": historical_unchanged,
        "scientific_cells_initiated": payload["scientific_cells_initiated"],
        "provider_attempts_total": budget.used,
    }


def _study1_sqlite_snapshot(
    path: Path, request_keys: Sequence[str]
) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "n_requests": 0, "n_attempts": 0, "smoke_rows": []}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        n_requests = int(
            connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        )
        n_attempts = int(
            connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        )
        smoke_rows = []
        if request_keys:
            placeholders = ",".join("?" for _ in request_keys)
            rows = connection.execute(
                "SELECT request_key, status, record_json, attempt_count "
                f"FROM requests WHERE request_key IN ({placeholders})",
                list(request_keys),
            ).fetchall()
            for row in rows:
                record = json.loads(row[2]) if row[2] else {}
                smoke_rows.append(
                    {
                        "request_key": row[0],
                        "status": row[1],
                        "attempt_count": row[3],
                        "reported_confidence": record.get("reported_confidence"),
                        "displayed_confidence": record.get("displayed_confidence"),
                        "parsed_action": record.get("parsed_action"),
                        "frozen_answer": record.get("frozen_answer"),
                    }
                )
    except sqlite3.Error as error:
        return {"exists": True, "error": str(error)}
    finally:
        connection.close()
    return {
        "exists": True,
        "n_requests": n_requests,
        "n_attempts": n_attempts,
        "smoke_rows": smoke_rows,
    }


def _counts(results: Sequence[dict[str, Any]], budget: ProviderAttemptBudget) -> dict[str, Any]:
    successful = sum(1 for item in results if item.get("ok"))
    failed = sum(1 for item in results if not item.get("ok"))
    repair_attempts = sum(int(item.get("parse_repairs") or 0) for item in results)
    network_cells = sum(
        1 for item in results if int(item.get("provider_attempts") or 0) > 0
    )
    retry_attempts = max(0, budget.used - network_cells - repair_attempts)
    total_cost = sum(float(item.get("estimated_cost_usd") or 0.0) for item in results)
    metadata: dict[str, list[str]] = {}
    for item in results:
        alias = item["row"].model_alias
        returned = item.get("returned_model_id")
        if returned:
            metadata.setdefault(alias, [])
            if returned not in metadata[alias]:
                metadata[alias].append(returned)
    return {
        "successful": successful,
        "failed": failed,
        "repair_attempts": repair_attempts,
        "retry_attempts": retry_attempts,
        "network_cells": network_cells,
        "total_cost": total_cost,
        "metadata": metadata,
    }


def _ready_for_task_003(payload: Mapping[str, Any], post: Mapping[str, Any], counts: Mapping[str, Any]) -> bool:
    skipped = sum(
        1
        for item in payload.get("results", [])
        if item.get("skipped_existing_success")
    )
    initiated = int(payload.get("scientific_cells_initiated") or 0)
    return bool(
        payload.get("preflight_ok")
        and post.get("ok")
        and counts["successful"] == 12
        and counts["failed"] == 0
        and payload["budget"].used <= SMOKE_PROVIDER_ATTEMPT_CAP
        and initiated <= SMOKE_SCIENTIFIC_CELL_CAP
        and initiated + skipped == 12
        and payload.get("stopped_reason") is None
        and all(
            item.get("parse_status") == "success"
            and item.get("parsed_action") in {"VERIFY_FIRST", "USE_UNVERIFIED"}
            for item in payload.get("results", [])
        )
    )


def _write_changed_files(output: Path) -> None:
    paths: list[str] = []
    seen: set[str] = set()
    for relative in (
        "from_gpt/002_smoke_test_study1.md",
        "src/study1_smoke.py",
        "src/study1_cli.py",
        "src/study1_runner.py",
        "tests/test_study1.py",
        "docs/D1_DECISION_LOG.md",
    ):
        full = PROJECT_ROOT / relative
        if full.exists() and relative not in seen:
            paths.append(relative)
            seen.add(relative)
    if output.exists():
        for found in sorted(output.rglob("*")):
            if found.is_file():
                relative = str(found.relative_to(PROJECT_ROOT))
                if relative not in seen:
                    paths.append(relative)
                    seen.add(relative)
    (output / "changed_files.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")


def write_smoke_artifacts(payload: Mapping[str, Any], output: Path | None = None) -> None:
    output = output or RETURN_DIR
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw_responses").mkdir(parents=True, exist_ok=True)
    if not payload.get("preflight_ok"):
        errors = list(payload.get("errors") or [])
        (output / "report.md").write_text(
            "\n".join(
                [
                    "# Task 002 smoke test FAILED pre-flight",
                    "",
                    "Zero API calls were made.",
                    "",
                    f"- Question ID selected: `{payload.get('question_id')}`",
                    "- Pre-flight validation passed: **False**",
                    "- Scientific cells attempted: 0",
                    "- Raw provider request attempts: 0",
                    "- Historical V2 untouched: yes (no writes attempted)",
                    "",
                    "Validation errors:",
                    *[f"- {error}" for error in errors],
                    "",
                    "READY_FOR_TASK_003 = NO",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (output / "run_manifest.json").write_text(
            json.dumps(
                {
                    "task_id": "002_smoke_test_study1",
                    "status": "PREFLIGHT_FAILED",
                    "git_commit": _git_commit(),
                    "paid_calls_authorized": True,
                    "scientific_cell_cap": 12,
                    "provider_attempt_cap": 20,
                    "selected_question_id": payload.get("question_id"),
                    "models": ["openai_gpt56_sol", "anthropic_sonnet5"],
                    "L_values": [10.0, 20.0],
                    "planned_cells": 12,
                    "successful_cells": 0,
                    "failed_cells": 0,
                    "provider_attempts_total": 0,
                    "retry_attempts": 0,
                    "repair_attempts": 0,
                    "api_cost_usd": 0.0,
                    "historical_artifacts_modified": False,
                    "study1_checkpoint_path": None,
                    "provider_model_metadata": {},
                    "ready_for_task_003": False,
                    "errors": errors,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "smoke_test_cells.csv").write_text(
            "model,L,condition,reported_confidence,displayed_confidence,parsed_action,attempts,latency,tokens,cost\n",
            encoding="utf-8",
        )
        (output / "provider_attempts.csv").write_text(
            "attempt_index,provider,requested_model_id,model_alias\n",
            encoding="utf-8",
        )
        (output / "sqlite_validation.md").write_text(
            "# SQLite validation\n\nPre-flight failed. No Study 1 writes. Historical V2 was not opened for write.\n",
            encoding="utf-8",
        )
        (output / "cost_summary.md").write_text(
            "# Cost summary\n\n- Estimated API cost (USD): 0.000000\n- Provider request attempts: 0\n",
            encoding="utf-8",
        )
        _write_changed_files(output)
        return
    calls: Sequence[Study1PlannedCall] = payload["calls"]
    results: Sequence[dict[str, Any]] = payload["results"]
    budget: ProviderAttemptBudget = payload["budget"]
    post = postvalidate_smoke(payload)
    raw_dir = output / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "L",
        "condition",
        "reported_confidence",
        "displayed_confidence",
        "parsed_action",
        "parse_status",
        "ok",
        "attempts",
        "parse_repairs",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "returned_model_id",
        "request_key",
    ]
    with (output / "smoke_test_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row: Study1PlannedCall = item["row"]
            writer.writerow(
                {
                    "model": row.model_alias,
                    "L": row.L,
                    "condition": row.display_condition.value,
                    "reported_confidence": row.reported_confidence,
                    "displayed_confidence": (
                        "" if row.displayed_confidence is None else row.displayed_confidence
                    ),
                    "parsed_action": item.get("parsed_action") or "",
                    "parse_status": item.get("parse_status"),
                    "ok": item.get("ok"),
                    "attempts": item.get("provider_attempts"),
                    "parse_repairs": item.get("parse_repairs"),
                    "latency_ms": item.get("latency_ms"),
                    "input_tokens": item.get("input_tokens"),
                    "output_tokens": item.get("output_tokens"),
                    "cost_usd": item.get("estimated_cost_usd"),
                    "returned_model_id": item.get("returned_model_id") or "",
                    "request_key": row.request_key,
                }
            )
            raw = item.get("raw_response")
            if raw:
                name = (
                    f"{row.model_alias}__L{int(row.L)}__"
                    f"{row.display_condition.value}.txt"
                )
                (raw_dir / name).write_text(str(raw) + "\n", encoding="utf-8")
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
    counts = _counts(results, budget)
    successful = counts["successful"]
    failed = counts["failed"]
    repair_attempts = counts["repair_attempts"]
    retry_attempts = counts["retry_attempts"]
    total_cost = counts["total_cost"]
    metadata = counts["metadata"]
    ready = _ready_for_task_003(payload, post, counts)
    leaked = any("historical V2 sqlite" in issue for issue in post["issues"])
    snapshot = _study1_sqlite_snapshot(
        Path(payload["checkpoint_path"]),
        [row.request_key for row in calls],
    )
    snapshot_lines = [
        f"- Study 1 sqlite exists: **{snapshot.get('exists')}**",
        f"- Study 1 requests table rows: {snapshot.get('n_requests')}",
        f"- Study 1 attempts table rows: {snapshot.get('n_attempts')}",
        f"- Smoke request keys found in Study 1 sqlite: {len(snapshot.get('smoke_rows') or [])}",
    ]
    if snapshot.get("error"):
        snapshot_lines.append(f"- Study 1 sqlite read error: {snapshot['error']}")
    for row in snapshot.get("smoke_rows") or []:
        snapshot_lines.append(
            f"- `{row['request_key'][:16]}…` status={row['status']} "
            f"reported={row['reported_confidence']} displayed={row['displayed_confidence']} "
            f"action={row['parsed_action']}"
        )
    (output / "sqlite_validation.md").write_text(
        "\n".join(
            [
                "# SQLite validation",
                "",
                f"- Historical V2 unchanged: **{post['historical_unchanged']}**",
                f"- Before SHA-256: `{payload['historical_before']['sha256']}`",
                f"- After SHA-256: `{payload['historical_after']['sha256']}`",
                f"- Before size: {payload['historical_before']['size_bytes']}",
                f"- After size: {payload['historical_after']['size_bytes']}",
                f"- Before mtime_ns: {payload['historical_before']['mtime_ns']}",
                f"- After mtime_ns: {payload['historical_after']['mtime_ns']}",
                f"- Study 1 checkpoint: `{payload['checkpoint_path']}`",
                f"- Historical leaked keys: {'YES' if leaked else 'no'}",
                "",
                "## Study 1 checkpoint contents",
                "",
                *snapshot_lines,
                "",
                "Issues:" if post["issues"] else "No SQLite validation issues.",
                *[f"- {issue}" for issue in post["issues"]],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "cost_summary.md").write_text(
        "\n".join(
            [
                "# Cost summary",
                "",
                f"- Successful cells: {successful}",
                f"- Failed cells: {failed}",
                f"- Provider request attempts: {budget.used}",
                f"- Parse repairs: {repair_attempts}",
                f"- Transport retries beyond first attempt/repair: {retry_attempts}",
                f"- Estimated API cost (USD): {total_cost:.6f}",
                f"- Wall-clock seconds: {payload['wall_clock_seconds']:.3f}",
                "",
                "Pricing from `config/models.yaml` (`openai_gpt56_sol` $4/$20 per million input/output; `anthropic_sonnet5` $2/$10).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    gpt_conf = next(
        row.reported_confidence
        for row in calls
        if row.model_alias == "openai_gpt56_sol"
    )
    claude_conf = next(
        row.reported_confidence
        for row in calls
        if row.model_alias == "anthropic_sonnet5"
    )
    valid_actions = all(
        (not item.get("ok"))
        or item.get("parsed_action") in {"VERIFY_FIRST", "USE_UNVERIFIED"}
        for item in results
    )
    table_lines = [
        "| model | L | condition | reported | displayed | action | attempts | latency_ms | tokens | cost |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in results:
        row = item["row"]
        tokens = (item.get("input_tokens") or 0) + (item.get("output_tokens") or 0)
        displayed = (
            "null" if row.displayed_confidence is None else row.displayed_confidence
        )
        table_lines.append(
            "| "
            + " | ".join(
                [
                    row.model_alias,
                    str(int(row.L)),
                    row.display_condition.value,
                    str(row.reported_confidence),
                    str(displayed),
                    str(item.get("parsed_action") or item.get("parse_status")),
                    str(item.get("provider_attempts")),
                    "n/a"
                    if item.get("latency_ms") is None
                    else f"{item['latency_ms']:.0f}",
                    str(tokens),
                    f"{float(item.get('estimated_cost_usd') or 0):.6f}",
                ]
            )
            + " |"
        )
    engineering_issue = None
    if not ready:
        engineering_issue = (
            payload.get("stopped_reason")
            or ("; ".join(post["issues"]) if post["issues"] else "smoke test incomplete")
        )
    report = [
        "# Task 002 report — Study 1 API smoke test",
        "",
        "Engineering inspection only. n = 1 question. These 12 cells are not a scientific test.",
        "Do not treat action changes across fake confidence values as evidence for or against the hypothesis.",
        "",
        "## Required answers",
        "",
        f"- What question ID was used? `{payload['question_id']}` (first ID in frozen `data/development/study1_ids.json`)",
        f"- GPT historical reported confidence: `{gpt_conf}`",
        f"- Claude historical reported confidence: `{claude_conf}`",
        f"- Did pre-flight validation pass? **{payload['preflight_ok']}**",
        f"- Were exactly 12 scientific cells attempted? **{payload['scientific_cells_initiated'] == 12}** (initiated={payload['scientific_cells_initiated']})",
        f"- How many completed successfully? **{successful}**",
        f"- How many raw provider request attempts occurred? **{budget.used}**",
        f"- Were any retries needed? **{retry_attempts > 0}** (retry attempts={retry_attempts})",
        f"- Were any parse repairs needed? **{repair_attempts > 0}** (repair attempts={repair_attempts})",
        f"- Did every valid response parse into the two-action schema? **{valid_actions}**",
        f"- Were reported_confidence and displayed_confidence stored correctly? **{post['ok'] and not any('confidence' in issue for issue in post['issues'])}**",
        f"- Was historical V2 untouched? **{post['historical_unchanged']}**",
        f"- Exact provider model/version metadata returned: `{json.dumps(metadata)}`",
        f"- Total measured/estimated API cost: **${total_cost:.6f}**",
        f"- Wall-clock runtime: **{payload['wall_clock_seconds']:.3f} seconds**",
        f"- Did the call caps work? Scientific cells initiated={payload['scientific_cells_initiated']} (cap 12); provider attempts={budget.used} (cap 20). 13th cell and 21st attempt are refused.",
        f"- Is the pipeline technically ready for the full 2,800-call primary pilot? **{'YES, engineering-only' if ready else 'NO'}**. This does not authorize Task 003.",
        f"- Engineering issue GPT must resolve first: {engineering_issue or 'none observed in this smoke test'}",
        "",
        "## Compact raw action table",
        "",
        *table_lines,
        "",
        "This table is for engineering inspection only. It is not evidence for or against the hypothesis.",
        "",
        "## Caps",
        "",
        "The runner refuses a 13th scientific cell and refuses a 21st provider attempt.",
        "`study1-run --yes` remains blocked for the 2,800-call primary and 1,120-call repeats.",
        "",
        "## Engineering readiness",
        "",
        (
            "The live GPT and Claude verification path, parser, Study-1 sqlite writes, "
            "and call caps worked on this 12-cell subset."
            if ready
            else "The smoke test did not complete cleanly enough to recommend the 2,800-call primary pilot yet."
        ),
        "",
        "Issues:" if post["issues"] or payload.get("stopped_reason") else "No post-run issues recorded.",
        *[f"- {issue}" for issue in post["issues"]],
        *([f"- stopped: {payload.get('stopped_reason')}"] if payload.get("stopped_reason") else []),
        "",
        "READY_FOR_TASK_003 = YES" if ready else "READY_FOR_TASK_003 = NO",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "002_smoke_test_study1",
                "status": "SUCCESS" if ready else "PARTIAL_OR_FAILED",
                "git_commit": payload.get("code_commit"),
                "paid_calls_authorized": True,
                "scientific_cell_cap": 12,
                "provider_attempt_cap": 20,
                "selected_question_id": payload["question_id"],
                "models": ["openai_gpt56_sol", "anthropic_sonnet5"],
                "L_values": [10.0, 20.0],
                "planned_cells": 12,
                "successful_cells": successful,
                "failed_cells": failed,
                "provider_attempts_total": budget.used,
                "retry_attempts": retry_attempts,
                "repair_attempts": repair_attempts,
                "api_cost_usd": total_cost,
                "historical_artifacts_modified": not post["historical_unchanged"],
                "study1_checkpoint_path": payload["checkpoint_path"],
                "provider_model_metadata": metadata,
                "ready_for_task_003": ready,
                "wall_clock_seconds": payload["wall_clock_seconds"],
                "stopped_reason": payload.get("stopped_reason"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_changed_files(output)


def execute_smoke_task() -> dict[str, Any]:
    """Task 002 entry: pre-flight, then at most 12 paid cells / 20 attempts."""
    payload = asyncio.run(
        run_smoke_test(
            allow_paid=True,
            scientific_cell_cap=SMOKE_SCIENTIFIC_CELL_CAP,
            provider_attempt_cap=SMOKE_PROVIDER_ATTEMPT_CAP,
        )
    )
    write_smoke_artifacts(payload)
    if payload.get("preflight_ok"):
        post = postvalidate_smoke(payload)
        counts = _counts(payload["results"], payload["budget"])
        ready = _ready_for_task_003(payload, post, counts)
    else:
        ready = False
    print("READY_FOR_TASK_003 = YES" if ready else "READY_FOR_TASK_003 = NO")
    return payload
