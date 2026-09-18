"""Shared paid qualitative runner for Task 007 Lanes A and B.

Reuses the frozen Task-006 prompt bodies and QualCall layout. Separate sqlite files.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .checkpointing import CheckpointStore, deterministic_request_key
from .config import PROJECT_ROOT, load_models_config
from .model_adapters import create_adapter
from .study1_analysis import _is_verify
from .study1_sample import assert_not_v2_write_target, load_v2b_examples
from .study1_schemas import load_study1_config
from .study1_smoke import (
    ProviderAttemptBudget,
    ProviderAttemptCapReached,
    _git_commit,
    _install_attempt_cap,
    _response_value,
    estimate_cost_usd,
)
from .task006_common import SCORE_CONDITIONS, STAKES_FAMILIES
from .task006_lane_a import QualCall, _audit_diffs, _condition_displayed, family_template_hash
from .task006_prompts import (
    audit_prompt_body,
    build_qualitative_prompt,
    build_qualitative_repair_prompt,
    format_displayed_probability,
    parse_qualitative_response,
    prompt_family_id,
)
from .task007_common import (
    CONCURRENCY,
    MAX_PARSE_REPAIRS,
    PROMPT_HASH_MODERATE,
    PROMPT_HASH_STRONGER,
    PROMPT_VERSION,
    assert_007_write_target,
    load_historical_aliases,
    prompt_sha256,
    protected_fingerprints,
)


def build_qual_calls(
    *,
    question_ids: Sequence[str],
    aliases: Sequence[str],
    sample_hash: str,
    experiment_version: str,
) -> list[QualCall]:
    config = load_study1_config()
    models = load_models_config(config.resolve_path(config.models_config_path))
    examples = {row.example_id: row for row in load_v2b_examples(config)}
    historical = load_historical_aliases(question_ids, aliases)
    calls: list[QualCall] = []
    for qid in question_ids:
        example = examples[qid]
        for alias in aliases:
            spec = models.models[alias]
            hist = historical[(qid, alias)]
            for family in STAKES_FAMILIES:
                for condition in SCORE_CONDITIONS:
                    displayed = _condition_displayed(condition)
                    prompt = build_qualitative_prompt(
                        question=example.question,
                        choices=example.choices,
                        frozen_answer=str(hist["frozen_answer"]),
                        family=family,
                        displayed_confidence=displayed,
                    )
                    request_key = deterministic_request_key(
                        stage="verification",
                        dataset="mmlu_pro",
                        example_id=qid,
                        model_id=alias,
                        prompt_version=PROMPT_VERSION,
                        experiment_version=experiment_version,
                        prompt_family=prompt_family_id(family),
                        stake={
                            "stakes_family": family,
                            "score_condition": condition,
                            "displayed_confidence": displayed,
                        },
                        dependency={
                            "frozen_answer": hist["frozen_answer"],
                            "historical_answer_request_key": hist[
                                "historical_answer_request_key"
                            ],
                        },
                    )
                    calls.append(
                        QualCall(
                            question_id=qid,
                            model_alias=alias,
                            model_endpoint=spec.api_model,
                            provider=spec.provider,
                            family=family,
                            score_condition=condition,
                            displayed_confidence=displayed,
                            displayed_token=(
                                None
                                if displayed is None
                                else format_displayed_probability(displayed)
                            ),
                            frozen_answer=str(hist["frozen_answer"]),
                            stage1_correct=bool(hist["stage1_correct"]),
                            reported_confidence=float(hist["reported_confidence"]),
                            prompt=prompt,
                            prompt_hash=prompt_sha256(prompt),
                            prompt_family=prompt_family_id(family),
                            request_key=request_key,
                            historical_answer_request_key=str(
                                hist["historical_answer_request_key"]
                            ),
                            historical_confidence_request_key=str(
                                hist["historical_confidence_request_key"]
                            ),
                            choices=dict(example.choices),
                            question=example.question,
                            selected_sample_hash=sample_hash,
                        )
                    )
    return calls


def validate_calls(
    calls: Sequence[QualCall],
    *,
    expected_n: int,
    expected_ids: Sequence[str],
    expected_hash: str,
    aliases: Sequence[str],
    endpoints: dict[str, str],
) -> dict[str, Any]:
    errors: list[str] = []
    if len(calls) != expected_n:
        errors.append(f"planned calls {len(calls)} != {expected_n}")
    keys = [row.request_key for row in calls]
    if len(set(keys)) != len(keys):
        errors.append("duplicate request keys")
    unique: list[str] = []
    for row in calls:
        if row.question_id not in unique:
            unique.append(row.question_id)
    if unique != list(expected_ids):
        errors.append("call-plan question order does not match expected IDs")
    from .study1_sample import hash_id_list

    if hash_id_list(unique) != expected_hash:
        errors.append("sample hash mismatch")
    if {row.model_alias for row in calls} != set(aliases):
        errors.append("unexpected model aliases")
    for alias, endpoint in endpoints.items():
        got = {row.model_endpoint for row in calls if row.model_alias == alias}
        if got and got != {endpoint}:
            errors.append(f"{alias} endpoint {got} != {endpoint}")
    for row in calls:
        errors.extend(
            f"{row.question_id}/{row.model_alias}/{row.family}/{row.score_condition}: {msg}"
            for msg in audit_prompt_body(
                row.prompt,
                question=row.question,
                choices=row.choices,
                frozen_answer=row.frozen_answer,
                family=row.family,
                displayed_confidence=row.displayed_confidence,
            )
        )
    errors.extend(_audit_diffs(calls))
    moderate = family_template_hash("moderate")
    stronger = family_template_hash("stronger")
    if moderate != PROMPT_HASH_MODERATE:
        errors.append(f"moderate template hash {moderate} != frozen Task 006")
    if stronger != PROMPT_HASH_STRONGER:
        errors.append(f"stronger template hash {stronger} != frozen Task 006")
    return {
        "ok": not errors,
        "errors": errors[:80],
        "n_errors": len(errors),
        "n_calls": len(calls),
        "n_questions": len(unique),
        "question_ids": unique,
        "sample_hash": expected_hash,
        "prompt_hash_moderate": moderate,
        "prompt_hash_stronger": stronger,
        "prompt_version": PROMPT_VERSION,
        "visible_tokens": {
            "0.70": format_displayed_probability(0.70),
            "0.90": format_displayed_probability(0.90),
            "0.99": format_displayed_probability(0.99),
        },
        "stakes_families": list(STAKES_FAMILIES),
        "score_conditions": list(SCORE_CONDITIONS),
        "will_not_modify_paperDirection": True,
        "reuse_task006_gpt_claude_20": True,
    }


async def _execute_qual_cell(
    *,
    row: QualCall,
    adapter: Any,
    spec: Any,
    checkpoint: CheckpointStore,
    budget: ProviderAttemptBudget,
    code_commit: str | None,
    experiment_version: str,
    run_id: str,
) -> dict[str, Any]:
    attempts_before = budget.used
    prompt = row.prompt
    last_raw = ""
    parse_repairs = 0
    base_record = {
        "study_id": experiment_version,
        "experiment_version": experiment_version,
        "pilot_or_confirmatory": "exploratory",
        "run_id": run_id,
        "request_key": row.request_key,
        "question_id": row.question_id,
        "model_alias": row.model_alias,
        "model_endpoint": row.model_endpoint,
        "provider": row.provider,
        "stakes_family": row.family,
        "score_condition": row.score_condition,
        "displayed_confidence": row.displayed_confidence,
        "displayed_token": row.displayed_token,
        "reported_confidence": row.reported_confidence,
        "frozen_answer": row.frozen_answer,
        "stage1_correct": row.stage1_correct,
        "prompt_version": PROMPT_VERSION,
        "prompt_family": row.prompt_family,
        "prompt_hash": row.prompt_hash,
        "selected_sample_hash": row.selected_sample_hash,
        "historical_answer_request_key": row.historical_answer_request_key,
        "historical_confidence_request_key": row.historical_confidence_request_key,
        "code_commit": code_commit,
        "model_settings": {
            "requested_model_id": spec.api_model,
            "provider": spec.provider,
            "api_style": spec.api_style,
            "max_output_tokens": spec.max_output_tokens,
            "reasoning_effort": spec.reasoning_effort,
            "thinking": (
                spec.thinking.model_dump()
                if getattr(spec, "thinking", None) is not None
                else None
            ),
            "thinking_level": getattr(spec, "thinking_level", None),
        },
    }
    for parse_attempt in range(MAX_PARSE_REPAIRS + 1):
        attempt_kind = "generation" if parse_attempt == 0 else "parse_repair"
        if parse_attempt:
            parse_repairs += 1
            prompt = build_qualitative_repair_prompt(row.prompt, last_raw)
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
                "row": row,
                "error": str(error),
                "parse_status": "failed",
                "parsed_action": None,
                "raw_response": None,
                "provider_attempts": budget.used - attempts_before,
                "parse_repairs": parse_repairs,
                "estimated_cost_usd": 0.0,
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
            payload = parse_qualitative_response(raw)
            action = payload.action.value
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
            if parse_attempt < MAX_PARSE_REPAIRS:
                continue
            checkpoint.mark_failed(row.request_key, error)
            return {
                "ok": False,
                "row": row,
                "error": str(error),
                "parse_status": "parse_failed" if not refusal else "refused",
                "parsed_action": None,
                "raw_response": raw,
                "provider_attempts": budget.used - attempts_before,
                "parse_repairs": parse_repairs,
                "estimated_cost_usd": estimate_cost_usd(spec, input_tokens, output_tokens),
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
        record = {
            **base_record,
            "parsed_action": action,
            "parse_status": "success",
            "raw_response": raw,
            "returned_model_id": returned_model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_seconds * 1000,
            "estimated_cost_usd": estimate_cost_usd(spec, input_tokens, output_tokens),
            "api_timestamp": datetime.now(UTC).isoformat(),
            "attempt_number": int(attempt_number),
            "parse_repairs": parse_repairs,
        }
        checkpoint.mark_success(row.request_key, record)
        return {
            "ok": True,
            "row": row,
            "parsed_action": action,
            "parse_status": "success",
            "raw_response": raw,
            "provider_attempts": budget.used - attempts_before,
            "parse_repairs": parse_repairs,
            "estimated_cost_usd": record["estimated_cost_usd"],
            "record": record,
        }
    raise RuntimeError("qualitative parse-repair loop ended without a result")


async def run_paid_grid(
    *,
    calls: Sequence[QualCall],
    aliases: Sequence[str],
    sqlite_path: Path,
    experiment_version: str,
    run_id: str,
    scientific_cap: int,
    provider_cap: int,
    concurrency: int = CONCURRENCY,
    max_transient_retries: int = 3,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 30.0,
) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    if len(calls) != scientific_cap:
        raise RuntimeError(f"paid grid {len(calls)} != cap {scientific_cap}")
    assert_007_write_target(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    config = load_study1_config()
    models = load_models_config(config.resolve_path(config.models_config_path))
    adapters = {
        alias: create_adapter(
            alias,
            models.models[alias],
            max_transient_retries=max_transient_retries,
            backoff_base_seconds=backoff_base_seconds,
            backoff_max_seconds=backoff_max_seconds,
        )
        for alias in aliases
    }
    budget = ProviderAttemptBudget(provider_cap)
    for adapter in adapters.values():
        _install_attempt_cap(adapter, budget)
    code_commit = _git_commit()
    before = protected_fingerprints()
    started = time.perf_counter()
    todo: list[QualCall] = []
    reused = 0
    results: list[dict[str, Any]] = []
    with CheckpointStore(sqlite_path) as checkpoint:
        for row in calls:
            checkpoint.register_request(
                request_key=row.request_key,
                run_id=run_id,
                stage="verification",
                dataset="mmlu_pro",
                example_id=row.question_id,
                model_alias=row.model_alias,
                requested_model_id=row.model_endpoint,
                prompt_version=PROMPT_VERSION,
                stake={
                    "stakes_family": row.family,
                    "score_condition": row.score_condition,
                    "displayed_confidence": row.displayed_confidence,
                    "frozen_answer": row.frozen_answer,
                },
            )
            if checkpoint.request_status(row.request_key) == "success":
                existing = checkpoint.get_record(row.request_key) or {}
                reused += 1
                results.append(
                    {
                        "ok": True,
                        "row": row,
                        "parsed_action": existing.get("parsed_action"),
                        "parse_status": existing.get("parse_status", "success"),
                        "raw_response": existing.get("raw_response"),
                        "provider_attempts": 0,
                        "parse_repairs": existing.get("parse_repairs") or 0,
                        "estimated_cost_usd": float(existing.get("estimated_cost_usd") or 0),
                        "record": existing,
                        "reused": True,
                    }
                )
            else:
                todo.append(row)
        if len(todo) > scientific_cap:
            raise RuntimeError(f"todo {len(todo)} exceeds cap {scientific_cap}")
        semaphore = asyncio.Semaphore(max(1, int(concurrency)))

        async def execute_one(row: QualCall) -> dict[str, Any]:
            async with semaphore:
                return await _execute_qual_cell(
                    row=row,
                    adapter=adapters[row.model_alias],
                    spec=models.models[row.model_alias],
                    checkpoint=checkpoint,
                    budget=budget,
                    code_commit=code_commit,
                    experiment_version=experiment_version,
                    run_id=run_id,
                )

        if todo:
            gathered = await asyncio.gather(*(execute_one(row) for row in todo))
            results.extend(gathered)
    wall = time.perf_counter() - started
    after = protected_fingerprints()
    for name in ("v2", "study1", "q2", "task006_sqlite", "paperDirection"):
        before_sha = (before.get(name) or {}).get("sha256")
        after_sha = (after.get(name) or {}).get("sha256")
        if before_sha and before_sha != after_sha:
            raise RuntimeError(f"{name} hash changed during Task 007 paid run")
    successful = [item for item in results if item.get("ok")]
    failed = [item for item in results if not item.get("ok")]
    return {
        "ok": len(successful) == len(calls) and not failed,
        "planned": len(calls),
        "reused": reused,
        "new_scientific_calls": len(todo),
        "successful": len(successful),
        "failed": len(failed),
        "provider_attempts": budget.used,
        "wall_clock_seconds": wall,
        "results": results,
        "code_commit": code_commit,
        "sqlite": str(sqlite_path),
        "estimated_cost_usd": sum(float(item.get("estimated_cost_usd") or 0) for item in results),
    }


def results_to_rows(payload: Mapping[str, Any], *, source: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    table: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        row: QualCall = item["row"]
        if not item.get("ok") or not item.get("parsed_action"):
            failures.append(
                {
                    "source": source,
                    "question_id": row.question_id,
                    "model_alias": row.model_alias,
                    "family": row.family,
                    "score_condition": row.score_condition,
                    "error": item.get("error"),
                    "parse_status": item.get("parse_status"),
                }
            )
            continue
        rec = item.get("record") or {}
        action = str(item["parsed_action"])
        table.append(
            {
                "source": source,
                "question_id": row.question_id,
                "model_alias": row.model_alias,
                "model_endpoint": row.model_endpoint,
                "family": row.family,
                "score_condition": row.score_condition,
                "displayed_confidence": row.displayed_confidence,
                "displayed_token": row.displayed_token,
                "frozen_answer": row.frozen_answer,
                "stage1_correct": row.stage1_correct,
                "parsed_action": action,
                "verify": int(_is_verify(action)),
                "raw_response": item.get("raw_response"),
                "input_tokens": rec.get("input_tokens"),
                "output_tokens": rec.get("output_tokens"),
                "estimated_cost_usd": item.get("estimated_cost_usd"),
                "prompt_hash": row.prompt_hash,
                "request_key": row.request_key,
            }
        )
    return table, failures
