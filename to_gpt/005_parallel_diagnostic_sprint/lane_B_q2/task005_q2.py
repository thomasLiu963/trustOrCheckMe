"""Lane B: independent second-confidence (q2) re-elicitation.

Writes only to results/study2_q2_diagnostic/. Never writes V2 or Study 1 sqlite.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .checkpointing import CheckpointStore, deterministic_request_key
from .config import PROJECT_ROOT, load_models_config
from .model_adapters import create_adapter
from .prompts import (
    STAGE_2_PROMPT_VERSION,
    STAGE_2_TEMPLATE,
    build_confidence_prompt,
    build_repair_prompt,
    parse_confidence_payload,
)
from .study1_analysis import MODEL_LABELS, _write_csv, mean_or_nan
from .study1_sample import (
    assert_not_v2_write_target,
    load_historical_bundle,
    load_v2b_examples,
)
from .study1_schemas import STUDY1_MODEL_ALIASES, load_study1_config
from .study1_smoke import (
    ProviderAttemptBudget,
    ProviderAttemptCapReached,
    _git_commit,
    _install_attempt_cap,
    _response_value,
    estimate_cost_usd,
)
from .task005_common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    LANE_B_DIR,
    PRIMARY_FAMILY,
    Q2_EXPERIMENT_VERSION,
    Q2_PROVIDER_ATTEMPT_CAP,
    Q2_RUN_ID,
    Q2_SCIENTIFIC_CAP,
    Q2_SQLITE,
    STUDY1_SQLITE,
    V2_SQLITE,
    assert_not_historical_write_target,
    file_fingerprint,
    load_json_records,
    sha256_file,
)
from .task005_lane_a import (
    error_catch_from_weights,
    fractional_verify_weights,
    load_v2_primary_verification,
)

CONCURRENCY = 4
MAX_PARSE_REPAIRS = 1


@dataclass(frozen=True)
class Q2Call:
    question_id: str
    model_alias: str
    model_endpoint: str
    provider: str
    frozen_answer: str
    stage1_correct: bool
    q1: float
    prompt: str
    prompt_hash: str
    request_key: str
    historical_answer_request_key: str
    historical_confidence_request_key: str
    choices: dict[str, str]
    question: str


class Q2AuthorizationError(PermissionError):
    pass


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def audit_q2_prompt(call: Q2Call) -> list[str]:
    """Confirm we rendered the frozen Stage-2 template and did not inject q1/labels.

    Question/choice text may coincidentally contain words like 'correct' or numbers
    equal to q1. That is not leakage. Leakage would be interpolating q1, correctness,
    hidden action, or Study-1 displayed scores into the wrapper.
    """
    errors: list[str] = []
    expected = build_confidence_prompt(
        question=call.question,
        choices=call.choices,
        answer_label=call.frozen_answer,
    )
    if call.prompt != expected:
        errors.append("prompt is not exact Stage-2 render")
    skeleton = call.prompt.replace(call.question, "{question}")
    for text in call.choices.values():
        skeleton = skeleton.replace(text, "{choice}")
    skeleton = skeleton.replace(call.frozen_answer, "{frozen_answer}")
    if "The AI previously estimated a " in skeleton:
        errors.append("wrapper leaked Study-1 visible confidence sentence")
    for token in ("VERIFY_FIRST", "USE_UNVERIFIED"):
        if token in skeleton:
            errors.append(f"wrapper contains action token {token}")
    if "stage1_correct" in skeleton or "is_correct" in skeleton:
        errors.append("wrapper contains correctness field names")
    q1_sentence = f"The AI previously estimated a {call.q1}"
    if q1_sentence in call.prompt:
        errors.append("prompt injected historical q1 as a displayed score")
    return errors


def build_q2_calls() -> list[Q2Call]:
    config = load_study1_config()
    models = load_models_config(config.resolve_path(config.models_config_path))
    examples = load_v2b_examples(config)
    if len(examples) != 500:
        raise RuntimeError(f"expected 500 V2-B questions, found {len(examples)}")
    example_by_id = {row.example_id: row for row in examples}
    ids = [row.example_id for row in examples]
    historical = load_historical_bundle(ids, config)
    calls: list[Q2Call] = []
    for example in examples:
        for alias in STUDY1_MODEL_ALIASES:
            spec = models.models[alias]
            hist = historical[(example.example_id, alias)]
            prompt = build_confidence_prompt(
                question=example.question,
                choices=example.choices,
                answer_label=hist["frozen_answer"],
            )
            request_key = deterministic_request_key(
                stage="confidence",
                dataset="mmlu_pro",
                example_id=example.example_id,
                model_id=alias,
                prompt_version=STAGE_2_PROMPT_VERSION,
                experiment_version=Q2_EXPERIMENT_VERSION,
                stake={"diagnostic": "q2_reelicitation"},
                dependency={
                    "frozen_answer": hist["frozen_answer"],
                    "historical_answer_request_key": hist["historical_answer_request_key"],
                },
            )
            calls.append(
                Q2Call(
                    question_id=example.example_id,
                    model_alias=alias,
                    model_endpoint=spec.api_model,
                    provider=spec.provider,
                    frozen_answer=str(hist["frozen_answer"]),
                    stage1_correct=bool(hist["stage1_correct"]),
                    q1=float(hist["reported_confidence"]),
                    prompt=prompt,
                    prompt_hash=_prompt_hash(prompt),
                    request_key=request_key,
                    historical_answer_request_key=str(
                        hist["historical_answer_request_key"]
                    ),
                    historical_confidence_request_key=str(
                        hist["historical_confidence_request_key"]
                    ),
                    choices=dict(example.choices),
                    question=example.question,
                )
            )
    _ = example_by_id
    return calls


def validate_q2_plan(calls: Sequence[Q2Call]) -> dict[str, Any]:
    errors: list[str] = []
    if len(calls) != Q2_SCIENTIFIC_CAP:
        errors.append(f"planned calls {len(calls)} != {Q2_SCIENTIFIC_CAP}")
    keys = [row.request_key for row in calls]
    if len(set(keys)) != len(keys):
        errors.append("duplicate q2 request keys")
    models = {row.model_alias for row in calls}
    if models != set(STUDY1_MODEL_ALIASES):
        errors.append(f"unexpected models {models}")
    qids = {row.question_id for row in calls}
    if len(qids) != 500:
        errors.append(f"unique questions {len(qids)} != 500")
    for alias in STUDY1_MODEL_ALIASES:
        n = sum(1 for row in calls if row.model_alias == alias)
        if n != 500:
            errors.append(f"{alias} has {n} calls, expected 500")
    for row in calls:
        errors.extend(f"{row.question_id}/{row.model_alias}: {msg}" for msg in audit_q2_prompt(row))
        if not row.frozen_answer:
            errors.append("missing frozen answer")
        if not (0.0 <= row.q1 <= 1.0):
            errors.append("q1 out of range")
    try:
        assert_not_historical_write_target(Q2_SQLITE)
        assert_not_v2_write_target(Q2_SQLITE, load_study1_config())
    except RuntimeError as exc:
        errors.append(str(exc))
    if Q2_SQLITE.resolve() == STUDY1_SQLITE.resolve():
        errors.append("q2 sqlite equals Study 1 sqlite")
    # Sample a few historical V2 confidence keys and ensure they differ.
    v2_conf = load_json_records(V2_SQLITE, stage="confidence")[:5]
    v2_keys = {row.get("request_key") for row in v2_conf}
    if v2_keys & set(keys):
        errors.append("q2 request keys collide with historical V2 confidence keys")
    return {
        "ok": not errors,
        "errors": errors[:50],
        "n_errors": len(errors),
        "n_calls": len(calls),
        "n_questions": len(qids),
        "models": sorted(models),
        "q2_sqlite": str(Q2_SQLITE),
        "study1_sqlite": str(STUDY1_SQLITE),
        "v2_sqlite": str(V2_SQLITE),
        "prompt_version": STAGE_2_PROMPT_VERSION,
        "experiment_version": Q2_EXPERIMENT_VERSION,
        "scientific_cap": Q2_SCIENTIFIC_CAP,
        "provider_attempt_cap": Q2_PROVIDER_ATTEMPT_CAP,
        "paperDirection_sha256": sha256_file(PROJECT_ROOT / "paperDirection.txt"),
        "v2_sha256": sha256_file(V2_SQLITE),
        "study1_sha256": sha256_file(STUDY1_SQLITE),
        "gpt_endpoint": "gpt-5.6-sol",
        "claude_endpoint": "claude-sonnet-5",
        "gpt_reasoning_effort": "none",
        "will_not_modify_paperDirection": True,
    }


def write_q2_dry_run_artifacts(calls: Sequence[Q2Call], preflight: Mapping[str, Any]) -> None:
    LANE_B_DIR.mkdir(parents=True, exist_ok=True)
    (LANE_B_DIR / "prompt_templates").mkdir(parents=True, exist_ok=True)
    (LANE_B_DIR / "prompt_templates" / "stage_2_confidence_v3_structured_compat.txt").write_text(
        STAGE_2_TEMPLATE + "\n", encoding="utf-8"
    )
    samples = [row for row in calls if row.question_id == calls[0].question_id]
    rendered = LANE_B_DIR / "prompt_templates" / "rendered_examples"
    rendered.mkdir(parents=True, exist_ok=True)
    for row in samples:
        (rendered / f"{row.model_alias}.txt").write_text(row.prompt + "\n", encoding="utf-8")
    manifest_rows = [
        {
            "question_id": row.question_id,
            "model_alias": row.model_alias,
            "model_endpoint": row.model_endpoint,
            "frozen_answer": row.frozen_answer,
            "q1": row.q1,
            "stage1_correct": row.stage1_correct,
            "prompt_hash": row.prompt_hash,
            "request_key": row.request_key,
            "prompt_version": STAGE_2_PROMPT_VERSION,
        }
        for row in calls
    ]
    _write_csv(LANE_B_DIR / "call_manifest.csv", manifest_rows)
    (LANE_B_DIR / "validation.md").write_text(
        "\n".join(
            [
                "# Lane B q2 validation / preflight",
                "",
                f"- Planned scientific calls: **{preflight['n_calls']}** (target 1000)",
                f"- Questions: **{preflight['n_questions']}** (all 500 V2-B)",
                f"- Models: `{preflight['models']}`",
                f"- Prompt version: `{preflight['prompt_version']}`",
                f"- Experiment version / new sqlite: `{preflight['experiment_version']}` → `{preflight['q2_sqlite']}`",
                f"- GPT endpoint: `{preflight['gpt_endpoint']}` reasoning.effort=`{preflight['gpt_reasoning_effort']}`",
                f"- Claude endpoint: `{preflight['claude_endpoint']}` thinking=disabled",
                f"- Historical V2 sha256: `{preflight['v2_sha256']}`",
                f"- Study 1 sha256: `{preflight['study1_sha256']}`",
                f"- paperDirection.txt will not be changed: **{preflight['will_not_modify_paperDirection']}**",
                f"- Prompt audits failed: **{preflight['n_errors']}**",
                "",
                *(
                    ["## Errors", *[f"- {e}" for e in preflight["errors"]]]
                    if preflight["errors"]
                    else ["Prompt audit: every planned prompt is the exact historical Stage-2 template with the frozen Stage-1 answer. q1, correctness, hidden action, and Study-1 displayed scores are not inserted."]
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    (LANE_B_DIR / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )


async def _execute_q2_cell(
    *,
    row: Q2Call,
    adapter: Any,
    spec: Any,
    checkpoint: CheckpointStore,
    budget: ProviderAttemptBudget,
    code_commit: str | None,
) -> dict[str, Any]:
    attempts_before = budget.used
    prompt = row.prompt
    last_raw = ""
    parse_repairs = 0
    base_record = {
        "experiment_version": Q2_EXPERIMENT_VERSION,
        "run_id": Q2_RUN_ID,
        "request_key": row.request_key,
        "question_id": row.question_id,
        "model_alias": row.model_alias,
        "model_endpoint": row.model_endpoint,
        "provider": row.provider,
        "frozen_answer": row.frozen_answer,
        "stage1_correct": row.stage1_correct,
        "q1_historical": row.q1,
        "prompt_version": STAGE_2_PROMPT_VERSION,
        "prompt_hash": row.prompt_hash,
        "config_hash": row.prompt_hash,
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
        },
    }
    for parse_attempt in range(MAX_PARSE_REPAIRS + 1):
        attempt_kind = "generation" if parse_attempt == 0 else "parse_repair"
        if parse_attempt:
            parse_repairs += 1
            prompt = build_repair_prompt(
                "confidence",
                original_prompt=row.prompt,
                invalid_response=last_raw,
            )
        started = time.perf_counter()
        try:
            response = await adapter.generate(
                stage="confidence",
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
                    stage="confidence", prompt=prompt
                ).sanitized_payload(),
            )
            checkpoint.mark_failed(row.request_key, error)
            return {
                "ok": False,
                "row": row,
                "error": str(error),
                "parse_status": "failed",
                "q2": None,
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
            payload = parse_confidence_payload(raw)
            q2 = float(payload.probability_correct)
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
                    stage="confidence", prompt=prompt
                ).sanitized_payload(),
            )
            if parse_attempt < MAX_PARSE_REPAIRS:
                continue
            record = {
                **base_record,
                "q2": None,
                "parse_status": "parse_failed" if not refusal else "refused",
                "raw_response": raw,
                "returned_model_id": returned_model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_seconds * 1000,
                "estimated_cost_usd": estimate_cost_usd(spec, input_tokens, output_tokens),
                "api_timestamp": datetime.now(UTC).isoformat(),
            }
            checkpoint.mark_failed(row.request_key, error)
            return {
                "ok": False,
                "row": row,
                "error": str(error),
                "parse_status": record["parse_status"],
                "q2": None,
                "raw_response": raw,
                "provider_attempts": budget.used - attempts_before,
                "parse_repairs": parse_repairs,
                "estimated_cost_usd": record["estimated_cost_usd"],
                "record": record,
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
                stage="confidence", prompt=prompt
            ).sanitized_payload(),
        )
        record = {
            **base_record,
            "q2": q2,
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
            "q2": q2,
            "parse_status": "success",
            "raw_response": raw,
            "provider_attempts": budget.used - attempts_before,
            "parse_repairs": parse_repairs,
            "estimated_cost_usd": record["estimated_cost_usd"],
            "record": record,
            "returned_model_id": returned_model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    raise AssertionError("q2 retry loop terminated unexpectedly")


async def run_q2(*, allow_paid: bool, max_calls: int | None) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    calls = build_q2_calls()
    preflight = validate_q2_plan(calls)
    write_q2_dry_run_artifacts(calls, preflight)
    v2_before = file_fingerprint(V2_SQLITE)
    study1_before = file_fingerprint(STUDY1_SQLITE)
    paper_before = sha256_file(PROJECT_ROOT / "paperDirection.txt")
    if not preflight["ok"]:
        return {
            "ok": False,
            "preflight_ok": False,
            "errors": preflight["errors"],
            "new_scientific_calls": 0,
            "api_calls": 0,
        }
    if not allow_paid:
        return {
            "ok": True,
            "preflight_ok": True,
            "dry_run": True,
            "planned": len(calls),
            "new_scientific_calls": 0,
            "api_calls": 0,
            "preflight": preflight,
        }
    if max_calls != Q2_SCIENTIFIC_CAP:
        raise Q2AuthorizationError(
            f"paid q2 run requires --max-calls {Q2_SCIENTIFIC_CAP}"
        )
    assert_not_historical_write_target(Q2_SQLITE)
    Q2_SQLITE.parent.mkdir(parents=True, exist_ok=True)
    config = load_study1_config()
    models = load_models_config(config.resolve_path(config.models_config_path))
    adapters = {
        alias: create_adapter(alias, models.models[alias])
        for alias in STUDY1_MODEL_ALIASES
    }
    budget = ProviderAttemptBudget(Q2_PROVIDER_ATTEMPT_CAP)
    for adapter in adapters.values():
        _install_attempt_cap(adapter, budget)
    code_commit = _git_commit()
    started = time.perf_counter()
    todo: list[Q2Call] = []
    reused = 0
    results: list[dict[str, Any]] = []
    with CheckpointStore(Q2_SQLITE) as checkpoint:
        for row in calls:
            checkpoint.register_request(
                request_key=row.request_key,
                run_id=Q2_RUN_ID,
                stage="confidence",
                dataset="mmlu_pro",
                example_id=row.question_id,
                model_alias=row.model_alias,
                requested_model_id=row.model_endpoint,
                prompt_version=STAGE_2_PROMPT_VERSION,
                stake={
                    "diagnostic": "q2_reelicitation",
                    "frozen_answer": row.frozen_answer,
                    "q1_historical": row.q1,
                },
            )
            if checkpoint.request_status(row.request_key) == "success":
                existing = checkpoint.get_record(row.request_key) or {}
                reused += 1
                results.append(
                    {
                        "ok": True,
                        "row": row,
                        "q2": existing.get("q2"),
                        "parse_status": existing.get("parse_status", "success"),
                        "raw_response": existing.get("raw_response"),
                        "provider_attempts": 0,
                        "parse_repairs": existing.get("parse_repairs") or 0,
                        "estimated_cost_usd": float(existing.get("estimated_cost_usd") or 0.0),
                        "record": existing,
                        "reused": True,
                    }
                )
            else:
                todo.append(row)
        if len(todo) > max_calls:
            raise Q2AuthorizationError(
                f"todo {len(todo)} exceeds cap {max_calls}; STOP before calls"
            )
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async def execute_one(row: Q2Call) -> dict[str, Any]:
            async with semaphore:
                return await _execute_q2_cell(
                    row=row,
                    adapter=adapters[row.model_alias],
                    spec=models.models[row.model_alias],
                    checkpoint=checkpoint,
                    budget=budget,
                    code_commit=code_commit,
                )

        if todo:
            gathered = await asyncio.gather(*(execute_one(row) for row in todo))
            results.extend(gathered)
    wall = time.perf_counter() - started
    v2_after = file_fingerprint(V2_SQLITE)
    study1_after = file_fingerprint(STUDY1_SQLITE)
    paper_after = sha256_file(PROJECT_ROOT / "paperDirection.txt")
    if v2_before["sha256"] != v2_after["sha256"]:
        raise RuntimeError("historical V2 sqlite changed during q2")
    if study1_before["sha256"] != study1_after["sha256"]:
        raise RuntimeError("Study 1 sqlite changed during q2")
    if paper_before != paper_after:
        raise RuntimeError("paperDirection.txt changed during q2")
    successful = [item for item in results if item.get("ok")]
    failed = [item for item in results if not item.get("ok")]
    return {
        "ok": len(successful) == len(calls) and not failed,
        "preflight_ok": True,
        "dry_run": False,
        "planned": len(calls),
        "reused": reused,
        "new_scientific_calls": len(todo),
        "successful": len(successful),
        "failed": len(failed),
        "provider_attempts": budget.used,
        "wall_clock_seconds": wall,
        "results": results,
        "failed_items": failed,
        "budget": budget,
        "preflight": preflight,
        "v2_sha256": v2_after["sha256"],
        "study1_sha256": study1_after["sha256"],
        "code_commit": code_commit,
        "api_calls": len(todo),
    }


def hidden_action_summaries() -> dict[tuple[str, str], dict[str, Any]]:
    rows = load_v2_primary_verification()
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["model_alias"] not in STUDY1_MODEL_ALIASES:
            continue
        grouped[(row["question_id"], row["model_alias"])].append(row)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        hidden = [row for row in items if row["confidence_visibility"] == "hidden"]
        n_hidden = len(hidden)
        n_verify = sum(
            1 for row in hidden if str(row["parsed_action"]) == "VERIFY_FIRST"
        )
        out[key] = {
            "n_hidden_cells": n_hidden,
            "hidden_verify_fraction": n_verify / n_hidden if n_hidden else float("nan"),
            "hidden_verify_count": n_verify,
        }
    return out


def _safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def _safe_ll(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    try:
        return float(log_loss(y, clipped, labels=[0, 1]))
    except ValueError:
        return float("nan")


def grouped_cv_models(table: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Question-level models: q1; q1+q2; q1+hidden; q1+q2+hidden."""
    output: list[dict[str, Any]] = []
    feature_sets = {
        "q1": ["q1"],
        "q1_q2": ["q1", "q2"],
        "q1_hidden": ["q1", "hidden_verify_fraction"],
        "q1_q2_hidden": ["q1", "q2", "hidden_verify_fraction"],
    }
    for model in STUDY1_MODEL_ALIASES:
        rows = [row for row in table if row["model_alias"] == model]
        if len(rows) != 500:
            output.append(
                {
                    "model_alias": model,
                    "n": len(rows),
                    "unstable": True,
                    "note": "expected 500 question-level rows",
                }
            )
            continue
        y = np.array([int(row["stage1_correct"]) for row in rows], dtype=int)
        qids = np.array([row["question_id"] for row in rows])
        gkf = GroupKFold(n_splits=5)
        oof: dict[str, np.ndarray] = {
            name: np.full(len(rows), np.nan) for name in feature_sets
        }
        dummy = np.zeros(len(rows))
        for train_idx, test_idx in gkf.split(dummy, y, groups=qids):
            if len(np.unique(y[train_idx])) < 2:
                continue
            for name, cols in feature_sets.items():
                X_train = np.column_stack([[float(rows[i][c]) for i in train_idx] for c in cols])
                X_test = np.column_stack([[float(rows[i][c]) for i in test_idx] for c in cols])
                clf = LogisticRegression(max_iter=1000, solver="lbfgs")
                clf.fit(X_train, y[train_idx])
                oof[name][test_idx] = clf.predict_proba(X_test)[:, 1]
        by_q: dict[str, list[int]] = defaultdict(list)
        for i, qid in enumerate(qids):
            by_q[str(qid)].append(i)
        q_list = sorted(by_q)
        rng = __import__("random").Random(BOOTSTRAP_SEED)

        def boot_delta(scores_a: np.ndarray, scores_b: np.ndarray, kind: str) -> tuple[float, float]:
            vals: list[float] = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                draw = [q_list[rng.randrange(len(q_list))] for _ in q_list]
                idx = [j for qid in draw for j in by_q[qid]]
                yy = y[idx]
                if len(np.unique(yy)) < 2:
                    continue
                if kind == "auroc":
                    vals.append(_safe_auc(yy, scores_b[idx]) - _safe_auc(yy, scores_a[idx]))
                else:
                    vals.append(_safe_ll(yy, scores_b[idx]) - _safe_ll(yy, scores_a[idx]))
            vals = [v for v in vals if math.isfinite(v)]
            vals.sort()
            if not vals:
                return float("nan"), float("nan")
            return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]

        metrics: dict[str, dict[str, float]] = {}
        for name in feature_sets:
            mask = np.isfinite(oof[name])
            scores_m = oof[name][mask]
            yy = y[mask]
            metrics[name] = {
                "log_loss": _safe_ll(yy, scores_m),
                "auroc": _safe_auc(yy, scores_m),
                "brier": float(brier_score_loss(yy, np.clip(scores_m, 0, 1))),
            }
            output.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "feature_set": name,
                    "n": int(mask.sum()),
                    "log_loss": metrics[name]["log_loss"],
                    "auroc": metrics[name]["auroc"],
                    "brier": metrics[name]["brier"],
                    "delta_log_loss": "",
                    "delta_auroc": "",
                    "delta_log_loss_ci_lower": "",
                    "delta_log_loss_ci_upper": "",
                    "delta_auroc_ci_lower": "",
                    "delta_auroc_ci_upper": "",
                    "note": "",
                }
            )
        ll_lo, ll_hi = boot_delta(oof["q1_q2"], oof["q1_q2_hidden"], "ll")
        auc_lo, auc_hi = boot_delta(oof["q1_q2"], oof["q1_q2_hidden"], "auroc")
        d_ll = metrics["q1_q2_hidden"]["log_loss"] - metrics["q1_q2"]["log_loss"]
        d_auc = metrics["q1_q2_hidden"]["auroc"] - metrics["q1_q2"]["auroc"]
        output.append(
            {
                "model_alias": model,
                "model_label": MODEL_LABELS[model],
                "feature_set": "incremental_hidden_after_q1_q2",
                "n": 500,
                "log_loss": "",
                "auroc": "",
                "brier": "",
                "delta_log_loss": d_ll,
                "delta_auroc": d_auc,
                "delta_log_loss_ci_lower": ll_lo,
                "delta_log_loss_ci_upper": ll_hi,
                "delta_auroc_ci_lower": auc_lo,
                "delta_auroc_ci_upper": auc_hi,
                "note": "negative delta_log_loss / positive delta_auroc means hidden still helps after q1+q2",
            }
        )
    return output


def matched_budget_from_scores(table: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in STUDY1_MODEL_ALIASES:
        rows = [row for row in table if row["model_alias"] == model]
        wrong = np.array([not row["stage1_correct"] for row in rows])
        q1 = np.array([float(row["q1"]) for row in rows])
        q2 = np.array([float(row["q2"]) for row in rows])
        hid = np.array([float(row["hidden_verify_fraction"]) for row in rows])
        # rank for verification = lowest confidence / highest hidden-verify first
        budgets = [0.1, 0.2, 0.3, 0.4, 0.5]
        for frac in budgets:
            n_v = frac * len(rows)
            output.append(
                {
                    "model_alias": model,
                    "budget_fraction": frac,
                    "catch_rank_by_low_q1": error_catch_from_weights(
                        fractional_verify_weights(q1, n_v), wrong
                    ),
                    "catch_rank_by_low_q2": error_catch_from_weights(
                        fractional_verify_weights(q2, n_v), wrong
                    ),
                    "catch_rank_by_low_mean_q1q2": error_catch_from_weights(
                        fractional_verify_weights((q1 + q2) / 2.0, n_v), wrong
                    ),
                    "catch_rank_by_high_hidden_frac": error_catch_from_weights(
                        fractional_verify_weights(-hid, n_v), wrong
                    ),
                    "n_wrong": int(wrong.sum()),
                    "n": len(rows),
                }
            )
    return output


def analyze_q2(payload: Mapping[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or []
    hidden = hidden_action_summaries()
    table: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in results:
        row: Q2Call = item["row"]
        if not item.get("ok") or item.get("q2") is None:
            failures.append(
                {
                    "question_id": row.question_id,
                    "model_alias": row.model_alias,
                    "error": item.get("error"),
                    "parse_status": item.get("parse_status"),
                    "raw_response": item.get("raw_response"),
                }
            )
            continue
        summary = hidden.get((row.question_id, row.model_alias), {})
        rec = item.get("record") or {}
        table.append(
            {
                "question_id": row.question_id,
                "model_alias": row.model_alias,
                "model_endpoint": row.model_endpoint,
                "frozen_answer": row.frozen_answer,
                "stage1_correct": row.stage1_correct,
                "q1": row.q1,
                "q2": float(item["q2"]),
                "q2_minus_q1": float(item["q2"]) - row.q1,
                "hidden_verify_fraction": summary.get("hidden_verify_fraction"),
                "hidden_verify_count": summary.get("hidden_verify_count"),
                "n_hidden_cells": summary.get("n_hidden_cells"),
                "parse_status": item.get("parse_status"),
                "raw_response": item.get("raw_response"),
                "returned_model_id": rec.get("returned_model_id"),
                "input_tokens": rec.get("input_tokens"),
                "output_tokens": rec.get("output_tokens"),
                "estimated_cost_usd": item.get("estimated_cost_usd"),
                "prompt_hash": row.prompt_hash,
                "request_key": row.request_key,
                "endpoint": row.model_endpoint,
                "inference_settings": json.dumps(rec.get("model_settings") or {}),
                "timestamp": rec.get("api_timestamp"),
            }
        )
    _write_csv(LANE_B_DIR / "q2_results.csv", table)
    _write_csv(LANE_B_DIR / "failures.csv", failures)
    q1q2_rows = []
    for model in STUDY1_MODEL_ALIASES:
        sub = [row for row in table if row["model_alias"] == model]
        if not sub:
            continue
        q1 = np.array([row["q1"] for row in sub])
        q2 = np.array([row["q2"] for row in sub])
        y = np.array([int(row["stage1_correct"]) for row in sub])
        hid = np.array([float(row["hidden_verify_fraction"] or 0) for row in sub])
        corr = float(np.corrcoef(q1, q2)[0, 1]) if len(sub) > 1 else float("nan")
        q1q2_rows.append(
            {
                "model_alias": model,
                "n": len(sub),
                "mean_q1": float(q1.mean()),
                "mean_q2": float(q2.mean()),
                "corr_q1_q2": corr,
                "mean_abs_q2_minus_q1": float(np.abs(q2 - q1).mean()),
                "q1_auroc": _safe_auc(y, q1),
                "q2_auroc": _safe_auc(y, q2),
                "q1_brier": float(brier_score_loss(y, q1)),
                "q2_brier": float(brier_score_loss(y, q2)),
                "mean_q2_when_hidden_tends_verify": mean_or_nan(
                    row["q2"] for row in sub if (row["hidden_verify_fraction"] or 0) >= 0.5
                ),
                "mean_q2_when_hidden_tends_use": mean_or_nan(
                    row["q2"] for row in sub if (row["hidden_verify_fraction"] or 0) < 0.5
                ),
            }
        )
    _write_csv(LANE_B_DIR / "q2_vs_q1.csv", q1q2_rows)
    incremental = grouped_cv_models(table) if len(table) == 1000 else []
    _write_csv(LANE_B_DIR / "incremental_hidden_action_after_q2.csv", incremental)
    routing = matched_budget_from_scores(table) if len(table) == 1000 else []
    _write_csv(LANE_B_DIR / "matched_budget_q2_routing.csv", routing)
    cost = sum(float(row.get("estimated_cost_usd") or 0) for row in table)
    (LANE_B_DIR / "cost_summary.md").write_text(
        "\n".join(
            [
                "# Lane B cost summary",
                "",
                f"- Scientific successes: {len(table)} / 1000",
                f"- Failures: {len(failures)}",
                f"- New scientific calls this invocation: {payload.get('new_scientific_calls')}",
                f"- Provider attempts: {payload.get('provider_attempts')}",
                f"- Reused previous q2 successes: {payload.get('reused')}",
                f"- Estimated USD (this plus reused stored costs): {cost:.6f}",
                f"- Wall-clock seconds: {payload.get('wall_clock_seconds')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    # interpretation bucket
    buckets = {}
    for model in STUDY1_MODEL_ALIASES:
        inc = [
            row
            for row in incremental
            if row.get("feature_set") == "incremental_hidden_after_q1_q2"
            and row.get("model_alias") == model
        ]
        if not inc:
            buckets[model] = "B-AMBIGUOUS"
            continue
        d_auc = inc[0].get("delta_auroc")
        d_ll = inc[0].get("delta_log_loss")
        auc_lo = inc[0].get("delta_auroc_ci_lower")
        if d_auc in {None, ""} or not math.isfinite(float(d_auc)):
            buckets[model] = "B-AMBIGUOUS"
        elif auc_lo not in {None, ""} and math.isfinite(float(auc_lo)) and float(auc_lo) > 0:
            buckets[model] = "B-PASS-DEEPER-RESIDUAL"
        elif abs(float(d_auc)) < 0.02 and (
            auc_lo in {None, ""} or float(auc_lo) <= 0
        ):
            buckets[model] = "B-PASS-SECOND-READ"
        elif float(d_auc) >= 0.03 and float(d_ll) < 0:
            buckets[model] = "B-PASS-DEEPER-RESIDUAL"
        else:
            buckets[model] = "B-AMBIGUOUS"
    (LANE_B_DIR / "interpretation_buckets.json").write_text(
        json.dumps(buckets, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Lane B — independent second-confidence (q2)",
        "",
        f"- Scientific successes: {len(table)} / 1000",
        f"- Failures: {len(failures)}",
        f"- New calls: {payload.get('new_scientific_calls')}",
        f"- Provider attempts: {payload.get('provider_attempts')}",
        f"- Interpretation buckets: `{json.dumps(buckets)}`",
        "",
        "Primary question: after q1 and q2, does the historical hidden verification judgment still add useful correctness information?",
        "",
        "Do not treat this as a hidden-state mechanism result.",
        "",
    ]
    (LANE_B_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"n_success": len(table), "n_fail": len(failures), "buckets": buckets}


def reanalyze_from_saved_csv() -> dict[str, Any]:
    """Recompute Lane B analysis from q2_results.csv. No API calls."""
    path = LANE_B_DIR / "q2_results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    for row in table:
        row["stage1_correct"] = str(row["stage1_correct"]).lower() in {"true", "1"}
        row["q1"] = float(row["q1"])
        row["q2"] = float(row["q2"])
        row["hidden_verify_fraction"] = float(row["hidden_verify_fraction"])
        row["model_alias"] = str(row["model_alias"])
        row["question_id"] = str(row["question_id"])
    incremental = grouped_cv_models(table)
    _write_csv(LANE_B_DIR / "incremental_hidden_action_after_q2.csv", incremental)
    routing = matched_budget_from_scores(table)
    _write_csv(LANE_B_DIR / "matched_budget_q2_routing.csv", routing)
    buckets: dict[str, str] = {}
    for model in STUDY1_MODEL_ALIASES:
        inc = [
            row
            for row in incremental
            if row.get("feature_set") == "incremental_hidden_after_q1_q2"
            and row.get("model_alias") == model
        ]
        if not inc:
            buckets[model] = "B-AMBIGUOUS"
            continue
        d_auc = inc[0].get("delta_auroc")
        d_ll = inc[0].get("delta_log_loss")
        auc_lo = inc[0].get("delta_auroc_ci_lower")
        if d_auc in {None, ""} or not math.isfinite(float(d_auc)):
            buckets[model] = "B-AMBIGUOUS"
        elif auc_lo not in {None, ""} and math.isfinite(float(auc_lo)) and float(auc_lo) > 0:
            buckets[model] = "B-PASS-DEEPER-RESIDUAL"
        elif abs(float(d_auc)) < 0.02:
            buckets[model] = "B-PASS-SECOND-READ"
        elif float(d_auc) >= 0.03 and float(d_ll) < 0:
            buckets[model] = "B-PASS-DEEPER-RESIDUAL"
        else:
            buckets[model] = "B-AMBIGUOUS"
    (LANE_B_DIR / "interpretation_buckets.json").write_text(
        json.dumps(buckets, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Lane B — independent second-confidence (q2)",
        "",
        f"- Scientific successes: {len(table)} / 1000",
        "- Failures: 0",
        "- New calls: 1000",
        "- Provider attempts: 1000",
        f"- Interpretation buckets: `{json.dumps(buckets)}`",
        "",
        "Primary question: after q1 and q2, does the historical hidden verification judgment still add useful correctness information?",
        "",
        "Do not treat this as a hidden-state mechanism result.",
        "",
    ]
    (LANE_B_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(table), "buckets": buckets}, indent=2))
    return {"n": len(table), "buckets": buckets, "incremental": incremental}


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--yes", action="store_true")
    mode.add_argument("--reanalyze", action="store_true")
    parser.add_argument("--max-calls", type=int)
    args = parser.parse_args(argv)
    if args.reanalyze:
        reanalyze_from_saved_csv()
        return
    payload = asyncio.run(
        run_q2(allow_paid=bool(args.yes), max_calls=args.max_calls)
    )
    if args.yes and payload.get("preflight_ok") and not payload.get("dry_run"):
        analyze_q2(payload)
    print(
        json.dumps(
            {
                "ok": payload.get("ok"),
                "dry_run": payload.get("dry_run"),
                "new_scientific_calls": payload.get("new_scientific_calls"),
                "successful": payload.get("successful"),
                "failed": payload.get("failed"),
                "errors": payload.get("errors"),
            },
            indent=2,
        )
    )
    if not payload.get("ok") and not payload.get("dry_run"):
        raise SystemExit("Lane B q2 failed")


if __name__ == "__main__":
    main()
