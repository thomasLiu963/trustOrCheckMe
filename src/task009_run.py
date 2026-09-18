"""Task 009 paid runner: fresh Stage-1/q1 plus moderate qualitative Stage-3."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .checkpointing import CheckpointStore, deterministic_request_key
from .config import PROJECT_ROOT, load_experiment_config, load_models_config
from .model_adapters import create_adapter
from .runner import ExperimentRunner
from .schemas import BenchmarkExample
from .study1_smoke import (
    ProviderAttemptBudget,
    ProviderAttemptCapReached,
    _git_commit,
    _install_attempt_cap,
    _response_value,
    estimate_cost_usd,
)
from .task006_prompts import (
    audit_prompt_body,
    build_qualitative_prompt,
    build_qualitative_repair_prompt,
    format_displayed_probability,
    parse_qualitative_response,
    prompt_family_id,
)
from .task009_common import (
    CONCURRENCY,
    ENDPOINTS,
    EXPERIMENT_VERSION,
    FAMILY,
    GEMINI_GROK,
    GPT_CLAUDE,
    MAX_PARSE_REPAIRS,
    MAX_TRANSIENT_RETRIES,
    N_REPEAT_GENERATIONS,
    PILOT_OR_CONFIRMATORY,
    PROMPT_FAMILY,
    PROMPT_HASH_MODERATE,
    PROMPT_VERSION,
    PROVIDER_ATTEMPT_CAP,
    RUN_ID,
    SCIENTIFIC_CAP,
    SCORE_CONDITIONS,
    SQLITE_PATH,
    assert_009_write_target,
    freeze_paths,
    prompt_sha256,
    protected_fingerprints,
)
from .task009_sample import freeze_sample, load_frozen_ids


class Task009AuthorizationError(PermissionError):
    pass


@dataclass(frozen=True)
class Stage3Call:
    question_id: str
    model_alias: str
    model_endpoint: str
    provider: str
    family: str
    score_condition: str
    displayed_confidence: float | None
    displayed_token: str | None
    frozen_answer: str
    stage1_correct: bool
    reported_confidence: float
    prompt: str
    prompt_hash: str
    prompt_family: str
    request_key: str
    answer_request_key: str
    confidence_request_key: str
    choices: dict[str, str]
    question: str
    selected_sample_hash: str
    repeat_index: int
    roster: str


class ScientificCallBudget:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.used = 0

    def consume(self, n: int, *, label: str) -> None:
        if n < 0:
            raise ValueError("cannot consume a negative call count")
        if self.used + n > self.cap:
            raise Task009AuthorizationError(
                f"{label}: scientific cap {self.cap} would be exceeded "
                f"({self.used} used + {n} new)"
            )
        self.used += n

    def sync_from(self, checkpoint: CheckpointStore, *, label: str) -> int:
        with checkpoint._lock:
            row = checkpoint._connection.execute(
                "SELECT COUNT(*) AS n FROM requests"
            ).fetchone()
        n = int(row["n"])
        if n > self.cap:
            raise Task009AuthorizationError(
                f"{label}: scientific requests {n} exceed cap {self.cap}"
            )
        self.used = n
        return n


def condition_displayed(condition: str, q1: float) -> float | None:
    if condition == "hidden":
        return None
    if condition == "true_q_visible":
        return float(q1)
    if condition.startswith("displayed_"):
        return float(condition.split("_", 1)[1])
    raise ValueError(f"unknown score condition {condition!r}")


def load_examples(path: Path) -> list[BenchmarkExample]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(BenchmarkExample.model_validate_json(line))
    return rows


def _stage12_bundle(
    checkpoint: CheckpointStore,
    examples: Sequence[BenchmarkExample],
    aliases: Sequence[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    bundle: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[str] = []
    for example in examples:
        for alias in aliases:
            answer = checkpoint.find_success(
                stage="answer",
                dataset=example.dataset_name,
                example_id=example.example_id,
                model_alias=alias,
            )
            confidence = checkpoint.find_success(
                stage="confidence",
                dataset=example.dataset_name,
                example_id=example.example_id,
                model_alias=alias,
            )
            if answer is None or confidence is None:
                missing.append(f"{example.example_id}:{alias}")
                continue
            frozen = str(
                answer.get("answer_label")
                or answer.get("answer")
                or confidence.get("frozen_answer_label")
            )
            bundle[(example.example_id, alias)] = {
                "frozen_answer": frozen,
                "stage1_correct": bool(answer.get("is_correct")),
                "reported_confidence": float(
                    confidence.get("probability_correct")
                    if confidence.get("probability_correct") is not None
                    else confidence.get("confidence")
                ),
                "answer_request_key": str(answer["request_key"]),
                "confidence_request_key": str(confidence["request_key"]),
                "answer_raw": answer.get("raw_response"),
                "confidence_raw": confidence.get("raw_response"),
                "correct_label": answer.get("correct_label"),
                "category": example.category,
            }
    if missing:
        print(
            f"Stage-1/q1 incomplete for {len(missing)} cells; first={missing[0]}",
            flush=True,
        )
    return bundle


def build_stage3_calls(
    *,
    examples: Sequence[BenchmarkExample],
    aliases: Sequence[str],
    bundle: Mapping[tuple[str, str], Mapping[str, Any]],
    sample_hash: str,
    roster: str,
    repeat_indices: Sequence[int],
) -> list[Stage3Call]:
    models = load_models_config()
    calls: list[Stage3Call] = []
    for example in examples:
        for alias in aliases:
            spec = models.models[alias]
            hist = bundle.get((example.example_id, alias))
            if hist is None:
                continue
            q1 = float(hist["reported_confidence"])
            for repeat_index in repeat_indices:
                for condition in SCORE_CONDITIONS:
                    displayed = condition_displayed(condition, q1)
                    prompt = build_qualitative_prompt(
                        question=example.question,
                        choices=example.choices,
                        frozen_answer=str(hist["frozen_answer"]),
                        family=FAMILY,
                        displayed_confidence=displayed,
                    )
                    problems = audit_prompt_body(
                        prompt,
                        question=example.question,
                        choices=example.choices,
                        frozen_answer=str(hist["frozen_answer"]),
                        family=FAMILY,
                        displayed_confidence=displayed,
                    )
                    if problems:
                        raise Task009AuthorizationError(
                            f"prompt audit failed {example.example_id} {condition}: {problems}"
                        )
                    request_key = deterministic_request_key(
                        stage="verification",
                        dataset=example.dataset_name,
                        example_id=example.example_id,
                        model_id=alias,
                        prompt_version=PROMPT_VERSION,
                        experiment_version=EXPERIMENT_VERSION,
                        prompt_family=prompt_family_id(FAMILY),
                        stake={
                            "stakes_family": FAMILY,
                            "score_condition": condition,
                            "displayed_confidence": displayed,
                            "repeat_index": repeat_index,
                        },
                        dependency={
                            "frozen_answer": hist["frozen_answer"],
                            "answer_request_key": hist["answer_request_key"],
                        },
                    )
                    calls.append(
                        Stage3Call(
                            question_id=example.example_id,
                            model_alias=alias,
                            model_endpoint=spec.api_model,
                            provider=spec.provider,
                            family=FAMILY,
                            score_condition=condition,
                            displayed_confidence=displayed,
                            displayed_token=(
                                None
                                if displayed is None
                                else format_displayed_probability(displayed)
                            ),
                            frozen_answer=str(hist["frozen_answer"]),
                            stage1_correct=bool(hist["stage1_correct"]),
                            reported_confidence=q1,
                            prompt=prompt,
                            prompt_hash=prompt_sha256(prompt),
                            prompt_family=prompt_family_id(FAMILY),
                            request_key=request_key,
                            answer_request_key=str(hist["answer_request_key"]),
                            confidence_request_key=str(hist["confidence_request_key"]),
                            choices=dict(example.choices),
                            question=example.question,
                            selected_sample_hash=sample_hash,
                            repeat_index=int(repeat_index),
                            roster=roster,
                        )
                    )
    keys = [row.request_key for row in calls]
    if len(set(keys)) != len(keys):
        raise Task009AuthorizationError("duplicate Stage-3 request keys")
    return calls


async def _execute_stage3_cell(
    *,
    row: Stage3Call,
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
        "study_id": EXPERIMENT_VERSION,
        "experiment_version": EXPERIMENT_VERSION,
        "pilot_or_confirmatory": PILOT_OR_CONFIRMATORY,
        "run_id": RUN_ID,
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
        "answer_request_key": row.answer_request_key,
        "confidence_request_key": row.confidence_request_key,
        "repeat_index": row.repeat_index,
        "roster": row.roster,
        "code_commit": code_commit,
        "model_settings": {
            "requested_model_id": spec.api_model,
            "provider": spec.provider,
            "api_style": spec.api_style,
            "max_output_tokens": spec.max_output_tokens,
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
                "reused": False,
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
                "reused": False,
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
            "reused": False,
        }
    raise RuntimeError("Stage-3 parse-repair loop ended without a result")


def _cap_runner_adapters(
    runner: ExperimentRunner,
    aliases: Sequence[str],
    budget: ProviderAttemptBudget,
) -> None:
    models = runner.models_config
    configured = models.models if hasattr(models, "models") else models
    for alias in aliases:
        spec = configured[alias]
        _install_attempt_cap(runner._adapter(alias, spec), budget)


async def _run_stage12(
    *,
    examples: Sequence[BenchmarkExample],
    aliases: Sequence[str],
    checkpoint: CheckpointStore,
    budget: ProviderAttemptBudget,
    scientific: ScientificCallBudget,
    phase: str,
    retry_failed: bool,
) -> dict[str, Any]:
    expected = len(examples) * len(aliases)
    runner = ExperimentRunner(
        examples=examples,
        checkpoint=checkpoint,
        models_config=load_models_config(),
        experiment_config=load_experiment_config(),
        run_id=RUN_ID,
        concurrency=CONCURRENCY,
        max_transient_retries=MAX_TRANSIENT_RETRIES,
        max_parse_repairs=MAX_PARSE_REPAIRS,
    )
    _cap_runner_adapters(runner, aliases, budget)
    answers = await runner.run_answers(
        model_aliases=list(aliases),
        dry_run=False,
        allow_paid=True,
        retry_failed=retry_failed,
    )
    scientific.sync_from(checkpoint, label=f"{phase}:answer")
    ready = [
        example
        for example in examples
        if all(
            checkpoint.find_success(
                stage="answer",
                dataset=example.dataset_name,
                example_id=example.example_id,
                model_alias=alias,
            )
            is not None
            for alias in aliases
        )
    ]
    if not ready:
        raise Task009AuthorizationError(f"{phase}: no successful Stage-1 answers")
    conf_runner = ExperimentRunner(
        examples=ready,
        checkpoint=checkpoint,
        models_config=load_models_config(),
        experiment_config=load_experiment_config(),
        run_id=RUN_ID,
        concurrency=CONCURRENCY,
        max_transient_retries=MAX_TRANSIENT_RETRIES,
        max_parse_repairs=MAX_PARSE_REPAIRS,
    )
    _cap_runner_adapters(conf_runner, aliases, budget)
    confidence = await conf_runner.run_confidence(
        model_aliases=list(aliases),
        dry_run=False,
        allow_paid=True,
        retry_failed=retry_failed,
    )
    scientific.sync_from(checkpoint, label=f"{phase}:confidence")
    print(
        f"{phase}: answers new={answers.calls_needed} ok={answers.completed} "
        f"fail={answers.failed}; q1 new={confidence.calls_needed} "
        f"ok={confidence.completed} fail={confidence.failed} "
        f"q1_items={len(ready)}/{len(examples)}",
        flush=True,
    )
    return {
        "phase": phase,
        "expected_pairs": expected,
        "answer_completed": int(answers.completed),
        "answer_failed": int(answers.failed),
        "answer_new": int(answers.calls_needed),
        "confidence_completed": int(confidence.completed),
        "confidence_failed": int(confidence.failed),
        "confidence_new": int(confidence.calls_needed),
    }


async def _run_stage3(
    *,
    calls: Sequence[Stage3Call],
    checkpoint: CheckpointStore,
    budget: ProviderAttemptBudget,
    scientific: ScientificCallBudget,
    adapters: Mapping[str, Any],
    models: Any,
    phase: str,
) -> dict[str, Any]:
    todo: list[Stage3Call] = []
    reused = 0
    results: list[dict[str, Any]] = []
    for row in calls:
        checkpoint.register_request(
            request_key=row.request_key,
            run_id=RUN_ID,
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
                "repeat_index": row.repeat_index,
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
    scientific.sync_from(checkpoint, label=phase)
    code_commit = _git_commit()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    started = time.perf_counter()
    print(
        f"{phase}: sqlite reused={reused} todo={len(todo)} concurrency={CONCURRENCY}",
        flush=True,
    )
    progress_lock = asyncio.Lock()
    done = 0
    ok_count = 0
    fail_count = 0
    total = len(todo)

    async def execute_one(row: Stage3Call) -> dict[str, Any]:
        nonlocal done, ok_count, fail_count
        async with semaphore:
            result = await _execute_stage3_cell(
                row=row,
                adapter=adapters[row.model_alias],
                spec=models.models[row.model_alias],
                checkpoint=checkpoint,
                budget=budget,
                code_commit=code_commit,
            )
            async with progress_lock:
                done += 1
                if result.get("ok"):
                    ok_count += 1
                else:
                    fail_count += 1
                if done % 25 == 0 or done == total:
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed if elapsed else 0.0
                    print(
                        f"{phase}: {done}/{total} processed; {ok_count} success; "
                        f"{fail_count} failed; {rate:.2f}/s; {total - done} remaining",
                        flush=True,
                    )
            return result

    if todo:
        chunk_size = 64
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            gathered = await asyncio.gather(*(execute_one(row) for row in chunk))
            results.extend(gathered)
    successful = [item for item in results if item.get("ok")]
    failed_items = [item for item in results if not item.get("ok")]
    print(
        f"{phase}: planned={len(calls)} new={len(todo)} reused={reused} "
        f"success={len(successful)} failed={len(failed_items)} "
        f"wall={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return {
        "phase": phase,
        "planned": len(calls),
        "reused": reused,
        "new_scientific_calls": len(todo),
        "successful": len(successful),
        "failed": len(failed_items),
        "ok": len(failed_items) == 0,
    }


def _assert_endpoints(models: Any) -> None:
    observed = {alias: models.models[alias].api_model for alias in ENDPOINTS}
    if observed != dict(ENDPOINTS):
        raise Task009AuthorizationError(
            f"endpoint lock failed: {observed} vs {dict(ENDPOINTS)}"
        )


def _assert_prompt_hash() -> None:
    from .task006_lane_a import family_template_hash

    digest = family_template_hash(FAMILY)
    if digest != PROMPT_HASH_MODERATE:
        raise Task009AuthorizationError(
            f"moderate prompt hash {digest} != {PROMPT_HASH_MODERATE}"
        )
    if prompt_family_id(FAMILY) != PROMPT_FAMILY:
        raise Task009AuthorizationError("prompt family drifted")


async def run_paid(*, allow_paid: bool) -> dict[str, Any]:
    if not allow_paid:
        raise Task009AuthorizationError("paid Task 009 requires allow_paid=True")
    load_dotenv(PROJECT_ROOT / ".env")
    before = protected_fingerprints()
    freeze_sample()
    manifest = load_frozen_ids()
    paths = freeze_paths()
    assert_009_write_target(SQLITE_PATH)
    _assert_prompt_hash()
    models = load_models_config()
    _assert_endpoints(models)
    primary = load_examples(paths["examples_500"])
    secondary = load_examples(paths["examples_200"])
    repeats = load_examples(paths["examples_100"])
    if [row.example_id for row in primary] != manifest["sample"]["primary_ids"]:
        raise Task009AuthorizationError("primary JSONL order drifted from freeze")
    if [row.example_id for row in secondary] != manifest["sample"]["secondary_ids"]:
        raise Task009AuthorizationError("secondary JSONL order drifted from freeze")
    if [row.example_id for row in repeats] != manifest["sample"]["repeat_ids"]:
        raise Task009AuthorizationError("repeat JSONL order drifted from freeze")

    adapters = {alias: create_adapter(alias, models.models[alias]) for alias in ENDPOINTS}
    budget = ProviderAttemptBudget(PROVIDER_ATTEMPT_CAP)
    for adapter in adapters.values():
        _install_attempt_cap(adapter, budget)
    scientific = ScientificCallBudget(SCIENTIFIC_CAP)
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    phase_log: list[dict[str, Any]] = []
    started = time.perf_counter()
    with CheckpointStore(SQLITE_PATH) as checkpoint:
        for retry_failed in (False, True):
            label = "retry" if retry_failed else "first"
            print(f"Task 009 Stage-1/q1 primary {label}", flush=True)
            phase_log.append(
                await _run_stage12(
                    examples=primary,
                    aliases=GPT_CLAUDE,
                    checkpoint=checkpoint,
                    budget=budget,
                    scientific=scientific,
                    phase=f"primary_stage12_{label}",
                    retry_failed=retry_failed,
                )
            )
        primary_bundle = _stage12_bundle(checkpoint, primary, GPT_CLAUDE)
        primary_calls = build_stage3_calls(
            examples=primary,
            aliases=GPT_CLAUDE,
            bundle=primary_bundle,
            sample_hash=manifest["sample"]["primary_id_list_sha256"],
            roster="primary",
            repeat_indices=(0,),
        )
        if len(primary_calls) != 7000:
            print(
                f"WARNING: primary Stage-3 planned {len(primary_calls)} != 7000",
                flush=True,
            )
        for attempt_name in ("primary_stage3", "primary_stage3_retry"):
            print(f"Task 009 Stage-3 {attempt_name}", flush=True)
            phase_log.append(
                await _run_stage3(
                    calls=primary_calls,
                    checkpoint=checkpoint,
                    budget=budget,
                    scientific=scientific,
                    adapters=adapters,
                    models=models,
                    phase=attempt_name,
                )
            )
        extra_calls = build_stage3_calls(
            examples=repeats,
            aliases=GPT_CLAUDE,
            bundle=primary_bundle,
            sample_hash=manifest["sample"]["repeat_id_list_sha256"],
            roster="repeat_extra",
            repeat_indices=tuple(range(1, N_REPEAT_GENERATIONS)),
        )
        if len(extra_calls) != 2800:
            print(
                f"WARNING: repeat extras planned {len(extra_calls)} != 2800",
                flush=True,
            )
        for attempt_name in ("repeat_extras", "repeat_extras_retry"):
            print(f"Task 009 Stage-3 {attempt_name}", flush=True)
            phase_log.append(
                await _run_stage3(
                    calls=extra_calls,
                    checkpoint=checkpoint,
                    budget=budget,
                    scientific=scientific,
                    adapters=adapters,
                    models=models,
                    phase=attempt_name,
                )
            )
        for retry_failed in (False, True):
            label = "retry" if retry_failed else "first"
            print(f"Task 009 Stage-1/q1 secondary {label}", flush=True)
            phase_log.append(
                await _run_stage12(
                    examples=secondary,
                    aliases=GEMINI_GROK,
                    checkpoint=checkpoint,
                    budget=budget,
                    scientific=scientific,
                    phase=f"secondary_stage12_{label}",
                    retry_failed=retry_failed,
                )
            )
        secondary_bundle = _stage12_bundle(checkpoint, secondary, GEMINI_GROK)
        secondary_calls = build_stage3_calls(
            examples=secondary,
            aliases=GEMINI_GROK,
            bundle=secondary_bundle,
            sample_hash=manifest["sample"]["secondary_id_list_sha256"],
            roster="secondary",
            repeat_indices=(0,),
        )
        if len(secondary_calls) != 2800:
            print(
                f"WARNING: secondary Stage-3 planned {len(secondary_calls)} != 2800",
                flush=True,
            )
        for attempt_name in ("secondary_stage3", "secondary_stage3_retry"):
            print(f"Task 009 Stage-3 {attempt_name}", flush=True)
            phase_log.append(
                await _run_stage3(
                    calls=secondary_calls,
                    checkpoint=checkpoint,
                    budget=budget,
                    scientific=scientific,
                    adapters=adapters,
                    models=models,
                    phase=attempt_name,
                )
            )
    after = protected_fingerprints()
    if after["paperDirection"]["sha256"] != before["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection.txt changed during Task 009 paid run")
    for name in ("v2", "study1", "q2", "qual006", "qual007a", "qual007b"):
        if before[name].get("sha256") != after[name].get("sha256"):
            raise RuntimeError(f"{name} changed during Task 009 paid run")
    primary_stage3 = next(item for item in phase_log if item.get("phase") == "primary_stage3")
    return {
        "ok": bool(primary_stage3.get("ok")),
        "primary_complete": bool(primary_stage3.get("ok")),
        "scientific_used": scientific.used,
        "scientific_cap": SCIENTIFIC_CAP,
        "provider_attempts": budget.used,
        "provider_cap": PROVIDER_ATTEMPT_CAP,
        "wall_clock_seconds": time.perf_counter() - started,
        "phases": phase_log,
        "code_commit": _git_commit(),
    }

