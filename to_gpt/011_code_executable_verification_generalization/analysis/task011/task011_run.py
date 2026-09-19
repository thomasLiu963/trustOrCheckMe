"""Optimized Task 011 paid runner: pipelined code → q1 → routing."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv

from .checkpointing import CheckpointStore, deterministic_request_key
from .config import PROJECT_ROOT, load_models_config
from .model_adapters import create_adapter
from .prompts import build_repair_prompt
from .study1_smoke import (
    ProviderAttemptBudget,
    ProviderAttemptCapReached,
    _git_commit,
    _install_attempt_cap,
    _response_value,
    estimate_cost_usd,
)
from .task011_common import (
    CONCURRENCY_CODE,
    CONCURRENCY_PER_PROVIDER,
    CONCURRENCY_ROUTE,
    DATASET_NAME,
    ENDPOINTS,
    EXPERIMENT_VERSION,
    FAMILY,
    GPT_CLAUDE,
    MAX_PARSE_REPAIRS,
    MAX_TRANSIENT_RETRIES,
    PILOT_CONDITIONS,
    PROMPT_FAMILY,
    PROMPT_VERSION,
    PROVIDER_ATTEMPT_CAP,
    RETRY_FAILED_PASSES,
    RUN_ID,
    SCIENTIFIC_CAP,
    SCORE_CONDITIONS,
    SQLITE_PATH,
    VISIBLE_FIXED_CONDITIONS,
    prompt_sha256,
    protected_fingerprints,
)
from .task011_prompts import (
    audit_stage3_prompt,
    build_code_prompt,
    build_q1_prompt,
    build_qualitative_repair_prompt,
    build_stage3_prompt,
    extract_python_code,
    parse_q1,
    parse_qualitative_response,
)


class Task011AuthorizationError(PermissionError):
    pass


class ScientificCallBudget:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.used = 0

    def sync_from(self, checkpoint: CheckpointStore, *, label: str) -> int:
        with checkpoint._lock:
            row = checkpoint._connection.execute(
                "SELECT COUNT(*) AS n FROM requests"
            ).fetchone()
        n = int(row["n"])
        if n > self.cap:
            raise Task011AuthorizationError(
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


def code_request_key(question_id: str, alias: str) -> str:
    return deterministic_request_key(
        stage="code",
        dataset=DATASET_NAME,
        example_id=question_id,
        model_id=alias,
        prompt_version=PROMPT_VERSION,
        experiment_version=EXPERIMENT_VERSION,
        prompt_family=PROMPT_FAMILY,
        stake={"stage": "code"},
    )


def q1_request_key(question_id: str, alias: str, code_key: str) -> str:
    return deterministic_request_key(
        stage="confidence",
        dataset=DATASET_NAME,
        example_id=question_id,
        model_id=alias,
        prompt_version=PROMPT_VERSION,
        experiment_version=EXPERIMENT_VERSION,
        prompt_family=PROMPT_FAMILY,
        stake={"stage": "q1"},
        dependency={"code_request_key": code_key},
    )


def stage3_request_key(
    *,
    question_id: str,
    alias: str,
    condition: str,
    displayed: float | None,
    repeat_index: int,
    code_key: str,
) -> str:
    return deterministic_request_key(
        stage="verification",
        dataset=DATASET_NAME,
        example_id=question_id,
        model_id=alias,
        prompt_version=PROMPT_VERSION,
        experiment_version=EXPERIMENT_VERSION,
        prompt_family=PROMPT_FAMILY,
        stake={
            "stakes_family": FAMILY,
            "score_condition": condition,
            "displayed_confidence": displayed,
            "repeat_index": repeat_index,
        },
        dependency={"code_request_key": code_key},
    )


def _provider_for(alias: str) -> str:
    return "openai" if alias.startswith("openai") else "anthropic"


@dataclass
class Semaphores:
    code: asyncio.Semaphore
    route: asyncio.Semaphore
    per_provider: dict[str, asyncio.Semaphore]


def _semaphores() -> Semaphores:
    return Semaphores(
        code=asyncio.Semaphore(CONCURRENCY_CODE),
        route=asyncio.Semaphore(CONCURRENCY_ROUTE),
        per_provider={
            "openai": asyncio.Semaphore(CONCURRENCY_PER_PROVIDER),
            "anthropic": asyncio.Semaphore(CONCURRENCY_PER_PROVIDER),
        },
    )


async def _call_model(
    *,
    adapter: Any,
    spec: Any,
    checkpoint: CheckpointStore,
    request_key: str,
    stage: str,
    prompt: str,
    parse,
    repair_prompt,
    question_id: str,
    model_alias: str,
    extra_record: Mapping[str, Any],
) -> dict[str, Any]:
    last_raw = ""
    parse_repairs = 0
    prompt_to_send = prompt
    for parse_attempt in range(MAX_PARSE_REPAIRS + 1):
        attempt_kind = "generation" if parse_attempt == 0 else "parse_repair"
        if parse_attempt:
            parse_repairs += 1
            prompt_to_send = repair_prompt(prompt, last_raw)
        started = time.perf_counter()
        try:
            mapped = {
                "code": "code",
                "confidence": "confidence",
                "verification": "verification",
            }.get(stage, stage)
            response = await adapter.generate(
                stage=mapped,
                prompt=prompt_to_send,
                request_key=request_key,
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
                        request_key=request_key,
                        attempt_kind=attempt_kind,
                        success=False,
                        provider=adapter.provider,
                        requested_model_id=adapter.api_model,
                        latency_seconds=time.perf_counter() - started,
                        error=cause,
                    )
                    checkpoint.mark_failed(request_key, cause)
                    raise cause
                cause = cause.__cause__ or getattr(cause, "__context__", None)
            checkpoint.record_attempt(
                request_key=request_key,
                attempt_kind=attempt_kind,
                success=False,
                provider=adapter.provider,
                requested_model_id=adapter.api_model,
                latency_seconds=time.perf_counter() - started,
                error=error,
            )
            checkpoint.mark_failed(request_key, error)
            return {"ok": False, "error": str(error), "request_key": request_key}
        raw = str(_response_value(response, "raw_response", default="") or "")
        last_raw = raw
        refusal = bool(_response_value(response, "refused", "refusal", default=False))
        input_tokens = _response_value(response, "input_tokens")
        output_tokens = _response_value(response, "output_tokens")
        latency_seconds = float(_response_value(response, "latency_seconds") or 0.0)
        try:
            if refusal:
                raise ValueError("Provider returned a refusal")
            parsed = parse(raw)
        except (TypeError, ValueError) as error:
            checkpoint.record_attempt(
                request_key=request_key,
                attempt_kind=attempt_kind,
                success=False,
                provider=adapter.provider,
                requested_model_id=adapter.api_model,
                latency_seconds=latency_seconds,
                raw_output=raw,
                parse_error=str(error),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                refusal=refusal,
            )
            if parse_attempt < MAX_PARSE_REPAIRS:
                continue
            checkpoint.mark_failed(request_key, error)
            return {
                "ok": False,
                "error": str(error),
                "raw_response": raw,
                "request_key": request_key,
            }
        attempt_number = checkpoint.record_attempt(
            request_key=request_key,
            attempt_kind=attempt_kind,
            success=True,
            provider=adapter.provider,
            requested_model_id=adapter.api_model,
            latency_seconds=latency_seconds,
            raw_output=raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=_response_value(response, "total_tokens"),
            finish_reason=_response_value(response, "finish_reason"),
            refusal=False,
        )
        record = {
            "study_id": EXPERIMENT_VERSION,
            "run_id": RUN_ID,
            "request_key": request_key,
            "question_id": question_id,
            "model_alias": model_alias,
            "model_endpoint": spec.api_model,
            "provider": spec.provider,
            "prompt_version": PROMPT_VERSION,
            "prompt_family": PROMPT_FAMILY,
            "prompt_hash": prompt_sha256(prompt),
            "raw_response": raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_seconds * 1000,
            "estimated_cost_usd": estimate_cost_usd(spec, input_tokens, output_tokens),
            "api_timestamp": datetime.now(UTC).isoformat(),
            "attempt_number": int(attempt_number),
            "parse_repairs": parse_repairs,
            "code_commit": _git_commit(),
            **dict(extra_record),
            **(parsed if isinstance(parsed, dict) else {"parsed": parsed}),
        }
        checkpoint.mark_success(request_key, record)
        return {"ok": True, "record": record, "request_key": request_key, "parsed": parsed}
    return {"ok": False, "error": "parse loop ended", "request_key": request_key}


def _reuse_or_register(
    checkpoint: CheckpointStore,
    *,
    request_key: str,
    stage: str,
    question_id: str,
    alias: str,
    endpoint: str,
    stake: Mapping[str, Any],
) -> dict[str, Any] | None:
    checkpoint.register_request(
        request_key=request_key,
        run_id=RUN_ID,
        stage=stage,
        dataset=DATASET_NAME,
        example_id=question_id,
        model_alias=alias,
        requested_model_id=endpoint,
        prompt_version=PROMPT_VERSION,
        stake=stake,
    )
    if checkpoint.request_status(request_key) == "success":
        return checkpoint.get_record(request_key) or {}
    return None


async def _generate_code(
    *,
    item: Mapping[str, Any],
    alias: str,
    adapter: Any,
    spec: Any,
    checkpoint: CheckpointStore,
    sems: Semaphores,
) -> dict[str, Any]:
    question_id = str(item["question_id"])
    key = code_request_key(question_id, alias)
    existing = _reuse_or_register(
        checkpoint,
        request_key=key,
        stage="code",
        question_id=question_id,
        alias=alias,
        endpoint=spec.api_model,
        stake={"stage": "code"},
    )
    if existing is not None:
        return {"ok": True, "record": existing, "request_key": key, "reused": True}

    prompt = build_code_prompt(item)

    def parse(raw: str) -> dict[str, Any]:
        code, status = extract_python_code(raw)
        return {
            "frozen_code": code,
            "extraction_status": status,
            "malformed": int(not bool(code.strip())),
        }

    provider = _provider_for(alias)
    async with sems.per_provider[provider]:
        async with sems.code:
            result = await _call_model(
                adapter=adapter,
                spec=spec,
                checkpoint=checkpoint,
                request_key=key,
                stage="code",
                prompt=prompt,
                parse=parse,
                repair_prompt=lambda original, raw: original
                + "\n\nReturn only a complete Python program in a ```python fence.",
                question_id=question_id,
                model_alias=alias,
                extra_record={},
            )
    result["reused"] = False
    return result


async def _generate_q1(
    *,
    item: Mapping[str, Any],
    alias: str,
    frozen_code: str,
    code_key: str,
    adapter: Any,
    spec: Any,
    checkpoint: CheckpointStore,
    sems: Semaphores,
) -> dict[str, Any]:
    question_id = str(item["question_id"])
    key = q1_request_key(question_id, alias, code_key)
    existing = _reuse_or_register(
        checkpoint,
        request_key=key,
        stage="confidence",
        question_id=question_id,
        alias=alias,
        endpoint=spec.api_model,
        stake={"stage": "q1", "code_request_key": code_key},
    )
    if existing is not None:
        return {"ok": True, "record": existing, "request_key": key, "reused": True}
    prompt = build_q1_prompt(
        question=str(item.get("question_content") or ""),
        frozen_code=frozen_code,
    )

    def parse(raw: str) -> dict[str, Any]:
        return {"probability_correct": parse_q1(raw), "frozen_code": frozen_code}

    provider = _provider_for(alias)
    async with sems.per_provider[provider]:
        async with sems.route:
            result = await _call_model(
                adapter=adapter,
                spec=spec,
                checkpoint=checkpoint,
                request_key=key,
                stage="confidence",
                prompt=prompt,
                parse=parse,
                repair_prompt=lambda original, raw: build_repair_prompt(
                    "confidence", original, raw, "invalid probability_correct json"
                ),
                question_id=question_id,
                model_alias=alias,
                extra_record={"code_request_key": code_key},
            )
    result["reused"] = False
    return result


async def _generate_stage3(
    *,
    item: Mapping[str, Any],
    alias: str,
    frozen_code: str,
    q1: float,
    code_key: str,
    condition: str,
    repeat_index: int,
    adapter: Any,
    spec: Any,
    checkpoint: CheckpointStore,
    sems: Semaphores,
) -> dict[str, Any]:
    question_id = str(item["question_id"])
    displayed = condition_displayed(condition, q1)
    key = stage3_request_key(
        question_id=question_id,
        alias=alias,
        condition=condition,
        displayed=displayed,
        repeat_index=repeat_index,
        code_key=code_key,
    )
    existing = _reuse_or_register(
        checkpoint,
        request_key=key,
        stage="verification",
        question_id=question_id,
        alias=alias,
        endpoint=spec.api_model,
        stake={
            "score_condition": condition,
            "displayed_confidence": displayed,
            "repeat_index": repeat_index,
        },
    )
    if existing is not None:
        return {"ok": True, "record": existing, "request_key": key, "reused": True}
    question = str(item.get("question_content") or "")
    prompt = build_stage3_prompt(
        question=question,
        frozen_code=frozen_code,
        displayed_confidence=displayed,
    )
    problems = audit_stage3_prompt(
        prompt,
        question=question,
        frozen_code=frozen_code,
        displayed_confidence=displayed,
    )
    if problems:
        raise Task011AuthorizationError(
            f"prompt audit failed {question_id} {condition}: {problems}"
        )

    def parse(raw: str) -> dict[str, Any]:
        payload = parse_qualitative_response(raw)
        return {
            "parsed_action": payload.action.value,
            "probability_correct": q1,
            "frozen_code": frozen_code,
            "score_condition": condition,
            "displayed_confidence": displayed,
            "repeat_index": repeat_index,
        }

    provider = _provider_for(alias)
    async with sems.per_provider[provider]:
        async with sems.route:
            result = await _call_model(
                adapter=adapter,
                spec=spec,
                checkpoint=checkpoint,
                request_key=key,
                stage="verification",
                prompt=prompt,
                parse=parse,
                repair_prompt=build_qualitative_repair_prompt,
                question_id=question_id,
                model_alias=alias,
                extra_record={
                    "code_request_key": code_key,
                    "score_condition": condition,
                    "repeat_index": repeat_index,
                },
            )
    result["reused"] = False
    return result


async def process_pair(
    *,
    item: Mapping[str, Any],
    alias: str,
    conditions: Sequence[str],
    extra_visible_repeats: bool,
    adapter: Any,
    spec: Any,
    checkpoint: CheckpointStore,
    sems: Semaphores,
    scientific: ScientificCallBudget,
) -> dict[str, Any]:
    code_res = await _generate_code(
        item=item,
        alias=alias,
        adapter=adapter,
        spec=spec,
        checkpoint=checkpoint,
        sems=sems,
    )
    scientific.sync_from(checkpoint, label="code")
    if not code_res.get("ok"):
        return {
            "ok": False,
            "question_id": item["question_id"],
            "model_alias": alias,
            "error": code_res.get("error"),
            "stage": "code",
        }
    record = code_res["record"]
    frozen = str(record.get("frozen_code") or "")
    code_key = str(code_res["request_key"])
    q1_res = await _generate_q1(
        item=item,
        alias=alias,
        frozen_code=frozen,
        code_key=code_key,
        adapter=adapter,
        spec=spec,
        checkpoint=checkpoint,
        sems=sems,
    )
    scientific.sync_from(checkpoint, label="q1")
    if not q1_res.get("ok"):
        return {
            "ok": False,
            "question_id": item["question_id"],
            "model_alias": alias,
            "error": q1_res.get("error"),
            "stage": "q1",
        }
    q1 = float(q1_res["record"]["probability_correct"])
    stage3 = []
    for condition in conditions:
        stage3.append(
            _generate_stage3(
                item=item,
                alias=alias,
                frozen_code=frozen,
                q1=q1,
                code_key=code_key,
                condition=condition,
                repeat_index=0,
                adapter=adapter,
                spec=spec,
                checkpoint=checkpoint,
                sems=sems,
            )
        )
    if extra_visible_repeats:
        for repeat_index in (1, 2):
            for condition in VISIBLE_FIXED_CONDITIONS:
                stage3.append(
                    _generate_stage3(
                        item=item,
                        alias=alias,
                        frozen_code=frozen,
                        q1=q1,
                        code_key=code_key,
                        condition=condition,
                        repeat_index=repeat_index,
                        adapter=adapter,
                        spec=spec,
                        checkpoint=checkpoint,
                        sems=sems,
                    )
                )
    gathered = await asyncio.gather(*stage3)
    scientific.sync_from(checkpoint, label="stage3")
    failed = [row for row in gathered if not row.get("ok")]
    return {
        "ok": not failed,
        "question_id": item["question_id"],
        "model_alias": alias,
        "frozen_code": frozen,
        "q1": q1,
        "stage3_ok": len(gathered) - len(failed),
        "stage3_fail": len(failed),
        "error": failed[0].get("error") if failed else None,
    }


async def run_pairs(
    *,
    items: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    conditions: Sequence[str],
    repeat_ids: Sequence[str],
    checkpoint: CheckpointStore,
    phase: str,
) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    models = load_models_config()
    observed = {alias: models.models[alias].api_model for alias in ENDPOINTS}
    if observed != dict(ENDPOINTS):
        raise Task011AuthorizationError(f"endpoint lock failed: {observed}")
    budget = ProviderAttemptBudget(PROVIDER_ATTEMPT_CAP)
    adapters = {}
    for alias in aliases:
        spec = models.models[alias]
        adapter = create_adapter(
            alias,
            spec,
            max_transient_retries=MAX_TRANSIENT_RETRIES,
        )
        _install_attempt_cap(adapter, budget)
        adapters[alias] = adapter
    scientific = ScientificCallBudget(SCIENTIFIC_CAP)
    scientific.sync_from(checkpoint, label=phase)
    sems = _semaphores()
    repeat_set = set(repeat_ids)
    started = time.perf_counter()
    jobs = []
    for item in items:
        qid = str(item["question_id"])
        for alias in aliases:
            jobs.append((item, alias, qid in repeat_set))
    print(
        f"{phase}: pairs={len(jobs)} items={len(items)} aliases={list(aliases)} "
        f"code={CONCURRENCY_CODE} route={CONCURRENCY_ROUTE} "
        f"per_provider={CONCURRENCY_PER_PROVIDER} sqlite_used={scientific.used}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    lock = asyncio.Lock()
    done = 0
    ok_count = 0
    fail_count = 0
    total = len(jobs)

    async def one(item, alias, extra_visible_repeats) -> dict[str, Any]:
        nonlocal done, ok_count, fail_count
        result = await process_pair(
            item=item,
            alias=alias,
            conditions=conditions,
            extra_visible_repeats=extra_visible_repeats,
            adapter=adapters[alias],
            spec=models.models[alias],
            checkpoint=checkpoint,
            sems=sems,
            scientific=scientific,
        )
        async with lock:
            done += 1
            if result.get("ok"):
                ok_count += 1
            else:
                fail_count += 1
            if done % 5 == 0 or done == total:
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                print(
                    f"{phase}: {done}/{total} pairs; {ok_count} ok; {fail_count} fail; "
                    f"{rate:.2f}/s; {total - done} remaining; "
                    f"scientific={scientific.used}",
                    flush=True,
                )
        return result

    # All pairs in flight; inner semaphores cap provider/stage concurrency.
    gathered = await asyncio.gather(
        *(one(item, alias, extra) for item, alias, extra in jobs),
        return_exceptions=True,
    )
    results = []
    for job, row in zip(jobs, gathered):
        if isinstance(row, Exception):
            results.append(
                {
                    "ok": False,
                    "question_id": job[0]["question_id"],
                    "model_alias": job[1],
                    "error": str(row),
                    "stage": "pair",
                }
            )
        else:
            results.append(row)
    for retry_pass in range(RETRY_FAILED_PASSES):
        failed_jobs = [
            job
            for job, row in zip(jobs, results)
            if not row.get("ok")
        ]
        if not failed_jobs:
            break
        print(
            f"{phase}: retry pass {retry_pass + 1} of {len(failed_jobs)} failed pairs",
            flush=True,
        )
        retried_raw = await asyncio.gather(
            *(one(item, alias, extra) for item, alias, extra in failed_jobs),
            return_exceptions=True,
        )
        retried = []
        for job, row in zip(failed_jobs, retried_raw):
            if isinstance(row, Exception):
                retried.append(
                    {
                        "ok": False,
                        "question_id": job[0]["question_id"],
                        "model_alias": job[1],
                        "error": str(row),
                        "stage": "pair",
                    }
                )
            else:
                retried.append(row)
        by_key = {
            (str(row.get("question_id")), str(row.get("model_alias"))): row
            for row in retried
        }
        results = [
            by_key.get((str(job[0]["question_id"]), str(job[1])), row)
            for job, row in zip(jobs, results)
        ]
    failed = [row for row in results if not row.get("ok")]
    print(
        f"{phase}: done pairs={len(results)} fail={len(failed)} "
        f"scientific={scientific.used} provider_attempts={budget.used} "
        f"wall={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return {
        "phase": phase,
        "pairs": len(results),
        "failed": len(failed),
        "ok": len(failed) == 0,
        "scientific_used": scientific.used,
        "provider_attempts": budget.used,
        "results": results,
        "paperDirection_sha256": protected_fingerprints()["paperDirection"]["sha256"],
    }


async def run_pilot_paid(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointStore(SQLITE_PATH)
    return await run_pairs(
        items=items,
        aliases=GPT_CLAUDE,
        conditions=PILOT_CONDITIONS,
        repeat_ids=(),
        checkpoint=checkpoint,
        phase="pilot",
    )


async def run_main_paid(
    items: Sequence[Mapping[str, Any]],
    repeat_ids: Sequence[str],
) -> dict[str, Any]:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointStore(SQLITE_PATH)
    return await run_pairs(
        items=items,
        aliases=GPT_CLAUDE,
        conditions=SCORE_CONDITIONS,
        repeat_ids=repeat_ids,
        checkpoint=checkpoint,
        phase="confirmatory",
    )


def frozen_code_jobs(
    items: Mapping[str, Mapping[str, Any]],
    aliases: Sequence[str],
) -> list[dict[str, Any]]:
    checkpoint = CheckpointStore(SQLITE_PATH)
    jobs = []
    for qid, item in items.items():
        for alias in aliases:
            record = checkpoint.find_success(
                stage="code",
                dataset=DATASET_NAME,
                example_id=qid,
                model_alias=alias,
            )
            jobs.append(
                {
                    "question_id": qid,
                    "model_alias": alias,
                    "item": item,
                    "code": "" if record is None else str(record.get("frozen_code") or ""),
                }
            )
    return jobs
