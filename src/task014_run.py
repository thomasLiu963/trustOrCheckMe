"""Task 014 Stage-3 runner. New calls only for nonzero deltas."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv

from .checkpointing import CheckpointStore, deterministic_request_key
from .config import PROJECT_ROOT, load_models_config
from .model_adapters import create_adapter
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
    build_qualitative_repair_prompt,
    parse_qualitative_response,
)
from .task011_prompts import audit_stage3_prompt
from .task014_common import (
    CONCURRENCY_PER_PROVIDER,
    CONCURRENCY_ROUTE,
    ENDPOINTS,
    EXPERIMENT_VERSION,
    FAMILY,
    MAX_PARSE_REPAIRS,
    MAX_TRANSIENT_RETRIES,
    NEW_DELTAS,
    PROMPT_FAMILY_009,
    PROMPT_FAMILY_011,
    PROMPT_VERSION_009,
    PROMPT_VERSION_011,
    PROVIDER_ATTEMPT_CAP,
    RUN_ID,
    SCIENTIFIC_CAP,
    SQLITE_PATH,
)
from .task014_data import FrozenItem, offset_prompt
from .task014_transform import render_q, transform_q


class Task014AuthorizationError(PermissionError):
    pass


class ScientificCallBudget:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.used = 0

    def claim(self) -> None:
        if self.used >= self.cap:
            raise Task014AuthorizationError(f"scientific cap {self.cap} reached")
        self.used += 1


def request_key(item: FrozenItem, delta: float) -> str:
    displayed = transform_q(item.q1, delta)
    if item.task == "mmlu":
        version, family, dataset = PROMPT_VERSION_009, PROMPT_FAMILY_009, "mmlu_pro"
    else:
        version, family, dataset = PROMPT_VERSION_011, PROMPT_FAMILY_011, "livecodebench_code_generation_lite"
    return deterministic_request_key(
        stage="verification",
        dataset=dataset,
        example_id=item.question_id,
        model_id=item.model_alias,
        prompt_version=version,
        experiment_version=EXPERIMENT_VERSION,
        prompt_family=family,
        stake={
            "stakes_family": FAMILY,
            "score_condition": f"offset_{delta}",
            "displayed_confidence": displayed,
            "delta": delta,
        },
    )


def _provider(alias: str) -> str:
    return "openai" if alias.startswith("openai") else "anthropic"


async def _call_model(
    *,
    adapter: Any,
    spec: Any,
    checkpoint: CheckpointStore,
    request_key: str,
    prompt: str,
    item: FrozenItem,
    delta: float,
) -> dict[str, Any]:
    last_raw = ""
    parse_repairs = 0
    prompt_to_send = prompt
    displayed = transform_q(item.q1, delta)
    version = PROMPT_VERSION_009 if item.task == "mmlu" else PROMPT_VERSION_011
    family = PROMPT_FAMILY_009 if item.task == "mmlu" else PROMPT_FAMILY_011

    def parse(raw: str) -> dict[str, Any]:
        payload = parse_qualitative_response(raw)
        return {
            "parsed_action": payload.action.value,
            "verify": payload.action.value == "VERIFY_FIRST",
            "probability_correct": item.q1,
            "q_delta": displayed,
            "delta": delta,
            "task": item.task,
        }

    for parse_attempt in range(MAX_PARSE_REPAIRS + 1):
        attempt_kind = "generation" if parse_attempt == 0 else "parse_repair"
        if parse_attempt:
            parse_repairs += 1
            prompt_to_send = build_qualitative_repair_prompt(prompt, last_raw)
        started = time.perf_counter()
        try:
            response = await adapter.generate(
                stage="verification",
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
            return {"ok": False, "error": str(error), "raw_response": raw, "request_key": request_key}
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
            "question_id": item.question_id,
            "model_alias": item.model_alias,
            "model_endpoint": spec.api_model,
            "provider": spec.provider,
            "prompt_version": version,
            "prompt_family": family,
            "raw_response": raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_seconds * 1000,
            "estimated_cost_usd": estimate_cost_usd(spec, input_tokens, output_tokens),
            "api_timestamp": datetime.now(UTC).isoformat(),
            "attempt_number": int(attempt_number),
            "parse_repairs": parse_repairs,
            "code_commit": _git_commit(),
            "incorrect": item.incorrect,
            "reused_delta0": False,
            **parsed,
        }
        checkpoint.mark_success(request_key, record)
        return {"ok": True, "record": record, "request_key": request_key}
    return {"ok": False, "error": "parse loop ended", "request_key": request_key}


async def _one(
    item: FrozenItem,
    delta: float,
    adapters: Mapping[str, Any],
    specs: Mapping[str, Any],
    checkpoint: CheckpointStore,
    sems: dict[str, asyncio.Semaphore],
    scientific: ScientificCallBudget,
) -> dict[str, Any]:
    key = request_key(item, delta)
    displayed = transform_q(item.q1, delta)
    spec = specs[item.model_alias]
    dataset = "mmlu_pro" if item.task == "mmlu" else "livecodebench_code_generation_lite"
    version = PROMPT_VERSION_009 if item.task == "mmlu" else PROMPT_VERSION_011
    checkpoint.register_request(
        request_key=key,
        run_id=RUN_ID,
        stage="verification",
        dataset=dataset,
        example_id=item.question_id,
        model_alias=item.model_alias,
        requested_model_id=spec.api_model,
        prompt_version=version,
        stake={"delta": delta, "displayed_confidence": displayed},
    )
    if checkpoint.request_status(key) == "success":
        return {"ok": True, "record": checkpoint.get_record(key) or {}, "reused": True}
    prompt = offset_prompt(item, delta)
    if item.task == "mmlu":
        problems = audit_prompt_body(
            prompt,
            question=item.question,
            choices=item.choices or {},
            frozen_answer=item.frozen_output,
            family=FAMILY,
            displayed_confidence=displayed,
        )
    else:
        problems = audit_stage3_prompt(
            prompt,
            question=item.question,
            frozen_code=item.frozen_output,
            displayed_confidence=displayed,
        )
    if problems:
        return {"ok": False, "error": f"prompt audit failed: {problems}", "request_key": key}
    scientific.claim()
    provider = _provider(item.model_alias)
    try:
        async with sems[provider]:
            async with sems["route"]:
                return await _call_model(
                    adapter=adapters[item.model_alias],
                    spec=spec,
                    checkpoint=checkpoint,
                    request_key=key,
                    prompt=prompt,
                    item=item,
                    delta=delta,
                )
    except Exception as error:
        return {"ok": False, "error": str(error), "request_key": key}


def reused_delta0_record(item: FrozenItem) -> dict[str, Any]:
    displayed = transform_q(item.q1, 0.0)
    return {
        "study_id": EXPERIMENT_VERSION,
        "run_id": RUN_ID,
        "request_key": "reused_true_q_visible",
        "question_id": item.question_id,
        "model_alias": item.model_alias,
        "model_endpoint": ENDPOINTS[item.model_alias],
        "task": item.task,
        "delta": 0.0,
        "q_delta": displayed,
        "rendered_q_delta": render_q(displayed),
        "parsed_action": item.true_q_action,
        "verify": item.true_q_verify,
        "incorrect": item.incorrect,
        "reused_delta0": True,
        "estimated_cost_usd": 0.0,
    }


async def run_stage3(items: list[FrozenItem]) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    models = load_models_config()
    for alias, endpoint in ENDPOINTS.items():
        if models.models[alias].api_model != endpoint:
            raise Task014AuthorizationError(f"MODEL_VERSION_BLOCKER: {alias} is {models.models[alias].api_model}")
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointStore(SQLITE_PATH)
    budget = ProviderAttemptBudget(PROVIDER_ATTEMPT_CAP)
    adapters = {}
    specs = {}
    for alias in ENDPOINTS:
        spec = models.models[alias]
        adapter = create_adapter(alias, spec, max_transient_retries=MAX_TRANSIENT_RETRIES)
        _install_attempt_cap(adapter, budget)
        adapters[alias] = adapter
        specs[alias] = spec
    scientific = ScientificCallBudget(SCIENTIFIC_CAP)
    with checkpoint._lock:
        existing = int(checkpoint._connection.execute("SELECT COUNT(*) AS n FROM requests").fetchone()["n"])
    scientific.used = existing
    sems = {
        "route": asyncio.Semaphore(CONCURRENCY_ROUTE),
        "openai": asyncio.Semaphore(CONCURRENCY_PER_PROVIDER),
        "anthropic": asyncio.Semaphore(CONCURRENCY_PER_PROVIDER),
    }
    jobs = [(item, delta) for item in items for delta in NEW_DELTAS]
    results = await asyncio.gather(
        *[_one(item, delta, adapters, specs, checkpoint, sems, scientific) for item, delta in jobs],
        return_exceptions=False,
    )
    ok = sum(1 for r in results if r.get("ok"))
    fail = [r for r in results if not r.get("ok")]
    return {
        "n_jobs": len(jobs),
        "n_ok": ok,
        "n_fail": len(fail),
        "scientific_used": scientific.used,
        "failures": fail[:20],
    }
