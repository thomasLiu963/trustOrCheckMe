"""V2 reuse, planning, and matched-factorial execution."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import math
import subprocess
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from .checkpointing import CheckpointStore, deterministic_request_key
from .config import (
    ModelsConfig,
    V2ExperimentConfig,
    load_models_config,
    load_v2_experiment_config,
)
from .model_adapters import ModelAdapter, create_adapter
from .prompts import (
    ANSWER_PROMPT_VERSION,
    CONFIDENCE_PROMPT_VERSION,
    build_answer_prompt,
    build_confidence_prompt,
)
from .schemas import (
    AnswerRecord,
    BenchmarkExample,
    ConfidenceRecord,
    ConfidenceVisibility,
    DecisionOwner,
    VerificationDecisionRecord,
)
from .v2_prompts import (
    PRIMARY_PROMPT_FAMILY,
    build_verification_prompt,
    build_verification_repair_prompt,
    parse_verification_response,
)

V1_REUSABLE_MODELS = {
    "openai_gpt56_sol": "gpt-5.6-sol",
    "anthropic_sonnet5": "claude-sonnet-5",
}


def _v1_config_compatible(alias: str, spec: Any) -> bool:
    if V1_REUSABLE_MODELS.get(alias) != spec.api_model:
        return False
    if spec.max_output_tokens != 64:
        return False
    if alias == "openai_gpt56_sol":
        return (
            spec.provider == "openai"
            and spec.api_style == "responses"
            and spec.reasoning_effort == "none"
        )
    return (
        spec.provider == "anthropic"
        and spec.api_style == "messages"
        and spec.thinking is not None
        and spec.thinking.type == "disabled"
    )


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _response_value(response: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = _value(response, name, None)
        if value is not None:
            return value
    return default


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _sample_sha256(examples: Sequence[BenchmarkExample]) -> str:
    content = "\n".join(
        example.model_dump_json()
        for example in sorted(examples, key=lambda row: row.example_id)
    )
    return hashlib.sha256((content + "\n").encode("utf-8")).hexdigest()


def _answer_key(example: BenchmarkExample, alias: str, api_model: str) -> str:
    return deterministic_request_key(
        stage="answer",
        dataset=example.dataset_name,
        example_id=example.example_id,
        model_id=api_model,
        prompt_version=ANSWER_PROMPT_VERSION,
        experiment_version="v2",
    )


def _confidence_key(
    example: BenchmarkExample,
    alias: str,
    api_model: str,
    answer_request_key: str,
) -> str:
    return deterministic_request_key(
        stage="confidence",
        dataset=example.dataset_name,
        example_id=example.example_id,
        model_id=api_model,
        prompt_version=CONFIDENCE_PROMPT_VERSION,
        dependency=answer_request_key,
        experiment_version="v2",
    )


@dataclasses.dataclass(frozen=True)
class V1ReuseAudit:
    reusable_answers: int
    reusable_confidence: int
    missing_answers: int
    missing_confidence: int


@dataclasses.dataclass(frozen=True)
class V2Plan:
    examples: int
    models: int
    answer_requests: int
    confidence_requests: int
    verification_requests: int
    v1_answer_reuse: int
    v1_confidence_reuse: int
    v2_cache_hits: int
    new_requests: int
    approximate_cost_usd: float
    estimated_input_tokens: int
    maximum_output_tokens: int
    cost_by_stage: dict[str, float]
    remaining_requests_by_model: dict[str, int]
    estimated_runtime_seconds: float | None
    runtime_basis_successful_attempts: int
    runtime_assumed_concurrency: int
    unknown_pricing_models: tuple[str, ...]
    factor_counts: dict[str, int]


@dataclasses.dataclass(frozen=True)
class V2RunSummary:
    run_id: str
    planned_requests: int
    calls_needed: int
    completed: int
    failed: int
    skipped_successes: int
    approximate_cost_usd: float
    dry_run: bool
    sample_payloads: tuple[dict[str, Any], ...] = ()


def _selected_models(
    models_config: ModelsConfig, aliases: Sequence[str] | None
) -> list[tuple[str, Any]]:
    selected = list(aliases) if aliases is not None else list(models_config.models)
    unknown = [alias for alias in selected if alias not in models_config.models]
    if unknown:
        raise KeyError(f"Unknown model aliases: {', '.join(unknown)}")
    return [(alias, models_config.models[alias]) for alias in selected]


def audit_v1_reuse(
    *,
    examples: Sequence[BenchmarkExample],
    v1_checkpoint: str | Path,
    models_config: ModelsConfig | None = None,
    model_aliases: Sequence[str] | None = None,
) -> V1ReuseAudit:
    """Count exact-prompt/model V1 Stage-1/2 reuse candidates."""
    models = models_config or load_models_config()
    reusable_answers = reusable_confidence = 0
    missing_answers = missing_confidence = 0
    with CheckpointStore(v1_checkpoint) as source:
        for example in examples:
            for alias, spec in _selected_models(models, model_aliases):
                if not _v1_config_compatible(alias, spec):
                    missing_answers += 1
                    missing_confidence += 1
                    continue
                answer = source.find_success(
                    stage="answer",
                    dataset=example.dataset_name,
                    example_id=example.example_id,
                    model_alias=alias,
                    requested_model_id=spec.api_model,
                    prompt_version=ANSWER_PROMPT_VERSION,
                )
                if answer is None:
                    missing_answers += 1
                    missing_confidence += 1
                    continue
                reusable_answers += 1
                confidence = source.find_success(
                    stage="confidence",
                    dataset=example.dataset_name,
                    example_id=example.example_id,
                    model_alias=alias,
                    requested_model_id=spec.api_model,
                    prompt_version=CONFIDENCE_PROMPT_VERSION,
                )
                if (
                    confidence is not None
                    and confidence.get("frozen_answer_label")
                    == answer.get("answer_label")
                ):
                    reusable_confidence += 1
                else:
                    missing_confidence += 1
    return V1ReuseAudit(
        reusable_answers=reusable_answers,
        reusable_confidence=reusable_confidence,
        missing_answers=missing_answers,
        missing_confidence=missing_confidence,
    )


def materialize_v1_reuse(
    *,
    examples: Sequence[BenchmarkExample],
    v1_checkpoint: str | Path,
    v2_checkpoint: str | Path,
    run_id: str = "v2-v1-reuse",
    models_config: ModelsConfig | None = None,
    model_aliases: Sequence[str] | None = None,
) -> V1ReuseAudit:
    """Copy compatible V1 Stage-1/2 records into V2 under V2 request keys."""
    models = models_config or load_models_config()
    reusable_answers = reusable_confidence = 0
    missing_answers = missing_confidence = 0
    with (
        CheckpointStore(v1_checkpoint) as source,
        CheckpointStore(v2_checkpoint) as target,
    ):
        for example in examples:
            for alias, spec in _selected_models(models, model_aliases):
                if not _v1_config_compatible(alias, spec):
                    missing_answers += 1
                    missing_confidence += 1
                    continue
                source_answer = source.find_success(
                    stage="answer",
                    dataset=example.dataset_name,
                    example_id=example.example_id,
                    model_alias=alias,
                    requested_model_id=spec.api_model,
                    prompt_version=ANSWER_PROMPT_VERSION,
                )
                if source_answer is None:
                    missing_answers += 1
                    missing_confidence += 1
                    continue
                answer_key = _answer_key(example, alias, spec.api_model)
                target.register_request(
                    request_key=answer_key,
                    run_id=run_id,
                    stage="answer",
                    dataset=example.dataset_name,
                    example_id=example.example_id,
                    model_alias=alias,
                    requested_model_id=spec.api_model,
                    prompt_version=ANSWER_PROMPT_VERSION,
                )
                copied_answer = dict(source_answer)
                copied_answer.update(
                    {
                        "experiment_version": "v2",
                        "reused_from_v1": True,
                        "run_id": run_id,
                        "request_key": answer_key,
                    }
                )
                target.mark_success(
                    answer_key, AnswerRecord.model_validate(copied_answer)
                )
                reusable_answers += 1

                source_confidence = source.find_success(
                    stage="confidence",
                    dataset=example.dataset_name,
                    example_id=example.example_id,
                    model_alias=alias,
                    requested_model_id=spec.api_model,
                    prompt_version=CONFIDENCE_PROMPT_VERSION,
                )
                if (
                    source_confidence is None
                    or source_confidence.get("frozen_answer_label")
                    != source_answer.get("answer_label")
                ):
                    missing_confidence += 1
                    continue
                confidence_key = _confidence_key(
                    example, alias, spec.api_model, answer_key
                )
                target.register_request(
                    request_key=confidence_key,
                    run_id=run_id,
                    stage="confidence",
                    dataset=example.dataset_name,
                    example_id=example.example_id,
                    model_alias=alias,
                    requested_model_id=spec.api_model,
                    prompt_version=CONFIDENCE_PROMPT_VERSION,
                )
                copied_confidence = dict(source_confidence)
                copied_confidence.update(
                    {
                        "experiment_version": "v2",
                        "reused_from_v1": True,
                        "run_id": run_id,
                        "request_key": confidence_key,
                    }
                )
                target.mark_success(
                    confidence_key, ConfidenceRecord.model_validate(copied_confidence)
                )
                reusable_confidence += 1
    return V1ReuseAudit(
        reusable_answers=reusable_answers,
        reusable_confidence=reusable_confidence,
        missing_answers=missing_answers,
        missing_confidence=missing_confidence,
    )


@dataclasses.dataclass(frozen=True)
class _VerificationTask:
    example: BenchmarkExample
    model_alias: str
    model_config: Any
    owner: DecisionOwner
    visibility: ConfidenceVisibility
    prompt_family: str
    verification_cost: float
    error_cost: float
    answer_record: Mapping[str, Any]
    confidence_record: Mapping[str, Any]
    prompt: str
    request_key: str


class V2VerificationRunner:
    """Execute the 2×2×4 V2 Stage-3 factorial with strict dependencies."""

    def __init__(
        self,
        *,
        examples: Sequence[BenchmarkExample],
        checkpoint: CheckpointStore | str | Path,
        models_config: ModelsConfig | None = None,
        experiment_config: V2ExperimentConfig | None = None,
        run_id: str | None = None,
        concurrency: int | None = None,
    ) -> None:
        if not examples:
            raise ValueError("V2 requires a non-empty sample")
        self.examples = tuple(examples)
        self.models_config = models_config or load_models_config()
        self.config = experiment_config or load_v2_experiment_config()
        self.run_id = run_id or (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        self.concurrency = concurrency or self.config.model_inference.concurrency
        self._owns_checkpoint = not isinstance(checkpoint, CheckpointStore)
        self.checkpoint = (
            checkpoint
            if isinstance(checkpoint, CheckpointStore)
            else CheckpointStore(checkpoint)
        )
        self._adapters: dict[str, ModelAdapter] = {}

    def close(self) -> None:
        if self._owns_checkpoint:
            self.checkpoint.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _adapter(self, alias: str, spec: Any) -> ModelAdapter:
        if alias not in self._adapters:
            self._adapters[alias] = create_adapter(
                alias,
                spec,
                max_transient_retries=self.config.model_inference.max_transient_retries,
            )
        return self._adapters[alias]

    def _upstream(
        self,
        example: BenchmarkExample,
        alias: str,
        spec: Any,
        *,
        dry_run: bool,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        answer = self.checkpoint.find_success(
            stage="answer",
            dataset=example.dataset_name,
            example_id=example.example_id,
            model_alias=alias,
            requested_model_id=spec.api_model,
            prompt_version=ANSWER_PROMPT_VERSION,
        )
        confidence = self.checkpoint.find_success(
            stage="confidence",
            dataset=example.dataset_name,
            example_id=example.example_id,
            model_alias=alias,
            requested_model_id=spec.api_model,
            prompt_version=CONFIDENCE_PROMPT_VERSION,
        )
        if answer is not None and confidence is not None:
            if confidence.get("frozen_answer_label") != answer.get("answer_label"):
                raise RuntimeError("Frozen confidence dependency does not match answer")
            return answer, confidence
        if dry_run:
            label = next(iter(example.choices))
            return (
                {"answer_label": label, "request_key": "dry-run-answer"},
                {
                    "frozen_answer_label": label,
                    "probability_correct": 0.5,
                    "request_key": "dry-run-confidence",
                },
            )
        raise RuntimeError(
            f"Missing V2 answer/confidence dependency for {example.example_id} / {alias}"
        )

    def tasks(
        self,
        *,
        model_aliases: Sequence[str] | None = None,
        dry_run: bool,
        prompt_family: str | None = None,
        confidence_states: Sequence[ConfidenceVisibility | str] | None = None,
    ) -> list[_VerificationTask]:
        family = prompt_family or self.config.prompt_families.primary
        states = [
            ConfidenceVisibility(value)
            for value in (
                confidence_states
                if confidence_states is not None
                else self.config.factors.confidence_visibility
            )
        ]
        output: list[_VerificationTask] = []
        for example in self.examples:
            for alias, spec in _selected_models(self.models_config, model_aliases):
                answer, confidence = self._upstream(
                    example, alias, spec, dry_run=dry_run
                )
                for owner_value in self.config.factors.decision_owners:
                    owner = DecisionOwner(owner_value)
                    for visibility in states:
                        for error_cost in self.config.costs.error_costs:
                            prompt = build_verification_prompt(
                                question=example.question,
                                choices=example.choices,
                                answer_label=str(answer["answer_label"]),
                                probability_correct=float(
                                    confidence["probability_correct"]
                                ),
                                decision_owner=owner,
                                confidence_visibility=visibility,
                                verification_cost=self.config.costs.verification_cost,
                                error_cost=error_cost,
                                prompt_family=family,
                            )
                            key = deterministic_request_key(
                                stage="verification",
                                dataset=example.dataset_name,
                                example_id=example.example_id,
                                model_id=spec.api_model,
                                prompt_version=family,
                                stake={
                                    "verification_cost": (
                                        self.config.costs.verification_cost
                                    ),
                                    "error_cost": error_cost,
                                },
                                dependency={
                                    "answer": answer["request_key"],
                                    "confidence": confidence["request_key"],
                                },
                                experiment_version="v2",
                                prompt_family=family,
                                decision_owner=owner.value,
                                confidence_visibility=visibility.value,
                            )
                            output.append(
                                _VerificationTask(
                                    example=example,
                                    model_alias=alias,
                                    model_config=spec,
                                    owner=owner,
                                    visibility=visibility,
                                    prompt_family=family,
                                    verification_cost=(
                                        self.config.costs.verification_cost
                                    ),
                                    error_cost=error_cost,
                                    answer_record=answer,
                                    confidence_record=confidence,
                                    prompt=prompt,
                                    request_key=key,
                                )
                            )
        return output

    def _estimate(self, tasks: Sequence[_VerificationTask]) -> float:
        total = 0.0
        for task in tasks:
            pricing = task.model_config.pricing_per_million_tokens
            total += (
                math.ceil(len(task.prompt) / 4) * pricing.input
                + task.model_config.max_output_tokens * pricing.output
            ) / 1_000_000
        return total

    def _usage_manifest(
        self, selected_aliases: Sequence[str]
    ) -> tuple[dict[str, Any], float]:
        attempts = self.checkpoint.attempt_usage(run_id=self.run_id)
        aliases = sorted(
            set(selected_aliases)
            | {
                str(row["model_alias"])
                for row in attempts
                if row.get("model_alias") in self.models_config.models
            }
        )
        by_model: dict[str, dict[str, Any]] = {}
        observed_cost = 0.0
        for alias in aliases:
            spec = self.models_config.models[alias]
            model_attempts = [
                row for row in attempts if row.get("model_alias") == alias
            ]
            input_tokens = sum(
                int(row["input_tokens"] or 0) for row in model_attempts
            )
            output_tokens = sum(
                int(row["output_tokens"] or 0) for row in model_attempts
            )
            total_tokens = sum(
                int(row["total_tokens"] or 0) for row in model_attempts
            )
            model_cost = (
                input_tokens * spec.pricing_per_million_tokens.input
                + output_tokens * spec.pricing_per_million_tokens.output
            ) / 1_000_000
            observed_cost += model_cost
            latencies = [
                float(row["latency_seconds"])
                for row in model_attempts
                if row.get("latency_seconds") is not None
            ]
            by_model[alias] = {
                "provider": spec.provider,
                "requested_model_id": spec.api_model,
                "returned_model_ids": sorted(
                    {
                        str(row["returned_model_id"])
                        for row in model_attempts
                        if row.get("returned_model_id")
                    }
                ),
                "attempts": len(model_attempts),
                "successful_attempts": sum(
                    bool(row["success"]) for row in model_attempts
                ),
                "failed_attempts": sum(
                    not bool(row["success"]) for row in model_attempts
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "mean_latency_seconds": (
                    sum(latencies) / len(latencies) if latencies else None
                ),
                "observed_cost_usd": model_cost,
            }
        return {
            "attempt_count": len(attempts),
            "input_tokens": sum(int(row["input_tokens"] or 0) for row in attempts),
            "output_tokens": sum(
                int(row["output_tokens"] or 0) for row in attempts
            ),
            "total_tokens": sum(int(row["total_tokens"] or 0) for row in attempts),
            "by_model": by_model,
        }, observed_cost

    async def run(
        self,
        *,
        model_aliases: Sequence[str] | None = None,
        dry_run: bool = True,
        allow_paid: bool = False,
        retry_failed: bool = False,
        prompt_family: str | None = None,
        confidence_states: Sequence[ConfidenceVisibility | str] | None = None,
    ) -> V2RunSummary:
        if not dry_run and not allow_paid:
            raise PermissionError("Paid V2 execution requires explicit authorization")
        started_at = datetime.now(UTC).isoformat()
        tasks = self.tasks(
            model_aliases=model_aliases,
            dry_run=dry_run,
            prompt_family=prompt_family,
            confidence_states=confidence_states,
        )
        statuses = {
            task.request_key: self.checkpoint.request_status(task.request_key)
            for task in tasks
        }
        runnable = [
            task
            for task in tasks
            if statuses[task.request_key] != "success"
            and not (
                statuses[task.request_key] == "failed" and not retry_failed
            )
        ]
        estimate = self._estimate(runnable)
        sample_tasks: list[_VerificationTask] = []
        seen_providers: set[str] = set()
        for task in runnable:
            provider = str(task.model_config.provider)
            if provider not in seen_providers:
                sample_tasks.append(task)
                seen_providers.add(provider)
        payloads = tuple(
            self._adapter(task.model_alias, task.model_config)
            .prepare_request(stage="verification", prompt=task.prompt)
            .sanitized_payload()
            for task in sample_tasks
        )
        if dry_run:
            return V2RunSummary(
                run_id=self.run_id,
                planned_requests=len(tasks),
                calls_needed=len(runnable),
                completed=0,
                failed=0,
                skipped_successes=sum(
                    status == "success" for status in statuses.values()
                ),
                approximate_cost_usd=estimate,
                dry_run=True,
                sample_payloads=payloads,
            )

        for task in runnable:
            self.checkpoint.register_request(
                request_key=task.request_key,
                run_id=self.run_id,
                stage="verification",
                dataset=task.example.dataset_name,
                example_id=task.example.example_id,
                model_alias=task.model_alias,
                requested_model_id=task.model_config.api_model,
                prompt_version=task.prompt_family,
                stake={
                    "verification_cost": task.verification_cost,
                    "error_cost": task.error_cost,
                    "decision_owner": task.owner.value,
                    "confidence_visibility": task.visibility.value,
                    "prompt_family": task.prompt_family,
                },
            )
        semaphore = asyncio.Semaphore(self.concurrency)
        completed = failed = 0

        async def execute(task: _VerificationTask) -> bool:
            async with semaphore:
                return await self._execute(task, allow_paid=allow_paid)

        for ok in await asyncio.gather(*(execute(task) for task in runnable)):
            completed += int(ok)
            failed += int(not ok)
        selected_aliases = [alias for alias, _ in _selected_models(
            self.models_config, model_aliases
        )]
        usage, observed_cost = self._usage_manifest(selected_aliases)
        manifest = self.checkpoint.get_manifest(self.run_id) or {}
        manifest.update(
            {
                "experiment_id": self.config.experiment_id,
                "experiment_version": "v2",
                "run_id": self.run_id,
                "git_commit": _git_commit(),
                "start_time": manifest.get("start_time", started_at),
                "end_time": datetime.now(UTC).isoformat(),
                "dataset": self.config.dataset.name,
                "dataset_revision": self.config.dataset.revision,
                "sample_sha256": _sample_sha256(self.examples),
                "sample_ids": [example.example_id for example in self.examples],
                "model_configs": {
                    alias: self.models_config.models[alias].model_dump(mode="json")
                    for alias in usage["by_model"]
                },
                "prompt_family": prompt_family
                or self.config.prompt_families.primary,
                "prompt_version": prompt_family
                or self.config.prompt_families.primary,
                "decision_owners": list(self.config.factors.decision_owners),
                "confidence_visibility": [
                    state.value for state in (
                        [
                            ConfidenceVisibility(value)
                            for value in confidence_states
                        ]
                        if confidence_states is not None
                        else [
                            ConfidenceVisibility(value)
                            for value in self.config.factors.confidence_visibility
                        ]
                    )
                ],
                "verification_cost": self.config.costs.verification_cost,
                "error_costs": list(self.config.costs.error_costs),
                "cache_hits": sum(
                    status == "success" for status in statuses.values()
                ),
                "v1_reuse_counts": {
                    "answer": len(
                        {
                            task.answer_record["request_key"]
                            for task in tasks
                            if task.answer_record.get("reused_from_v1")
                        }
                    ),
                    "confidence": len(
                        {
                            task.confidence_record["request_key"]
                            for task in tasks
                            if task.confidence_record.get("reused_from_v1")
                        }
                    ),
                },
                "estimated_new_request_cost_usd": estimate,
                "observed_attempt_cost_usd": observed_cost,
                "usage": usage,
                "factor_counts": dict(
                    Counter(
                        f"{task.owner.value}/{task.visibility.value}/L={task.error_cost:g}"
                        for task in tasks
                    )
                ),
            }
        )
        manifest.setdefault("stages", {})["verification"] = {
            "start_time": started_at,
            "end_time": manifest["end_time"],
            "request_counts": {
                "planned": len(tasks),
                "called": len(runnable),
                "cache_hits": sum(
                    status == "success" for status in statuses.values()
                ),
                "success": completed,
                "failed": failed,
            },
            "prompt_family": manifest["prompt_family"],
            "factor_counts": manifest["factor_counts"],
        }
        self.checkpoint.upsert_manifest(
            self.run_id,
            manifest,
            status="completed" if not failed else "failed",
        )
        return V2RunSummary(
            run_id=self.run_id,
            planned_requests=len(tasks),
            calls_needed=len(runnable),
            completed=completed,
            failed=failed,
            skipped_successes=sum(
                status == "success" for status in statuses.values()
            ),
            approximate_cost_usd=estimate,
            dry_run=False,
        )

    async def _execute(
        self, task: _VerificationTask, *, allow_paid: bool
    ) -> bool:
        adapter = self._adapter(task.model_alias, task.model_config)
        prompt = task.prompt
        for parse_attempt in range(
            self.config.model_inference.max_parse_repairs + 1
        ):
            started = time.perf_counter()
            attempt_kind = (
                "generation"
                if parse_attempt == 0
                else f"parse_repair_{parse_attempt}"
            )
            try:
                response = await adapter.generate(
                    stage="verification",
                    prompt=prompt,
                    request_key=task.request_key,
                    allow_paid=allow_paid,
                )
            except Exception as error:  # provider transport containment boundary
                self.checkpoint.record_attempt(
                    request_key=task.request_key,
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
                self.checkpoint.mark_failed(task.request_key, error)
                return False

            raw = str(_response_value(response, "raw_response", default=""))
            refusal = bool(
                _response_value(response, "refused", "refusal", default=False)
            )
            try:
                if refusal:
                    raise ValueError("Provider returned a refusal")
                payload = parse_verification_response(raw)
            except (TypeError, ValueError) as error:
                self.checkpoint.record_attempt(
                    request_key=task.request_key,
                    attempt_kind=attempt_kind,
                    success=False,
                    provider=adapter.provider,
                    requested_model_id=adapter.api_model,
                    returned_model_id=_response_value(
                        response, "provider_model_id", "returned_model_id"
                    ),
                    latency_seconds=_response_value(response, "latency_seconds"),
                    raw_output=raw,
                    parse_error=str(error),
                    input_tokens=_response_value(response, "input_tokens"),
                    output_tokens=_response_value(response, "output_tokens"),
                    total_tokens=_response_value(response, "total_tokens"),
                    finish_reason=_response_value(response, "finish_reason"),
                    refusal=refusal,
                    sanitized_payload=adapter.prepare_request(
                        stage="verification", prompt=prompt
                    ).sanitized_payload(),
                )
                if parse_attempt < self.config.model_inference.max_parse_repairs:
                    prompt = build_verification_repair_prompt(task.prompt, raw)
                    continue
                self.checkpoint.mark_failed(task.request_key, error)
                return False

            self.checkpoint.record_attempt(
                request_key=task.request_key,
                attempt_kind=attempt_kind,
                success=True,
                provider=adapter.provider,
                requested_model_id=adapter.api_model,
                returned_model_id=_response_value(
                    response, "provider_model_id", "returned_model_id"
                ),
                latency_seconds=_response_value(response, "latency_seconds"),
                raw_output=raw,
                input_tokens=_response_value(response, "input_tokens"),
                output_tokens=_response_value(response, "output_tokens"),
                total_tokens=_response_value(response, "total_tokens"),
                finish_reason=_response_value(response, "finish_reason"),
                refusal=_response_value(response, "refused", "refusal", default=False),
                sanitized_payload=adapter.prepare_request(
                    stage="verification", prompt=prompt
                ).sanitized_payload(),
            )
            record = VerificationDecisionRecord(
                experiment_version="v2",
                run_id=self.run_id,
                request_key=task.request_key,
                example_id=task.example.example_id,
                model_id=task.model_alias,
                provider=adapter.provider,
                requested_model_id=adapter.api_model,
                returned_model_id=_response_value(
                    response, "provider_model_id", "returned_model_id"
                ),
                frozen_answer_label=str(task.answer_record["answer_label"]),
                probability_correct=float(
                    task.confidence_record["probability_correct"]
                ),
                decision_owner=task.owner,
                confidence_visibility=task.visibility,
                prompt_family=task.prompt_family,
                verification_cost=task.verification_cost,
                error_cost=task.error_cost,
                action=payload.action,
                raw_response=raw,
                input_tokens=_response_value(response, "input_tokens"),
                output_tokens=_response_value(response, "output_tokens"),
                total_tokens=_response_value(response, "total_tokens"),
                latency_seconds=_response_value(response, "latency_seconds"),
                finish_reason=_response_value(response, "finish_reason"),
                refused=bool(
                    _response_value(response, "refused", "refusal", default=False)
                ),
                prompt_version=task.prompt_family,
                provider_metadata={
                    "confidence_shown": (
                        task.visibility == ConfidenceVisibility.VISIBLE
                    )
                },
            )
            self.checkpoint.mark_success(task.request_key, record)
            return True
        return False


def plan_v2(
    *,
    examples: Sequence[BenchmarkExample],
    v1_checkpoint: str | Path,
    v2_checkpoint: str | Path,
    models_config: ModelsConfig | None = None,
    experiment_config: V2ExperimentConfig | None = None,
    model_aliases: Sequence[str] | None = None,
) -> V2Plan:
    """Calculate exact core counts from current V1/V2 checkpoints."""
    models = models_config or load_models_config()
    config = experiment_config or load_v2_experiment_config()
    selected = _selected_models(models, model_aliases)
    total_pairs = len(examples) * len(selected)
    verification_requests = total_pairs * 16
    stage12_cache_hits = 0
    planned_answer_reuse = planned_confidence_reuse = 0
    new_answers = new_confidence = 0
    stage12_input_tokens = 0
    stage12_output_tokens = 0
    answer_cost = 0.0
    confidence_cost = 0.0
    remaining_by_model: Counter[str] = Counter()
    with (
        CheckpointStore(v1_checkpoint) as source,
        CheckpointStore(v2_checkpoint) as target,
    ):
        for example in examples:
            for alias, spec in selected:
                answer_key = _answer_key(example, alias, spec.api_model)
                confidence_key = _confidence_key(
                    example, alias, spec.api_model, answer_key
                )
                answer_cached = target.request_status(answer_key) == "success"
                confidence_cached = (
                    target.request_status(confidence_key) == "success"
                )
                stage12_cache_hits += int(answer_cached) + int(confidence_cached)
                reusable_answer = reusable_confidence = False
                source_answer = None
                if _v1_config_compatible(alias, spec):
                    source_answer = source.find_success(
                        stage="answer",
                        dataset=example.dataset_name,
                        example_id=example.example_id,
                        model_alias=alias,
                        requested_model_id=spec.api_model,
                        prompt_version=ANSWER_PROMPT_VERSION,
                    )
                    reusable_answer = source_answer is not None
                    if source_answer is not None:
                        source_confidence = source.find_success(
                            stage="confidence",
                            dataset=example.dataset_name,
                            example_id=example.example_id,
                            model_alias=alias,
                            requested_model_id=spec.api_model,
                            prompt_version=CONFIDENCE_PROMPT_VERSION,
                        )
                        reusable_confidence = (
                            source_confidence is not None
                            and source_confidence.get("frozen_answer_label")
                            == source_answer.get("answer_label")
                        )
                if not answer_cached:
                    if reusable_answer:
                        planned_answer_reuse += 1
                    else:
                        new_answers += 1
                        remaining_by_model[alias] += 1
                        prompt = build_answer_prompt(
                            question=example.question, choices=example.choices
                        )
                        input_tokens = math.ceil(len(prompt) / 4)
                        output_tokens = spec.max_output_tokens
                        stage12_input_tokens += input_tokens
                        stage12_output_tokens += output_tokens
                        answer_cost += (
                            input_tokens
                            * spec.pricing_per_million_tokens.input
                            + output_tokens
                            * spec.pricing_per_million_tokens.output
                        ) / 1_000_000
                if not confidence_cached:
                    if reusable_confidence:
                        planned_confidence_reuse += 1
                    else:
                        new_confidence += 1
                        remaining_by_model[alias] += 1
                        answer_label = (
                            source_answer.get("answer_label")
                            if source_answer is not None
                            else next(iter(example.choices))
                        )
                        prompt = build_confidence_prompt(
                            question=example.question,
                            choices=example.choices,
                            answer_label=str(answer_label),
                        )
                        input_tokens = math.ceil(len(prompt) / 4)
                        output_tokens = spec.max_output_tokens
                        stage12_input_tokens += input_tokens
                        stage12_output_tokens += output_tokens
                        confidence_cost += (
                            input_tokens
                            * spec.pricing_per_million_tokens.input
                            + output_tokens
                            * spec.pricing_per_million_tokens.output
                        ) / 1_000_000
        with V2VerificationRunner(
            examples=examples,
            checkpoint=target,
            models_config=models,
            experiment_config=config,
        ) as runner:
            verification_tasks = runner.tasks(
                model_aliases=[alias for alias, _ in selected], dry_run=True
            )
            verification_cache_hits = sum(
                target.request_status(task.request_key) == "success"
                for task in verification_tasks
            )
            verification_cost = runner._estimate(  # noqa: SLF001
                [
                    task
                    for task in verification_tasks
                    if target.request_status(task.request_key) != "success"
                ]
            )
            verification_pending = [
                task
                for task in verification_tasks
                if target.request_status(task.request_key) != "success"
            ]
            remaining_by_model.update(
                task.model_alias for task in verification_pending
            )
            verification_input_tokens = sum(
                math.ceil(len(task.prompt) / 4) for task in verification_pending
            )
            verification_output_tokens = sum(
                task.model_config.max_output_tokens
                for task in verification_pending
            )
            successful_latency_rows = [
                row
                for row in target.attempt_usage()
                if bool(row.get("success"))
                and row.get("latency_seconds") is not None
            ]
            latencies_by_model: dict[str, list[float]] = defaultdict(list)
            for row in successful_latency_rows:
                latencies_by_model[str(row["model_alias"])].append(
                    float(row["latency_seconds"])
                )
            all_latencies = [
                latency
                for values in latencies_by_model.values()
                for latency in values
            ]
            global_mean_latency = (
                sum(all_latencies) / len(all_latencies)
                if all_latencies
                else None
            )
            runtime_work_seconds = 0.0
            runtime_estimable = global_mean_latency is not None
            for alias, count in remaining_by_model.items():
                model_latencies = latencies_by_model.get(alias)
                mean_latency = (
                    sum(model_latencies) / len(model_latencies)
                    if model_latencies
                    else global_mean_latency
                )
                if mean_latency is None:
                    runtime_estimable = False
                    break
                runtime_work_seconds += count * mean_latency
            estimated_runtime_seconds = (
                runtime_work_seconds / config.model_inference.concurrency
                if runtime_estimable
                else None
            )

    answer_requests = total_pairs
    confidence_requests = total_pairs
    v2_cache_hits = stage12_cache_hits + verification_cache_hits
    new_requests = (
        new_answers
        + new_confidence
        + verification_requests
        - verification_cache_hits
    )
    unknown = tuple(
        alias
        for alias, spec in selected
        if spec.pricing_per_million_tokens.input == 0
        and spec.pricing_per_million_tokens.output == 0
    )
    factor_counts = {
        f"{owner}/{visibility}/L={error_cost:g}": total_pairs
        for owner in config.factors.decision_owners
        for visibility in config.factors.confidence_visibility
        for error_cost in config.costs.error_costs
    }
    return V2Plan(
        examples=len(examples),
        models=len(selected),
        answer_requests=answer_requests,
        confidence_requests=confidence_requests,
        verification_requests=verification_requests,
        v1_answer_reuse=planned_answer_reuse,
        v1_confidence_reuse=planned_confidence_reuse,
        v2_cache_hits=v2_cache_hits,
        new_requests=new_requests,
        approximate_cost_usd=answer_cost + confidence_cost + verification_cost,
        estimated_input_tokens=stage12_input_tokens + verification_input_tokens,
        maximum_output_tokens=stage12_output_tokens + verification_output_tokens,
        cost_by_stage={
            "answer": answer_cost,
            "confidence": confidence_cost,
            "verification": verification_cost,
        },
        remaining_requests_by_model=dict(sorted(remaining_by_model.items())),
        estimated_runtime_seconds=estimated_runtime_seconds,
        runtime_basis_successful_attempts=len(successful_latency_rows),
        runtime_assumed_concurrency=config.model_inference.concurrency,
        unknown_pricing_models=unknown,
        factor_counts=factor_counts,
    )
