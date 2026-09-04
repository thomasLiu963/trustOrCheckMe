"""Resumable async runners for answer, confidence, and trust stages."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import math
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from . import prompts as prompt_module
from .checkpointing import (
    CheckpointStore,
    deterministic_request_key,
    sanitize_payload,
)
from .config import load_experiment_config, load_models_config
from .model_adapters import ModelAdapter, create_adapter
from .schemas import (
    AnswerRecord,
    BenchmarkExample,
    ConfidenceRecord,
    TrustDecisionRecord,
)


@dataclasses.dataclass(frozen=True)
class RunSummary:
    run_id: str
    stage: str
    planned_requests: int
    calls_needed: int
    completed: int
    failed: int
    skipped_successes: int
    skipped_failures: int
    approximate_cost_usd: float
    dry_run: bool
    sanitized_payloads: tuple[dict[str, Any], ...] = ()


@dataclasses.dataclass(frozen=True)
class _Task:
    example: BenchmarkExample
    model_alias: str
    model_config: Any
    stage: str
    prompt: str
    prompt_version: str
    request_key: str
    stake: Any
    upstream: Mapping[str, Any] | None


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _nested(source: Any, *path: str, default: Any = None) -> Any:
    current = source
    for name in path:
        missing = object()
        current = _value(current, name, missing)
        if current is missing:
            return default
    return current


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(
        f"Expected a mapping-like configuration, got {type(value).__name__}"
    )


def _schema_instance(schema: type[Any], values: Mapping[str, Any]) -> Any:
    fields = getattr(schema, "model_fields", None)
    if fields is None:
        fields = getattr(schema, "__fields__", None)
    if fields is None and dataclasses.is_dataclass(schema):
        fields = getattr(schema, "__dataclass_fields__", None)
    if fields is not None:
        accepted = set(fields)
        return schema(
            **{key: value for key, value in values.items() if key in accepted}
        )
    signature = inspect.signature(schema)
    accepted = set(signature.parameters)
    return schema(**{key: value for key, value in values.items() if key in accepted})


def _call_compatible(
    function: Callable[..., Any], candidates: Mapping[str, Any]
) -> Any:
    signature = inspect.signature(function)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**dict(candidates))
    kwargs = {
        name: candidates[name] for name in signature.parameters if name in candidates
    }
    return function(**kwargs)


def _find_callable(names: Sequence[str]) -> Callable[..., Any]:
    for name in names:
        function = getattr(prompt_module, name, None)
        if callable(function):
            return function
    raise AttributeError(f"src.prompts must expose one of: {', '.join(names)}")


_BUILDERS = {
    "answer": ("build_answer_prompt", "build_stage1_prompt", "answer_prompt"),
    "confidence": (
        "build_confidence_prompt",
        "build_stage2_prompt",
        "confidence_prompt",
    ),
    "trust": ("build_trust_prompt", "build_stage3_prompt", "trust_prompt"),
}

_PARSERS = {
    "answer": (
        "parse_answer_response",
        "parse_answer_output",
        "parse_stage1_response",
        "parse_answer",
    ),
    "confidence": (
        "parse_confidence_response",
        "parse_confidence_output",
        "parse_stage2_response",
        "parse_confidence",
    ),
    "trust": (
        "parse_trust_response",
        "parse_trust_output",
        "parse_stage3_response",
        "parse_recommendation",
    ),
}

_VERSION_CONSTANTS = {
    "answer": ("ANSWER_PROMPT_VERSION", "STAGE1_PROMPT_VERSION"),
    "confidence": ("CONFIDENCE_PROMPT_VERSION", "STAGE2_PROMPT_VERSION"),
    "trust": ("TRUST_PROMPT_VERSION", "STAGE3_PROMPT_VERSION"),
}


def _prompt_version(stage: str, experiment_config: Any) -> str:
    configured = _nested(experiment_config, "prompt_versions", stage, default=None)
    if configured is not None:
        return str(configured)
    for name in _VERSION_CONSTANTS[stage]:
        value = getattr(prompt_module, name, None)
        if value is not None:
            return str(value)
    return f"{stage}-v1"


def _example_values(example: BenchmarkExample) -> dict[str, Any]:
    return {
        "example": example,
        "benchmark_example": example,
        "example_id": str(_value(example, "example_id")),
        "dataset": str(
            _value(example, "dataset_name", _value(example, "dataset", "unknown"))
        ),
        "dataset_name": str(
            _value(example, "dataset_name", _value(example, "dataset", "unknown"))
        ),
        "question": _value(example, "question"),
        "choices": _value(example, "choices"),
    }


def _extract_builder_result(
    result: Any,
    *,
    default_version: str,
) -> tuple[str, str]:
    if isinstance(result, str):
        return result, default_version
    if isinstance(result, tuple) and len(result) == 2:
        return str(result[0]), str(result[1])
    text = _value(result, "text", _value(result, "prompt", None))
    version = _value(result, "version", default_version)
    if text is None:
        raise TypeError(
            "Prompt builder must return text, (text, version), or prompt object"
        )
    return str(text), str(version)


def _build_prompt(
    stage: str,
    example: BenchmarkExample,
    experiment_config: Any,
    *,
    frozen_answer: str | None = None,
    verification_cost: float | None = None,
    error_cost: float | None = None,
) -> tuple[str, str]:
    candidates = _example_values(example)
    candidates.update(
        {
            "answer": frozen_answer,
            "answer_label": frozen_answer,
            "frozen_answer": frozen_answer,
            "frozen_answer_label": frozen_answer,
            "verification_cost": verification_cost,
            "error_cost": error_cost,
            "loss": error_cost,
        }
    )
    default_version = _prompt_version(stage, experiment_config)
    result = _call_compatible(_find_callable(_BUILDERS[stage]), candidates)
    return _extract_builder_result(result, default_version=default_version)


def _repair_prompt(
    *,
    stage: str,
    original_prompt: str,
    invalid_response: str,
    parse_error: str,
) -> str:
    for name in (
        "build_repair_prompt",
        "build_parse_repair_prompt",
        f"build_{stage}_repair_prompt",
    ):
        function = getattr(prompt_module, name, None)
        if callable(function):
            return str(
                _call_compatible(
                    function,
                    {
                        "stage": stage,
                        "original_prompt": original_prompt,
                        "prompt": original_prompt,
                        "invalid_response": invalid_response,
                        "raw_response": invalid_response,
                        "parse_error": parse_error,
                        "error": parse_error,
                    },
                )
            )
    return (
        f"{original_prompt}\n\nYour previous response was not valid for the required "
        "JSON format. Return only corrected JSON. Do not change the substantive "
        f"answer.\nPrevious response:\n{invalid_response}"
    )


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _parse(stage: str, raw_text: str, example: BenchmarkExample) -> Any:
    parser = _find_callable(_PARSERS[stage])
    result = _call_compatible(
        parser,
        {
            "text": raw_text,
            "raw_text": raw_text,
            "raw_response": raw_text,
            "response": raw_text,
            "choices": _value(example, "choices"),
            "example": example,
        },
    )
    if stage == "answer":
        value = _value(result, "answer", _value(result, "answer_label", result))
        value = str(_enum_value(value)).strip().upper()
        choices = _value(example, "choices")
        labels = (
            {str(key).upper() for key in choices}
            if isinstance(choices, Mapping)
            else {chr(ord("A") + index) for index in range(len(choices))}
        )
        if value not in labels:
            raise ValueError(f"Answer {value!r} is not a provided choice label")
        return value
    if stage == "confidence":
        value = _value(
            result,
            "probability_correct",
            _value(result, "confidence", result),
        )
        probability = float(_enum_value(value))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability_correct must be finite and within [0, 1]")
        return probability
    value = _value(
        result,
        "recommendation",
        _value(result, "action", result),
    )
    recommendation = str(_enum_value(value)).strip().upper()
    if recommendation not in {"RELY", "VERIFY"}:
        raise ValueError("recommendation must be exactly RELY or VERIFY")
    return recommendation


def _response_value(response: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = _value(response, name, None)
        if value is not None:
            return value
    return default


def _models(config: Any) -> Mapping[str, Any]:
    models = _value(config, "models", config)
    return _as_mapping(models)


def _frozen_answer(record: Mapping[str, Any]) -> str:
    answer = record.get("answer_label", record.get("answer"))
    if answer is None:
        raise ValueError("Completed answer record has no frozen answer label")
    return str(_enum_value(answer)).upper()


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


class ExperimentRunner:
    def __init__(
        self,
        *,
        examples: Sequence[BenchmarkExample],
        checkpoint: CheckpointStore | str | Path,
        models_config: Any | None = None,
        experiment_config: Any | None = None,
        run_id: str | None = None,
        concurrency: int | None = None,
        max_transient_retries: int | None = None,
        max_parse_repairs: int | None = None,
    ) -> None:
        if not examples:
            raise ValueError("The answer stage requires a non-empty benchmark sample")
        self.examples = tuple(examples)
        self.models_config = (
            models_config if models_config is not None else load_models_config()
        )
        self.experiment_config = (
            experiment_config
            if experiment_config is not None
            else load_experiment_config()
        )
        self.run_id = run_id or (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        configured_concurrency = _nested(
            self.experiment_config,
            "model_inference",
            "concurrency",
            default=_value(self.experiment_config, "concurrency", 4),
        )
        self.concurrency = max(
            1, int(concurrency if concurrency is not None else configured_concurrency)
        )
        configured_transport_retries = _nested(
            self.experiment_config,
            "model_inference",
            "max_transient_retries",
            default=3,
        )
        self.max_transient_retries = max(
            0,
            int(
                max_transient_retries
                if max_transient_retries is not None
                else configured_transport_retries
            ),
        )
        configured_repairs = _nested(
            self.experiment_config,
            "model_inference",
            "max_parse_repairs",
            default=2,
        )
        self.max_parse_repairs = max(
            0,
            int(
                max_parse_repairs
                if max_parse_repairs is not None
                else configured_repairs
            ),
        )
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

    def _adapter(self, alias: str, config: Any) -> ModelAdapter:
        if alias not in self._adapters:
            self._adapters[alias] = create_adapter(
                alias,
                config,
                max_transient_retries=self.max_transient_retries,
            )
        return self._adapters[alias]

    def _selected_models(self, aliases: Sequence[str] | None) -> list[tuple[str, Any]]:
        configured = _models(self.models_config)
        selected = list(aliases) if aliases is not None else list(configured)
        unknown = [alias for alias in selected if alias not in configured]
        if unknown:
            raise KeyError(f"Unknown model aliases: {', '.join(unknown)}")
        if not selected:
            raise ValueError("At least one model must be selected")
        return [(alias, configured[alias]) for alias in selected]

    def _costs(self) -> tuple[float, tuple[float, ...]]:
        verification_cost = float(
            _nested(
                self.experiment_config,
                "costs",
                "verification_cost",
                default=_value(self.experiment_config, "verification_cost", 1.0),
            )
        )
        error_costs = _nested(
            self.experiment_config,
            "costs",
            "error_costs",
            default=_value(self.experiment_config, "error_costs", (2, 5, 10, 20)),
        )
        result = tuple(float(value) for value in error_costs)
        if result != (2.0, 5.0, 10.0, 20.0):
            raise ValueError("Frozen pilot error costs must be [2, 5, 10, 20]")
        if verification_cost != 1.0:
            raise ValueError("Frozen pilot verification cost must be 1")
        return verification_cost, result

    def _upstream(
        self,
        *,
        stage: str,
        example: BenchmarkExample,
        model_alias: str,
        model_config: Any,
        allow_dry_run_placeholder: bool = False,
    ) -> Mapping[str, Any]:
        values = _example_values(example)
        _, upstream_prompt_version = _build_prompt(
            stage,
            example,
            self.experiment_config,
        )
        requested_model_id = str(
            _value(model_config, "api_model", _value(model_config, "model_id", ""))
        )
        record = self.checkpoint.find_success(
            stage=stage,
            dataset=values["dataset"],
            example_id=values["example_id"],
            model_alias=model_alias,
            requested_model_id=requested_model_id,
            prompt_version=upstream_prompt_version,
        )
        if record is None:
            if allow_dry_run_placeholder:
                choices = _value(example, "choices")
                first_label = next(iter(choices))
                return {
                    "answer_label": str(first_label),
                    "_dry_run_placeholder": True,
                }
            raise RuntimeError(
                f"Strict dependency missing: {stage} success for dataset="
                f"{values['dataset']!r}, example={values['example_id']!r}, "
                f"model={model_alias!r}"
            )
        return record

    def _tasks(
        self,
        stage: str,
        aliases: Sequence[str] | None,
        *,
        dry_run: bool = False,
    ) -> list[_Task]:
        tasks: list[_Task] = []
        selected_models = self._selected_models(aliases)
        verification_cost, error_costs = self._costs()
        for example in self.examples:
            values = _example_values(example)
            for model_alias, model_config in selected_models:
                upstream = None
                frozen_answer = None
                if stage in {"confidence", "trust"}:
                    upstream = self._upstream(
                        stage="answer",
                        example=example,
                        model_alias=model_alias,
                        model_config=model_config,
                        allow_dry_run_placeholder=dry_run,
                    )
                    frozen_answer = _frozen_answer(upstream)
                stakes = error_costs if stage == "trust" else (None,)
                for error_cost in stakes:
                    prompt, version = _build_prompt(
                        stage,
                        example,
                        self.experiment_config,
                        frozen_answer=frozen_answer,
                        verification_cost=(
                            verification_cost if stage == "trust" else None
                        ),
                        error_cost=error_cost,
                    )
                    api_model = str(
                        _value(
                            model_config,
                            "api_model",
                            _value(model_config, "model_id", ""),
                        )
                    )
                    stake = (
                        {
                            "verification_cost": verification_cost,
                            "error_cost": error_cost,
                        }
                        if stage == "trust"
                        else None
                    )
                    key = deterministic_request_key(
                        stage=stage,
                        dataset=values["dataset"],
                        example_id=values["example_id"],
                        model_id=api_model,
                        prompt_version=version,
                        stake=stake,
                    )
                    tasks.append(
                        _Task(
                            example=example,
                            model_alias=model_alias,
                            model_config=model_config,
                            stage=stage,
                            prompt=prompt,
                            prompt_version=version,
                            request_key=key,
                            stake=stake,
                            upstream=upstream,
                        )
                    )
        return tasks

    def _estimate(self, tasks: Sequence[_Task]) -> float:
        total = 0.0
        for task in tasks:
            input_tokens = max(1, math.ceil(len(task.prompt) / 4))
            output_tokens = min(
                int(_value(task.model_config, "max_output_tokens", 64)), 64
            )
            pricing = _value(task.model_config, "pricing_per_million_tokens", {})
            input_price = float(_value(pricing, "input", 0.0))
            output_price = float(_value(pricing, "output", 0.0))
            total += (
                input_tokens * input_price + output_tokens * output_price
            ) / 1_000_000
        return total

    def _manifest(self, stage: str, tasks: Sequence[_Task]) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        manifest = self.checkpoint.get_manifest(self.run_id) or {
            "run_id": self.run_id,
            "start_time": now,
            "end_time": None,
            "git_commit": _git_commit(),
            "experiment_config_snapshot": sanitize_payload(
                dict(_as_mapping(self.experiment_config))
            ),
            "model_config_snapshot": sanitize_payload(
                dict(_as_mapping(self.models_config))
            ),
            "dataset_sample_ids": [
                str(_value(example, "example_id")) for example in self.examples
            ],
            "stages": {},
        }
        manifest["end_time"] = None
        manifest["stage"] = stage
        manifest["prompt_versions"] = sorted(
            set(manifest.get("prompt_versions", ()))
            | {task.prompt_version for task in tasks}
        )
        manifest.setdefault("stages", {})[stage] = {
            "start_time": now,
            "end_time": None,
            "prompt_versions": sorted({task.prompt_version for task in tasks}),
            "request_counts": {"planned": len(tasks)},
            "failure_counts": {},
        }
        return manifest

    async def run_answers(
        self,
        *,
        model_aliases: Sequence[str] | None = None,
        dry_run: bool = True,
        allow_paid: bool = False,
        retry_failed: bool = False,
        show_prompts: bool = False,
    ) -> RunSummary:
        return await self._run(
            "answer",
            model_aliases=model_aliases,
            dry_run=dry_run,
            allow_paid=allow_paid,
            retry_failed=retry_failed,
            show_prompts=show_prompts,
        )

    async def run_confidence(
        self,
        *,
        model_aliases: Sequence[str] | None = None,
        dry_run: bool = True,
        allow_paid: bool = False,
        retry_failed: bool = False,
        show_prompts: bool = False,
    ) -> RunSummary:
        return await self._run(
            "confidence",
            model_aliases=model_aliases,
            dry_run=dry_run,
            allow_paid=allow_paid,
            retry_failed=retry_failed,
            show_prompts=show_prompts,
        )

    async def run_trust(
        self,
        *,
        model_aliases: Sequence[str] | None = None,
        dry_run: bool = True,
        allow_paid: bool = False,
        retry_failed: bool = False,
        show_prompts: bool = False,
    ) -> RunSummary:
        return await self._run(
            "trust",
            model_aliases=model_aliases,
            dry_run=dry_run,
            allow_paid=allow_paid,
            retry_failed=retry_failed,
            show_prompts=show_prompts,
        )

    async def _run(
        self,
        stage: str,
        *,
        model_aliases: Sequence[str] | None,
        dry_run: bool,
        allow_paid: bool,
        retry_failed: bool,
        show_prompts: bool,
    ) -> RunSummary:
        if not dry_run and not allow_paid:
            raise PermissionError("Non-dry runs require explicit allow_paid=True.")
        tasks = self._tasks(stage, model_aliases, dry_run=dry_run)
        estimate = self._estimate(tasks)
        statuses = {
            task.request_key: self.checkpoint.request_status(task.request_key)
            for task in tasks
        }
        skipped_successes = sum(status == "success" for status in statuses.values())
        skipped_failures = sum(
            status == "failed" and not retry_failed for status in statuses.values()
        )
        runnable = [
            task
            for task in tasks
            if statuses[task.request_key] not in {"success"}
            and not (statuses[task.request_key] == "failed" and not retry_failed)
        ]
        print(
            f"{stage}: {len(tasks)} planned; {len(runnable)} calls needed; "
            f"{skipped_successes} completed successes; approximate full-stage "
            f"cost ${estimate:.4f}"
        )
        if (
            dry_run
            and stage in {"confidence", "trust"}
            and any(
                bool(task.upstream and task.upstream.get("_dry_run_placeholder"))
                for task in tasks
            )
        ):
            print(
                "dry-run note: no completed frozen answers were available for some "
                "requests, so the first valid choice label was used only to inspect "
                "request construction."
            )

        payloads = tuple(
            self._adapter(task.model_alias, task.model_config)
            .prepare_request(prompt=task.prompt)
            .sanitized_payload()
            for task in runnable
        )
        if show_prompts:
            for index, task in enumerate(runnable[:3], start=1):
                print(f"\nSample prompt {index}:\n{task.prompt}")
        if dry_run:
            return RunSummary(
                run_id=self.run_id,
                stage=stage,
                planned_requests=len(tasks),
                calls_needed=len(runnable),
                completed=0,
                failed=0,
                skipped_successes=skipped_successes,
                skipped_failures=skipped_failures,
                approximate_cost_usd=estimate,
                dry_run=True,
                sanitized_payloads=payloads,
            )

        manifest = self._manifest(stage, tasks)
        self.checkpoint.upsert_manifest(self.run_id, manifest, status="running")
        for task in runnable:
            values = _example_values(task.example)
            self.checkpoint.register_request(
                request_key=task.request_key,
                run_id=self.run_id,
                stage=stage,
                dataset=values["dataset"],
                example_id=values["example_id"],
                model_alias=task.model_alias,
                requested_model_id=str(
                    _value(
                        task.model_config,
                        "api_model",
                        _value(task.model_config, "model_id", ""),
                    )
                ),
                prompt_version=task.prompt_version,
                stake=task.stake,
            )

        completed = 0
        failed = 0
        processed = 0
        started = time.perf_counter()
        progress_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(task: _Task) -> None:
            nonlocal completed, failed, processed
            async with semaphore:
                ok = await self._execute_task(task, allow_paid=allow_paid)
            async with progress_lock:
                processed += 1
                completed += int(ok)
                failed += int(not ok)
                if (
                    processed == len(runnable)
                    or processed % max(1, min(25, len(runnable) // 20 or 1)) == 0
                ):
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    rate = processed / elapsed
                    remaining = len(runnable) - processed
                    print(
                        f"{stage}: {processed}/{len(runnable)} processed; "
                        f"{completed} success; {failed} failed; {rate:.2f}/s; "
                        f"{remaining} remaining"
                    )

        try:
            await asyncio.gather(*(run_one(task) for task in runnable))
        except BaseException:
            manifest["end_time"] = datetime.now(UTC).isoformat()
            manifest["request_counts"] = {
                "planned": len(tasks),
                "processed": processed,
                "success": completed,
            }
            manifest["failure_counts"] = {"failed": failed}
            manifest["stages"][stage].update(
                {
                    "end_time": manifest["end_time"],
                    "request_counts": manifest["request_counts"],
                    "failure_counts": manifest["failure_counts"],
                }
            )
            self.checkpoint.upsert_manifest(self.run_id, manifest, status="failed")
            raise

        manifest["end_time"] = datetime.now(UTC).isoformat()
        manifest["request_counts"] = {
            "planned": len(tasks),
            "processed": processed,
            "success": completed,
            "skipped_successes": skipped_successes,
            "skipped_failures": skipped_failures,
        }
        manifest["failure_counts"] = {"failed": failed}
        manifest["stages"][stage].update(
            {
                "end_time": manifest["end_time"],
                "request_counts": manifest["request_counts"],
                "failure_counts": manifest["failure_counts"],
            }
        )
        self.checkpoint.upsert_manifest(self.run_id, manifest, status="completed")
        return RunSummary(
            run_id=self.run_id,
            stage=stage,
            planned_requests=len(tasks),
            calls_needed=len(runnable),
            completed=completed,
            failed=failed,
            skipped_successes=skipped_successes,
            skipped_failures=skipped_failures,
            approximate_cost_usd=estimate,
            dry_run=False,
        )

    async def _execute_task(self, task: _Task, *, allow_paid: bool) -> bool:
        adapter = self._adapter(task.model_alias, task.model_config)
        prompt = task.prompt
        original_prompt = task.prompt

        def record_transport_failure(event: Mapping[str, Any]) -> None:
            self.checkpoint.record_attempt(
                request_key=task.request_key,
                attempt_kind=str(event["attempt_kind"]),
                success=False,
                started_at=_value(event, "started_at"),
                finished_at=_value(event, "finished_at"),
                provider=_value(event, "provider"),
                requested_model_id=_value(event, "requested_model_id"),
                latency_seconds=_value(event, "latency_seconds"),
                error=_value(event, "error"),
                sanitized_payload=_value(event, "sanitized_payload"),
            )

        for parse_attempt in range(self.max_parse_repairs + 1):
            attempt_kind = (
                "generation" if parse_attempt == 0 else f"parse_repair_{parse_attempt}"
            )
            try:
                response = await adapter.generate(
                    stage=task.stage,
                    prompt=prompt,
                    request_key=task.request_key,
                    allow_paid=allow_paid,
                    attempt_callback=record_transport_failure,
                    attempt_kind=attempt_kind,
                )
            # Provider SDKs expose different exception hierarchies; this is the
            # per-request containment boundary, and the error is persisted.
            except Exception as error:  # noqa: BLE001
                self.checkpoint.mark_failed(task.request_key, error)
                return False

            raw_text = str(
                _response_value(
                    response,
                    "raw_response",
                    "raw_text",
                    "text",
                    default="",
                )
            )
            refusal = bool(
                _response_value(response, "refusal", "is_refusal", default=False)
            )
            parse_error: str | None = None
            parsed: Any = None
            try:
                if refusal:
                    raise ValueError("Provider returned a refusal")
                parsed = _parse(task.stage, raw_text, task.example)
            except (TypeError, ValueError, KeyError) as error:
                parse_error = str(error)

            self.checkpoint.record_attempt(
                request_key=task.request_key,
                attempt_kind=attempt_kind,
                success=parse_error is None,
                provider=str(
                    _response_value(response, "provider", default=adapter.provider)
                ),
                requested_model_id=str(
                    _response_value(
                        response,
                        "requested_model_id",
                        "requested_api_model_id",
                        default=adapter.api_model,
                    )
                ),
                returned_model_id=_response_value(
                    response, "returned_model_id", "provider_model_id"
                ),
                latency_seconds=_response_value(response, "latency_seconds", "latency"),
                raw_output=raw_text,
                parse_error=parse_error,
                input_tokens=_response_value(response, "input_tokens"),
                output_tokens=_response_value(response, "output_tokens"),
                total_tokens=_response_value(response, "total_tokens"),
                finish_reason=_response_value(response, "finish_reason", "stop_reason"),
                refusal=refusal,
                sanitized_payload=adapter.prepare_request(
                    prompt=prompt
                ).sanitized_payload(),
            )
            if parse_error is None:
                record = self._record(task, parsed, response)
                self.checkpoint.mark_success(task.request_key, record)
                return True
            if parse_attempt < self.max_parse_repairs:
                prompt = _repair_prompt(
                    stage=task.stage,
                    original_prompt=original_prompt,
                    invalid_response=raw_text,
                    parse_error=parse_error,
                )

        self.checkpoint.mark_failed(
            task.request_key,
            f"Parsing failed after {self.max_parse_repairs + 1} outputs",
        )
        return False

    def _record(self, task: _Task, parsed: Any, response: Any) -> Any:
        values = _example_values(task.example)
        timestamp = datetime.now(UTC)
        common = {
            "run_id": self.run_id,
            "request_key": task.request_key,
            "example_id": values["example_id"],
            "dataset": values["dataset"],
            "dataset_name": values["dataset"],
            "model_id": task.model_alias,
            "model_alias": task.model_alias,
            "provider": _response_value(response, "provider"),
            "requested_model_id": _response_value(
                response, "requested_model_id", "requested_api_model_id"
            ),
            "returned_model_id": _response_value(
                response, "returned_model_id", "provider_model_id"
            ),
            "raw_response": _response_value(
                response, "raw_response", "raw_text", "text", default=""
            ),
            "prompt_version": task.prompt_version,
            "timestamp": timestamp,
            "input_tokens": _response_value(response, "input_tokens"),
            "output_tokens": _response_value(response, "output_tokens"),
            "total_tokens": _response_value(response, "total_tokens"),
            "latency_seconds": _response_value(response, "latency_seconds", "latency"),
            "finish_reason": _response_value(response, "finish_reason", "stop_reason"),
            "refusal": _response_value(
                response, "refusal", "is_refusal", default=False
            ),
            "provider_metadata": {
                "provider": _response_value(response, "provider"),
                "requested_model_id": _response_value(
                    response, "requested_model_id", "requested_api_model_id"
                ),
                "provider_model_id": _response_value(
                    response, "returned_model_id", "provider_model_id"
                ),
                "input_tokens": _response_value(response, "input_tokens"),
                "output_tokens": _response_value(response, "output_tokens"),
                "total_tokens": _response_value(response, "total_tokens"),
                "latency_seconds": _response_value(
                    response, "latency_seconds", "latency"
                ),
                "finish_reason": _response_value(
                    response, "finish_reason", "stop_reason"
                ),
                "refusal": _response_value(
                    response, "refusal", "is_refusal", default=False
                ),
            },
        }
        if task.stage == "answer":
            correct = str(_enum_value(_value(task.example, "correct_label"))).upper()
            common.update(
                {
                    "answer": parsed,
                    "answer_label": parsed,
                    "is_correct": parsed == correct,
                    "category": _value(task.example, "category"),
                    "question": _value(task.example, "question"),
                    "choices": _value(task.example, "choices"),
                    "correct_label": correct,
                }
            )
            return _schema_instance(AnswerRecord, common)
        frozen_answer = _frozen_answer(task.upstream or {})
        if task.stage == "confidence":
            common.update(
                {
                    "frozen_answer": frozen_answer,
                    "frozen_answer_label": frozen_answer,
                    "probability_correct": parsed,
                    "confidence": parsed,
                }
            )
            return _schema_instance(ConfidenceRecord, common)
        common.update(
            {
                "frozen_answer": frozen_answer,
                "frozen_answer_label": frozen_answer,
                "verification_cost": float(task.stake["verification_cost"]),
                "error_cost": float(task.stake["error_cost"]),
                "recommendation": parsed,
                "action": parsed,
            }
        )
        return _schema_instance(TrustDecisionRecord, common)


async def run_answers(**kwargs: Any) -> RunSummary:
    """Convenience entry point; accepts ExperimentRunner constructor/run arguments."""
    run_names = {
        "model_aliases",
        "dry_run",
        "allow_paid",
        "retry_failed",
        "show_prompts",
    }
    run_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in run_names}
    with ExperimentRunner(**kwargs) as runner:
        return await runner.run_answers(**run_kwargs)


async def run_confidence(**kwargs: Any) -> RunSummary:
    run_names = {
        "model_aliases",
        "dry_run",
        "allow_paid",
        "retry_failed",
        "show_prompts",
    }
    run_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in run_names}
    with ExperimentRunner(**kwargs) as runner:
        return await runner.run_confidence(**run_kwargs)


async def run_trust(**kwargs: Any) -> RunSummary:
    run_names = {
        "model_aliases",
        "dry_run",
        "allow_paid",
        "retry_failed",
        "show_prompts",
    }
    run_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in run_names}
    with ExperimentRunner(**kwargs) as runner:
        return await runner.run_trust(**run_kwargs)
