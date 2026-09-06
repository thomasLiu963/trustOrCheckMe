"""Command-line workflow for the frozen MMLU-Pro pilot."""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .analysis import analyze
from .checkpointing import CheckpointStore
from .config import (
    DEFAULT_V2_EXPERIMENT_CONFIG,
    PROJECT_ROOT,
    ExperimentConfig,
    load_experiment_config,
    load_models_config,
    load_v2_experiment_config,
)
from .datasets import load_local_sample, load_or_create_pilot_sample
from .runner import ExperimentRunner, RunSummary
from .schemas import ConfidenceVisibility, DecisionOwner
from .v2_analysis import analyze_v2, analyze_v2_robustness
from .v2_datasets import load_v2_sample, prepare_v2_sample
from .v2_prompts import (
    PRIMARY_PROMPT_FAMILY,
    ROBUSTNESS_PROMPT_FAMILY,
    build_verification_prompt,
)
from .v2_runner import (
    V2VerificationRunner,
    audit_v1_reuse,
    materialize_v1_reuse,
    plan_v2,
)

DEFAULT_MODELS = ("openai_gpt56_sol", "anthropic_sonnet5")
V2_MODELS = (
    "openai_gpt56_sol",
    "anthropic_sonnet5",
    "google_gemini38_flash",
    "xai_grok420_nonreasoning",
)


def _sample(config: ExperimentConfig, *, limit: int | None = None) -> list[Any]:
    path = config.resolve_path(config.dataset.pilot_sample_path)
    manifest = config.resolve_path(config.dataset.pilot_manifest_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pilot sample not found at {path}. Run prepare-sample first."
        )
    examples = load_local_sample(
        path,
        expected_size=config.dataset.pilot_size,
        expected_revision=config.dataset.revision,
        manifest_path=manifest,
    )
    if limit is not None:
        if limit < 1 or limit > len(examples):
            raise ValueError(f"--limit must be between 1 and {len(examples)}")
        examples = examples[:limit]
    return examples


def _checkpoint_path(config: ExperimentConfig, override: str | None) -> Path:
    return (
        Path(override)
        if override
        else config.resolve_path(config.results.checkpoint_path)
    )


def _print_summary(summary: RunSummary) -> None:
    print(
        json.dumps(
            {
                "run_id": summary.run_id,
                "stage": summary.stage,
                "planned_requests": summary.planned_requests,
                "calls_needed": summary.calls_needed,
                "completed": summary.completed,
                "failed": summary.failed,
                "skipped_successes": summary.skipped_successes,
                "skipped_failures": summary.skipped_failures,
                "approximate_full_stage_cost_usd": round(
                    summary.approximate_cost_usd, 4
                ),
                "dry_run": summary.dry_run,
            },
            indent=2,
        )
    )


def _assert_frozen_payloads(
    summaries: Sequence[RunSummary], expected_examples: int
) -> None:
    expected_counts = {
        "answer": expected_examples * 2,
        "confidence": expected_examples * 2,
        "trust": expected_examples * 2 * 4,
    }
    for summary in summaries:
        if summary.planned_requests != expected_counts[summary.stage]:
            raise RuntimeError(
                f"{summary.stage} planned {summary.planned_requests}; expected "
                f"{expected_counts[summary.stage]}"
            )
        for payload in summary.sanitized_payloads:
            model = payload.get("model")
            if model == "gpt-5.6-sol":
                if payload.get("reasoning") != {"effort": "none"}:
                    raise RuntimeError("OpenAI reasoning effort is not frozen to none")
                if any(key in payload for key in ("tools", "temperature", "top_p")):
                    raise RuntimeError("Unexpected OpenAI optional inference parameter")
            elif model == "claude-sonnet-5":
                if payload.get("thinking") != {"type": "disabled"}:
                    raise RuntimeError("Anthropic thinking is not explicitly disabled")
                if any(
                    key in payload for key in ("tools", "temperature", "top_p", "top_k")
                ):
                    raise RuntimeError("Unexpected Anthropic sampling/tool parameter")
            else:
                raise RuntimeError(f"Unexpected frozen pilot model: {model!r}")


async def _run_stage(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.experiment_config)
    models = load_models_config(args.models_config)
    examples = _sample(config, limit=args.limit)
    dry_run = bool(args.dry_run)
    if not dry_run and not args.yes:
        raise PermissionError("Paid execution requires the explicit --yes flag.")
    if not dry_run:
        load_dotenv(PROJECT_ROOT / ".env")
    with ExperimentRunner(
        examples=examples,
        checkpoint=_checkpoint_path(config, args.checkpoint),
        models_config=models,
        experiment_config=config,
        run_id=args.run_id,
        concurrency=args.concurrency,
    ) as runner:
        method = {
            "run-answers": runner.run_answers,
            "run-confidence": runner.run_confidence,
            "run-trust": runner.run_trust,
        }[args.command]
        summary = await method(
            model_aliases=args.models,
            dry_run=dry_run,
            allow_paid=bool(args.yes),
            retry_failed=args.retry_failed,
            show_prompts=args.show_prompts,
        )
    _print_summary(summary)
    if summary.failed:
        raise RuntimeError(
            f"{summary.failed} request(s) failed; inspect status and use "
            "--retry-failed after resolving permanent errors."
        )


async def _plan(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.experiment_config)
    models = load_models_config(args.models_config)
    examples = _sample(config)
    checkpoint = _checkpoint_path(config, args.checkpoint)
    summaries: list[RunSummary] = []
    with ExperimentRunner(
        examples=examples,
        checkpoint=checkpoint,
        models_config=models,
        experiment_config=config,
        run_id=args.run_id,
    ) as runner:
        for method in (
            runner.run_answers,
            runner.run_confidence,
            runner.run_trust,
        ):
            summaries.append(
                await method(
                    model_aliases=list(DEFAULT_MODELS),
                    dry_run=True,
                    show_prompts=args.show_prompts,
                )
            )
    _assert_frozen_payloads(summaries, len(examples))
    print("\nFrozen pilot plan")
    print(f"  examples: {len(examples)}")
    print("  models: gpt-5.6-sol, claude-sonnet-5")
    for summary in summaries:
        print(
            f"  {summary.stage}: {summary.planned_requests} requests, "
            f"approximately ${summary.approximate_cost_usd:.4f}"
        )
    print(f"  total: {sum(item.planned_requests for item in summaries)} requests")
    print(
        "  approximate baseline cost estimate: "
        f"${sum(item.approximate_cost_usd for item in summaries):.4f}"
    )
    print(
        "  estimate uses a character-based input-token approximation and the "
        "configured output cap; retries, repairs, and pricing changes can increase it"
    )
    print(f"  checkpoint: {checkpoint}")
    print("  payload validation: PASS")
    print("No provider API calls were made.")


def _prepare(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.experiment_config)
    examples = load_or_create_pilot_sample(config, force_resample=args.force_resample)
    counts = Counter(example.category for example in examples)
    print(f"Prepared {len(examples)} examples at revision {config.dataset.revision}")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")
    print(
        f"Sample: {config.resolve_path(config.dataset.pilot_sample_path)}\n"
        f"Manifest: {config.resolve_path(config.dataset.pilot_manifest_path)}"
    )


def _inspect(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.experiment_config)
    examples = _sample(config)
    print(f"sample_size={len(examples)}")
    print(f"revision={config.dataset.revision}")
    print(f"unique_ids={len({example.example_id for example in examples})}")
    for category, count in sorted(Counter(e.category for e in examples).items()):
        print(f"{category}={count}")
    if args.show_ids:
        for example in examples:
            print(example.example_id)


def _status(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.experiment_config)
    path = _checkpoint_path(config, args.checkpoint)
    with CheckpointStore(path) as store:
        counts = store.counts(run_id=args.run_id if args.only_run else None)
        manifest = store.get_manifest(args.run_id)
    print(json.dumps({"checkpoint": str(path), "counts": counts}, indent=2))
    if manifest:
        print(json.dumps(manifest, indent=2, default=str))


def _analyze(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.experiment_config)
    models = load_models_config(args.models_config)
    checkpoint = _checkpoint_path(config, args.checkpoint)
    pricing = {
        alias: {
            "input": spec.pricing_per_million_tokens.input,
            "output": spec.pricing_per_million_tokens.output,
        }
        for alias, spec in models.models.items()
    }
    result = analyze(
        checkpoint,
        output_directory=config.resolve_path(config.results.analysis_directory),
        paper_output_directory=config.resolve_path(
            config.results.paper_output_directory
        ),
        seed=config.seed,
        calibration_fraction=config.calibration.calibration_fraction,
        n_resamples=(
            args.bootstrap_resamples
            if args.bootstrap_resamples is not None
            else config.bootstrap.n_resamples
        ),
        confidence_level=config.bootstrap.confidence_level,
        make_plots=not args.no_plots,
        pricing_per_million_tokens=pricing,
    )
    print(
        f"Analyzed {len(result.direct_rows)} direct decisions. "
        f"Paper outputs: {config.resolve_path(config.results.paper_output_directory)}"
    )
    for warning in result.warnings:
        print(f"WARNING: {warning}")


def _v2_config(args: argparse.Namespace):
    return load_v2_experiment_config(args.experiment_config)


def _v2_checkpoint(config, override: str | None) -> Path:
    return (
        Path(override)
        if override
        else config.resolve_path(config.results.checkpoint_path)
    )


def _audit_v1(args: argparse.Namespace) -> None:
    config = _v2_config(args)
    models = load_models_config(args.models_config)
    examples = load_local_sample(
        config.resolve_path(config.dataset.v1_sample_path),
        expected_size=config.dataset.v2a_size,
        expected_revision=config.dataset.revision,
    )
    checkpoint = config.resolve_path(config.results.v1_checkpoint_path)
    audit = audit_v1_reuse(
        examples=examples,
        v1_checkpoint=checkpoint,
        models_config=models,
        model_aliases=args.models,
    )
    print(
        json.dumps(
            {
                "v1_checkpoint": str(checkpoint),
                "v1_checkpoint_exists": checkpoint.exists(),
                "sample_size": len(examples),
                **audit.__dict__,
                "v1_stage3_reuse_allowed": False,
            },
            indent=2,
        )
    )


def _import_v1_reuse(args: argparse.Namespace) -> None:
    config = _v2_config(args)
    models = load_models_config(args.models_config)
    examples = load_v2_sample(config, args.sample)
    result = materialize_v1_reuse(
        examples=examples,
        v1_checkpoint=config.resolve_path(config.results.v1_checkpoint_path),
        v2_checkpoint=_v2_checkpoint(config, args.checkpoint),
        run_id=args.run_id,
        models_config=models,
        model_aliases=args.models,
    )
    print(
        json.dumps(
            {
                **result.__dict__,
                "source": str(
                    config.resolve_path(config.results.v1_checkpoint_path)
                ),
                "target": str(_v2_checkpoint(config, args.checkpoint)),
                "provider_calls_made": False,
                "v1_stage3_imported": False,
            },
            indent=2,
        )
    )


def _prepare_v2(args: argparse.Namespace) -> None:
    config = _v2_config(args)
    examples = prepare_v2_sample(args.size, config)
    sample = "v2a" if args.size == config.dataset.v2a_size else "v2b"
    print(f"Prepared {sample}: {len(examples)} deterministic MMLU-Pro questions")
    if sample == "v2b":
        print(
            "Also prepared deterministic "
            f"{config.dataset.robustness_size}-question robustness subset"
        )
    print("Selection did not use observed V1 or V2 results.")


def _plan_v2(args: argparse.Namespace) -> None:
    config = _v2_config(args)
    models = load_models_config(args.models_config)
    sample = "v2a" if args.size == config.dataset.v2a_size else "v2b"
    examples = load_v2_sample(config, sample)
    plan = plan_v2(
        examples=examples,
        v1_checkpoint=config.resolve_path(config.results.v1_checkpoint_path),
        v2_checkpoint=_v2_checkpoint(config, args.checkpoint),
        models_config=models,
        experiment_config=config,
        model_aliases=args.models,
    )
    print(
        json.dumps(
            {
                **plan.__dict__,
                "unknown_pricing_models": list(plan.unknown_pricing_models),
                "factor_counts": plan.factor_counts,
                "pricing_warning": (
                    "Cost estimate is incomplete until zero-valued provider "
                    "pricing entries are updated."
                    if plan.unknown_pricing_models
                    else None
                ),
                "paid_calls_made": False,
            },
            indent=2,
        )
    )


def _inspect_v2_prompts(args: argparse.Namespace) -> None:
    config = _v2_config(args)
    examples = load_v2_sample(config, args.sample, limit=args.limit)
    output = config.resolve_path(config.results.analysis_directory) / "prompt_diffs"
    output.mkdir(parents=True, exist_ok=True)
    for index, example in enumerate(examples, start=1):
        prompts: dict[str, str] = {}
        for owner in DecisionOwner:
            for visibility in ConfidenceVisibility:
                name = f"{owner.value}_{visibility.value}"
                prompts[name] = build_verification_prompt(
                    question=example.question,
                    choices=example.choices,
                    answer_label=next(iter(example.choices)),
                    probability_correct=0.73,
                    decision_owner=owner,
                    confidence_visibility=visibility,
                    verification_cost=config.costs.verification_cost,
                    error_cost=config.costs.error_costs[1],
                    prompt_family=PRIMARY_PROMPT_FAMILY,
                )
        sections = []
        for left, right in (
            ("human_hidden", "ai_system_hidden"),
            ("human_hidden", "human_visible"),
            ("ai_system_hidden", "ai_system_visible"),
        ):
            diff = difflib.unified_diff(
                prompts[left].splitlines(),
                prompts[right].splitlines(),
                fromfile=left,
                tofile=right,
                lineterm="",
            )
            sections.append("\n".join(diff))
        path = output / f"example_{index:02d}.diff"
        path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote matched prompt diffs for {len(examples)} example(s) to {output}")
    print("No provider API calls were made.")


async def _run_v2_stage(args: argparse.Namespace) -> None:
    config = _v2_config(args)
    models = load_models_config(args.models_config)
    examples = load_v2_sample(config, args.sample, limit=args.limit)
    checkpoint = _v2_checkpoint(config, args.checkpoint)
    dry_run = bool(args.dry_run)
    if not dry_run and not args.yes:
        raise PermissionError("Paid execution requires the explicit --yes flag.")
    if not dry_run:
        load_dotenv(PROJECT_ROOT / ".env")
    if not dry_run and args.command in {
        "run-v2-answers",
        "run-v2-confidence",
    }:
        materialize_v1_reuse(
            examples=examples,
            v1_checkpoint=config.resolve_path(config.results.v1_checkpoint_path),
            v2_checkpoint=checkpoint,
            models_config=models,
            model_aliases=args.models,
        )

    if args.command in {"run-v2-answers", "run-v2-confidence"}:
        with ExperimentRunner(
            examples=examples,
            checkpoint=checkpoint,
            models_config=models,
            experiment_config=config,
            run_id=args.run_id,
            concurrency=args.concurrency,
        ) as runner:
            method = (
                runner.run_answers
                if args.command == "run-v2-answers"
                else runner.run_confidence
            )
            summary = await method(
                model_aliases=args.models,
                dry_run=dry_run,
                allow_paid=bool(args.yes),
                retry_failed=args.retry_failed,
                show_prompts=args.show_prompts,
            )
        _print_summary(summary)
        if summary.failed:
            raise RuntimeError(
                f"{summary.failed} V2 {summary.stage} request(s) failed; "
                "inspect the checkpoint and retry with --retry-failed."
            )
        return

    with V2VerificationRunner(
        examples=examples,
        checkpoint=checkpoint,
        models_config=models,
        experiment_config=config,
        run_id=args.run_id,
        concurrency=args.concurrency,
    ) as runner:
        summary = await runner.run(
            model_aliases=args.models,
            dry_run=dry_run,
            allow_paid=bool(args.yes),
            retry_failed=args.retry_failed,
        )
    summary_data = dict(summary.__dict__)
    payloads = summary_data.pop("sample_payloads")
    if payloads:
        payload_path = (
            config.resolve_path(config.results.analysis_directory)
            / "plans"
            / f"{args.command}_sample_payloads.json"
        )
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(
            json.dumps(payloads, indent=2, default=str) + "\n", encoding="utf-8"
        )
        summary_data["sample_payloads_path"] = str(payload_path)
    print(json.dumps(summary_data, indent=2, default=str))
    if summary.failed:
        raise RuntimeError(
            f"{summary.failed} V2 verification request(s) failed; "
            "inspect the checkpoint and retry with --retry-failed."
        )


async def _run_v2_robustness(args: argparse.Namespace) -> None:
    config = _v2_config(args)
    models = load_models_config(args.models_config)
    examples = load_v2_sample(config, "robustness", limit=args.limit)
    dry_run = bool(args.dry_run)
    if not dry_run and not args.yes:
        raise PermissionError("Paid execution requires the explicit --yes flag.")
    if not dry_run:
        load_dotenv(PROJECT_ROOT / ".env")
    with V2VerificationRunner(
        examples=examples,
        checkpoint=_v2_checkpoint(config, args.checkpoint),
        models_config=models,
        experiment_config=config,
        run_id=args.run_id,
        concurrency=args.concurrency,
    ) as runner:
        summary = await runner.run(
            model_aliases=args.models,
            dry_run=dry_run,
            allow_paid=bool(args.yes),
            retry_failed=args.retry_failed,
            prompt_family=ROBUSTNESS_PROMPT_FAMILY,
            confidence_states=[
                ConfidenceVisibility(args.confidence_visibility)
            ],
        )
    summary_data = dict(summary.__dict__)
    payloads = summary_data.pop("sample_payloads")
    if payloads:
        payload_path = (
            config.resolve_path(config.results.analysis_directory)
            / "plans"
            / "run-v2-robustness_sample_payloads.json"
        )
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(
            json.dumps(payloads, indent=2, default=str) + "\n", encoding="utf-8"
        )
        summary_data["sample_payloads_path"] = str(payload_path)
    print(json.dumps(summary_data, indent=2, default=str))
    if summary.failed:
        raise RuntimeError(
            f"{summary.failed} V2 robustness request(s) failed; "
            "inspect the checkpoint and retry with --retry-failed."
        )


def _analyze_v2(args: argparse.Namespace) -> None:
    config = _v2_config(args)
    result = analyze_v2(
        _v2_checkpoint(config, args.checkpoint),
        output_directory=config.resolve_path(config.results.paper_output_directory),
        seed=config.seed,
        calibration_fraction=config.calibration.calibration_fraction,
        n_resamples=(
            args.bootstrap_resamples
            if args.bootstrap_resamples is not None
            else config.bootstrap.n_resamples
        ),
        confidence_level=config.bootstrap.confidence_level,
    )
    print(
        f"Analyzed {len(result.scored_rows)} V2 decisions; "
        f"{len(result.completeness_issues)} factor-completeness issue(s). "
        f"Outputs: {result.output_directory}"
    )


def _analyze_v2_robustness(args: argparse.Namespace) -> None:
    config = _v2_config(args)
    result = analyze_v2_robustness(
        _v2_checkpoint(config, args.checkpoint),
        output_directory=config.resolve_path(config.results.paper_output_directory),
        seed=config.seed,
        n_resamples=(
            args.bootstrap_resamples
            if args.bootstrap_resamples is not None
            else config.bootstrap.n_resamples
        ),
        confidence_level=config.bootstrap.confidence_level,
    )
    print(
        f"Analyzed paraphrase robustness on {result.n_questions} questions; "
        f"{result.paraphrase_decisions} paraphrase decisions; "
        f"{len(result.completeness_issues)} completeness issue(s). "
        f"Wrote {len(result.written_files)} robustness-only file(s) to "
        f"{result.output_directory}"
    )


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-config", default=str(PROJECT_ROOT / "config/experiment.yaml")
    )
    parser.add_argument(
        "--models-config", default=str(PROJECT_ROOT / "config/models.yaml")
    )


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    _add_config_options(parser)
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-id", default="mmlu-pro-pilot-v1")
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--show-prompts", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--yes",
        action="store_true",
        help="Explicitly authorize provider API calls and associated charges.",
    )


def _add_v2_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-config", default=str(DEFAULT_V2_EXPERIMENT_CONFIG)
    )
    parser.add_argument(
        "--models-config", default=str(PROJECT_ROOT / "config/models.yaml")
    )


def _add_v2_runtime_options(parser: argparse.ArgumentParser) -> None:
    _add_v2_config_options(parser)
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-id", default="mmlu-pro-v2")
    parser.add_argument("--sample", choices=("v2a", "v2b"), default="v2a")
    parser.add_argument("--models", nargs="+", choices=V2_MODELS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--show-prompts", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--yes",
        action="store_true",
        help="Explicitly authorize provider API calls and associated charges.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trust-or-check-me",
        description="Reproducible V1 pilot and matched-owner V2 experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-sample")
    _add_config_options(prepare)
    prepare.add_argument("--force-resample", action="store_true")

    inspect = subparsers.add_parser("inspect-sample")
    _add_config_options(inspect)
    inspect.add_argument("--show-ids", action="store_true")

    plan = subparsers.add_parser("plan")
    _add_config_options(plan)
    plan.add_argument("--checkpoint")
    plan.add_argument("--run-id", default="mmlu-pro-pilot-v1")
    plan.add_argument("--show-prompts", action="store_true")

    for name in ("run-answers", "run-confidence", "run-trust"):
        _add_runtime_options(subparsers.add_parser(name))

    status = subparsers.add_parser("status")
    _add_config_options(status)
    status.add_argument("--checkpoint")
    status.add_argument("--run-id", default="mmlu-pro-pilot-v1")
    status.add_argument("--only-run", action="store_true")

    analysis = subparsers.add_parser("analyze")
    _add_config_options(analysis)
    analysis.add_argument("--checkpoint")
    analysis.add_argument("--bootstrap-resamples", type=int)
    analysis.add_argument("--no-plots", action="store_true")

    audit_v1 = subparsers.add_parser("audit-v1")
    _add_v2_config_options(audit_v1)
    audit_v1.add_argument("--models", nargs="+", choices=V2_MODELS)

    import_v1 = subparsers.add_parser("import-v1-reuse")
    _add_v2_config_options(import_v1)
    import_v1.add_argument("--checkpoint")
    import_v1.add_argument("--run-id", default="v2-v1-reuse")
    import_v1.add_argument("--sample", choices=("v2a", "v2b"), default="v2a")
    import_v1.add_argument("--models", nargs="+", choices=V2_MODELS)

    prepare_v2 = subparsers.add_parser("prepare-v2-sample")
    _add_v2_config_options(prepare_v2)
    prepare_v2.add_argument("--size", type=int, choices=(200, 500), required=True)

    plan_v2_parser = subparsers.add_parser("plan-v2")
    _add_v2_config_options(plan_v2_parser)
    plan_v2_parser.add_argument("--size", type=int, choices=(200, 500), required=True)
    plan_v2_parser.add_argument("--checkpoint")
    plan_v2_parser.add_argument("--models", nargs="+", choices=V2_MODELS)

    prompt_inspection = subparsers.add_parser("inspect-v2-prompts")
    _add_v2_config_options(prompt_inspection)
    prompt_inspection.add_argument(
        "--sample", choices=("v2a", "v2b"), default="v2a"
    )
    prompt_inspection.add_argument("--limit", type=int, default=5)

    for name in ("run-v2-answers", "run-v2-confidence", "run-v2-decisions"):
        _add_v2_runtime_options(subparsers.add_parser(name))

    robustness = subparsers.add_parser("run-v2-robustness")
    _add_v2_config_options(robustness)
    robustness.add_argument("--checkpoint")
    robustness.add_argument("--run-id", default="mmlu-pro-v2-robustness")
    robustness.add_argument("--models", nargs="+", choices=V2_MODELS)
    robustness.add_argument("--limit", type=int)
    robustness.add_argument("--concurrency", type=int)
    robustness.add_argument("--retry-failed", action="store_true")
    robustness.add_argument(
        "--confidence-visibility",
        choices=("hidden", "visible"),
        default="hidden",
        help=(
            "hidden is the preregistered robustness test. "
            "visible is the post-primary confidence-visibility extension."
        ),
    )
    robustness_mode = robustness.add_mutually_exclusive_group(required=True)
    robustness_mode.add_argument("--dry-run", action="store_true")
    robustness_mode.add_argument("--yes", action="store_true")

    analysis_v2 = subparsers.add_parser("analyze-v2")
    _add_v2_config_options(analysis_v2)
    analysis_v2.add_argument("--checkpoint")
    analysis_v2.add_argument("--bootstrap-resamples", type=int)

    analysis_robustness = subparsers.add_parser("analyze-v2-robustness")
    _add_v2_config_options(analysis_robustness)
    analysis_robustness.add_argument("--checkpoint")
    analysis_robustness.add_argument("--bootstrap-resamples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-sample":
            _prepare(args)
        elif args.command == "inspect-sample":
            _inspect(args)
        elif args.command == "plan":
            asyncio.run(_plan(args))
        elif args.command in {"run-answers", "run-confidence", "run-trust"}:
            asyncio.run(_run_stage(args))
        elif args.command == "status":
            _status(args)
        elif args.command == "analyze":
            _analyze(args)
        elif args.command == "audit-v1":
            _audit_v1(args)
        elif args.command == "import-v1-reuse":
            _import_v1_reuse(args)
        elif args.command == "prepare-v2-sample":
            _prepare_v2(args)
        elif args.command == "plan-v2":
            _plan_v2(args)
        elif args.command == "inspect-v2-prompts":
            _inspect_v2_prompts(args)
        elif args.command in {
            "run-v2-answers",
            "run-v2-confidence",
            "run-v2-decisions",
        }:
            asyncio.run(_run_v2_stage(args))
        elif args.command == "run-v2-robustness":
            asyncio.run(_run_v2_robustness(args))
        elif args.command == "analyze-v2":
            _analyze_v2(args)
        elif args.command == "analyze-v2-robustness":
            _analyze_v2_robustness(args)
        else:  # pragma: no cover
            parser.error(f"Unknown command: {args.command}")
    except (
        FileNotFoundError,
        KeyError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
