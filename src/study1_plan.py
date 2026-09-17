"""Offline Study 1 call-plan construction, validation, and prompt audit."""

from __future__ import annotations

import csv
import dataclasses
import difflib
import hashlib
import json
import re
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .checkpointing import deterministic_request_key
from .config import PROJECT_ROOT, load_models_config
from .study1_prompts import (
    HISTORICAL_PROMPT_SOURCE,
    STUDY1_PROMPT_FAMILY,
    STUDY1_PROMPT_VERSION,
    build_study1_verification_prompt,
    format_displayed_probability,
    historical_visible_matches_v2_builder,
    prompt_sha256,
)
from .study1_sample import (
    REPEAT_SIZE,
    assert_not_v2_write_target,
    load_frozen_ids,
    load_historical_bundle,
    load_v2b_examples,
    write_study1_ids,
)
from .study1_schemas import (
    GRIDS_BY_L,
    HISTORICAL_V2_SQLITE,
    MANIPULATED_CONDITIONS,
    PRIMARY_DISPLAY_CONDITIONS,
    STUDY1_ERROR_COSTS,
    STUDY1_MODEL_ALIASES,
    STUDY1_PRIMARY_CALLS,
    STUDY1_REPEAT_EXTRA_CALLS,
    STUDY1_TOTAL_CALLS,
    STUDY1_VERIFICATION_COST,
    DisplayCondition,
    DisplaySourceCondition,
    Study1DecisionRecord,
    Study1ExperimentConfig,
    Study1Phase,
    display_source_for,
    grid_value,
    load_study1_config,
)

_CONFIDENCE_LINE = re.compile(
    r"The AI previously estimated a [0-9.]+ probability that this frozen answer is correct\."
)


@dataclasses.dataclass(frozen=True)
class Study1PlannedCall:
    phase: Study1Phase
    question_id: str
    model_alias: str
    model_endpoint: str
    provider: str
    L: float
    C: float
    display_condition: DisplayCondition
    display_source_condition: DisplaySourceCondition
    reported_confidence: float
    displayed_confidence: float | None
    frozen_answer: str
    stage1_correct: bool
    repeat_index: int
    request_key: str
    prompt: str
    prompt_hash: str
    historical_answer_request_key: str
    historical_confidence_request_key: str
    selected_sample_hash: str

    def planned_record(self, *, code_commit: str | None) -> Study1DecisionRecord:
        spec_settings = {
            "requested_model_id": self.model_endpoint,
            "provider": self.provider,
        }
        return Study1DecisionRecord(
            phase=self.phase,
            run_id="study1-plan-offline",
            request_key=self.request_key,
            question_id=self.question_id,
            model_id=self.model_alias,
            model_alias=self.model_alias,
            model_endpoint=self.model_endpoint,
            provider=self.provider,  # type: ignore[arg-type]
            L=self.L,
            C=self.C,
            reported_confidence=self.reported_confidence,
            displayed_confidence=self.displayed_confidence,
            display_condition=self.display_condition,
            display_source_condition=self.display_source_condition,
            frozen_answer=self.frozen_answer,
            stage1_correct=self.stage1_correct,
            prompt_version=STUDY1_PROMPT_VERSION,
            prompt_family=STUDY1_PROMPT_FAMILY,
            prompt_hash=self.prompt_hash,
            selected_sample_hash=self.selected_sample_hash,
            code_commit=code_commit,
            repeat_index=self.repeat_index,
            attempt_number=1,
            parse_status="not_run",
            historical_answer_request_key=self.historical_answer_request_key,
            historical_confidence_request_key=self.historical_confidence_request_key,
            model_settings=spec_settings,
        )


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


def displayed_for(
    condition: DisplayCondition,
    *,
    reported_confidence: float,
    error_cost: float,
) -> float | None:
    if condition == DisplayCondition.HIDDEN:
        return None
    if condition == DisplayCondition.TRUE_CONFIDENCE_VISIBLE:
        return reported_confidence
    return grid_value(error_cost, condition)


def study1_request_key(
    *,
    example,
    model_endpoint: str,
    error_cost: float,
    condition: DisplayCondition,
    displayed_confidence: float | None,
    repeat_index: int,
    historical_answer_key: str,
    historical_confidence_key: str,
) -> str:
    return deterministic_request_key(
        stage="verification",
        dataset=example.dataset_name,
        example_id=example.example_id,
        model_id=model_endpoint,
        prompt_version=STUDY1_PROMPT_VERSION,
        stake={
            "verification_cost": STUDY1_VERIFICATION_COST,
            "error_cost": error_cost,
            "display_condition": condition.value,
            "displayed_confidence": displayed_confidence,
            "repeat_index": repeat_index,
            "authority": "ai_system",
        },
        dependency={
            "historical_answer": historical_answer_key,
            "historical_confidence": historical_confidence_key,
        },
        experiment_version="study1",
        prompt_family=STUDY1_PROMPT_FAMILY,
        decision_owner="ai_system",
        confidence_visibility=condition.value,
    )


def build_one_call(
    *,
    example,
    model_alias: str,
    model_endpoint: str,
    provider: str,
    historical: dict[str, Any],
    error_cost: float,
    condition: DisplayCondition,
    repeat_index: int,
    phase: Study1Phase,
    selected_sample_hash: str,
) -> Study1PlannedCall:
    reported = float(historical["reported_confidence"])
    displayed = displayed_for(
        condition, reported_confidence=reported, error_cost=error_cost
    )
    prompt = build_study1_verification_prompt(
        question=example.question,
        choices=example.choices,
        frozen_answer=str(historical["frozen_answer"]),
        verification_cost=STUDY1_VERIFICATION_COST,
        error_cost=error_cost,
        displayed_confidence=displayed,
    )
    key = study1_request_key(
        example=example,
        model_endpoint=model_endpoint,
        error_cost=error_cost,
        condition=condition,
        displayed_confidence=displayed,
        repeat_index=repeat_index,
        historical_answer_key=historical["historical_answer_request_key"],
        historical_confidence_key=historical["historical_confidence_request_key"],
    )
    return Study1PlannedCall(
        phase=phase,
        question_id=example.example_id,
        model_alias=model_alias,
        model_endpoint=model_endpoint,
        provider=provider,
        L=float(error_cost),
        C=STUDY1_VERIFICATION_COST,
        display_condition=condition,
        display_source_condition=display_source_for(condition),
        reported_confidence=reported,
        displayed_confidence=displayed,
        frozen_answer=str(historical["frozen_answer"]),
        stage1_correct=bool(historical["stage1_correct"]),
        repeat_index=repeat_index,
        request_key=key,
        prompt=prompt,
        prompt_hash=prompt_sha256(prompt),
        historical_answer_request_key=historical["historical_answer_request_key"],
        historical_confidence_request_key=historical[
            "historical_confidence_request_key"
        ],
        selected_sample_hash=selected_sample_hash,
    )


def build_call_plans(
    config: Study1ExperimentConfig | None = None,
) -> tuple[list[Study1PlannedCall], list[Study1PlannedCall]]:
    config = config or load_study1_config()
    selected, repeats, selected_hash, _repeat_hash = load_frozen_ids(config)
    examples = {
        row.example_id: row
        for row in load_v2b_examples(config)
        if row.example_id in set(selected)
    }
    historical = load_historical_bundle(selected, config)
    models = load_models_config(config.resolve_path(config.models_config_path))
    primary: list[Study1PlannedCall] = []
    for question_id in selected:
        example = examples[question_id]
        for alias in STUDY1_MODEL_ALIASES:
            spec = models.models[alias]
            hist = historical[(question_id, alias)]
            for error_cost in STUDY1_ERROR_COSTS:
                for condition in PRIMARY_DISPLAY_CONDITIONS:
                    primary.append(
                        build_one_call(
                            example=example,
                            model_alias=alias,
                            model_endpoint=spec.api_model,
                            provider=spec.provider,
                            historical=hist,
                            error_cost=error_cost,
                            condition=condition,
                            repeat_index=0,
                            phase=Study1Phase.PRIMARY,
                            selected_sample_hash=selected_hash,
                        )
                    )
    extra: list[Study1PlannedCall] = []
    for question_id in repeats:
        example = examples[question_id]
        for alias in STUDY1_MODEL_ALIASES:
            spec = models.models[alias]
            hist = historical[(question_id, alias)]
            for error_cost in STUDY1_ERROR_COSTS:
                for condition in PRIMARY_DISPLAY_CONDITIONS:
                    for repeat_index in (1, 2):
                        extra.append(
                            build_one_call(
                                example=example,
                                model_alias=alias,
                                model_endpoint=spec.api_model,
                                provider=spec.provider,
                                historical=hist,
                                error_cost=error_cost,
                                condition=condition,
                                repeat_index=repeat_index,
                                phase=Study1Phase.REPEATS,
                                selected_sample_hash=selected_hash,
                            )
                        )
    return primary, extra


def validate_call_plans(
    primary: Sequence[Study1PlannedCall],
    extra: Sequence[Study1PlannedCall],
    config: Study1ExperimentConfig | None = None,
) -> dict[str, Any]:
    config = config or load_study1_config()
    if len(primary) != STUDY1_PRIMARY_CALLS:
        raise RuntimeError(f"primary plan has {len(primary)} rows; expected 2800")
    if len(extra) != STUDY1_REPEAT_EXTRA_CALLS:
        raise RuntimeError(f"repeat-extra plan has {len(extra)} rows; expected 1120")
    if len(primary) + len(extra) != STUDY1_TOTAL_CALLS:
        raise RuntimeError("total planned calls != 3920")

    primary_keys = [row.request_key for row in primary]
    extra_keys = [row.request_key for row in extra]
    if len(set(primary_keys)) != len(primary_keys):
        raise RuntimeError("primary request keys are not unique")
    if len(set(extra_keys)) != len(extra_keys):
        raise RuntimeError("repeat-extra request keys are not unique")
    if set(primary_keys) & set(extra_keys):
        raise RuntimeError("primary and repeat-extra request keys overlap")

    identity_primary = [
        (
            row.question_id,
            row.model_alias,
            row.L,
            row.display_condition.value,
            row.repeat_index,
        )
        for row in primary
    ]
    if len(set(identity_primary)) != len(identity_primary):
        raise RuntimeError("primary scientific identities are not unique")

    commit = _git_commit()
    for row in list(primary) + list(extra):
        row.planned_record(code_commit=commit)
        if row.C != STUDY1_VERIFICATION_COST:
            raise RuntimeError("unexpected C")
        if row.L not in STUDY1_ERROR_COSTS:
            raise RuntimeError("unexpected L")
        if row.model_alias not in STUDY1_MODEL_ALIASES:
            raise RuntimeError("unexpected model")
        if "human" in row.prompt.lower() and "HUMAN USER" in row.prompt:
            raise RuntimeError("human-authority wording leaked into Study 1")

    for row in primary:
        if row.repeat_index != 0 or row.phase != Study1Phase.PRIMARY:
            raise RuntimeError("primary row has wrong phase/repeat_index")
        _validate_grid_and_fields(row)
    for row in extra:
        if row.repeat_index not in {1, 2} or row.phase != Study1Phase.REPEATS:
            raise RuntimeError("repeat row has wrong phase/repeat_index")
        _validate_grid_and_fields(row)

    question_counts = Counter(row.question_id for row in primary)
    if set(question_counts.values()) != {2 * 2 * 7}:
        raise RuntimeError("primary per-question cell count is not 28")
    model_counts = Counter(row.model_alias for row in primary)
    if model_counts != {
        "openai_gpt56_sol": 1400,
        "anthropic_sonnet5": 1400,
    }:
        raise RuntimeError(f"unexpected primary model counts: {model_counts}")

    assert_not_v2_write_target(config.study1_sqlite(), config)
    historical = config.historical_sqlite().resolve()
    if historical != (PROJECT_ROOT / HISTORICAL_V2_SQLITE).resolve():
        raise RuntimeError("historical sqlite path drifted")
    if config.study1_sqlite().resolve() == historical:
        raise RuntimeError("Study 1 checkpoint path equals V2 sqlite")

    return {
        "primary_calls": len(primary),
        "repeat_extra_calls": len(extra),
        "total_calls": len(primary) + len(extra),
        "unique_primary_keys": len(set(primary_keys)),
        "unique_repeat_keys": len(set(extra_keys)),
        "historical_write_target": False,
        "study1_checkpoint": str(config.study1_sqlite()),
        "historical_checkpoint": str(historical),
    }


def _validate_grid_and_fields(row: Study1PlannedCall) -> None:
    reported_token = format_displayed_probability(row.reported_confidence)
    sentence_prefix = "The AI previously estimated a "
    if row.display_condition == DisplayCondition.HIDDEN:
        if row.displayed_confidence is not None:
            raise RuntimeError("hidden row stored a displayed_confidence")
        if sentence_prefix in row.prompt:
            raise RuntimeError("hidden prompt contains the confidence sentence")
        if _CONFIDENCE_LINE.search(row.prompt):
            raise RuntimeError("hidden prompt contains an explicit confidence number")
    elif row.display_condition == DisplayCondition.TRUE_CONFIDENCE_VISIBLE:
        if row.displayed_confidence != row.reported_confidence:
            raise RuntimeError("true-visible displayed != reported")
        expected = (
            sentence_prefix
            + reported_token
            + " probability that this frozen answer is correct."
        )
        if expected not in row.prompt:
            raise RuntimeError("true-visible prompt missing reported confidence")
        if row.prompt.count(sentence_prefix) != 1:
            raise RuntimeError("true-visible prompt has multiple confidence sentences")
    else:
        expected_value = grid_value(row.L, row.display_condition)
        if row.displayed_confidence != expected_value:
            raise RuntimeError("manipulated displayed_confidence off-grid")
        expected = (
            sentence_prefix
            + format_displayed_probability(expected_value)
            + " probability that this frozen answer is correct."
        )
        if expected not in row.prompt:
            raise RuntimeError("manipulated prompt missing assigned displayed value")
        if row.prompt.count(sentence_prefix) != 1:
            raise RuntimeError("manipulated prompt has multiple confidence sentences")
        if (
            row.displayed_confidence != row.reported_confidence
            and expected.replace(
                format_displayed_probability(expected_value), reported_token
            )
            == expected
            and reported_token != format_displayed_probability(expected_value)
            and (
                sentence_prefix + reported_token + " probability that this frozen answer is correct."
            )
            in row.prompt
        ):
            raise RuntimeError("manipulated prompt still shows reported_confidence")
    if f"Frozen answer:\n{row.frozen_answer}" not in row.prompt:
        raise RuntimeError("frozen answer missing from prompt")
    if '{"action":"USE_UNVERIFIED"}' not in row.prompt:
        raise RuntimeError("binary action schema missing")
    if '{"action":"VERIFY_FIRST"}' not in row.prompt:
        raise RuntimeError("binary action schema missing")


def _strip_confidence_number(prompt: str) -> str:
    return _CONFIDENCE_LINE.sub(
        "The AI previously estimated a <DISPLAYED_CONFIDENCE> probability "
        "that this frozen answer is correct.",
        prompt,
    )


def assert_manipulated_prompts_differ_only_by_number(
    calls: Sequence[Study1PlannedCall],
) -> None:
    grouped: dict[tuple[str, str, float], list[Study1PlannedCall]] = {}
    for row in calls:
        if row.display_condition not in MANIPULATED_CONDITIONS:
            continue
        grouped.setdefault((row.question_id, row.model_alias, row.L), []).append(row)
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item.display_condition.value)
        if len(rows) != 5:
            raise RuntimeError(f"expected 5 manipulated rows for {key}, found {len(rows)}")
        stripped = [_strip_confidence_number(row.prompt) for row in rows]
        if len(set(stripped)) != 1:
            raise RuntimeError(
                f"manipulated prompts differ by more than the confidence number: {key}"
            )
        numbers = [format_displayed_probability(row.displayed_confidence or 0) for row in rows]
        if len(set(numbers)) != 5:
            raise RuntimeError(f"manipulated displayed numbers are not unique: {key}")


def choose_audit_question_ids(
    primary: Sequence[Study1PlannedCall],
) -> tuple[str, str]:
    """Presentation-only: high vs lower GPT reported confidence among frozen IDs.

    Does not use correctness or historical Stage-3 actions.
    Does not change the experimental sample.
    """
    gpt_hidden = [
        row
        for row in primary
        if row.model_alias == "openai_gpt56_sol"
        and row.display_condition == DisplayCondition.HIDDEN
        and row.L == 10.0
        and row.repeat_index == 0
    ]
    by_id = {row.question_id: row.reported_confidence for row in gpt_hidden}
    ranked = sorted(by_id.items(), key=lambda item: (-item[1], item[0]))
    high_id = ranked[0][0]
    low_id = ranked[-1][0]
    if high_id == low_id:
        remaining = [question_id for question_id, _ in ranked if question_id != high_id]
        if not remaining:
            raise RuntimeError("need two question IDs for prompt audit")
        low_id = remaining[0]
    return high_id, low_id


def write_csv(path: Path, rows: Sequence[Study1PlannedCall]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "repeat_index",
        "question_id",
        "model_alias",
        "model_endpoint",
        "provider",
        "L",
        "C",
        "display_condition",
        "display_source_condition",
        "reported_confidence",
        "displayed_confidence",
        "frozen_answer",
        "stage1_correct",
        "request_key",
        "prompt_hash",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "phase": row.phase.value,
                    "repeat_index": row.repeat_index,
                    "question_id": row.question_id,
                    "model_alias": row.model_alias,
                    "model_endpoint": row.model_endpoint,
                    "provider": row.provider,
                    "L": row.L,
                    "C": row.C,
                    "display_condition": row.display_condition.value,
                    "display_source_condition": row.display_source_condition.value,
                    "reported_confidence": row.reported_confidence,
                    "displayed_confidence": (
                        "" if row.displayed_confidence is None else row.displayed_confidence
                    ),
                    "frozen_answer": row.frozen_answer,
                    "stage1_correct": row.stage1_correct,
                    "request_key": row.request_key,
                    "prompt_hash": row.prompt_hash,
                }
            )


def write_prompt_audit(
    primary: Sequence[Study1PlannedCall],
    audit_dir: Path,
) -> dict[str, Any]:
    high_id, low_id = choose_audit_question_ids(primary)
    audit_dir.mkdir(parents=True, exist_ok=True)
    diffs_dir = audit_dir / "diffs"
    diffs_dir.mkdir(parents=True, exist_ok=True)
    selected = {high_id, low_id}
    rendered = 0
    for row in primary:
        if row.question_id not in selected:
            continue
        slug = row.question_id.replace(":", "_")
        path = (
            audit_dir
            / slug
            / row.model_alias
            / f"L{int(row.L)}"
            / f"{row.display_condition.value}.txt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(row.prompt + "\n", encoding="utf-8")
        rendered += 1

    assert_manipulated_prompts_differ_only_by_number(
        [row for row in primary if row.question_id in selected]
    )
    for question_id in (high_id, low_id):
        for alias in STUDY1_MODEL_ALIASES:
            for error_cost in STUDY1_ERROR_COSTS:
                subset = [
                    row
                    for row in primary
                    if row.question_id == question_id
                    and row.model_alias == alias
                    and row.L == error_cost
                    and row.display_condition in MANIPULATED_CONDITIONS
                ]
                subset = sorted(subset, key=lambda item: MANIPULATED_CONDITIONS.index(item.display_condition))
                sections: list[str] = []
                for left, right in zip(subset, subset[1:]):
                    diff = "\n".join(
                        difflib.unified_diff(
                            left.prompt.splitlines(),
                            right.prompt.splitlines(),
                            fromfile=left.display_condition.value,
                            tofile=right.display_condition.value,
                            lineterm="",
                        )
                    )
                    sections.append(diff)
                    changed_lines = [
                        line
                        for line in diff.splitlines()
                        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
                    ]
                    for line in changed_lines:
                        if "The AI previously estimated a " not in line:
                            raise RuntimeError(
                                "manipulated prompt diff contains a non-confidence change: "
                                f"{question_id} {alias} L={error_cost}: {line}"
                            )
                hidden = next(
                    row
                    for row in primary
                    if row.question_id == question_id
                    and row.model_alias == alias
                    and row.L == error_cost
                    and row.display_condition == DisplayCondition.HIDDEN
                )
                visible = next(
                    row
                    for row in primary
                    if row.question_id == question_id
                    and row.model_alias == alias
                    and row.L == error_cost
                    and row.display_condition
                    == DisplayCondition.TRUE_CONFIDENCE_VISIBLE
                )
                hv = "\n".join(
                    difflib.unified_diff(
                        hidden.prompt.splitlines(),
                        visible.prompt.splitlines(),
                        fromfile="hidden",
                        tofile="true_confidence_visible",
                        lineterm="",
                    )
                )
                slug = question_id.replace(":", "_")
                (diffs_dir / f"{slug}__{alias}__L{int(error_cost)}__manipulated.diff").write_text(
                    "\n\n".join(sections) + "\n", encoding="utf-8"
                )
                (
                    diffs_dir / f"{slug}__{alias}__L{int(error_cost)}__hidden_vs_true_visible.diff"
                ).write_text(hv + "\n", encoding="utf-8")

    example = next(row for row in primary if row.question_id == high_id)
    hist_example = next(
        row
        for row in load_v2b_examples()
        if row.example_id == high_id
    )
    if not historical_visible_matches_v2_builder(
        question=hist_example.question,
        choices=hist_example.choices,
        frozen_answer=example.frozen_answer,
        probability=example.reported_confidence,
        verification_cost=STUDY1_VERIFICATION_COST,
        error_cost=10.0,
    ):
        # Compare using a matching GPT L=10 true-visible row.
        gpt_visible = next(
            row
            for row in primary
            if row.question_id == high_id
            and row.model_alias == "openai_gpt56_sol"
            and row.L == 10.0
            and row.display_condition == DisplayCondition.TRUE_CONFIDENCE_VISIBLE
        )
        if not historical_visible_matches_v2_builder(
            question=hist_example.question,
            choices=hist_example.choices,
            frozen_answer=gpt_visible.frozen_answer,
            probability=gpt_visible.reported_confidence,
            verification_cost=STUDY1_VERIFICATION_COST,
            error_cost=10.0,
        ):
            raise RuntimeError(
                "Study 1 true-visible prompt does not match historical V2 AI-visible wording"
            )

    gpt_conf = {
        row.question_id: row.reported_confidence
        for row in primary
        if row.model_alias == "openai_gpt56_sol"
        and row.display_condition == DisplayCondition.HIDDEN
        and row.L == 10.0
    }
    return {
        "audit_question_selection": "presentation_only_gpt_reported_confidence_extremes",
        "does_not_alter_experimental_sample": True,
        "selection_uses_correctness": False,
        "selection_uses_historical_stage3": False,
        "high_reported_confidence_question_id": high_id,
        "high_gpt_reported_confidence": gpt_conf[high_id],
        "lower_reported_confidence_question_id": low_id,
        "lower_gpt_reported_confidence": gpt_conf[low_id],
        "rendered_prompt_files": rendered,
        "historical_prompt_source": HISTORICAL_PROMPT_SOURCE,
    }


def write_plan_summary(
    path: Path,
    *,
    primary: Sequence[Study1PlannedCall],
    extra: Sequence[Study1PlannedCall],
    selected_hash: str,
    repeat_hash: str,
    validation: dict[str, Any],
    audit_meta: dict[str, Any],
) -> None:
    grids = {
        str(int(level)): list(values) for level, values in GRIDS_BY_L.items()
    }
    lines = [
        "# Study 1 call-plan summary",
        "",
        "Offline plan only. No model API calls were made.",
        "",
        f"- Primary calls: **{len(primary)}** (required 2800)",
        f"- Repeat-extra calls: **{len(extra)}** (required 1120)",
        f"- Total possible later pilot: **{len(primary) + len(extra)}** (required 3920)",
        f"- Models: `{', '.join(STUDY1_MODEL_ALIASES)}`",
        "- Authority: AI-system only",
        "- L: 10 and 20; C = 1",
        f"- L=10 grid: {list(GRIDS_BY_L[10.0])}",
        f"- L=20 grid: {list(GRIDS_BY_L[20.0])}",
        "- L=10 local contrast: 0.89 vs 0.91",
        "- L=20 local contrast: 0.94 vs 0.96",
        f"- Primary ID list SHA-256: `{selected_hash}`",
        f"- Repeat ID list SHA-256: `{repeat_hash}`",
        f"- Study 1 checkpoint path: `{validation['study1_checkpoint']}`",
        f"- Historical checkpoint path: `{validation['historical_checkpoint']}`",
        "- Historical write target: no",
        "",
        "## Prompt-audit presentation IDs",
        "",
        "These two IDs were chosen only to display high vs lower GPT reported confidence.",
        "They do not change the frozen 100-question experimental sample.",
        "They were not chosen using correctness or historical Stage-3 actions.",
        "",
        f"- High GPT reported confidence: `{audit_meta['high_reported_confidence_question_id']}` "
        f"({audit_meta['high_gpt_reported_confidence']})",
        f"- Lower GPT reported confidence: `{audit_meta['lower_reported_confidence_question_id']}` "
        f"({audit_meta['lower_gpt_reported_confidence']})",
        "",
        f"Historical wording source: `{audit_meta['historical_prompt_source']}`",
        "",
        f"Confidence grids JSON: `{json.dumps(grids)}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_study1_offline(
    config: Study1ExperimentConfig | None = None,
) -> dict[str, Any]:
    config = config or load_study1_config()
    primary_payload, repeat_payload = write_study1_ids(config)
    primary, extra = build_call_plans(config)
    validation = validate_call_plans(primary, extra, config)
    assert_manipulated_prompts_differ_only_by_number(primary)
    analysis_dir = config.resolve_path(config.results["analysis_directory"])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_dir / "study1_primary_call_plan.csv", primary)
    write_csv(analysis_dir / "study1_repeat_call_plan.csv", extra)
    audit_dir = config.resolve_path(config.results["prompt_audit_directory"])
    audit_meta = write_prompt_audit(primary, audit_dir)
    write_plan_summary(
        analysis_dir / "study1_call_plan_summary.md",
        primary=primary,
        extra=extra,
        selected_hash=primary_payload["selected_id_list_sha256"],
        repeat_hash=repeat_payload["repeat_id_list_sha256"],
        validation=validation,
        audit_meta=audit_meta,
    )
    return {
        "primary_payload": primary_payload,
        "repeat_payload": repeat_payload,
        "validation": validation,
        "audit_meta": audit_meta,
        "primary_count": len(primary),
        "repeat_extra_count": len(extra),
    }
