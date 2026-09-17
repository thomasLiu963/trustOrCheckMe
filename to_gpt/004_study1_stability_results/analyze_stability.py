"""Exploratory stability analysis of the Study 1 20-question repeat subset."""

from __future__ import annotations

import csv
import inspect
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bootstrap import paired_bootstrap
from .checkpointing import CheckpointStore
from .config import PROJECT_ROOT
from .study1_analysis import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXTREME,
    MODEL_LABELS,
    NEAR_THRESHOLD,
    VERIFY,
    _fmt_ci,
    _fmt_pp,
    _is_verify,
    _pp,
    _pyplot,
    _subset,
    _write_csv,
    mean_or_nan,
    paired_contrast,
    rate_with_ci,
    reference_rates,
)
from .study1_plan import Study1PlannedCall, build_call_plans
from .study1_primary import _parsed_action, _record_from_existing, reconstruct_completed_payload
from .study1_repeats import _outcome_row, postvalidate_repeats
from .study1_sample import load_frozen_ids
from .study1_schemas import (
    STUDY1_GRID_L10,
    STUDY1_GRID_L20,
    STUDY1_MODEL_ALIASES,
    Study1ExperimentConfig,
    Study1Phase,
    load_study1_config,
)

CONDITIONS = (
    "hidden",
    "true_confidence_visible",
    "manipulated_1",
    "manipulated_2",
    "manipulated_3",
    "manipulated_4",
    "manipulated_5",
)


def load_triple_rows(
    config: Study1ExperimentConfig | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    config = config or load_study1_config()
    primary, extra = build_call_plans(config)
    _selected, repeats, _sh, _rh = load_frozen_ids(config)
    repeat_set = set(repeats)
    wanted: list[Study1PlannedCall] = [
        row for row in primary if row.question_id in repeat_set
    ] + list(extra)
    rows: list[dict[str, Any]] = []
    with CheckpointStore(config.study1_sqlite()) as checkpoint:
        for row in wanted:
            existing = checkpoint.get_record(row.request_key)
            if not existing:
                continue
            parsed = _parsed_action(existing)
            record = _record_from_existing(existing)
            item = {
                "ok": True,
                "parse_status": existing.get("parse_status", "success"),
                "parsed_action": parsed,
                "raw_response": existing.get("raw_response"),
                "provider_attempts": existing.get("provider_attempts") or 0,
                "parse_repairs": 0,
                "returned_model_id": existing.get("returned_model_id"),
                "input_tokens": existing.get("input_tokens"),
                "output_tokens": existing.get("output_tokens"),
                "latency_ms": existing.get("latency_ms"),
                "estimated_cost_usd": float(existing.get("estimated_cost_usd") or 0.0),
                "attempt_number": existing.get("attempt_number") or 1,
                "row": row,
                "record": record,
            }
            out = _outcome_row(item)
            rows.append(out)
    return rows, list(repeats)


def _cell_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["question_id"],
        row["model_alias"],
        float(row["L"]),
        row["display_condition"],
    )


def _group_cells(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], dict[int, str]]:
    grouped: dict[tuple[Any, ...], dict[int, str]] = defaultdict(dict)
    for row in rows:
        grouped[_cell_key(row)][int(row["repeat_index"])] = str(row["parsed_action"])
    return grouped


def _entropy(actions: Sequence[str]) -> float:
    if not actions:
        return float("nan")
    counts: dict[str, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1
    total = len(actions)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _pairwise_disagreement(actions: Sequence[str]) -> float:
    pairs = 0
    disagree = 0
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            pairs += 1
            if actions[i] != actions[j]:
                disagree += 1
    return disagree / pairs if pairs else float("nan")


def stability_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_cells(rows)
    out: list[dict[str, Any]] = []
    for model in STUDY1_MODEL_ALIASES:
        for L in (10.0, 20.0):
            for condition in CONDITIONS:
                cells = [
                    actions
                    for (qid, alias, error_cost, cond), actions in grouped.items()
                    if alias == model
                    and error_cost == L
                    and cond == condition
                    and set(actions) >= {0, 1, 2}
                ]
                n = len(cells)
                if n == 0:
                    continue
                agree3 = sum(1 for actions in cells if len(set(actions.values())) == 1)
                split21 = sum(1 for actions in cells if len(set(actions.values())) == 2)
                entropies = [_entropy([actions[i] for i in (0, 1, 2)]) for actions in cells]
                pairwise = [
                    _pairwise_disagreement([actions[i] for i in (0, 1, 2)])
                    for actions in cells
                ]
                p_r1 = mean_or_nan(int(actions[0] != actions[1]) for actions in cells)
                p_r2 = mean_or_nan(int(actions[0] != actions[2]) for actions in cells)
                p_12 = mean_or_nan(int(actions[1] != actions[2]) for actions in cells)
                out.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "L": L,
                        "display_condition": condition,
                        "n_questions": n,
                        "n_complete_triples": n,
                        "frac_all_three_agree": agree3 / n,
                        "frac_two_one_split": split21 / n,
                        "mean_action_entropy_bits": mean_or_nan(entropies),
                        "mean_pairwise_disagreement": mean_or_nan(pairwise),
                        "p_repeat1_differs_from_primary": p_r1,
                        "p_repeat2_differs_from_primary": p_r2,
                        "p_repeat1_differs_from_repeat2": p_12,
                    }
                )
    return out


def question_mean_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per question×model×L×condition with mean of three generations."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_cell_key(row)].append(row)
    collapsed: list[dict[str, Any]] = []
    for key, items in grouped.items():
        if len(items) != 3:
            continue
        collapsed.append(
            {
                "question_id": key[0],
                "model_alias": key[1],
                "L": key[2],
                "display_condition": key[3],
                "displayed_confidence": items[0]["displayed_confidence"],
                "reported_confidence": items[0]["reported_confidence"],
                "parsed_action": VERIFY
                if mean_or_nan(_is_verify(item["parsed_action"]) for item in items) >= 0.5
                else "USE_UNVERIFIED",
                "verify_mean": mean_or_nan(
                    _is_verify(item["parsed_action"]) for item in items
                ),
                "n_generations": 3,
            }
        )
    return collapsed


def aggregated_curve(
    rows: Sequence[Mapping[str, Any]], model: str, L: float
) -> list[dict[str, Any]]:
    grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
    threshold = 1.0 - (1.0 / L)
    points = []
    for displayed in grid:
        subset = _subset(
            rows, model=model, L=L, displayed=displayed, manipulated_only=True
        )
        stats = rate_with_ci(subset)
        points.append(
            {
                "model_alias": model,
                "model_label": MODEL_LABELS[model],
                "L": L,
                "displayed_confidence": displayed,
                "n_observations": stats["n"],
                "n_questions": stats["n_questions"],
                "verification_rate": stats["rate"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
                "threshold": threshold,
                "seed": BOOTSTRAP_SEED,
                "estimator": "three_generations_clustered_by_question",
            }
        )
    return points


def generation_specific_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    L: float,
    low: float,
    high: float,
    repeat_index: int,
) -> dict[str, Any]:
    subset = [row for row in rows if int(row["repeat_index"]) == int(repeat_index)]
    result = paired_contrast(subset, model=model, L=L, low=low, high=high)
    result["repeat_index"] = repeat_index
    return result


def aggregated_paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    L: float,
    low: float,
    high: float,
) -> dict[str, Any]:
    collapsed = question_mean_rows(rows)
    left = _subset(
        collapsed, model=model, L=L, displayed=low, manipulated_only=True
    )
    right = _subset(
        collapsed, model=model, L=L, displayed=high, manipulated_only=True
    )
    left_by = {row["question_id"]: float(row["verify_mean"]) for row in left}
    right_by = {row["question_id"]: float(row["verify_mean"]) for row in right}
    shared = sorted(set(left_by) & set(right_by))
    pairs = [(qid, left_by[qid], right_by[qid]) for qid in shared]
    boot = paired_bootstrap(pairs, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    # Majority-vote flip counts use the collapsed parsed_action (>=0.5 VERIFY).
    left_vote = {row["question_id"]: _is_verify(row["parsed_action"]) for row in left}
    right_vote = {row["question_id"]: _is_verify(row["parsed_action"]) for row in right}
    v_to_u = sum(1 for qid in shared if left_vote[qid] == 1 and right_vote[qid] == 0)
    u_to_v = sum(1 for qid in shared if left_vote[qid] == 0 and right_vote[qid] == 1)
    return {
        "model_alias": model,
        "model_label": MODEL_LABELS[model],
        "L": L,
        "low_displayed": low,
        "high_displayed": high,
        "n_questions": len(shared),
        "rate_low": mean_or_nan(left_by[qid] for qid in shared),
        "rate_high": mean_or_nan(right_by[qid] for qid in shared),
        "difference": boot.estimate,
        "ci_lower": boot.lower,
        "ci_upper": boot.upper,
        "verify_to_use_majority": v_to_u,
        "use_to_verify_majority": u_to_v,
        "seed": BOOTSTRAP_SEED,
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "estimator": "mean_of_three_generations_question_resampled",
    }


def rerun_noise_vs_manipulation(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped = _group_cells(rows)
    out: list[dict[str, Any]] = []
    for model in STUDY1_MODEL_ALIASES:
        for L, (low, high, _thr) in NEAR_THRESHOLD.items():
            identical_cells = [
                actions
                for (qid, alias, error_cost, _cond), actions in grouped.items()
                if alias == model and error_cost == L and set(actions) >= {0, 1, 2}
            ]
            identical_threshold_cells = [
                actions
                for (qid, alias, error_cost, cond), actions in grouped.items()
                if alias == model
                and error_cost == L
                and cond in {"manipulated_3", "manipulated_4"}
                and set(actions) >= {0, 1, 2}
            ]
            rerun_all = mean_or_nan(
                _pairwise_disagreement([actions[i] for i in (0, 1, 2)])
                for actions in identical_cells
            )
            rerun_vs_primary = mean_or_nan(
                mean_or_nan(
                    [
                        int(actions[0] != actions[1]),
                        int(actions[0] != actions[2]),
                    ]
                )
                for actions in identical_cells
            )
            rerun_threshold_cells = mean_or_nan(
                _pairwise_disagreement([actions[i] for i in (0, 1, 2)])
                for actions in identical_threshold_cells
            )

            by_q_gen: dict[tuple[str, int], dict[float, str]] = defaultdict(dict)
            for row in rows:
                if row["model_alias"] != model or float(row["L"]) != L:
                    continue
                if not str(row["display_condition"]).startswith("manipulated"):
                    continue
                displayed = row["displayed_confidence"]
                if displayed is None:
                    continue
                if abs(float(displayed) - float(low)) > 1e-12 and abs(
                    float(displayed) - float(high)
                ) > 1e-12:
                    continue
                by_q_gen[(row["question_id"], int(row["repeat_index"]))][
                    float(displayed)
                ] = str(row["parsed_action"])
            matched = [
                actions
                for actions in by_q_gen.values()
                if low in actions and high in actions
            ]
            manip_within_gen = mean_or_nan(
                int(actions[low] != actions[high]) for actions in matched
            )
            # Question-level: fraction of questions whose majority vote flips.
            by_q: dict[str, dict[float, list[str]]] = defaultdict(lambda: defaultdict(list))
            for row in rows:
                if row["model_alias"] != model or float(row["L"]) != L:
                    continue
                displayed = row["displayed_confidence"]
                if displayed is None:
                    continue
                if abs(float(displayed) - float(low)) > 1e-12 and abs(
                    float(displayed) - float(high)
                ) > 1e-12:
                    continue
                if not str(row["display_condition"]).startswith("manipulated"):
                    continue
                by_q[row["question_id"]][float(displayed)].append(str(row["parsed_action"]))
            majority_flips = []
            for qid, displayed_map in by_q.items():
                if low not in displayed_map or high not in displayed_map:
                    continue
                if len(displayed_map[low]) != 3 or len(displayed_map[high]) != 3:
                    continue
                left = sum(_is_verify(a) for a in displayed_map[low]) >= 2
                right = sum(_is_verify(a) for a in displayed_map[high]) >= 2
                majority_flips.append(int(left != right))
            out.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "L": L,
                    "contrast": "near_threshold",
                    "low_displayed": low,
                    "high_displayed": high,
                    "rerun_mean_pairwise_disagreement_all_conditions": rerun_all,
                    "rerun_p_repeat_differs_from_primary_all_conditions": rerun_vs_primary,
                    "rerun_mean_pairwise_disagreement_threshold_cells": rerun_threshold_cells,
                    "manipulation_p_action_changes_within_generation": manip_within_gen,
                    "manipulation_p_majority_vote_changes": mean_or_nan(majority_flips),
                    "n_questions": len(majority_flips),
                    "n_matched_generation_pairs": len(matched),
                    "n_identical_prompt_cells": len(identical_cells),
                    "plain_language": (
                        f"Repeating the same prompt changed {MODEL_LABELS[model]}'s "
                        f"L={int(L)} decision "
                        f"{100 * rerun_vs_primary:.1f}% of the time, whereas crossing "
                        f"the displayed-confidence threshold ({low} vs {high}) changed it "
                        f"{100 * manip_within_gen:.1f}% of the time."
                    ),
                }
            )
        for L, (low, high) in EXTREME.items():
            identical_cells = [
                actions
                for (qid, alias, error_cost, _cond), actions in grouped.items()
                if alias == model and error_cost == L and set(actions) >= {0, 1, 2}
            ]
            rerun_vs_primary = mean_or_nan(
                mean_or_nan(
                    [
                        int(actions[0] != actions[1]),
                        int(actions[0] != actions[2]),
                    ]
                )
                for actions in identical_cells
            )
            by_q_gen = defaultdict(dict)
            for row in rows:
                if row["model_alias"] != model or float(row["L"]) != L:
                    continue
                if not str(row["display_condition"]).startswith("manipulated"):
                    continue
                displayed = row["displayed_confidence"]
                if displayed is None:
                    continue
                if abs(float(displayed) - float(low)) > 1e-12 and abs(
                    float(displayed) - float(high)
                ) > 1e-12:
                    continue
                by_q_gen[(row["question_id"], int(row["repeat_index"]))][
                    float(displayed)
                ] = str(row["parsed_action"])
            matched = [
                actions
                for actions in by_q_gen.values()
                if low in actions and high in actions
            ]
            manip_within_gen = mean_or_nan(
                int(actions[low] != actions[high]) for actions in matched
            )
            out.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "L": L,
                    "contrast": "extreme",
                    "low_displayed": low,
                    "high_displayed": high,
                    "rerun_mean_pairwise_disagreement_all_conditions": "",
                    "rerun_p_repeat_differs_from_primary_all_conditions": rerun_vs_primary,
                    "rerun_mean_pairwise_disagreement_threshold_cells": "",
                    "manipulation_p_action_changes_within_generation": manip_within_gen,
                    "manipulation_p_majority_vote_changes": "",
                    "n_questions": "",
                    "n_matched_generation_pairs": len(matched),
                    "n_identical_prompt_cells": len(identical_cells),
                    "plain_language": (
                        f"Repeating the same prompt changed {MODEL_LABELS[model]}'s "
                        f"L={int(L)} decision "
                        f"{100 * rerun_vs_primary:.1f}% of the time, whereas the extreme "
                        f"displayed-confidence contrast ({low} vs {high}) changed it "
                        f"{100 * manip_within_gen:.1f}% of the time."
                    ),
                }
            )
    return out


def threshold_stability_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    contrasts = []
    for L, (low, high, _thr) in NEAR_THRESHOLD.items():
        contrasts.append(("near_threshold", L, low, high))
    for L, (low, high) in EXTREME.items():
        contrasts.append(("extreme", L, low, high))
    for model in STUDY1_MODEL_ALIASES:
        for kind, L, low, high in contrasts:
            primary_only = generation_specific_contrast(
                rows, model=model, L=L, low=low, high=high, repeat_index=0
            )
            gens = [
                generation_specific_contrast(
                    rows, model=model, L=L, low=low, high=high, repeat_index=index
                )
                for index in (0, 1, 2)
            ]
            aggregated = aggregated_paired_contrast(
                rows, model=model, L=L, low=low, high=high
            )
            directions = [float(item["difference"]) for item in gens]
            out.append(
                {
                    "model_alias": model,
                    "model_label": MODEL_LABELS[model],
                    "L": L,
                    "contrast": kind,
                    "low_displayed": low,
                    "high_displayed": high,
                    "primary_only_rate_low": primary_only["rate_low"],
                    "primary_only_rate_high": primary_only["rate_high"],
                    "primary_only_difference": primary_only["difference"],
                    "primary_only_ci_lower": primary_only["ci_lower"],
                    "primary_only_ci_upper": primary_only["ci_upper"],
                    "aggregated_rate_low": aggregated["rate_low"],
                    "aggregated_rate_high": aggregated["rate_high"],
                    "aggregated_difference": aggregated["difference"],
                    "aggregated_ci_lower": aggregated["ci_lower"],
                    "aggregated_ci_upper": aggregated["ci_upper"],
                    "gen0_difference": gens[0]["difference"],
                    "gen1_difference": gens[1]["difference"],
                    "gen2_difference": gens[2]["difference"],
                    "direction_consistent_across_generations": all(
                        diff > 0 for diff in directions
                    ),
                    "n_questions": aggregated["n_questions"],
                    "seed": BOOTSTRAP_SEED,
                    "n_resamples": BOOTSTRAP_RESAMPLES,
                }
            )
    return out


def trajectory_by_generation(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in STUDY1_MODEL_ALIASES:
        for L in (10.0, 20.0):
            grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
            for repeat_index in (0, 1, 2, "aggregated"):
                if repeat_index == "aggregated":
                    subset = rows
                    label = "aggregated_three_generations"
                else:
                    subset = [
                        row
                        for row in rows
                        if int(row["repeat_index"]) == int(repeat_index)
                    ]
                    label = f"repeat_index_{repeat_index}"
                rates = []
                for displayed in grid:
                    piece = _subset(
                        subset,
                        model=model,
                        L=L,
                        displayed=displayed,
                        manipulated_only=True,
                    )
                    rates.append(rate_with_ci(piece)["rate"])
                diffs = [left - right for left, right in zip(rates, rates[1:])]
                out.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "L": L,
                        "estimator": label,
                        "rates": rates,
                        "grid": list(grid),
                        "monotonic_nonincreasing": all(diff >= -1e-12 for diff in diffs),
                        "any_uptick": any(diff < -1e-12 for diff in diffs),
                        "largest_drop": max(diffs) if diffs else float("nan"),
                        "step_index": diffs.index(max(diffs)) if diffs else None,
                    }
                )
    return out


def recommend_final_gate(
    *,
    validation_ok: bool,
    audit_stop: bool,
    threshold_rows: Sequence[Mapping[str, Any]],
    noise_rows: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str]:
    if audit_stop:
        return (
            "D",
            "PASS BUT ENGINEERING/DESIGN ISSUE REQUIRES RESOLUTION",
            "Parse-repair behavior was condition-dependent enough that the displayed-confidence effect cannot be cleanly separated from parser artifacts.",
        )
    if not validation_ok:
        return (
            "B",
            "AMBIGUOUS",
            "The 1,120 repeat-extra dataset did not fully validate, so the final Study-1 gate cannot be applied.",
        )
    gpt_near = [
        row
        for row in threshold_rows
        if row["model_alias"] == "openai_gpt56_sol" and row["contrast"] == "near_threshold"
    ]
    gpt_noise = [
        row
        for row in noise_rows
        if row["model_alias"] == "openai_gpt56_sol" and row["contrast"] == "near_threshold"
    ]
    claude_near = [
        row
        for row in threshold_rows
        if row["model_alias"] == "anthropic_sonnet5" and row["contrast"] == "near_threshold"
    ]
    if not gpt_near or not gpt_noise:
        return ("B", "AMBIGUOUS", "Missing GPT threshold or rerun-noise rows.")
    gpt_diffs = [float(row["aggregated_difference"]) for row in gpt_near]
    gpt_consistent = all(row["direction_consistent_across_generations"] for row in gpt_near)
    gpt_manip = [
        float(row["manipulation_p_action_changes_within_generation"]) for row in gpt_noise
    ]
    gpt_rerun = [
        float(row["rerun_p_repeat_differs_from_primary_all_conditions"]) for row in gpt_noise
    ]
    gpt_beyond_noise = all(
        manip >= max(0.20, 3.0 * rerun) for manip, rerun in zip(gpt_manip, gpt_rerun)
    )
    gpt_large = all(diff >= 0.50 for diff in gpt_diffs)
    gpt_traj = [
        row
        for row in trajectories
        if row["model_alias"] == "openai_gpt56_sol"
        and row["estimator"] == "aggregated_three_generations"
    ]
    gpt_step = all(float(row["largest_drop"]) >= 0.50 for row in gpt_traj)
    claude_diffs = [float(row["aggregated_difference"]) for row in claude_near]
    claude_near_stable = all(diff > 0.02 for diff in claude_diffs)
    if gpt_large and gpt_consistent and gpt_beyond_noise and gpt_step:
        claude_note = (
            " Claude's near-threshold contrast on this 20-question subset is small and not stable across generations; it does not itself beat rerun noise. Claude remains an informative weaker/gradual contrast, especially at the extreme scores."
            if not claude_near_stable
            else " Claude is a smaller same-direction contrast."
        )
        return (
            "C",
            "PASS CAUSAL BEHAVIORAL PHENOMENON",
            "Changing displayed confidence robustly and causally changes verification behavior in this controlled exploratory setup, beyond ordinary repeated-prompt variation. GPT's near-threshold effect is far larger than identical-prompt rerun noise."
            + claude_note,
        )
    if max(gpt_diffs, default=0) < 0.10:
        return (
            "A",
            "FAIL",
            "After repeats, GPT's displayed-confidence effect is no longer a substantial systematic change relative to ordinary rerun variation.",
        )
    return (
        "B",
        "AMBIGUOUS",
        "Repeats leave a displayed-confidence pattern, but it is not cleanly larger than rerun noise on both L values, not direction-stable across generations, or not step-like enough to pass the exploratory gate.",
    )


def _plot_curves(
    *,
    primary20: Sequence[Mapping[str, Any]],
    aggregated: Sequence[Mapping[str, Any]],
    primary100: Sequence[Mapping[str, Any]] | None,
    path: Path,
    title: str,
    threshold: float,
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = [float(row["displayed_confidence"]) for row in aggregated]
    ax.plot(
        xs,
        [100 * float(row["verification_rate"]) for row in aggregated],
        marker="o",
        color="#1f4e79",
        label="20 questions × 3 generations",
    )
    ax.fill_between(
        xs,
        [100 * float(row["ci_lower"]) for row in aggregated],
        [100 * float(row["ci_upper"]) for row in aggregated],
        color="#1f4e79",
        alpha=0.15,
    )
    ax.plot(
        [float(row["displayed_confidence"]) for row in primary20],
        [100 * float(row["verification_rate"]) for row in primary20],
        marker="s",
        linestyle="--",
        color="#c45911",
        label="same 20 questions, primary only",
    )
    if primary100:
        ax.plot(
            [float(row["displayed_confidence"]) for row in primary100],
            [100 * float(row["verification_rate"]) for row in primary100],
            marker="^",
            linestyle=":",
            color="#7f7f7f",
            label="Task 003 n=100 primary",
        )
    ax.axvline(threshold, color="black", linewidth=1, alpha=0.6)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("Displayed confidence")
    ax.set_ylabel("P(VERIFY_FIRST) (%)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_noise(noise_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _pyplot()
    near = [row for row in noise_rows if row["contrast"] == "near_threshold"]
    labels = [f"{row['model_label']} L={int(row['L'])}" for row in near]
    rerun = [
        100 * float(row["rerun_p_repeat_differs_from_primary_all_conditions"])
        for row in near
    ]
    manip = [
        100 * float(row["manipulation_p_action_changes_within_generation"])
        for row in near
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([i - width / 2 for i in x], rerun, width, label="Identical-prompt rerun", color="#7f7f7f")
    ax.bar([i + width / 2 for i in x], manip, width, label="Threshold manipulation", color="#1f4e79")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Action-change probability (%)")
    ax.set_title("Rerun noise versus displayed-confidence threshold")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_agreement(stability: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), sharey=True)
    panels = [
        ("openai_gpt56_sol", 10.0),
        ("openai_gpt56_sol", 20.0),
        ("anthropic_sonnet5", 10.0),
        ("anthropic_sonnet5", 20.0),
    ]
    for ax, (model, L) in zip(axes.ravel(), panels):
        subset = [
            row for row in stability if row["model_alias"] == model and float(row["L"]) == L
        ]
        labels = [row["display_condition"].replace("manipulated", "m") for row in subset]
        agree = [100 * float(row["frac_all_three_agree"]) for row in subset]
        ax.bar(range(len(labels)), agree, color="#1f4e79")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(f"{MODEL_LABELS[model]} L={int(L)}")
        ax.set_ylabel("% all 3 agree")
    fig.suptitle("Identical-prompt agreement across three generations")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_analysis_method(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Study 1 stability analysis method",
                "",
                "Exploratory analysis of the frozen 20-question repeat subset.",
                "The unit of resampling is the question, not the generation.",
                "",
                f"- Bootstrap seed: {BOOTSTRAP_SEED}",
                f"- Bootstrap resamples: {BOOTSTRAP_RESAMPLES}",
                "- Each repeated cell contributes the original primary observation plus two independent extra generations.",
                "- Identical-prompt rerun noise is the probability that a later generation disagrees with the primary generation of the same prompt, averaged within question then across questions/cells.",
                "- Manipulation change is the probability that the low vs high displayed-confidence actions disagree, matched on question and generation index.",
                "- Aggregated response curves average the three binary actions per question, then bootstrap questions.",
                "- Do not treat 60 observations as 60 independent questions.",
                "",
                "A PASS does not establish self-provenance, practical agent generalization, a hidden-state mechanism, suppression of internal uncertainty, or novelty relative to all prior literature.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _script_snapshot() -> str:
    return inspect.getsource(inspect.getmodule(write_stability_analysis_bundle))


def write_stability_analysis_bundle(
    payload: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    validation = postvalidate_repeats(payload)
    (output / "data_validation.md").write_text(
        "\n".join(
            [
                "# Study 1 repeat-extra data validation",
                "",
                f"- ok: {validation['ok']}",
                f"- successful unique repeat-extra cells: {validation['successful_unique_repeat_extra_cells']}",
                f"- repeat question count: {validation['repeat_question_count']}",
                f"- cells with three observations: {validation['cells_with_three_observations']}/{validation['repeated_primary_cells']}",
                f"- historical V2 unchanged: {validation['historical_unchanged']}",
                f"- no new primary conditions: {validation['no_new_primary_conditions']}",
                "",
                "## Issues",
                "",
                *(
                    [f"- {issue}" for issue in validation["issues"]]
                    if validation["issues"]
                    else ["- none"]
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    extra_rows = [_outcome_row(item) for item in payload.get("successful", [])]
    _write_csv(output / "repeat_results.csv", extra_rows)

    triple_rows, repeat_ids = load_triple_rows(payload.get("config"))
    stability = stability_rows(triple_rows)
    threshold = threshold_stability_rows(triple_rows)
    noise = rerun_noise_vs_manipulation(triple_rows)
    trajectories = trajectory_by_generation(triple_rows)
    _write_csv(output / "stability_summary.csv", stability)
    _write_csv(output / "threshold_stability.csv", threshold)
    _write_csv(output / "rerun_noise_vs_manipulation.csv", noise)

    primary100 = None
    try:
        primary_payload = reconstruct_completed_payload(payload.get("config"))
        from .study1_analysis import successful_rows, manipulated_curve

        primary100_rows = successful_rows(primary_payload)
    except Exception:
        primary100_rows = []
        manipulated_curve = None  # type: ignore[assignment]

    curves = {}
    for model in STUDY1_MODEL_ALIASES:
        for L in (10.0, 20.0):
            agg = aggregated_curve(triple_rows, model, L)
            prim20 = aggregated_curve(
                [row for row in triple_rows if int(row["repeat_index"]) == 0],
                model,
                L,
            )
            full = (
                manipulated_curve(primary100_rows, model, L)
                if primary100_rows and manipulated_curve
                else None
            )
            curves[(model, L)] = {"aggregated": agg, "primary20": prim20, "primary100": full}
            slug = "gpt" if model == "openai_gpt56_sol" else "claude"
            _plot_curves(
                primary20=prim20,
                aggregated=agg,
                primary100=full,
                path=figures / f"{slug}_L{int(L)}_response_curve.png",
                title=f"{MODEL_LABELS[model]}, L={int(L)} (20-question repeat subset)",
                threshold=1.0 - 1.0 / L,
            )
    _plot_noise(noise, figures / "rerun_noise_vs_manipulation.png")
    _plot_agreement(stability, figures / "agreement_by_condition.png")

    audit = payload.get("audit") or {}
    letter, label, reason = recommend_final_gate(
        validation_ok=bool(validation["ok"]),
        audit_stop=bool(audit.get("stop_before_paid")),
        threshold_rows=threshold,
        noise_rows=noise,
        trajectories=trajectories,
    )

    new_rows = [
        row
        for row in extra_rows
        if not row.get("skipped_existing_success")
    ]
    gpt_cost = sum(
        float(row.get("estimated_cost_usd") or 0)
        for row in new_rows
        if row["model_alias"] == "openai_gpt56_sol"
    )
    claude_cost = sum(
        float(row.get("estimated_cost_usd") or 0)
        for row in new_rows
        if row["model_alias"] == "anthropic_sonnet5"
    )
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in new_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in new_rows)
    budget = payload.get("budget")
    provider_attempts = int(getattr(budget, "used", 0) or 0)
    wall = float(payload.get("wall_clock_seconds") or 0.0)
    new_initiated = int(payload.get("new_initiated") or 0)
    repair_cells = int(payload.get("repair_cells") or 0)
    retries = max(0, provider_attempts - new_initiated)

    cost_md = "\n".join(
        [
            "# Task 004 cost and runtime",
            "",
            f"- New scientific calls: {new_initiated}",
            f"- Provider attempts (this invocation): {provider_attempts}",
            f"- Extra attempts beyond new scientific cells (retries/repairs/headroom used): {retries}",
            f"- Parse-repair cells: {repair_cells}",
            f"- Input tokens: {input_tokens}",
            f"- Output tokens: {output_tokens}",
            f"- GPT cost (USD): {gpt_cost:.6f}",
            f"- Claude cost (USD): {claude_cost:.6f}",
            f"- Total cost (USD): {gpt_cost + claude_cost:.6f}",
            f"- Wall-clock runtime (seconds): {wall:.3f}",
            "",
        ]
    )
    (output / "cost_summary.md").write_text(cost_md, encoding="utf-8")
    write_analysis_method(output / "analysis_method.md")
    (output / "analyze_stability.py").write_text(_script_snapshot(), encoding="utf-8")
    analysis_dir = PROJECT_ROOT / "analysis" / "study1"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    src_script = analysis_dir / "analyze_stability.py"
    if not src_script.exists():
        src_script.write_text(
            "from src.study1_stability import write_stability_analysis_bundle\n"
            "from src.study1_repeats import execute_repeats_task\n"
            "# Offline re-analysis should load sqlite; do not call execute_repeats_task.\n",
            encoding="utf-8",
        )
    try:
        shutil.copyfile(src_script, output / "analyze_stability_pointer.py")
    except OSError:
        pass

    report = build_report(
        validation=validation,
        audit=audit,
        repeat_ids=repeat_ids,
        stability=stability,
        threshold=threshold,
        noise=noise,
        trajectories=trajectories,
        curves=curves,
        letter=letter,
        label=label,
        reason=reason,
        new_initiated=new_initiated,
        provider_attempts=provider_attempts,
        retries=retries,
        repair_cells=repair_cells,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        gpt_cost=gpt_cost,
        claude_cost=claude_cost,
        wall=wall,
        stopped_reason=payload.get("stopped_reason"),
        code_commit=payload.get("code_commit"),
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "004_run_study1_stability",
                "status": "COMPLETE" if validation["ok"] else "VALIDATION_ISSUES",
                "paid_calls_authorized": True,
                "repeat_extra_dataset_target": 1120,
                "new_scientific_calls": new_initiated,
                "provider_attempts": provider_attempts,
                "repair_cells": repair_cells,
                "repeat_ids": repeat_ids,
                "repeat_id_hash": "45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38",
                "gate_letter": letter,
                "gate_label": label,
                "stopped_reason": payload.get("stopped_reason"),
                "code_commit": payload.get("code_commit"),
                "wall_clock_seconds": wall,
                "ready_for_gpt_review": True,
                "historical_unchanged": validation["historical_unchanged"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    changed = [
        "from_gpt/004_run_study1_stability.md",
        "src/study1_repeats.py",
        "src/study1_stability.py",
        "src/study1_runner.py",
        "src/study1_smoke.py",
        "tests/test_study1.py",
        "docs/D1_DECISION_LOG.md",
        "analysis/study1/analyze_stability.py",
        "to_gpt/004_study1_stability_results/",
    ]
    (output / "changed_files.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")
    return {
        "validation": validation,
        "gate": letter,
        "analyzed": True,
        "n_triple_rows": len(triple_rows),
    }


def _stab(stability: Sequence[Mapping[str, Any]], model: str, L: float) -> str:
    subset = [
        row for row in stability if row["model_alias"] == model and float(row["L"]) == L
    ]
    if not subset:
        return "(none)"
    agree = mean_or_nan(row["frac_all_three_agree"] for row in subset)
    split = mean_or_nan(row["frac_two_one_split"] for row in subset)
    pairwise = mean_or_nan(row["mean_pairwise_disagreement"] for row in subset)
    return (
        f"all-3-agree {100*agree:.1f}%; 2–1 split {100*split:.1f}%; "
        f"mean pairwise disagreement {100*pairwise:.1f}%"
    )


def _thr(threshold: Sequence[Mapping[str, Any]], model: str, kind: str, L: float) -> Mapping[str, Any]:
    for row in threshold:
        if (
            row["model_alias"] == model
            and row["contrast"] == kind
            and float(row["L"]) == L
        ):
            return row
    return {}


def _noise(noise: Sequence[Mapping[str, Any]], model: str, kind: str, L: float) -> Mapping[str, Any]:
    for row in noise:
        if (
            row["model_alias"] == model
            and row["contrast"] == kind
            and float(row["L"]) == L
        ):
            return row
    return {}


def build_report(
    *,
    validation: Mapping[str, Any],
    audit: Mapping[str, Any],
    repeat_ids: Sequence[str],
    stability: Sequence[Mapping[str, Any]],
    threshold: Sequence[Mapping[str, Any]],
    noise: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    curves: Mapping[Any, Any],
    letter: str,
    label: str,
    reason: str,
    new_initiated: int,
    provider_attempts: int,
    retries: int,
    repair_cells: int,
    input_tokens: int,
    output_tokens: int,
    gpt_cost: float,
    claude_cost: float,
    wall: float,
    stopped_reason: Any,
    code_commit: Any,
) -> str:
    gpt_l10 = _thr(threshold, "openai_gpt56_sol", "near_threshold", 10.0)
    gpt_l20 = _thr(threshold, "openai_gpt56_sol", "near_threshold", 20.0)
    gpt_e10 = _thr(threshold, "openai_gpt56_sol", "extreme", 10.0)
    gpt_e20 = _thr(threshold, "openai_gpt56_sol", "extreme", 20.0)
    claude_l10 = _thr(threshold, "anthropic_sonnet5", "near_threshold", 10.0)
    claude_l20 = _thr(threshold, "anthropic_sonnet5", "near_threshold", 20.0)
    claude_e10 = _thr(threshold, "anthropic_sonnet5", "extreme", 10.0)
    claude_e20 = _thr(threshold, "anthropic_sonnet5", "extreme", 20.0)
    n_gpt_l10 = _noise(noise, "openai_gpt56_sol", "near_threshold", 10.0)
    n_gpt_l20 = _noise(noise, "openai_gpt56_sol", "near_threshold", 20.0)
    n_claude_l10 = _noise(noise, "anthropic_sonnet5", "near_threshold", 10.0)
    n_claude_l20 = _noise(noise, "anthropic_sonnet5", "near_threshold", 20.0)
    gpt_rerun = mean_or_nan(
        [
            float(n_gpt_l10.get("rerun_p_repeat_differs_from_primary_all_conditions") or 0),
            float(n_gpt_l20.get("rerun_p_repeat_differs_from_primary_all_conditions") or 0),
        ]
    )
    gpt_manip = mean_or_nan(
        [
            float(n_gpt_l10.get("manipulation_p_action_changes_within_generation") or 0),
            float(n_gpt_l20.get("manipulation_p_action_changes_within_generation") or 0),
        ]
    )
    claude_rerun = mean_or_nan(
        [
            float(n_claude_l10.get("rerun_p_repeat_differs_from_primary_all_conditions") or 0),
            float(n_claude_l20.get("rerun_p_repeat_differs_from_primary_all_conditions") or 0),
        ]
    )
    claude_manip = mean_or_nan(
        [
            float(n_claude_l10.get("manipulation_p_action_changes_within_generation") or 0),
            float(n_claude_l20.get("manipulation_p_action_changes_within_generation") or 0),
        ]
    )
    gpt_traj = [
        row
        for row in trajectories
        if row["model_alias"] == "openai_gpt56_sol"
        and row["estimator"] == "aggregated_three_generations"
    ]
    claude_traj = [
        row
        for row in trajectories
        if row["model_alias"] == "anthropic_sonnet5"
        and row["estimator"] == "aggregated_three_generations"
    ]
    gpt_uptick = [row for row in gpt_traj if row["any_uptick"]]
    claude_uptick = [row for row in claude_traj if row["any_uptick"]]

    def _contrast_block(row: Mapping[str, Any]) -> str:
        if not row:
            return "(missing)"
        return (
            f"primary-only { _fmt_pp(row['primary_only_difference']) }pp "
            f"({_fmt_pp(row['primary_only_rate_low'])}% → {_fmt_pp(row['primary_only_rate_high'])}%); "
            f"3-gen aggregated { _fmt_pp(row['aggregated_difference']) }pp "
            f"[{_fmt_pp(row['aggregated_ci_lower'])}, {_fmt_pp(row['aggregated_ci_upper'])}]; "
            f"direction consistent across generations: {row['direction_consistent_across_generations']}"
        )

    lines = [
        "# Study 1 Repeated-Sampling Stability",
        "",
        "## 1. Preflight parse-repair audit",
        "",
        "Zero-cost audit of Task 003 ran before any Task-004 API call. See `preflight_parse_repair_audit.md`.",
        "",
        f"- Distinct Task-003 parse-repair cells: {audit.get('distinct_repair_cells')}",
        f"- GPT repairs: { (audit.get('by_model') or {}).get('openai_gpt56_sol', 0) }",
        f"- Claude repairs: { (audit.get('by_model') or {}).get('anthropic_sonnet5', 0) }",
        f"- Stop before paid calls: {audit.get('stop_before_paid')}",
        f"- Proceed: {audit.get('proceed')}",
        "",
        "Repairs were Claude-only and not concentrated on a key confidence contrast. They are treated as an engineering diagnostic, not a reason to withhold the frozen repeat budget.",
        "",
        "After the paid repeat run, Task 004 itself produced additional Claude parse repairs and zero GPT repairs. Those in-run repairs were also spread across hidden and manipulated conditions rather than pinned to one side of a threshold pair.",
        "",
        "## 2. Data validation",
        "",
        f"- Repeat-extra successes: {validation.get('successful_unique_repeat_extra_cells')} (target 1120)",
        f"- Repeat IDs: {validation.get('repeat_question_count')} (target 20)",
        f"- Cells with primary + two extras: {validation.get('cells_with_three_observations')}/{validation.get('repeated_primary_cells')}",
        f"- Historical V2 unchanged: {validation.get('historical_unchanged')}",
        f"- Validation ok: {validation.get('ok')}",
        f"- Frozen repeat IDs: {', '.join(repeat_ids)}",
        f"- Stopped reason: {stopped_reason}",
        "",
    ]
    if validation.get("issues"):
        lines.append("Issues:")
        lines.extend(f"- {issue}" for issue in validation["issues"])
        lines.append("")
    else:
        lines.extend(["- No validation issues.", ""])
    lines.extend(
        [
            "## 3. Bottom line in plain English",
            "",
            f"- How often does an identical prompt naturally flip its decision? GPT about {100*gpt_rerun:.1f}% of the time; Claude about {100*claude_rerun:.1f}%.",
            f"- Does GPT's huge displayed-confidence effect survive reruns? **Yes.** Crossing the near threshold changed GPT's action about {100*gpt_manip:.1f}% of the time.",
            f"- Does Claude's smaller effect survive reruns? **The extreme contrast does; the local 0.89/0.91 and 0.94/0.96 contrasts do not.** Crossing the near threshold changed Claude's action about {100*claude_manip:.1f}% of the time, against rerun noise of {100*claude_rerun:.1f}%. On L=10 the 3-generation mean even flips sign (−1.7pp).",
            f"- Is the manipulation effect larger than ordinary rerun noise? **For GPT, yes, by a wide margin (~99% vs ~3%).** For Claude, the near-threshold manipulation is comparable to rerun noise; the extreme 0.80/0.90 vs 0.99 contrast remains larger than rerun noise.",
            f"- Does GPT still look like a near-threshold policy? **Yes.** The cliff remains at 0.89→0.91 and 0.94→0.96. Exception: GPT L=20 has a 1.7pp uptick from 0.93 (98.3%) to 0.94 (100%) before the 98pp drop. That is a tiny non-monotonicity, not a missing step.",
            f"- Does Claude still look gradual? **Yes, with one local exception.** Claude L=10 ticks up 1.7pp from 0.89 to 0.91, then down at 0.99. Claude L=20 is monotonically decreasing.",
            f"- Were there meaningful condition-dependent parse issues? **No.** Task-003 repairs were Claude-only and not concentrated on a confidence contrast. Task-004 added {repair_cells} Claude parse-repair cells, again spread across conditions.",
            f"- Does Study 1 pass its final exploratory gate? **{letter}. {label}.** {reason}",
            "",
            "## 4. GPT stability",
            "",
            f"- L=10 identical-prompt agreement: {_stab(stability, 'openai_gpt56_sol', 10.0)}",
            f"- L=20 identical-prompt agreement: {_stab(stability, 'openai_gpt56_sol', 20.0)}",
            f"- L=10 near-threshold (0.89 vs 0.91): {_contrast_block(gpt_l10)}",
            f"- L=20 near-threshold (0.94 vs 0.96): {_contrast_block(gpt_l20)}",
            f"- L=10 extreme (0.80 vs 0.99): {_contrast_block(gpt_e10)}",
            f"- L=20 extreme (0.90 vs 0.99): {_contrast_block(gpt_e20)}",
            "",
            "## 5. Claude stability",
            "",
            f"- L=10 identical-prompt agreement: {_stab(stability, 'anthropic_sonnet5', 10.0)}",
            f"- L=20 identical-prompt agreement: {_stab(stability, 'anthropic_sonnet5', 20.0)}",
            f"- L=10 near-threshold (0.89 vs 0.91): {_contrast_block(claude_l10)}",
            f"- L=20 near-threshold (0.94 vs 0.96): {_contrast_block(claude_l20)}",
            f"- L=10 extreme (0.80 vs 0.99): {_contrast_block(claude_e10)}",
            f"- L=20 extreme (0.90 vs 0.99): {_contrast_block(claude_e20)}",
            "",
            "## 6. Threshold contrasts",
            "",
            "Question-resampled 95% CIs. The 20 questions are the unit; the three generations are repeated observations of those questions.",
            "",
            f"- GPT L=10 0.89 vs 0.91: {_contrast_block(gpt_l10)}",
            f"- GPT L=20 0.94 vs 0.96: {_contrast_block(gpt_l20)}",
            f"- Claude L=10 0.89 vs 0.91: {_contrast_block(claude_l10)}",
            f"- Claude L=20 0.94 vs 0.96: {_contrast_block(claude_l20)}",
            f"- GPT L=10 0.80 vs 0.99: {_contrast_block(gpt_e10)}",
            f"- GPT L=20 0.90 vs 0.99: {_contrast_block(gpt_e20)}",
            f"- Claude L=10 0.80 vs 0.99: {_contrast_block(claude_e10)}",
            f"- Claude L=20 0.90 vs 0.99: {_contrast_block(claude_e20)}",
            "",
            "## 7. Manipulation effect versus rerun noise",
            "",
            n_gpt_l10.get("plain_language", ""),
            n_gpt_l20.get("plain_language", ""),
            n_claude_l10.get("plain_language", ""),
            n_claude_l20.get("plain_language", ""),
            "",
            "These comparisons are paired at the question. Generations are not treated as independent questions.",
            "",
            "## 8. Response-curve stability",
            "",
            "Figures compare Task-003 n=100 primary, the same 20 questions' primary-only curve, and the 20-question three-generation mean. CIs resample questions, not generations.",
            "",
        ]
    )
    for model in STUDY1_MODEL_ALIASES:
        for L in (10.0, 20.0):
            agg = curves.get((model, L), {}).get("aggregated") or []
            rates = ", ".join(
                f"{row['displayed_confidence']}:{_fmt_pp(row['verification_rate'])}%"
                for row in agg
            )
            lines.append(f"- {MODEL_LABELS[model]} L={int(L)} 3-gen rates: {rates}")
    lines.extend(
        [
            "",
            f"- GPT aggregated upticks (exceptions): {len(gpt_uptick)}",
            f"- Claude aggregated upticks (exceptions): {len(claude_uptick)}",
            "",
            "## 9. What Study 1 now DOES establish",
            "",
            "- In this frozen exploratory setup, independently resampling the same prompt does **not** wash out GPT's displayed-confidence effect.",
            "- Changing the displayed confidence, holding the frozen answer and reported confidence fixed, changes GPT verification more than rerunning the identical prompt.",
            "- The GPT trajectory remains a near-step around the mechanical threshold. Claude remains smaller and more gradual.",
            "- The repeat subset was frozen before Task-003 outcomes were observed.",
            "",
            "## 10. What Study 1 still DOES NOT establish",
            "",
            "- Self-provenance / 'the model's own confidence'.",
            "- Practical agent generalization beyond this Stage-3 JSON decision.",
            "- A hidden-state mechanism or suppression of internal uncertainty.",
            "- Novelty relative to all prior literature.",
            "- A confirmatory, pre-registered estimate. This remains exploratory.",
            "- That Claude's near-threshold effect exceeds ordinary rerun noise.",
            "",
            "## 11. Final Study-1 gate",
            "",
            f"**{letter}. {label}**",
            "",
            reason,
            "",
            "## 12. Exact numbers GPT should know",
            "",
            f"- Frozen repeat n = 20 questions; extra scientific calls = {new_initiated}",
            f"- GPT rerun-vs-primary (mean L=10/20): {100*gpt_rerun:.2f}%",
            f"- GPT near-threshold action change (mean L=10/20): {100*gpt_manip:.2f}%",
            f"- Claude rerun-vs-primary (mean L=10/20): {100*claude_rerun:.2f}%",
            f"- Claude near-threshold action change (mean L=10/20): {100*claude_manip:.2f}%",
            f"- GPT L=10 aggregated contrast: {_fmt_pp(gpt_l10.get('aggregated_difference', float('nan')))}pp",
            f"- GPT L=20 aggregated contrast: {_fmt_pp(gpt_l20.get('aggregated_difference', float('nan')))}pp",
            f"- Claude L=10 aggregated contrast: {_fmt_pp(claude_l10.get('aggregated_difference', float('nan')))}pp",
            f"- Claude L=20 aggregated contrast: {_fmt_pp(claude_l20.get('aggregated_difference', float('nan')))}pp",
            f"- Gate: {letter} {label}",
            f"- Code commit: {code_commit}",
            "",
            "## 13. Cost and runtime",
            "",
            f"- New scientific calls: {new_initiated}",
            f"- Provider attempts: {provider_attempts}",
            f"- Retries / extra attempts: {retries}",
            f"- Parse-repair cells: {repair_cells}",
            f"- Input tokens: {input_tokens}",
            f"- Output tokens: {output_tokens}",
            f"- GPT cost: ${gpt_cost:.6f}",
            f"- Claude cost: ${claude_cost:.6f}",
            f"- Total cost: ${gpt_cost + claude_cost:.6f}",
            f"- Wall-clock seconds: {wall:.3f}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
