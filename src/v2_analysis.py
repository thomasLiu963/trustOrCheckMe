"""Paired V2 analysis and paper-oriented tabular outputs."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bootstrap import paired_bootstrap
from .calibration import (
    assign_calibrated_probabilities,
    brier_score,
    deterministic_model_splits,
    fit_per_model_calibrators,
    persist_calibration_artifacts,
)
from .v2_scoring import (
    USE_UNVERIFIED,
    VERIFY_FIRST,
    confidence_policy_action,
    score_verification_decision,
    v2_monotonicity,
    validate_factor_completeness,
)
from .v2_plotting import generate_v2_figures


@dataclass(frozen=True)
class V2AnalysisResult:
    scored_rows: list[dict[str, Any]]
    model_summary: list[dict[str, Any]]
    factorial_metrics: list[dict[str, Any]]
    owner_effects: list[dict[str, Any]]
    confidence_visibility_effects: list[dict[str, Any]]
    owner_confidence_interactions: list[dict[str, Any]]
    policy_comparison: list[dict[str, Any]]
    prompt_robustness: list[dict[str, Any]]
    completeness_issues: list[dict[str, Any]]
    output_directory: Path


def _load_records(path: str | Path, stage: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(Path(path))
    try:
        rows = connection.execute(
            """
            SELECT record_json FROM requests
            WHERE stage = ? AND status = 'success' AND record_json IS NOT NULL
            ORDER BY example_id, model_alias, request_key
            """,
            (stage,),
        ).fetchall()
    finally:
        connection.close()
    return [json.loads(row[0]) for row in rows]


def _mean(values: Iterable[Any]) -> float:
    usable: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            usable.append(number)
    return sum(usable) / len(usable) if usable else float("nan")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _write_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _paired_values(
    rows: Iterable[Mapping[str, Any]],
    *,
    factor_key: str,
    left: str,
    right: str,
    value,
) -> list[tuple[str, Any, Any]]:
    paired: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        factor = row.get(factor_key)
        if factor in {left, right}:
            paired[str(row["example_id"])][str(factor)] = value(row)
    return [
        (identifier, values[left], values[right])
        for identifier, values in paired.items()
        if left in values and right in values
    ]


def _bootstrap_dict(result) -> dict[str, Any]:
    return {
        "estimate": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "n_questions": result.n_questions,
        "n_valid_resamples": result.n_valid_resamples,
    }


def _factorial_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "model_id",
        "prompt_family",
        "decision_owner",
        "confidence_visibility",
        "error_cost",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: repr(item[0])):
        wrong = [row for row in group if not row["is_correct"]]
        correct = [row for row in group if row["is_correct"]]
        output.append(
            {
                **dict(zip(keys, key)),
                "n": len(group),
                "wrong_n": len(wrong),
                "correct_n": len(correct),
                "verify_rate": _mean(row["verified_first"] for row in group),
                "unsafe_unverified_rate": _mean(
                    row["used_unverified"] for row in wrong
                ),
                "unnecessary_verification_rate": _mean(
                    row["verified_first"] for row in correct
                ),
                "mean_realized_cost": _mean(
                    row["realized_cost"] for row in group
                ),
                "mean_regret": _mean(row["regret"] for row in group),
                "direct_vs_raw_disagreement_rate": _mean(
                    row["direct_vs_raw_confidence_disagreement"] for row in group
                ),
            }
        )
    return output


def _owner_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, Any]]:
    keys = ("model_id", "prompt_family", "confidence_visibility", "error_cost")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for index, (key, group) in enumerate(
        sorted(grouped.items(), key=lambda item: repr(item[0]))
    ):
        pairs = _paired_values(
            group,
            factor_key="decision_owner",
            left="ai_system",
            right="human",
            value=lambda row: float(row["verified_first"]),
        )
        effect = paired_bootstrap(
            pairs,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed + index,
        )
        disagreement = paired_bootstrap(
            pairs,
            difference=lambda ai, human: float(ai != human),
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed + 10_000 + index,
        )
        wrong = [row for row in group if not row["is_correct"]]
        unsafe_pairs = _paired_values(
            wrong,
            factor_key="decision_owner",
            left="human",
            right="ai_system",
            value=lambda row: float(row["action"] == USE_UNVERIFIED),
        )
        unsafe = paired_bootstrap(
            unsafe_pairs,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed + 20_000 + index,
        )
        output.append(
            {
                **dict(zip(keys, key)),
                **{
                    f"owner_effect_{name}": value
                    for name, value in _bootstrap_dict(effect).items()
                },
                "paired_disagreement": disagreement.estimate,
                "paired_disagreement_ci_lower": disagreement.lower,
                "paired_disagreement_ci_upper": disagreement.upper,
                "unsafe_owner_gap": unsafe.estimate,
                "unsafe_owner_gap_ci_lower": unsafe.lower,
                "unsafe_owner_gap_ci_upper": unsafe.upper,
                "wrong_pair_n": unsafe.n_questions,
            }
        )
    return output


def _confidence_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, Any]]:
    keys = ("model_id", "prompt_family", "decision_owner", "error_cost")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for index, (key, group) in enumerate(
        sorted(grouped.items(), key=lambda item: repr(item[0]))
    ):
        pairs = _paired_values(
            group,
            factor_key="confidence_visibility",
            left="visible",
            right="hidden",
            value=lambda row: float(row["verified_first"]),
        )
        effect = paired_bootstrap(
            pairs,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed + index,
        )
        output.append({**dict(zip(keys, key)), **_bootstrap_dict(effect)})
    return output


def _interactions(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> list[dict[str, Any]]:
    keys = ("model_id", "prompt_family", "error_cost")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for index, (key, group) in enumerate(
        sorted(grouped.items(), key=lambda item: repr(item[0]))
    ):
        cells: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
        for row in group:
            cells[str(row["example_id"])][
                (str(row["decision_owner"]), str(row["confidence_visibility"]))
            ] = float(row["verified_first"])
        pairs = []
        for identifier, values in cells.items():
            required = {
                ("human", "hidden"),
                ("ai_system", "hidden"),
                ("human", "visible"),
                ("ai_system", "visible"),
            }
            if not required <= values.keys():
                continue
            hidden_gap = (
                values[("ai_system", "hidden")] - values[("human", "hidden")]
            )
            visible_gap = (
                values[("ai_system", "visible")] - values[("human", "visible")]
            )
            pairs.append((identifier, visible_gap, hidden_gap))
        interaction = paired_bootstrap(
            pairs,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            seed=seed + index,
        )
        output.append({**dict(zip(keys, key)), **_bootstrap_dict(interaction)})
    return output


def _policy_comparison(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = (
        "model_id",
        "prompt_family",
        "decision_owner",
        "confidence_visibility",
        "error_cost",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: repr(item[0])):
        policies: dict[str, list[float]] = defaultdict(list)
        for row in group:
            is_correct = bool(row["is_correct"])
            loss = float(row["error_cost"])
            cost = float(row["verification_cost"])
            policies["direct"].append(float(row["realized_cost"]))
            raw = confidence_policy_action(
                row["probability_correct"], loss, cost
            )
            policies["raw_confidence"].append(
                cost if raw == VERIFY_FIRST else (0.0 if is_correct else loss)
            )
            if row.get("calibrated_probability") is not None:
                calibrated = confidence_policy_action(
                    row["calibrated_probability"], loss, cost
                )
                policies["calibrated_confidence"].append(
                    cost
                    if calibrated == VERIFY_FIRST
                    else (0.0 if is_correct else loss)
                )
            policies["always_use_unverified"].append(
                0.0 if is_correct else loss
            )
            policies["always_verify"].append(cost)
            policies["oracle"].append(0.0 if is_correct else cost)
        for policy, costs in sorted(policies.items()):
            output.append(
                {
                    **dict(zip(keys, key)),
                    "policy": policy,
                    "n": len(costs),
                    "mean_cost": _mean(costs),
                }
            )
    return output


def _owner_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    keys = (
        "example_id",
        "model_id",
        "prompt_family",
        "confidence_visibility",
        "error_cost",
    )
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)][
            str(row["decision_owner"])
        ] = row
    disagreements: list[dict[str, Any]] = []
    for key, owners in sorted(grouped.items(), key=lambda item: repr(item[0])):
        if not {"human", "ai_system"} <= owners.keys():
            continue
        human = owners["human"]
        ai = owners["ai_system"]
        if human["action"] == ai["action"]:
            continue
        disagreements.append(
            {
                **dict(zip(keys, key)),
                "is_correct": human["is_correct"],
                "probability_correct": human["probability_correct"],
                "human_action": human["action"],
                "ai_action": ai["action"],
            }
        )
    return {
        "owner_disagreements": disagreements,
        "wrong_human_unverified_ai_verified": [
            row
            for row in disagreements
            if not row["is_correct"]
            and row["human_action"] == USE_UNVERIFIED
            and row["ai_action"] == VERIFY_FIRST
        ],
        "wrong_ai_unverified_human_verified": [
            row
            for row in disagreements
            if not row["is_correct"]
            and row["ai_action"] == USE_UNVERIFIED
            and row["human_action"] == VERIFY_FIRST
        ],
        "direct_vs_confidence_visible_disagreements": [
            dict(row)
            for row in rows
            if row["confidence_visibility"] == "visible"
            and row["direct_vs_raw_confidence_disagreement"]
        ],
    }


def _prompt_robustness(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = (
        "example_id",
        "model_id",
        "decision_owner",
        "confidence_visibility",
        "error_cost",
    )
    paired: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[tuple(row.get(key) for key in keys)][
            str(row["prompt_family"])
        ] = row
    grouped: dict[tuple[Any, ...], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    for key, families in paired.items():
        if {
            "v2_owner_match_v1",
            "v2_owner_match_paraphrase_v1",
        } <= families.keys():
            grouped[key[1:]].append(
                (
                    families["v2_owner_match_v1"],
                    families["v2_owner_match_paraphrase_v1"],
                )
            )
    output = []
    group_keys = keys[1:]
    for key, pairs in sorted(grouped.items(), key=lambda item: repr(item[0])):
        output.append(
            {
                **dict(zip(group_keys, key)),
                "n": len(pairs),
                "action_agreement": _mean(
                    primary["action"] == paraphrase["action"]
                    for primary, paraphrase in pairs
                ),
                "primary_verify_rate": _mean(
                    primary["action"] == VERIFY_FIRST
                    for primary, _ in pairs
                ),
                "paraphrase_verify_rate": _mean(
                    paraphrase["action"] == VERIFY_FIRST
                    for _, paraphrase in pairs
                ),
            }
        )
    return output


def analyze_v2(
    checkpoint: str | Path,
    *,
    output_directory: str | Path,
    seed: int = 20260904,
    calibration_fraction: float = 0.20,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    make_plots: bool = True,
) -> V2AnalysisResult:
    """Analyze completed V2 records without mutating the checkpoint."""
    answers = _load_records(checkpoint, "answer")
    decisions = _load_records(checkpoint, "verification")
    if not decisions:
        raise ValueError("No completed V2 verification decisions were found")
    answer_by_key = {
        (str(row["example_id"]), str(row["model_id"])): row for row in answers
    }
    joined: list[dict[str, Any]] = []
    for decision in decisions:
        answer = answer_by_key.get(
            (str(decision["example_id"]), str(decision["model_id"]))
        )
        if answer is None:
            raise ValueError(
                "Verification record has no matching frozen answer: "
                f"{decision['example_id']} / {decision['model_id']}"
            )
        joined.append(
            {
                **decision,
                "is_correct": bool(answer["is_correct"]),
                "category": answer.get("category"),
            }
        )

    unique_for_calibration = {
        (str(row["example_id"]), str(row["model_id"])): {
            "example_id": row["example_id"],
            "model_id": row["model_id"],
            "probability_correct": row["probability_correct"],
            "is_correct": row["is_correct"],
        }
        for row in joined
    }
    calibration_rows = list(unique_for_calibration.values())
    splits = deterministic_model_splits(
        calibration_rows,
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    calibrators, calibrator_metadata = fit_per_model_calibrators(
        calibration_rows, splits
    )
    calibrated_rows = assign_calibrated_probabilities(
        calibration_rows, splits, calibrators, evaluation_only=True
    )
    calibrated_by_key = {
        (str(row["example_id"]), str(row["model_id"])): row
        for row in calibrated_rows
    }

    scored: list[dict[str, Any]] = []
    for row in joined:
        calibration = calibrated_by_key[
            (str(row["example_id"]), str(row["model_id"]))
        ]
        scores = score_verification_decision(
            is_correct=row["is_correct"],
            action=row["action"],
            probability_correct=row["probability_correct"],
            error_cost=row["error_cost"],
            verification_cost=row["verification_cost"],
            calibrated_probability=calibration["calibrated_probability"],
        )
        scored.append(
            {
                **row,
                **scores,
                "calibration_partition": calibration["calibration_partition"],
                "calibrated_probability": calibration["calibrated_probability"],
            }
        )

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        by_model[str(row["model_id"])].append(row)
    model_summary = [
        {
            "model_id": model,
            "n_answers": len(rows),
            "wrong_n": sum(not row["is_correct"] for row in rows),
            "accuracy": _mean(row["is_correct"] for row in rows),
            "brier_score": brier_score(
                [row["probability_correct"] for row in rows],
                [row["is_correct"] for row in rows],
            ),
        }
        for model, rows in sorted(by_model.items())
    ]

    primary = [
        row for row in scored if row["prompt_family"] == "v2_owner_match_v1"
    ]
    completeness = validate_factor_completeness(primary)
    factorial = _factorial_metrics(scored)
    owners = _owner_effects(
        scored,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    confidence = _confidence_effects(
        scored,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed + 30_000,
    )
    interactions = _interactions(
        scored,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed + 40_000,
    )
    policy_comparison = _policy_comparison(scored)
    monotonicity = v2_monotonicity(scored)
    diagnostics = _owner_diagnostics(scored)
    prompt_robustness = _prompt_robustness(scored)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "scored_decisions": scored,
        "model_summary": model_summary,
        "factorial_metrics": factorial,
        "owner_effects": owners,
        "confidence_visibility_effects": confidence,
        "owner_confidence_interactions": interactions,
        "policy_comparison": policy_comparison,
        "prompt_robustness": prompt_robustness,
        "factor_completeness_issues": completeness,
        "monotonicity": monotonicity,
    }
    for name, rows in artifacts.items():
        _write_csv(output / f"{name}.csv", rows)
        _write_json(output / f"{name}.json", rows)
    for name, rows in diagnostics.items():
        _write_csv(output / "diagnostics" / f"{name}.csv", rows)
    persist_calibration_artifacts(
        output / "calibration", splits, calibrator_metadata
    )
    _write_json(
        output / "analysis_manifest.json",
        {
            "experiment_version": "v2",
            "checkpoint": str(checkpoint),
            "decision_count": len(scored),
            "factor_completeness_issue_count": len(completeness),
            "n_resamples": n_resamples,
            "confidence_level": confidence_level,
            "seed": seed,
        },
    )
    (output / "README_RESULTS.md").write_text(
        "# V2 results\n\n"
        f"- Completed verification decisions: {len(scored)}\n"
        f"- Models represented: {len(model_summary)}\n"
        f"- Primary factor-completeness issues: {len(completeness)}\n"
        f"- Bootstrap resamples: {n_resamples}\n"
        f"- Confidence level: {confidence_level:.0%}\n\n"
        "Machine-readable CSV and JSON tables in this directory are the "
        "canonical analysis outputs. Interpret effects under the claim limits "
        "in `EXPERIMENT_V2.md`.\n",
        encoding="utf-8",
    )
    if make_plots:
        generate_v2_figures(
            factorial_metrics=factorial,
            owner_effects=owners,
            interactions=interactions,
            policy_comparison=policy_comparison,
            output_directory=output,
        )
    return V2AnalysisResult(
        scored_rows=scored,
        model_summary=model_summary,
        factorial_metrics=factorial,
        owner_effects=owners,
        confidence_visibility_effects=confidence,
        owner_confidence_interactions=interactions,
        policy_comparison=policy_comparison,
        prompt_robustness=prompt_robustness,
        completeness_issues=completeness,
        output_directory=output,
    )
