"""Shared qualitative summaries for Task 007 Lanes A and B."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .study1_analysis import MODEL_LABELS
from .task006_common import SCORE_CONDITIONS, STAKES_FAMILIES
from .task007_common import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED

VISIBLE = ("displayed_0.70", "displayed_0.90", "displayed_0.99")
PAIRS = ((0.70, 0.90), (0.90, 0.99), (0.70, 0.99))


def _boot_mean(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(BOOTSTRAP_SEED)
    stats: list[float] = []
    arr = list(values)
    for _ in range(BOOTSTRAP_RESAMPLES):
        draw = [arr[rng.randrange(len(arr))] for _ in arr]
        stats.append(sum(draw) / len(draw))
    stats.sort()
    mean = sum(arr) / len(arr)
    return mean, stats[int(0.025 * (len(stats) - 1))], stats[int(0.975 * (len(stats) - 1))]


def _boot_question_rate(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float]:
    by_q = {row["question_id"]: int(row["verify"]) for row in rows}
    return _boot_mean(list(by_q.values()))


def rate_table(table: Sequence[Mapping[str, Any]], aliases: Sequence[str]) -> list[dict[str, Any]]:
    output = []
    by_cell: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in table:
        by_cell[(row["model_alias"], row["family"], row["score_condition"])].append(row)
    for model in aliases:
        for family in STAKES_FAMILIES:
            for condition in SCORE_CONDITIONS:
                cell = by_cell[(model, family, condition)]
                n_v = sum(int(r["verify"]) for r in cell)
                mean, lo, hi = _boot_question_rate(cell) if cell else (float("nan"),) * 3
                output.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "family": family,
                        "score_condition": condition,
                        "n": len(cell),
                        "n_verify": n_v,
                        "n_use": len(cell) - n_v,
                        "verify_rate": n_v / len(cell) if cell else float("nan"),
                        "ci_lower": lo,
                        "ci_upper": hi,
                    }
                )
    return output


def score_response_table(
    table: Sequence[Mapping[str, Any]], aliases: Sequence[str]
) -> list[dict[str, Any]]:
    output = []
    for model in aliases:
        for family in STAKES_FAMILIES:
            by_q: dict[str, dict[float, int]] = defaultdict(dict)
            for row in table:
                if row["model_alias"] != model or row["family"] != family:
                    continue
                if row["displayed_confidence"] is None:
                    continue
                by_q[row["question_id"]][float(row["displayed_confidence"])] = int(row["verify"])
            diffs = {pair: [] for pair in PAIRS}
            for scores in by_q.values():
                if set(scores) != {0.70, 0.90, 0.99}:
                    continue
                for lo, hi in PAIRS:
                    diffs[(lo, hi)].append(scores[hi] - scores[lo])
            for lo, hi in PAIRS:
                vals = diffs[(lo, hi)]
                mean, lo_ci, hi_ci = _boot_mean(vals)
                output.append(
                    {
                        "model_alias": model,
                        "family": family,
                        "from_score": lo,
                        "to_score": hi,
                        "n_questions": len(vals),
                        "mean_delta_verify": mean,
                        "ci_lower": lo_ci,
                        "ci_upper": hi_ci,
                        "n_increased": sum(1 for v in vals if v > 0),
                        "n_decreased": sum(1 for v in vals if v < 0),
                        "n_unchanged": sum(1 for v in vals if v == 0),
                    }
                )
    return output


def stakes_table(
    table: Sequence[Mapping[str, Any]], aliases: Sequence[str]
) -> list[dict[str, Any]]:
    output = []
    for model in aliases:
        for condition in SCORE_CONDITIONS:
            by_q: dict[str, dict[str, int]] = defaultdict(dict)
            for row in table:
                if row["model_alias"] != model or row["score_condition"] != condition:
                    continue
                by_q[row["question_id"]][row["family"]] = int(row["verify"])
            deltas = []
            n_stronger_only = n_moderate_only = 0
            for acts in by_q.values():
                if "moderate" not in acts or "stronger" not in acts:
                    continue
                deltas.append(acts["stronger"] - acts["moderate"])
                n_stronger_only += int(acts["stronger"] == 1 and acts["moderate"] == 0)
                n_moderate_only += int(acts["moderate"] == 1 and acts["stronger"] == 0)
            mean, lo, hi = _boot_mean(deltas)
            output.append(
                {
                    "model_alias": model,
                    "score_condition": condition,
                    "n": len(deltas),
                    "mean_stronger_minus_moderate": mean,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_stronger_only": n_stronger_only,
                    "n_moderate_only": n_moderate_only,
                }
            )
    return output


def split_pilot_vs_remaining(
    merged: Sequence[Mapping[str, Any]],
    remaining_ids: Sequence[str],
    repeat_ids: Sequence[str],
    aliases: Sequence[str],
) -> list[dict[str, Any]]:
    remaining_set = set(remaining_ids)
    repeat_set = set(repeat_ids)
    output = []
    for subset_name, wanted in (("pilot20", repeat_set), ("remaining80", remaining_set)):
        sub = [row for row in merged if row["question_id"] in wanted]
        rates = rate_table(sub, aliases)
        scores = score_response_table(sub, aliases)
        for row in rates:
            output.append({**row, "subset": subset_name, "kind": "rate"})
        for row in scores:
            output.append({**row, "subset": subset_name, "kind": "score_delta"})
    # direction replication: sign of 0.70→0.99 on remaining vs pilot
    for model in aliases:
        for family in STAKES_FAMILIES:
            bits = {}
            for subset_name in ("pilot20", "remaining80"):
                hits = [
                    row
                    for row in output
                    if row.get("kind") == "score_delta"
                    and row["subset"] == subset_name
                    and row["model_alias"] == model
                    and row["family"] == family
                    and abs(float(row["from_score"]) - 0.70) < 1e-12
                    and abs(float(row["to_score"]) - 0.99) < 1e-12
                ]
                bits[subset_name] = hits[0]["mean_delta_verify"] if hits else float("nan")
            output.append(
                {
                    "kind": "direction_replication",
                    "model_alias": model,
                    "family": family,
                    "pilot20_0.70_to_0.99": bits.get("pilot20"),
                    "remaining80_0.70_to_0.99": bits.get("remaining80"),
                    "same_sign": (
                        math.isfinite(bits.get("pilot20", float("nan")))
                        and math.isfinite(bits.get("remaining80", float("nan")))
                        and (
                            (bits["pilot20"] < 0 and bits["remaining80"] < 0)
                            or (bits["pilot20"] > 0 and bits["remaining80"] > 0)
                            or (bits["pilot20"] == 0 and bits["remaining80"] == 0)
                        )
                    ),
                }
            )
    return output


def saturation_table(
    table: Sequence[Mapping[str, Any]], aliases: Sequence[str]
) -> list[dict[str, Any]]:
    output = []
    by_cell: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in table:
        by_cell[(row["model_alias"], row["family"], row["score_condition"])].append(row)
    for model in aliases:
        for family in STAKES_FAMILIES:
            for condition in SCORE_CONDITIONS:
                cell = by_cell[(model, family, condition)]
                n_v = sum(int(r["verify"]) for r in cell)
                rate = n_v / len(cell) if cell else float("nan")
                both_actions = 0 < n_v < len(cell)
                output.append(
                    {
                        "model_alias": model,
                        "family": family,
                        "score_condition": condition,
                        "n": len(cell),
                        "verify_rate": rate,
                        "both_actions_present": both_actions,
                        "useful_intermediate_10_90": bool(
                            math.isfinite(rate) and 0.10 <= rate <= 0.90
                        ),
                        "label": (
                            ""
                            if both_actions
                            else "ITEM_ROUTING_NOT_IDENTIFIABLE_DUE_TO_ACTION_SATURATION"
                        ),
                    }
                )
    return output


def classify_score_response(
    score_rows: Sequence[Mapping[str, Any]], model: str
) -> dict[str, Any]:
    deltas = [
        float(row["mean_delta_verify"])
        for row in score_rows
        if row["model_alias"] == model
        and abs(float(row["from_score"]) - 0.70) < 1e-12
        and abs(float(row["to_score"]) - 0.99) < 1e-12
        and math.isfinite(float(row["mean_delta_verify"]))
    ]
    if not deltas:
        return {
            "model_alias": model,
            "class": "BLOCKED",
            "max_abs_070_to_099": float("nan"),
            "reason": "no paired 0.70→0.99 cells",
        }
    max_abs = max(abs(v) for v in deltas)
    if max_abs >= 0.25:
        label = "QUALITATIVE_SCORE_RESPONSE_STRONG"
    elif max_abs >= 0.10:
        label = "QUALITATIVE_SCORE_RESPONSE_MODERATE"
    else:
        label = "QUALITATIVE_SCORE_RESPONSE_WEAK_OR_NULL"
    return {
        "model_alias": model,
        "class": label,
        "max_abs_070_to_099": max_abs,
        "signed_deltas": deltas,
        "reason": f"max |0.70→0.99 VERIFY Δ| = {max_abs:.3f}",
    }
