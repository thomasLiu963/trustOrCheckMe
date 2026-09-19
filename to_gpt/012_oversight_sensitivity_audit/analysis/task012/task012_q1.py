"""Section 6–7: natural q1 range, calibration, hidden / true-q markers."""

from __future__ import annotations

from typing import Any

from .task012_common import ECE_BINS, GPT_CLAUDE, LABEL, write_csv
from .task012_data import ItemRow, filter_rows
from .task012_stats import (
    brier,
    coverage,
    ece,
    error_catch,
    leakage,
    percentiles,
    verify_error_precision,
)

INTERVALS = (("0.70_0.85", 0.70, 0.85), ("0.85_0.90", 0.85, 0.90), ("0.90_0.95", 0.90, 0.95), ("0.95_0.99", 0.95, 0.99))


def _q1_items(rows: list[ItemRow], task: str, model: str) -> list[ItemRow]:
    seen: set[str] = set()
    out: list[ItemRow] = []
    for row in filter_rows(rows, task=task, model=model, condition="displayed_0.70"):
        if row.question_id in seen or row.q1 is None:
            continue
        seen.add(row.question_id)
        out.append(row)
    return out


def _frac(values: list[float], lo: float, hi: float, right_closed: bool) -> float:
    if not values:
        return float("nan")
    if right_closed:
        return sum(lo <= v <= hi for v in values) / len(values)
    return sum(lo <= v < hi for v in values) / len(values)


def _slice_summary(task: str, model: str, subset: str, items: list[ItemRow]) -> dict[str, Any]:
    vals = [float(r.q1) for r in items if r.q1 is not None]
    pct = percentiles(vals)
    return {
        "label": LABEL,
        "task": task,
        "model_alias": model,
        "subset": subset,
        "n": len(vals),
        **pct,
        "frac_in_0.70_0.99": _frac(vals, 0.70, 0.99, True),
        "frac_0.70_0.85": _frac(vals, 0.70, 0.85, False),
        "frac_0.85_0.90": _frac(vals, 0.85, 0.90, False),
        "frac_0.90_0.95": _frac(vals, 0.90, 0.95, False),
        "frac_0.95_0.99": _frac(vals, 0.95, 0.99, True),
        "iqr_intersects_0.70_0.99": bool(pct["p25"] <= 0.99 and pct["p75"] >= 0.70),
    }


def run_q1_distribution(rows: list[ItemRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            items = _q1_items(rows, task, model)
            out.append(_slice_summary(task, model, "all", items))
            out.append(_slice_summary(task, model, "correct", [r for r in items if not r.incorrect]))
            out.append(_slice_summary(task, model, "incorrect", [r for r in items if r.incorrect]))
    return out


def run_q1_calibration(rows: list[ItemRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            items = _q1_items(rows, task, model)
            q1 = [float(r.q1) for r in items if r.q1 is not None]
            correct = [not r.incorrect for r in items if r.q1 is not None]
            if not q1:
                continue
            score, table = ece(q1, correct, ECE_BINS)
            inc = [float(r.q1) for r in items if r.incorrect and r.q1 is not None]
            cor = [float(r.q1) for r in items if not r.incorrect and r.q1 is not None]
            for bin_row in table:
                out.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "brier": brier(q1, correct),
                        "ece": score,
                        "mean_q1_incorrect": sum(inc) / len(inc) if inc else float("nan"),
                        "mean_q1_correct": sum(cor) / len(cor) if cor else float("nan"),
                        **bin_row,
                    }
                )
    return out


def run_natural_vs_counterfactual(rows: list[ItemRow], leakage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {(r["task"], r["model_alias"], r["score_condition"]): r for r in leakage_rows}
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            items = _q1_items(rows, task, model)
            mean_q1 = sum(float(r.q1) for r in items if r.q1 is not None) / max(1, len(items))
            for cond in ("hidden", "true_q_visible", "displayed_0.70", "displayed_0.85", "displayed_0.90", "displayed_0.95", "displayed_0.99"):
                src = by[(task, model, cond)]
                subset = filter_rows(rows, task=task, model=model, condition=cond)
                out.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "score_condition": cond,
                        "marker_only": cond in {"hidden", "true_q_visible"},
                        "do_not_plot_hidden_at_x": cond == "hidden",
                        "true_q_not_equivalent_to_mean_q1": cond == "true_q_visible",
                        "mean_natural_q1": mean_q1,
                        "verification_coverage": coverage(subset),
                        "unverified_error_rate": leakage(subset),
                        "error_catch_rate": error_catch(subset),
                        "verify_error_precision": verify_error_precision(subset),
                        "coverage_ci_lo": src["coverage_ci_lo"],
                        "coverage_ci_hi": src["coverage_ci_hi"],
                        "leakage_ci_lo": src["leakage_ci_lo"],
                        "leakage_ci_hi": src["leakage_ci_hi"],
                    }
                )
    return out


def write_q1(dist_path, cal_path, nat_path, dist, cal, nat) -> None:
    write_csv(dist_path, dist)
    write_csv(cal_path, cal)
    write_csv(nat_path, nat)
