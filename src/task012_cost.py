"""Sections 8–9 and optional H: cost/regret, marginal efficiency, matched-budget."""

from __future__ import annotations

import ast
import json
from typing import Any

from .task012_common import (
    FIXED_CONDS,
    GPT_CLAUDE,
    LABEL,
    LAMBDA_GRID,
    TASK011_DIR,
    write_csv,
    load_csv,
)
from .task012_data import ItemRow, filter_rows
from .task012_stats import coverage, error_rate, leakage


def _loss(leak: float, cov: float, lam: float) -> float:
    return leak + lam * cov


def _oracle(err: float, lam: float) -> tuple[float, float, float]:
    if lam < 1.0:
        return 0.0, err, _loss(0.0, err, lam)
    return err, 0.0, _loss(err, 0.0, lam)


def run_cost_sweep(rows: list[ItemRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            err = error_rate(filter_rows(rows, task=task, model=model, condition="displayed_0.70"))
            cond_stats = {}
            for cond in ("hidden", "true_q_visible") + FIXED_CONDS:
                subset = filter_rows(rows, task=task, model=model, condition=cond)
                cond_stats[cond] = (leakage(subset), coverage(subset))
            for lam in LAMBDA_GRID:
                fixed_losses = {c: _loss(*cond_stats[c], lam) for c in FIXED_CONDS}
                best_fixed = min(fixed_losses.values())
                o_leak, o_cov, o_loss = _oracle(err, lam)
                for cond, (leak, cov) in cond_stats.items():
                    loss = _loss(leak, cov, lam)
                    out.append(
                        {
                            "label": LABEL,
                            "task": task,
                            "model_alias": model,
                            "score_condition": cond,
                            "lambda": lam,
                            "leakage_rate": leak,
                            "verification_coverage": cov,
                            "normalized_loss": loss,
                            "regret_best_fixed": loss - best_fixed if cond in FIXED_CONDS else "",
                            "oracle_leakage": o_leak,
                            "oracle_coverage": o_cov,
                            "oracle_loss": o_loss,
                            "regret_oracle": loss - o_loss,
                            "best_fixed_loss": best_fixed,
                            "error_rate": err,
                        }
                    )
    return out


def _exec_seconds() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for row in load_csv(TASK011_DIR / "hidden_test_results.csv"):
        meta = row.get("metadata") or ""
        seconds = None
        if meta:
            try:
                parsed = json.loads(meta)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(meta)
            if isinstance(parsed, dict):
                seconds = parsed.get("execution time")
        if seconds is None:
            seconds = row.get("cpu_user")
        if seconds not in (None, ""):
            out[(row["question_id"], row["model_alias"])] = float(seconds)
    return out


def run_efficiency(rows: list[ItemRow]) -> list[dict[str, Any]]:
    exec_s = _exec_seconds()
    pairs = [
        ("displayed_0.99", "displayed_0.95"),
        ("displayed_0.95", "displayed_0.90"),
        ("displayed_0.90", "displayed_0.85"),
        ("displayed_0.85", "displayed_0.70"),
        ("displayed_0.99", "displayed_0.70"),
    ]
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            for low_cov, high_cov in pairs:
                a = filter_rows(rows, task=task, model=model, condition=low_cov)
                b = filter_rows(rows, task=task, model=model, condition=high_cov)
                by_a = {r.question_id: r for r in a}
                extra_v = extra_catch = extra_cpu = 0.0
                n = 0
                for r in b:
                    other = by_a.get(r.question_id)
                    if other is None:
                        continue
                    n += 1
                    added = int(r.verify) - int(other.verify)
                    extra_v += max(added, 0)
                    extra_catch += int(r.incorrect and r.verify) - int(other.incorrect and other.verify)
                    if task == "code" and added > 0:
                        extra_cpu += exec_s.get((r.question_id, model), 0.0)
                denom_v = extra_v if extra_v else float("nan")
                out.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "from_condition": low_cov,
                        "to_condition": high_cov,
                        "direction": "toward_lower_displayed_confidence",
                        "n": n,
                        "additional_verify_calls": extra_v,
                        "additional_errors_caught": extra_catch,
                        "errors_caught_per_added_verify": extra_catch / denom_v if extra_v else float("nan"),
                        "errors_caught_per_100_verify": 100.0 * extra_catch / denom_v if extra_v else float("nan"),
                        "added_execution_seconds": extra_cpu if task == "code" else "",
                        "errors_caught_per_exec_second": (extra_catch / extra_cpu) if task == "code" and extra_cpu else "",
                    }
                )
    return out


def run_matched_budget(rows: list[ItemRow]) -> list[dict[str, Any]]:
    """Optional H: random-subset standardization of leakage change."""
    pairs = [
        ("displayed_0.70", "displayed_0.99"),
        ("displayed_0.70", "displayed_0.85"),
        ("displayed_0.85", "displayed_0.90"),
        ("displayed_0.90", "displayed_0.95"),
        ("displayed_0.95", "displayed_0.99"),
    ]
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            for high_c, low_c in pairs:
                high = filter_rows(rows, task=task, model=model, condition=high_c)
                low = filter_rows(rows, task=task, model=model, condition=low_c)
                if coverage(high) < coverage(low):
                    high, low = low, high
                    high_c, low_c = low_c, high_c
                n = len(high)
                if n == 0:
                    continue
                n_high_v = sum(r.verify for r in high)
                n_low_v = sum(r.verify for r in low)
                caught_high = sum(r.incorrect and r.verify for r in high)
                obs_delta = leakage(low) - leakage(high)
                if n_high_v == 0:
                    expected_random_leak_low = error_rate(high)
                else:
                    expected_caught = caught_high * (n_low_v / n_high_v)
                    expected_random_leak_low = error_rate(high) - expected_caught / n
                out.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "higher_coverage_condition": high_c,
                        "lower_coverage_condition": low_c,
                        "observed_delta_leakage": obs_delta,
                        "random_subset_expected_delta_leakage": expected_random_leak_low - leakage(high),
                        "observed_minus_random": obs_delta - (expected_random_leak_low - leakage(high)),
                        "note": "positive observed_minus_random means the lower-coverage policy leaked more than a random down-sample of the higher-coverage VERIFY set",
                    }
                )
    return out


def write_cost(cost_path, eff_path, med_path, cost, eff, med) -> None:
    write_csv(cost_path, cost)
    write_csv(eff_path, eff)
    write_csv(med_path, med)
