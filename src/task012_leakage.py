"""Section 4: error leakage at each displayed score."""

from __future__ import annotations

from typing import Any

from .task012_common import ALL_CONDS, GPT_CLAUDE, LABEL, write_csv
from .task012_data import ItemRow, filter_rows
from .task012_stats import (
    bootstrap_pair,
    bootstrap_stat,
    coverage,
    error_catch,
    error_rate,
    leakage,
    unverified_error_share,
    verify_error_precision,
)


def condition_row(rows: list[ItemRow], task: str, model: str, cond: str) -> dict[str, Any]:
    subset = filter_rows(rows, task=task, model=model, condition=cond)
    n = len(subset)
    n_inc = sum(r.incorrect for r in subset)
    n_v = sum(r.verify for r in subset)
    n_inc_v = sum(r.incorrect and r.verify for r in subset)
    n_inc_u = sum(r.incorrect and not r.verify for r in subset)
    cov = bootstrap_stat(subset, coverage)
    leak = bootstrap_stat(subset, leakage)
    catch = bootstrap_stat(subset, error_catch)
    return {
        "label": LABEL,
        "task": task,
        "model_alias": model,
        "score_condition": cond,
        "displayed_confidence": subset[0].displayed_confidence if subset else "",
        "n_outputs": n,
        "n_incorrect": n_inc,
        "n_verify_first": n_v,
        "verification_coverage": cov.point,
        "coverage_ci_lo": cov.lo,
        "coverage_ci_hi": cov.hi,
        "n_incorrect_verify": n_inc_v,
        "n_incorrect_use_unverified": n_inc_u,
        "error_catch_rate": catch.point,
        "error_catch_ci_lo": catch.lo,
        "error_catch_ci_hi": catch.hi,
        "unverified_error_rate": leak.point,
        "leakage_ci_lo": leak.lo,
        "leakage_ci_hi": leak.hi,
        "unverified_errors_per_100": 100.0 * leak.point,
        "fraction_errors_unverified": unverified_error_share(subset),
        "error_rate": error_rate(subset),
        "verify_error_precision": verify_error_precision(subset),
        "bootstrap_n": leak.n_boot,
    }


def contrast_row(
    rows: list[ItemRow],
    task: str,
    model: str,
    cond_a: str,
    cond_b: str,
) -> dict[str, Any]:
    a = filter_rows(rows, task=task, model=model, condition=cond_a)
    b = filter_rows(rows, task=task, model=model, condition=cond_b)
    d_cov = bootstrap_pair(a, b, coverage)
    d_leak = bootstrap_pair(a, b, leakage)
    d_catch = bootstrap_pair(a, b, error_catch)
    return {
        "label": LABEL,
        "task": task,
        "model_alias": model,
        "from_condition": cond_a,
        "to_condition": cond_b,
        "delta_coverage": d_cov.point,
        "delta_coverage_ci_lo": d_cov.lo,
        "delta_coverage_ci_hi": d_cov.hi,
        "delta_unverified_error_rate": d_leak.point,
        "delta_leakage_ci_lo": d_leak.lo,
        "delta_leakage_ci_hi": d_leak.hi,
        "delta_unverified_errors_per_100": 100.0 * d_leak.point,
        "delta_error_catch_rate": d_catch.point,
        "delta_error_catch_ci_lo": d_catch.lo,
        "delta_error_catch_ci_hi": d_catch.hi,
        "bootstrap_n": d_leak.n_boot,
    }


def run_leakage(rows: list[ItemRow]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conditions = []
    contrasts = []
    pairs = [
        ("displayed_0.70", "displayed_0.85"),
        ("displayed_0.85", "displayed_0.90"),
        ("displayed_0.90", "displayed_0.95"),
        ("displayed_0.95", "displayed_0.99"),
        ("displayed_0.70", "displayed_0.99"),
    ]
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            for cond in ALL_CONDS:
                conditions.append(condition_row(rows, task, model, cond))
            for a, b in pairs:
                contrasts.append(contrast_row(rows, task, model, a, b))
    return conditions, contrasts


def write_leakage(path, conditions: list[dict[str, Any]]) -> None:
    write_csv(path, conditions)
