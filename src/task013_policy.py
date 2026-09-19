"""Sections 3–4 and 7: policy metrics, cost sweep, hidden vs true_q."""

from __future__ import annotations

from typing import Any

from .task012_data import ItemRow, filter_rows
from .task012_stats import (
    bootstrap_pair,
    coverage,
    error_catch,
    error_rate,
    leakage,
    verify_error_precision,
)
from .task013_common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    GPT_CLAUDE,
    LABEL,
    LAMBDA_GRID,
    POLICIES,
    TASK012_DIR,
    load_csv,
)


def _loss(leak: float, cov: float, lam: float) -> float:
    return leak + lam * cov


def policy_metrics(rows: list[ItemRow]) -> list[dict[str, Any]]:
    prior = {(r["task"], r["model_alias"], r["score_condition"]): r for r in load_csv(TASK012_DIR / "error_leakage_by_condition.csv")}
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            for cond in POLICIES:
                subset = filter_rows(rows, task=task, model=model, condition=cond)
                n = len(subset)
                n_inc = sum(r.incorrect for r in subset)
                n_v = sum(r.verify for r in subset)
                n_inc_v = sum(r.incorrect and r.verify for r in subset)
                n_inc_u = sum(r.incorrect and not r.verify for r in subset)
                prev = prior[(task, model, cond)]
                out.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "policy": cond,
                        "n": n,
                        "n_incorrect": n_inc,
                        "n_verify_first": n_v,
                        "verification_coverage": coverage(subset),
                        "coverage_ci_lo": float(prev["coverage_ci_lo"]),
                        "coverage_ci_hi": float(prev["coverage_ci_hi"]),
                        "n_incorrect_verified": n_inc_v,
                        "n_incorrect_unverified": n_inc_u,
                        "leakage_rate": leakage(subset),
                        "leakage_ci_lo": float(prev["leakage_ci_lo"]),
                        "leakage_ci_hi": float(prev["leakage_ci_hi"]),
                        "error_catch_rate": error_catch(subset),
                        "verification_precision": verify_error_precision(subset),
                        "errors_caught_per_100": 100.0 * n_inc_v / n if n else float("nan"),
                        "errors_unverified_per_100": 100.0 * n_inc_u / n if n else float("nan"),
                        "error_rate": error_rate(subset),
                        "ci_source": "task012_question_bootstrap_5000",
                    }
                )
    return out


def cost_sweep(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in metrics:
        by_cell.setdefault((row["task"], row["model_alias"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (task, model), policies in by_cell.items():
        for lam in LAMBDA_GRID:
            losses = {p["policy"]: _loss(p["leakage_rate"], p["verification_coverage"], lam) for p in policies}
            best = min(losses.values())
            best_name = min(losses, key=losses.get)
            for p in policies:
                loss = losses[p["policy"]]
                out.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "policy": p["policy"],
                        "lambda": lam,
                        "leakage_rate": p["leakage_rate"],
                        "verification_coverage": p["verification_coverage"],
                        "normalized_loss": loss,
                        "best_observed_loss": best,
                        "best_observed_policy": best_name,
                        "regret_vs_best_observed": loss - best,
                        "is_best_observed": p["policy"] == best_name,
                    }
                )
    return out


def hidden_vs_trueq(rows: list[ItemRow], metrics: list[dict[str, Any]], cost: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            hid = filter_rows(rows, task=task, model=model, condition="hidden")
            tq = filter_rows(rows, task=task, model=model, condition="true_q_visible")
            d_cov = bootstrap_pair(tq, hid, coverage, seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_RESAMPLES)
            d_leak = bootstrap_pair(tq, hid, leakage, seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_RESAMPLES)
            d_catch = bootstrap_pair(tq, hid, error_catch, seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_RESAMPLES)
            hm = next(r for r in metrics if r["task"] == task and r["model_alias"] == model and r["policy"] == "hidden")
            tm = next(r for r in metrics if r["task"] == task and r["model_alias"] == model and r["policy"] == "true_q_visible")
            loss_rows = []
            for lam in LAMBDA_GRID:
                h = next(r for r in cost if r["task"] == task and r["model_alias"] == model and r["policy"] == "hidden" and r["lambda"] == lam)
                t = next(r for r in cost if r["task"] == task and r["model_alias"] == model and r["policy"] == "true_q_visible" and r["lambda"] == lam)
                loss_rows.append(h["normalized_loss"] - t["normalized_loss"])
            out.append(
                {
                    "label": LABEL,
                    "task": task,
                    "model_alias": model,
                    "n": len(hid),
                    "hidden_coverage": hm["verification_coverage"],
                    "trueq_coverage": tm["verification_coverage"],
                    "delta_coverage_hidden_minus_trueq": d_cov.point,
                    "delta_coverage_ci_lo": d_cov.lo,
                    "delta_coverage_ci_hi": d_cov.hi,
                    "hidden_leakage": hm["leakage_rate"],
                    "trueq_leakage": tm["leakage_rate"],
                    "delta_leakage_hidden_minus_trueq": d_leak.point,
                    "delta_leakage_ci_lo": d_leak.lo,
                    "delta_leakage_ci_hi": d_leak.hi,
                    "hidden_error_catch": hm["error_catch_rate"],
                    "trueq_error_catch": tm["error_catch_rate"],
                    "delta_error_catch": d_catch.point,
                    "hidden_n_unverified_errors": hm["n_incorrect_unverified"],
                    "trueq_n_unverified_errors": tm["n_incorrect_unverified"],
                    "mean_delta_loss_over_lambda": sum(loss_rows) / len(loss_rows),
                    "min_delta_loss_over_lambda": min(loss_rows),
                    "max_delta_loss_over_lambda": max(loss_rows),
                    "hidden_more_leakage_than_trueq_pp": 100.0 * (hm["leakage_rate"] - tm["leakage_rate"]),
                    "bootstrap_n": d_leak.n_boot,
                }
            )
    return out
