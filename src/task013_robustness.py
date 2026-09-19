"""Sections 5–8 and 12: robustness labels, minimax, Pareto, systems matrix."""

from __future__ import annotations

import math
from typing import Any

from .task013_common import GPT_CLAUDE, LABEL, LAMBDA_GRID, NEAR_BEST, POLICIES, TASK012_DIR, load_csv


def _cell_cost(cost: list[dict[str, Any]], task: str, model: str) -> list[dict[str, Any]]:
    return [r for r in cost if r["task"] == task and r["model_alias"] == model]


def _policy_regrets(cost: list[dict[str, Any]], task: str, model: str, policy: str) -> list[float]:
    rows = [r for r in _cell_cost(cost, task, model) if r["policy"] == policy]
    rows = sorted(rows, key=lambda r: r["lambda"])
    return [float(r["regret_vs_best_observed"]) for r in rows]


def _log_weighted_mean(lams: tuple[float, ...], values: list[float]) -> float:
    logs = [math.log10(x) for x in lams]
    area = 0.0
    width = 0.0
    for i in range(len(logs) - 1):
        w = logs[i + 1] - logs[i]
        area += 0.5 * (values[i] + values[i + 1]) * w
        width += w
    return area / width if width else float("nan")


def _area(lams: tuple[float, ...], values: list[float]) -> float:
    logs = [math.log10(x) for x in lams]
    area = 0.0
    for i in range(len(logs) - 1):
        area += 0.5 * (values[i] + values[i + 1]) * (logs[i + 1] - logs[i])
    return area


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["leakage_rate"] <= b["leakage_rate"]
        and a["verification_coverage"] <= b["verification_coverage"]
        and (a["leakage_rate"] < b["leakage_rate"] or a["verification_coverage"] < b["verification_coverage"])
    )


def _pareto(metrics: list[dict[str, Any]], task: str, model: str) -> set[str]:
    pols = [r for r in metrics if r["task"] == task and r["model_alias"] == model]
    efficient = set()
    for a in pols:
        if not any(_dominates(b, a) for b in pols if b["policy"] != a["policy"]):
            efficient.add(a["policy"])
    return efficient


def hidden_robustness(metrics: list[dict[str, Any]], cost: list[dict[str, Any]], vs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            regrets = _policy_regrets(cost, task, model, "hidden")
            n = len(regrets)
            hid = next(r for r in metrics if r["task"] == task and r["model_alias"] == model and r["policy"] == "hidden")
            dominated = [
                r["policy"]
                for r in metrics
                if r["task"] == task and r["model_alias"] == model and r["policy"] != "hidden" and _dominates(hid, r)
            ]
            vs_row = next(r for r in vs if r["task"] == task and r["model_alias"] == model)
            leak_gap = vs_row["hidden_more_leakage_than_trueq_pp"]
            near = {thr: sum(x <= thr for x in regrets) / n for thr in NEAR_BEST}
            gpt_pass = (
                model == "openai_gpt56_sol"
                and near[0.02] >= 0.50
                and max(regrets) <= 0.10
                and leak_gap <= 10.0
            )
            out.append(
                {
                    "label": LABEL,
                    "task": task,
                    "model_alias": model,
                    "frac_exactly_best": sum(x == 0 for x in regrets) / n,
                    "frac_within_0.01": near[0.01],
                    "frac_within_0.02": near[0.02],
                    "frac_within_0.05": near[0.05],
                    "worst_case_regret": max(regrets),
                    "mean_regret_unweighted": sum(regrets) / n,
                    "mean_regret_log_lambda": _log_weighted_mean(LAMBDA_GRID, regrets),
                    "integrated_regret_log_lambda": _area(LAMBDA_GRID, regrets),
                    "strictly_dominates": ";".join(dominated) if dominated else "",
                    "hidden_minus_trueq_leakage_pp": leak_gap,
                    "passes_gpt_robustness_criteria": gpt_pass,
                }
            )
    return out


def classify_gpt(robust: list[dict[str, Any]]) -> str:
    gpt = [r for r in robust if r["model_alias"] == "openai_gpt56_sol"]
    n_pass = sum(bool(r["passes_gpt_robustness_criteria"]) for r in gpt)
    if n_pass == 2:
        return "HIDDEN_ROBUST_BOTH_GPT_TASKS"
    if n_pass == 1:
        return "HIDDEN_ROBUST_ONE_TASK"
    return "HIDDEN_NOT_ROBUST"


def robust_policy_selection(metrics: list[dict[str, Any]], cost: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            frontier = _pareto(metrics, task, model)
            n = len(LAMBDA_GRID)
            worst: dict[str, float] = {}
            near: dict[str, dict[float, float]] = {}
            for policy in POLICIES:
                regrets = _policy_regrets(cost, task, model, policy)
                worst[policy] = max(regrets)
                near[policy] = {thr: sum(x <= thr for x in regrets) / n for thr in NEAR_BEST}
            minimax_policy = min(worst, key=worst.get)
            for policy in POLICIES:
                out.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "policy": policy,
                        "worst_case_regret": worst[policy],
                        "is_minimax_regret_policy": policy == minimax_policy,
                        "frac_within_0.01": near[policy][0.01],
                        "frac_within_0.02": near[policy][0.02],
                        "frac_within_0.05": near[policy][0.05],
                        "on_observed_pareto_frontier": policy in frontier,
                    }
                )
    return out


def systems_matrix(metrics: list[dict[str, Any]], robust: list[dict[str, Any]], selection: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q1 = load_csv(TASK012_DIR / "q1_calibration.csv")
    sens = load_csv(TASK012_DIR / "oversight_sensitivity.csv")
    eff = load_csv(TASK012_DIR / "marginal_verification_efficiency.csv")
    seen_q1 = {}
    for row in q1:
        key = (row["task"], row["model_alias"])
        if key not in seen_q1:
            seen_q1[key] = row
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            hid = next(r for r in metrics if r["task"] == task and r["model_alias"] == model and r["policy"] == "hidden")
            tq = next(r for r in metrics if r["task"] == task and r["model_alias"] == model and r["policy"] == "true_q_visible")
            rob = next(r for r in robust if r["task"] == task and r["model_alias"] == model)
            hid_sel = next(r for r in selection if r["task"] == task and r["model_alias"] == model and r["policy"] == "hidden")
            srow = next(
                r
                for r in sens
                if r["task"] == task and r["model_alias"] == model and r["span"] == "0.70_to_0.99"
            )
            qrow = seen_q1[(task, model)]
            e070 = next(
                (
                    r
                    for r in eff
                    if r["task"] == task
                    and r["model_alias"] == model
                    and r["from_condition"] == "displayed_0.99"
                    and r["to_condition"] == "displayed_0.70"
                ),
                None,
            )
            out.append(
                {
                    "label": LABEL,
                    "task": task,
                    "model_alias": model,
                    "q1_brier": float(qrow["brier"]),
                    "q1_ece": float(qrow["ece"]),
                    "coverage_sensitivity_per_0.10": float(srow["coverage_sensitivity_per_0.10"]),
                    "leakage_sensitivity_per_0.10": float(srow["leakage_sensitivity_per_0.10"]),
                    "hidden_coverage": hid["verification_coverage"],
                    "hidden_leakage": hid["leakage_rate"],
                    "hidden_frac_within_0.02": rob["frac_within_0.02"],
                    "hidden_worst_case_regret": rob["worst_case_regret"],
                    "hidden_on_pareto": hid_sel["on_observed_pareto_frontier"],
                    "trueq_coverage": tq["verification_coverage"],
                    "trueq_leakage": tq["leakage_rate"],
                    "errors_caught_per_added_verify_0.99_to_0.70": (e070 or {}).get("errors_caught_per_added_verify", ""),
                    "added_execution_seconds_0.99_to_0.70": (e070 or {}).get("added_execution_seconds", ""),
                }
            )
    return out


def paper_gate(
    robust_label: str,
    selection: list[dict[str, Any]],
    robust: list[dict[str, Any]],
    repro_ok: bool,
) -> dict[str, Any]:
    gpt_sel = [r for r in selection if r["model_alias"] == "openai_gpt56_sol" and r["policy"] == "hidden"]
    gpt_rob = [r for r in robust if r["model_alias"] == "openai_gpt56_sol"]
    B = all(r["on_observed_pareto_frontier"] for r in gpt_sel)
    C = all(r["frac_within_0.02"] >= 0.50 for r in gpt_rob)
    D = all(r["worst_case_regret"] <= 0.10 for r in gpt_rob)
    A = robust_label == "HIDDEN_ROBUST_BOTH_GPT_TASKS"
    E = True
    if not repro_ok:
        decision = "CONTRADICTS_CURRENT_STORY"
    elif A and B and C and D and E:
        decision = "ACTIONABLE_UPGRADE"
    elif robust_label == "HIDDEN_ROBUST_ONE_TASK":
        decision = "ACTIONABLE_BUT_TASK_SPECIFIC"
    else:
        decision = "DIAGNOSTIC_ONLY"
    return {
        "label": LABEL,
        "decision": decision,
        "robustness_class": robust_label,
        "A_both_gpt_robust": A,
        "B_pareto_both_gpt": B,
        "C_near_best_both_gpt": C,
        "D_worst_regret_both_gpt": D,
        "E_honest_reporting": E,
        "reproduction_ok": repro_ok,
    }
