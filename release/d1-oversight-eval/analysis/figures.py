"""Main-candidate figures for the public D1 package."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .constants import DELTAS, FIGURES_DIR, FIXED_SCORES, LAMBDA_GRID, MODEL_LABELS, TASK_LABELS
from .io_csv import OffsetRow, RoutingRow, filter_offset, filter_routing
from .metrics import coverage, leakage, loss


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(fig: Any, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURES_DIR / name
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")


COLORS = {"openai_gpt56_sol": "#1f4e79", "anthropic_sonnet5": "#b85c38"}
DELTA_COLORS = {-1.5: "#1f4e79", -0.75: "#2e75b6", 0.0: "#7f7f7f", 0.75: "#c45911", 1.5: "#c00000"}


def fixed_score_coverage_leakage(rows: Sequence[RoutingRow]) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
    cells = (("mmlu", "openai_gpt56_sol"), ("code", "openai_gpt56_sol"), ("mmlu", "anthropic_sonnet5"), ("code", "anthropic_sonnet5"))
    for ax, (task, model) in zip(axes.ravel(), cells):
        xs, covs, leaks = [], [], []
        for score in FIXED_SCORES:
            cell = filter_routing(list(rows), task=task, model=model, condition=f"displayed_{score:.2f}")
            xs.append(score)
            covs.append(100 * coverage(cell))
            leaks.append(100 * leakage(cell))
        ax.plot(xs, covs, "-o", color=COLORS[model], label="coverage %")
        ax.plot(xs, leaks, "--s", color="#c00000", label="leakage / 100")
        ax.set_title(f"{MODEL_LABELS[model]} {TASK_LABELS[task]}")
        ax.set_xlabel("Displayed confidence")
        ax.set_ylabel("Percent")
        ax.legend(frameon=False)
    fig.suptitle("Fixed-score stress test: coverage and leakage")
    _save(fig, "fixed_score_coverage_leakage")
    plt.close(fig)


def rank_preserving_response(rows: Sequence[OffsetRow]) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
    cells = (("mmlu", "openai_gpt56_sol"), ("code", "openai_gpt56_sol"), ("mmlu", "anthropic_sonnet5"), ("code", "anthropic_sonnet5"))
    for ax, (task, model) in zip(axes.ravel(), cells):
        xs, covs, leaks = [], [], []
        for delta in DELTAS:
            cell = filter_offset(list(rows), task=task, model=model, delta=delta)
            xs.append(delta)
            covs.append(100 * coverage(cell))
            leaks.append(100 * leakage(cell))
        ax.plot(xs, covs, "-o", color=COLORS[model], label="coverage %")
        ax.plot(xs, leaks, "--s", color="#c00000", label="leakage / 100")
        ax.set_title(f"{MODEL_LABELS[model]} {TASK_LABELS[task]}")
        ax.set_xlabel("Rank-preserving δ")
        ax.set_ylabel("Percent")
        ax.legend(frameon=False)
    fig.suptitle("Rank-preserving level shift: coverage and leakage")
    _save(fig, "rank_preserving_response")
    plt.close(fig)


def loss_spread(rows: Sequence[OffsetRow]) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
    cells = (("mmlu", "openai_gpt56_sol"), ("code", "openai_gpt56_sol"), ("mmlu", "anthropic_sonnet5"), ("code", "anthropic_sonnet5"))
    for ax, (task, model) in zip(axes.ravel(), cells):
        xs, ys = [], []
        for lam in LAMBDA_GRID:
            losses = []
            for delta in DELTAS:
                cell = filter_offset(list(rows), task=task, model=model, delta=delta)
                losses.append(loss(leakage(cell), coverage(cell), lam))
            xs.append(lam)
            ys.append(max(losses) - min(losses))
        ax.plot(xs, ys, "-o", color="#1f4e79")
        ax.set_xscale("log")
        ax.set_title(f"{MODEL_LABELS[model]} {TASK_LABELS[task]}")
        ax.set_xlabel("λ")
        ax.set_ylabel("max L − min L")
        ax.set_ylim(bottom=0)
    fig.suptitle("Rank-preserving loss spread under L = leakage + λ·coverage")
    _save(fig, "rank_preserving_loss_spread")
    plt.close(fig)


def unmanipulated_escape_callout(escape_rows: Sequence[dict[str, Any]]) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    labels, vals, los, his = [], [], [], []
    order = (
        ("mmlu", "openai_gpt56_sol"),
        ("code", "openai_gpt56_sol"),
        ("mmlu", "anthropic_sonnet5"),
        ("code", "anthropic_sonnet5"),
    )
    for task, model in order:
        rec = next(r for r in escape_rows if r["task"] == task and r["model_alias"] == model)
        labels.append(f"{MODEL_LABELS[model]}\n{TASK_LABELS[task]}")
        vals.append(100 * rec["conditional_escape_rate"])
        los.append(100 * rec["ci_lo"])
        his.append(100 * rec["ci_hi"])
    xs = range(len(labels))
    ax.bar(xs, vals, color=["#1f4e79", "#1f4e79", "#b85c38", "#b85c38"], alpha=0.85)
    ax.errorbar(xs, vals, yerr=[np_sub(vals, los), np_sub(his, vals)], fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel("P(USE_UNVERIFIED | incorrect) %")
    ax.set_title("Unmanipulated q1-visible conditional escape")
    _save(fig, "unmanipulated_escape_callout")
    plt.close(fig)


def np_sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]
