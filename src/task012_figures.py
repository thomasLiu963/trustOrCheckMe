"""Task 012 figures. Hidden is a reference marker, never plotted at an x."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .plotting import _pyplot, _save
from .task012_common import FIGURES_DIR, FIXED_SCORES, MODEL_LABELS
from .task012_data import ItemRow, filter_rows

TASKS = (("mmlu", "MMLU-Pro"), ("code", "LiveCodeBench hard"))
COLORS = {"openai_gpt56_sol": "#1f4e79", "anthropic_sonnet5": "#b85c38"}


def _items_q1(rows: Sequence[ItemRow], task: str, model: str) -> tuple[list[float], list[float]]:
    seen: set[str] = set()
    correct: list[float] = []
    incorrect: list[float] = []
    for row in filter_rows(list(rows), task=task, model=model, condition="displayed_0.70"):
        if row.question_id in seen or row.q1 is None:
            continue
        seen.add(row.question_id)
        (incorrect if row.incorrect else correct).append(float(row.q1))
    return correct, incorrect


def _lookup(table: Sequence[dict[str, Any]], **keys: Any) -> dict[str, Any] | None:
    for row in table:
        if all(row.get(k) == v for k, v in keys.items()):
            return row
    return None


def _style() -> Any:
    plt = _pyplot()
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


def error_leakage_curve(leakage_rows: Sequence[dict[str, Any]]) -> None:
    plt = _style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for col, (task, title) in enumerate(TASKS):
        for row_i, metric in enumerate(("verification_coverage", "unverified_errors_per_100")):
            ax = axes[row_i, col]
            for model, color in COLORS.items():
                xs, ys, lo, hi = [], [], [], []
                for score in FIXED_SCORES:
                    rec = _lookup(leakage_rows, task=task, model_alias=model, score_condition=f"displayed_{score:.2f}")
                    if rec is None:
                        continue
                    xs.append(score)
                    if metric == "verification_coverage":
                        ys.append(100 * rec[metric])
                        lo.append(100 * rec["coverage_ci_lo"])
                        hi.append(100 * rec["coverage_ci_hi"])
                    else:
                        ys.append(rec[metric])
                        lo.append(100 * rec["leakage_ci_lo"])
                        hi.append(100 * rec["leakage_ci_hi"])
                ax.plot(xs, ys, "-o", color=color, label=MODEL_LABELS[model], linewidth=1.8)
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
            ax.set_xlim(0.66, 1.02)
            ax.set_title(title if row_i == 0 else "")
            ax.set_xlabel("Displayed confidence")
            ax.set_ylabel("Verification coverage (%)" if metric == "verification_coverage" else "Unverified errors / 100 outputs")
            if row_i == 0:
                ax.legend(frameon=False)
    fig.suptitle("POST_HOC_SYSTEMS_REANALYSIS: coverage vs error leakage", fontsize=11)
    _save(fig, FIGURES_DIR / "error_leakage_curve")
    plt.close(fig)


def oversight_sensitivity(sens_rows: Sequence[dict[str, Any]]) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    labels = ["0.70→0.85", "0.85→0.90", "0.90→0.95", "0.95→0.99"]
    x = range(len(labels))
    width = 0.18
    for ax, task, title in ((axes[0], "mmlu", "MMLU-Pro"), (axes[1], "code", "LiveCodeBench hard")):
        for i, model in enumerate(COLORS):
            vals = []
            for qa, qb in ((0.7, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 0.99)):
                rec = _lookup(sens_rows, task=task, model_alias=model, from_score=qa, to_score=qb, span="adjacent")
                vals.append(rec["coverage_sensitivity_per_0.10"] * 100 if rec else 0)
            ax.bar([p + (i - 0.5) * width for p in x], vals, width=width, color=COLORS[model], label=MODEL_LABELS[model])
        ax.axhline(0, color="#444", linewidth=0.8)
        ax.set_xticks(list(x), labels, rotation=20)
        ax.set_ylabel("Coverage pp per +0.10 displayed q")
        ax.set_title(title)
        ax.legend(frameon=False)
    fig.suptitle("POST_HOC_SYSTEMS_REANALYSIS: local coverage sensitivity", fontsize=11)
    _save(fig, FIGURES_DIR / "oversight_sensitivity")
    plt.close(fig)


def causal_with_q1(leakage_rows: Sequence[dict[str, Any]], item_rows: Sequence[ItemRow]) -> None:
    plt = _style()
    fig, axes = plt.subplots(4, 2, figsize=(9.4, 10.2), constrained_layout=True)
    for col, (task, title) in enumerate(TASKS):
        for row_i, model in enumerate(COLORS):
            cov_ax = axes[row_i * 2, col]
            hist_ax = axes[row_i * 2 + 1, col]
            xs, ys = [], []
            for score in FIXED_SCORES:
                rec = _lookup(leakage_rows, task=task, model_alias=model, score_condition=f"displayed_{score:.2f}")
                if rec:
                    xs.append(score)
                    ys.append(100 * rec["verification_coverage"])
            cov_ax.plot(xs, ys, "-o", color=COLORS[model], linewidth=1.8)
            hid = _lookup(leakage_rows, task=task, model_alias=model, score_condition="hidden")
            tq = _lookup(leakage_rows, task=task, model_alias=model, score_condition="true_q_visible")
            if hid:
                cov_ax.axhline(100 * hid["verification_coverage"], color="#666", linestyle="--", linewidth=1, label="hidden")
            if tq:
                cov_ax.axhline(100 * tq["verification_coverage"], color="#888", linestyle=":", linewidth=1, label="true-q visible")
            cov_ax.set_xlim(0, 1)
            cov_ax.set_ylabel("Coverage %")
            cov_ax.set_title(f"{title} · {MODEL_LABELS[model]}")
            cov_ax.legend(frameon=False, loc="best")
            good, bad = _items_q1(item_rows, task, model)
            bins = [i / 20 for i in range(21)]
            hist_ax.hist(good, bins=bins, alpha=0.55, color="#2a6f4e", label="correct", density=True)
            hist_ax.hist(bad, bins=bins, alpha=0.55, color="#8b2e2e", label="incorrect", density=True)
            hist_ax.axvspan(0.70, 0.99, color="#1f4e79", alpha=0.08)
            hist_ax.set_xlim(0, 1)
            hist_ax.set_xlabel("Reported q1 / displayed score")
            hist_ax.set_ylabel("Density")
            hist_ax.legend(frameon=False)
    fig.suptitle("Fixed-score causal curve vs natural q1 (POST_HOC_SYSTEMS_REANALYSIS)", fontsize=11)
    _save(fig, FIGURES_DIR / "causal_curve_with_q1_distribution")
    plt.close(fig)


def cost_regret(cost_rows: Sequence[dict[str, Any]], task: str, name: str) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True, sharey=True)
    lams = sorted({r["lambda"] for r in cost_rows if r["task"] == task})
    for ax, model in zip(axes, COLORS):
        for cond, style in (
            ("displayed_0.70", "-"),
            ("displayed_0.85", "--"),
            ("displayed_0.90", "-."),
            ("displayed_0.95", ":"),
            ("displayed_0.99", "-"),
        ):
            ys = []
            for lam in lams:
                rec = _lookup(cost_rows, task=task, model_alias=model, score_condition=cond, **{"lambda": lam})
                ys.append(rec["regret_best_fixed"] if rec and rec["regret_best_fixed"] != "" else 0)
            ax.plot(lams, ys, style, color=COLORS[model] if cond != "displayed_0.99" else "#222", label=cond.replace("displayed_", "q="), linewidth=1.6)
        ax.set_xscale("log")
        ax.set_xlabel("λ (verify cost / escaped-error cost)")
        ax.set_ylabel("Regret vs best observed fixed score")
        ax.set_title(MODEL_LABELS[model])
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{name} cost-regret sweep · POST_HOC_SYSTEMS_REANALYSIS", fontsize=11)
    _save(fig, FIGURES_DIR / f"cost_regret_sweep_{'mmlu' if task == 'mmlu' else 'code'}")
    plt.close(fig)


def main_candidate(leakage_rows: Sequence[dict[str, Any]]) -> None:
    plt = _style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), constrained_layout=True)
    for col, (task, title) in enumerate(TASKS):
        for row_i, model in enumerate(COLORS):
            ax = axes[row_i, col]
            xs, cov, leak = [], [], []
            for score in FIXED_SCORES:
                rec = _lookup(leakage_rows, task=task, model_alias=model, score_condition=f"displayed_{score:.2f}")
                if rec:
                    xs.append(score)
                    cov.append(100 * rec["verification_coverage"])
                    leak.append(rec["unverified_errors_per_100"])
            ax.plot(xs, cov, "-o", color=COLORS[model], label="coverage %")
            ax.set_xlim(0.66, 1.02)
            ax.set_ylabel("Verification coverage (%)", color=COLORS[model])
            ax2 = ax.twinx()
            ax2.plot(xs, leak, "-s", color="#8b2e2e", label="unverified errors / 100")
            ax2.set_ylabel("Unverified errors / 100 outputs", color="#8b2e2e")
            ax.set_title(f"{title} · {MODEL_LABELS[model]}")
            ax.set_xlabel("Displayed confidence")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, frameon=False)
    fig.suptitle("Main-candidate systems figure · POST_HOC_SYSTEMS_REANALYSIS", fontsize=11)
    _save(fig, FIGURES_DIR / "task012_main_candidate")
    plt.close(fig)


def write_all(
    leakage_rows: Sequence[dict[str, Any]],
    sens_rows: Sequence[dict[str, Any]],
    cost_rows: Sequence[dict[str, Any]],
    item_rows: Sequence[ItemRow],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    error_leakage_curve(leakage_rows)
    oversight_sensitivity(sens_rows)
    causal_with_q1(leakage_rows, item_rows)
    cost_regret(cost_rows, "mmlu", "MMLU-Pro")
    cost_regret(cost_rows, "code", "LiveCodeBench hard")
    main_candidate(leakage_rows)
