"""Task 013 figures. Hidden is never placed at an x-coordinate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .plotting import _pyplot, _save
from .task013_common import FIGURES_DIR, FIXED_CONDS, LAMBDA_GRID, MODEL_LABELS

TASKS = (("mmlu", "MMLU-Pro"), ("code", "LiveCodeBench hard"))
COLORS = {"openai_gpt56_sol": "#1f4e79", "anthropic_sonnet5": "#b85c38"}
POLICY_MARKERS = {
    "hidden": ("D", "#111111"),
    "true_q_visible": ("s", "#6b4c9a"),
    "displayed_0.70": ("o", "#1f4e79"),
    "displayed_0.85": ("o", "#3d7ab5"),
    "displayed_0.90": ("o", "#6aa0c8"),
    "displayed_0.95": ("o", "#9bbfdb"),
    "displayed_0.99": ("o", "#c5d9eb"),
}


def _style():
    plt = _pyplot()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _get(rows: Sequence[dict[str, Any]], **keys: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(k) == v for k, v in keys.items()):
            return row
    raise KeyError(keys)


def hidden_policy_summary(metrics: Sequence[dict[str, Any]], cost: Sequence[dict[str, Any]]) -> None:
    plt = _style()
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 9.4), constrained_layout=True)
    scores = [0.70, 0.85, 0.90, 0.95, 0.99]
    for col, (task, title) in enumerate(TASKS):
        ax = axes[0, col]
        for model, color in COLORS.items():
            ys = [100 * _get(metrics, task=task, model_alias=model, policy=f"displayed_{s:.2f}")["verification_coverage"] for s in scores]
            ax.plot(scores, ys, "-o", color=color, label=MODEL_LABELS[model], linewidth=1.7)
            hid = 100 * _get(metrics, task=task, model_alias=model, policy="hidden")["verification_coverage"]
            tq = 100 * _get(metrics, task=task, model_alias=model, policy="true_q_visible")["verification_coverage"]
            ax.axhline(hid, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
            ax.axhline(tq, color=color, linestyle=":", linewidth=1.0, alpha=0.8)
        ax.set_xlim(0.66, 1.02)
        ax.set_title(f"{title}: coverage (dashed=hidden, dotted=true-q)")
        ax.set_xlabel("Displayed confidence")
        ax.set_ylabel("Coverage %")
        ax.legend(frameon=False)

        ax = axes[1, col]
        for model, color in COLORS.items():
            for policy, (mk, _) in POLICY_MARKERS.items():
                rec = _get(metrics, task=task, model_alias=model, policy=policy)
                ax.scatter(
                    100 * rec["verification_coverage"],
                    rec["errors_unverified_per_100"],
                    marker=mk,
                    color=color,
                    s=55 if policy == "hidden" else 36,
                    zorder=4 if policy == "hidden" else 3,
                )
                if policy in {"hidden", "true_q_visible", "displayed_0.70", "displayed_0.99"}:
                    ax.annotate(
                        f"{MODEL_LABELS[model][:1]}:{policy.replace('displayed_', 'q=')}",
                        (100 * rec["verification_coverage"], rec["errors_unverified_per_100"]),
                        fontsize=6,
                        xytext=(3, 3),
                        textcoords="offset points",
                    )
        ax.set_xlabel("Coverage %")
        ax.set_ylabel("Unverified errors / 100")
        ax.set_title(f"{title}: observed policy map")

        ax = axes[2, col]
        for model, color in COLORS.items():
            for policy, style in (("hidden", "-"), ("true_q_visible", "--"), ("displayed_0.70", ":"), ("displayed_0.99", "-.")):
                ys = [
                    _get(cost, task=task, model_alias=model, policy=policy, **{"lambda": lam})["regret_vs_best_observed"]
                    for lam in LAMBDA_GRID
                ]
                ax.plot(LAMBDA_GRID, ys, style, color=color, linewidth=1.6, label=f"{MODEL_LABELS[model]} {policy.replace('displayed_', 'q=')}")
        ax.set_xscale("log")
        ax.set_xlabel("λ")
        ax.set_ylabel("Regret vs best observed")
        ax.set_title(f"{title}: regret vs λ")
        ax.legend(frameon=False, fontsize=6)
    fig.suptitle("POST_HOC_POLICY_REANALYSIS: hidden as a policy, not an x-value", fontsize=11)
    _save(fig, FIGURES_DIR / "hidden_policy_summary")
    plt.close(fig)


def policy_pareto(metrics: Sequence[dict[str, Any]], selection: Sequence[dict[str, Any]]) -> None:
    plt = _style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True)
    for ax, task, model, title in (
        (axes[0, 0], "mmlu", "openai_gpt56_sol", "MMLU-Pro · GPT"),
        (axes[0, 1], "mmlu", "anthropic_sonnet5", "MMLU-Pro · Claude"),
        (axes[1, 0], "code", "openai_gpt56_sol", "LiveCodeBench · GPT"),
        (axes[1, 1], "code", "anthropic_sonnet5", "LiveCodeBench · Claude"),
    ):
        for policy, (mk, color) in POLICY_MARKERS.items():
            rec = _get(metrics, task=task, model_alias=model, policy=policy)
            sel = _get(selection, task=task, model_alias=model, policy=policy)
            ax.scatter(
                100 * rec["verification_coverage"],
                rec["errors_unverified_per_100"],
                marker=mk,
                color="#111" if policy == "hidden" else color,
                s=80 if policy == "hidden" else 42,
                edgecolor="black" if sel["on_observed_pareto_frontier"] else "none",
                linewidths=1.1,
            )
            ax.annotate(policy.replace("displayed_", "q="), (100 * rec["verification_coverage"], rec["errors_unverified_per_100"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xlabel("Coverage %")
        ax.set_ylabel("Unverified errors / 100")
        ax.set_title(title)
    fig.suptitle("Observed Pareto frontier (edge = efficient). Hidden has no x-score.", fontsize=11)
    _save(fig, FIGURES_DIR / "policy_pareto_frontier")
    plt.close(fig)


def hidden_regret(cost: Sequence[dict[str, Any]]) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    for ax, task, title in ((axes[0], "mmlu", "MMLU-Pro"), (axes[1], "code", "LiveCodeBench hard")):
        for model, color in COLORS.items():
            for policy, style in (("hidden", "-"), ("true_q_visible", "--"), ("displayed_0.99", ":")):
                ys = [
                    _get(cost, task=task, model_alias=model, policy=policy, **{"lambda": lam})["regret_vs_best_observed"]
                    for lam in LAMBDA_GRID
                ]
                ax.plot(LAMBDA_GRID, ys, style, color=color, linewidth=1.7, label=f"{MODEL_LABELS[model]} {policy.replace('displayed_', 'q=')}")
        ax.set_xscale("log")
        ax.set_xlabel("λ (verify cost / escaped-error cost)")
        ax.set_ylabel("Regret vs best observed")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("POST_HOC_POLICY_REANALYSIS: hidden regret vs λ", fontsize=11)
    _save(fig, FIGURES_DIR / "hidden_regret_vs_lambda")
    plt.close(fig)


def coverage_leakage_map(metrics: Sequence[dict[str, Any]]) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), constrained_layout=True)
    for ax, task, title in ((axes[0], "mmlu", "MMLU-Pro"), (axes[1], "code", "LiveCodeBench hard")):
        for model, color in COLORS.items():
            xs, ys = [], []
            for cond in FIXED_CONDS:
                rec = _get(metrics, task=task, model_alias=model, policy=cond)
                xs.append(100 * rec["verification_coverage"])
                ys.append(rec["errors_unverified_per_100"])
            ax.plot(xs, ys, "-", color=color, alpha=0.45, linewidth=1.2)
            for policy, (mk, _) in POLICY_MARKERS.items():
                rec = _get(metrics, task=task, model_alias=model, policy=policy)
                ax.scatter(
                    100 * rec["verification_coverage"],
                    rec["errors_unverified_per_100"],
                    marker=mk,
                    color=color,
                    s=70 if policy == "hidden" else 36,
                )
            hid = _get(metrics, task=task, model_alias=model, policy="hidden")
            ax.annotate(f"{MODEL_LABELS[model]} hidden", (100 * hid["verification_coverage"], hid["errors_unverified_per_100"]), fontsize=8, xytext=(6, 6), textcoords="offset points")
        ax.set_xlabel("Coverage %")
        ax.set_ylabel("Unverified errors / 100")
        ax.set_title(title)
    fig.suptitle("Coverage–leakage policy map · POST_HOC_POLICY_REANALYSIS", fontsize=11)
    _save(fig, FIGURES_DIR / "coverage_leakage_policy_map")
    plt.close(fig)


def write_all(metrics, cost, selection) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    hidden_policy_summary(metrics, cost)
    policy_pareto(metrics, selection)
    hidden_regret(cost)
    coverage_leakage_map(metrics)
