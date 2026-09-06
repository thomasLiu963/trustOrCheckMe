"""Paper-ready V2 factorial figures."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _axes(plt, models: Sequence[str]):
    count = max(1, len(models))
    columns = 2 if count > 1 else 1
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.8 * columns, 3.3 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat = list(axes.flat)
    for axis in flat[count:]:
        axis.set_visible(False)
    return fig, flat


def _save(fig: Any, base: Path) -> dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    return {"pdf": str(pdf), "png": str(png)}


def plot_factor_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    ylabel: str,
    output_base: Path,
) -> dict[str, str]:
    plt = _pyplot()
    models = sorted({str(row["model_id"]) for row in rows})
    fig, axes = _axes(plt, models)
    styles = {
        ("human", "hidden"): ("o", "-"),
        ("ai_system", "hidden"): ("s", "-"),
        ("human", "visible"): ("o", "--"),
        ("ai_system", "visible"): ("s", "--"),
    }
    for axis, model in zip(axes, models):
        model_rows = [row for row in rows if str(row["model_id"]) == model]
        grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for row in model_rows:
            x = _number(row.get("error_cost"))
            y = _number(row.get(metric))
            if x is not None and y is not None:
                grouped[
                    (
                        str(row["decision_owner"]),
                        str(row["confidence_visibility"]),
                    )
                ].append((x, y))
        for key, values in sorted(grouped.items()):
            values.sort()
            marker, linestyle = styles[key]
            axis.plot(
                [item[0] for item in values],
                [item[1] for item in values],
                marker=marker,
                linestyle=linestyle,
                label=f"{key[0]} / {key[1]}",
            )
        axis.set_title(model)
        axis.set_xticks([2, 5, 10, 20])
        axis.set_xlabel("Error cost, L")
        axis.set_ylabel(ylabel)
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False)
    paths = _save(fig, output_base)
    plt.close(fig)
    return paths


def plot_owner_effect(
    rows: Sequence[Mapping[str, Any]], output_base: Path
) -> dict[str, str]:
    plt = _pyplot()
    models = sorted({str(row["model_id"]) for row in rows})
    fig, axes = _axes(plt, models)
    for axis, model in zip(axes, models):
        for visibility in ("hidden", "visible"):
            values = sorted(
                (
                    float(row["error_cost"]),
                    float(row["owner_effect_estimate"]),
                    float(row["owner_effect_ci_lower"]),
                    float(row["owner_effect_ci_upper"]),
                )
                for row in rows
                if str(row["model_id"]) == model
                and row["confidence_visibility"] == visibility
            )
            if not values:
                continue
            axis.errorbar(
                [item[0] for item in values],
                [item[1] for item in values],
                yerr=[
                    [item[1] - item[2] for item in values],
                    [item[3] - item[1] for item in values],
                ],
                marker="o",
                capsize=2,
                label=visibility,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(model)
        axis.set_xticks([2, 5, 10, 20])
        axis.set_xlabel("Error cost, L")
        axis.set_ylabel("AI minus human VERIFY_FIRST rate")
        axis.grid(axis="y", alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False)
    paths = _save(fig, output_base)
    plt.close(fig)
    return paths


def plot_interaction(
    rows: Sequence[Mapping[str, Any]], output_base: Path
) -> dict[str, str]:
    plt = _pyplot()
    models = sorted({str(row["model_id"]) for row in rows})
    fig, axes = _axes(plt, models)
    for axis, model in zip(axes, models):
        values = sorted(
            (
                float(row["error_cost"]),
                float(row["estimate"]),
                float(row["ci_lower"]),
                float(row["ci_upper"]),
            )
            for row in rows
            if str(row["model_id"]) == model
        )
        axis.errorbar(
            [item[0] for item in values],
            [item[1] for item in values],
            yerr=[
                [item[1] - item[2] for item in values],
                [item[3] - item[1] for item in values],
            ],
            marker="o",
            capsize=2,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(model)
        axis.set_xticks([2, 5, 10, 20])
        axis.set_xlabel("Error cost, L")
        axis.set_ylabel("Owner gap change (visible − hidden)")
        axis.grid(axis="y", alpha=0.25)
    paths = _save(fig, output_base)
    plt.close(fig)
    return paths


def plot_policy_cost(
    rows: Sequence[Mapping[str, Any]], output_base: Path
) -> dict[str, str]:
    plt = _pyplot()
    models = sorted({str(row["model_id"]) for row in rows})
    fig, axes = _axes(plt, models)
    preferred = (
        "direct",
        "raw_confidence",
        "calibrated_confidence",
        "always_use_unverified",
        "always_verify",
        "oracle",
    )
    for axis, model in zip(axes, models):
        grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
        for row in rows:
            if str(row["model_id"]) != model:
                continue
            value = _number(row.get("mean_cost"))
            if value is not None:
                grouped[(str(row["policy"]), float(row["error_cost"]))].append(value)
        for policy in preferred:
            values = sorted(
                (
                    error_cost,
                    sum(costs) / len(costs),
                )
                for (name, error_cost), costs in grouped.items()
                if name == policy
            )
            if values:
                axis.plot(
                    [item[0] for item in values],
                    [item[1] for item in values],
                    marker="o",
                    label=policy.replace("_", " "),
                )
        axis.set_title(model)
        axis.set_xticks([2, 5, 10, 20])
        axis.set_xlabel("Error cost, L")
        axis.set_ylabel("Mean realized cost")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False, ncol=2)
    paths = _save(fig, output_base)
    plt.close(fig)
    return paths


def plot_prompt_visibility_robustness(
    rows: Sequence[Mapping[str, Any]], output_base: Path
) -> dict[str, str]:
    """Four-panel primary vs paraphrase confidence-visibility effects."""
    plt = _pyplot()
    models = sorted({str(row["model_id"]) for row in rows})
    fig, axes = _axes(plt, models)
    labels = {
        "v2_owner_match_v1": ("primary", "o", "-"),
        "v2_owner_match_paraphrase_v1": ("paraphrase", "s", "--"),
    }
    for axis, model in zip(axes, models):
        model_rows = [row for row in rows if str(row["model_id"]) == model]
        for family, (label, marker, line) in labels.items():
            by_cost: dict[float, list[float]] = defaultdict(list)
            for row in model_rows:
                if str(row.get("prompt_family")) != family:
                    continue
                value = _number(row.get("estimate"))
                cost = _number(row.get("error_cost"))
                if value is None or cost is None:
                    continue
                by_cost[cost].append(value)
            if not by_cost:
                continue
            costs = sorted(by_cost)
            axis.plot(
                costs,
                [sum(by_cost[cost]) / len(by_cost[cost]) for cost in costs],
                marker=marker,
                linestyle=line,
                label=label,
            )
        axis.axhline(0, color="0.6", linewidth=0.8)
        axis.set_title(model)
        axis.set_xticks([2, 5, 10, 20])
        axis.set_xlabel("Error cost, L")
        axis.set_ylabel("Visibility effect")
        axis.grid(axis="y", alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False)
    paths = _save(fig, output_base)
    plt.close(fig)
    return paths


def generate_v2_figures(
    *,
    factorial_metrics: Sequence[Mapping[str, Any]],
    owner_effects: Sequence[Mapping[str, Any]],
    interactions: Sequence[Mapping[str, Any]],
    policy_comparison: Sequence[Mapping[str, Any]],
    output_directory: str | Path,
) -> dict[str, dict[str, str]]:
    directory = Path(output_directory)
    primary_factorial = [
        row
        for row in factorial_metrics
        if row.get("prompt_family") == "v2_owner_match_v1"
    ]
    primary_owner = [
        row
        for row in owner_effects
        if row.get("prompt_family") == "v2_owner_match_v1"
    ]
    primary_interactions = [
        row
        for row in interactions
        if row.get("prompt_family") == "v2_owner_match_v1"
    ]
    return {
        "owner_verify_rate": plot_factor_metric(
            primary_factorial,
            metric="verify_rate",
            ylabel="VERIFY_FIRST rate",
            output_base=directory / "figure_owner_verify_rate",
        ),
        "owner_unsafe_unverified": plot_factor_metric(
            primary_factorial,
            metric="unsafe_unverified_rate",
            ylabel="Unsafe unverified use among wrong answers",
            output_base=directory / "figure_owner_unsafe_unverified",
        ),
        "owner_effect": plot_owner_effect(
            primary_owner, directory / "figure_owner_effect"
        ),
        "confidence_interaction": plot_interaction(
            primary_interactions, directory / "figure_confidence_interaction"
        ),
        "policy_cost": plot_policy_cost(
            [
                row
                for row in policy_comparison
                if row.get("prompt_family") == "v2_owner_match_v1"
            ],
            directory / "figure_policy_cost",
        ),
    }
