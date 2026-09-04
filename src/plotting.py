"""Publication-quality matplotlib figures for analysis summaries."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib; tabular analysis does not."
        ) from exc
    return plt


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _save(fig: Any, output_base: str | Path) -> dict[str, str]:
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {"pdf": str(base.with_suffix(".pdf")), "png": str(base.with_suffix(".png"))}
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    return paths


def _empty_figure(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _metric_figure(
    records: Iterable[Mapping[str, Any]],
    *,
    metric_key: str,
    ylabel: str,
    output_base: str | Path,
    model_key: str = "model_id",
    error_cost_key: str = "error_cost",
    lower_key: str | None = None,
    upper_key: str | None = None,
    expected_error_costs: Sequence[float] = (2, 5, 10, 20),
) -> dict[str, str]:
    plt = _pyplot()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    grouped: dict[str, list[tuple[float, float, float | None, float | None]]] = (
        defaultdict(list)
    )
    for row in records:
        model = row.get(model_key)
        x_value = _number(row.get(error_cost_key))
        y_value = _number(row.get(metric_key))
        if model is None or x_value is None or y_value is None:
            continue
        lower = _number(row.get(lower_key)) if lower_key else None
        upper = _number(row.get(upper_key)) if upper_key else None
        grouped[str(model)].append((x_value, y_value, lower, upper))

    if not grouped:
        _empty_figure(ax, "No estimable data")
    else:
        for model, values in sorted(grouped.items()):
            values.sort(key=lambda item: item[0])
            x = [item[0] for item in values]
            y = [item[1] for item in values]
            line = ax.plot(x, y, marker="o", linewidth=1.7, label=model)[0]
            if any(item[2] is not None and item[3] is not None for item in values):
                lower_values = [
                    item[2] if item[2] is not None else item[1] for item in values
                ]
                upper_values = [
                    item[3] if item[3] is not None else item[1] for item in values
                ]
                ax.fill_between(
                    x,
                    lower_values,
                    upper_values,
                    alpha=0.16,
                    color=line.get_color(),
                    linewidth=0,
                )
        ax.set_xticks(list(expected_error_costs))
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Error cost, L")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False)
    paths = _save(fig, output_base)
    plt.close(fig)
    return paths


def plot_verify_rate(
    records: Iterable[Mapping[str, Any]],
    output_base: str | Path,
) -> dict[str, str]:
    """Create separate PDF and PNG VERIFY-rate figures."""
    return _metric_figure(
        records,
        metric_key="verify_rate",
        ylabel="VERIFY rate",
        output_base=output_base,
        lower_key="verify_rate_ci_lower",
        upper_key="verify_rate_ci_upper",
    )


def plot_unsafe_reliance(
    records: Iterable[Mapping[str, Any]],
    output_base: str | Path,
) -> dict[str, str]:
    """Create separate PDF and PNG unsafe-reliance figures."""
    return _metric_figure(
        records,
        metric_key="unsafe_reliance",
        ylabel="Unsafe reliance rate",
        output_base=output_base,
        lower_key="unsafe_reliance_ci_lower",
        upper_key="unsafe_reliance_ci_upper",
    )


def plot_policy_cost(
    records: Iterable[Mapping[str, Any]],
    output_base: str | Path,
    *,
    preferred_policy_order: Sequence[str] = (
        "direct",
        "raw_confidence",
        "calibrated_confidence",
        "always_rely",
        "always_verify",
        "oracle",
    ),
) -> dict[str, str]:
    """Plot expected cost by policy, retaining partial policy/stake series."""
    plt = _pyplot()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in records:
        model = row.get("model_id")
        policy = row.get("policy")
        error_cost = _number(row.get("error_cost"))
        cost = _number(row.get("mean_cost", row.get("realized_cost")))
        if model is None or policy is None or error_cost is None or cost is None:
            continue
        grouped[(str(model), str(policy))].append((error_cost, cost))

    if not grouped:
        _empty_figure(ax, "No estimable data")
    else:
        policy_rank = {
            policy: index for index, policy in enumerate(preferred_policy_order)
        }
        ordered_groups = sorted(
            grouped.items(),
            key=lambda item: (
                item[0][0],
                policy_rank.get(item[0][1], len(policy_rank)),
                item[0][1],
            ),
        )
        models = sorted({key[0] for key in grouped})
        for (model, policy), values in ordered_groups:
            values.sort()
            label = policy if len(models) == 1 else f"{model}: {policy}"
            ax.plot(
                [value[0] for value in values],
                [value[1] for value in values],
                marker="o",
                linewidth=1.5,
                label=label.replace("_", " "),
            )
        ax.set_xticks([2, 5, 10, 20])
        ax.set_xlabel("Error cost, L")
        ax.set_ylabel("Mean realized cost")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, ncol=2)
    paths = _save(fig, output_base)
    plt.close(fig)
    return paths


def generate_core_figures(
    decision_summary: Iterable[Mapping[str, Any]],
    policy_summary: Iterable[Mapping[str, Any]],
    output_directory: str | Path,
) -> dict[str, dict[str, str]]:
    """Generate the three required figures, each as PDF and PNG."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    decision_rows = list(decision_summary)
    policy_rows = list(policy_summary)
    return {
        "verify_rate": plot_verify_rate(
            decision_rows, directory / "figure_verify_rate"
        ),
        "unsafe_reliance": plot_unsafe_reliance(
            decision_rows, directory / "figure_unsafe_reliance"
        ),
        "policy_cost": plot_policy_cost(policy_rows, directory / "figure_policy_cost"),
    }
