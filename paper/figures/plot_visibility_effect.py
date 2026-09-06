"""Build paper Figure 1 from the frozen visibility-effect table.

Reads paper_outputs/v2/confidence_visibility_effects.csv only. Does not
write back to analysis outputs or alter that CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper_outputs" / "v2" / "confidence_visibility_effects.csv"
OUTPUT = Path(__file__).resolve().parent / "figure_visibility_effect.pdf"

MODEL_ORDER = (
    "anthropic_sonnet5",
    "google_gemini38_flash",
    "openai_gpt56_sol",
    "xai_grok420_nonreasoning",
)
MODEL_LABELS = {
    "anthropic_sonnet5": "Claude Sonnet 5",
    "google_gemini38_flash": "Gemini 3.8 Flash",
    "openai_gpt56_sol": "GPT-5.6 Sol",
    "xai_grok420_nonreasoning": "Grok 4.20",
}
# Okabe–Ito, colorblind-safe.
MODEL_STYLE = {
    "anthropic_sonnet5": ("#0072B2", "o"),
    "google_gemini38_flash": ("#009E73", "s"),
    "openai_gpt56_sol": ("#D55E00", "D"),
    "xai_grok420_nonreasoning": ("#CC79A7", "^"),
}
OWNERS = (("human", "Human authority"), ("ai_system", "AI authority"))
COSTS = (2.0, 5.0, 10.0, 20.0)


def _load_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row["prompt_family"] == "v2_owner_match_v1"
    ]


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"canonical table not found: {SOURCE}")

    grouped: dict[tuple[str, str, float], dict[str, float]] = {}
    for row in _load_rows():
        key = (
            row["model_id"],
            row["decision_owner"],
            float(row["error_cost"]),
        )
        grouped[key] = {
            "estimate": 100.0 * float(row["estimate"]),
            "ci_lower": 100.0 * float(row["ci_lower"]),
            "ci_upper": 100.0 * float(row["ci_upper"]),
        }

    expected = {
        (model, owner, cost)
        for model in MODEL_ORDER
        for owner, _ in OWNERS
        for cost in COSTS
    }
    missing = expected - set(grouped)
    if missing:
        raise ValueError(f"missing visibility-effect cells: {sorted(missing)}")

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.50), sharey=True)
    handles = []
    labels = []
    n_models = len(MODEL_ORDER)
    dodge = 0.22

    for axis, (owner, title) in zip(axes, OWNERS):
        axis.axhline(0.0, color="0.35", linewidth=0.8, zorder=0)
        for index, model in enumerate(MODEL_ORDER):
            color, marker = MODEL_STYLE[model]
            xs = []
            ys = []
            yerr_lo = []
            yerr_hi = []
            for cost in COSTS:
                cell = grouped[(model, owner, cost)]
                xs.append(cost + dodge * (index - (n_models - 1) / 2))
                ys.append(cell["estimate"])
                yerr_lo.append(cell["estimate"] - cell["ci_lower"])
                yerr_hi.append(cell["ci_upper"] - cell["estimate"])
            line = axis.errorbar(
                xs,
                ys,
                yerr=[yerr_lo, yerr_hi],
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.15,
                capsize=2.0,
                capthick=0.8,
                elinewidth=0.8,
                label=MODEL_LABELS[model],
            )
            if axis is axes[0]:
                handles.append(line)
                labels.append(MODEL_LABELS[model])
        axis.set_title(title)
        axis.set_xticks(list(COSTS))
        axis.set_xlabel(r"Error cost $L$")
        axis.set_xlim(0.4, 21.6)
        axis.grid(axis="y", color="0.85", linewidth=0.5)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    axes[0].set_ylabel("Visibility effect (pp)")
    axes[0].set_ylim(-30, 30)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        handlelength=1.6,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.30, wspace=0.10)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
