"""Build paper Figure 2 from frozen V2 tables and scored decisions.

Reads paper_outputs/v2/factorial_metrics.csv and scored_decisions.json.
Does not write back to analysis outputs or alter those files.

Panel (b) CIs use question_level_bootstrap on all 500 questions, with the
wrong-answer subset applied inside each replicate (EXPERIMENT_V2.md and
the paper statistical plan). This is not the same as first restricting to
the 87 wrong answers and resampling those IDs, which is what V2 owner-gap
code does.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.bootstrap import mean, question_level_bootstrap

ROOT = Path(__file__).resolve().parents[2]
FACTORIAL = ROOT / "paper_outputs" / "v2" / "factorial_metrics.csv"
SCORED = ROOT / "paper_outputs" / "v2" / "scored_decisions.json"
OUTPUT = Path(__file__).resolve().parent / "figure_openai_divergence.pdf"

MODEL_ID = "openai_gpt56_sol"
PROMPT_FAMILY = "v2_owner_match_v1"
COSTS = (2.0, 5.0, 10.0, 20.0)
SEED = 20260904
N_RESAMPLES = 5000
CONFIDENCE_LEVEL = 0.95

CANONICAL_DISAGREEMENT = {
    ("human", "hidden"): (19.6, 20.6, 24.8, 27.2),
    ("human", "visible"): (0.8, 0.2, 0.4, 2.2),
    ("ai_system", "hidden"): (20.8, 22.8, 24.2, 25.4),
    ("ai_system", "visible"): (0.6, 0.4, 0.6, 1.2),
}
CANONICAL_UNSAFE = {
    ("human", "hidden"): 32.18,
    ("human", "visible"): 56.32,
    ("ai_system", "hidden"): 33.33,
    ("ai_system", "visible"): 54.02,
}

# Hidden vs visible is the dominant contrast; owner is secondary.
VISIBILITY_COLOR = {"hidden": "#0072B2", "visible": "#D55E00"}
OWNER_MARKER = {"human": "o", "ai_system": "s"}
OWNER_LINE = {"human": "-", "ai_system": (0, (3, 1.4))}
OWNER_LABEL = {"human": "Human", "ai_system": "AI"}
VIS_LABEL = {"hidden": "Hidden", "visible": "Visible"}
SERIES = (
    ("human", "hidden"),
    ("ai_system", "hidden"),
    ("human", "visible"),
    ("ai_system", "visible"),
)


def _rounded(value: float, digits: int) -> float:
    return round(value + 1e-12, digits)


def _load_disagreement() -> dict[tuple[str, str], dict[float, float]]:
    with FACTORIAL.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], dict[float, float]] = {}
    for row in rows:
        if row["model_id"] != MODEL_ID or row["prompt_family"] != PROMPT_FAMILY:
            continue
        key = (row["decision_owner"], row["confidence_visibility"])
        grouped.setdefault(key, {})[float(row["error_cost"])] = (
            100.0 * float(row["direct_vs_raw_disagreement_rate"])
        )
    for key, expected in CANONICAL_DISAGREEMENT.items():
        if key not in grouped:
            raise ValueError(f"missing disagreement series {key}")
        observed = tuple(_rounded(grouped[key][cost], 1) for cost in COSTS)
        if observed != expected:
            raise ValueError(
                f"disagreement values for {key} {observed} != canonical {expected}"
            )
    return grouped


def _unsafe_ci(
    rows: list[dict],
    *,
    seed: int,
) -> tuple[float, float, float]:
    def statistic(sample) -> float:
        wrong = [row for row in sample if not row["is_correct"]]
        return mean(row["used_unverified"] for row in wrong)

    result = question_level_bootstrap(
        rows,
        statistic,
        n_resamples=N_RESAMPLES,
        confidence_level=CONFIDENCE_LEVEL,
        seed=seed,
    )
    if result.n_questions != 500:
        raise ValueError(
            f"expected 500 resampled questions, got {result.n_questions}"
        )
    return (
        100.0 * result.estimate,
        100.0 * result.lower,
        100.0 * result.upper,
    )


def _load_unsafe() -> dict[tuple[str, str], dict[str, float]]:
    records = json.loads(SCORED.read_text())
    cells: dict[tuple[str, str], list[dict]] = {key: [] for key in SERIES}
    for row in records:
        if row["model_id"] != MODEL_ID or row.get("prompt_family") != PROMPT_FAMILY:
            continue
        if float(row["error_cost"]) != 20.0:
            continue
        key = (row["decision_owner"], row["confidence_visibility"])
        if key in cells:
            cells[key].append(row)

    output: dict[tuple[str, str], dict[str, float]] = {}
    for index, key in enumerate(SERIES):
        rows = cells[key]
        if len(rows) != 500:
            raise ValueError(f"{key}: expected 500 decisions, got {len(rows)}")
        wrong_n = sum(not row["is_correct"] for row in rows)
        if wrong_n != 87:
            raise ValueError(f"{key}: expected 87 wrong answers, got {wrong_n}")
        estimate, lower, upper = _unsafe_ci(rows, seed=SEED + index)
        canonical = CANONICAL_UNSAFE[key]
        if _rounded(estimate, 2) != canonical:
            raise ValueError(
                f"unsafe rate for {key} {_rounded(estimate, 2)} != canonical {canonical}"
            )
        output[key] = {"estimate": estimate, "lower": lower, "upper": upper}
    return output


def main() -> None:
    disagreement = _load_disagreement()
    unsafe = _load_unsafe()

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.55))
    axis_a, axis_b = axes

    for owner, visibility in SERIES:
        color = VISIBILITY_COLOR[visibility]
        axis_a.plot(
            list(COSTS),
            [disagreement[(owner, visibility)][cost] for cost in COSTS],
            color=color,
            marker=OWNER_MARKER[owner],
            markersize=4.5,
            linewidth=1.2,
            linestyle=OWNER_LINE[owner],
            markerfacecolor=color,
            markeredgecolor=color,
        )
    axis_a.set_xticks(list(COSTS))
    axis_a.set_xlabel(r"Error cost $L$")
    axis_a.set_ylabel("Disagreement with confidence-implied policy (%)")
    axis_a.set_ylim(-1, 32)
    axis_a.set_xlim(0.4, 21.6)
    axis_a.grid(axis="y", color="0.85", linewidth=0.5)
    axis_a.set_axisbelow(True)
    axis_a.text(
        0.02,
        0.96,
        "(a)",
        transform=axis_a.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )

    positions = {
        ("human", "hidden"): 0.0,
        ("human", "visible"): 0.22,
        ("ai_system", "hidden"): 1.00,
        ("ai_system", "visible"): 1.22,
    }
    bar_width = 0.20
    for key in SERIES:
        cell = unsafe[key]
        x = positions[key]
        axis_b.bar(
            x,
            cell["estimate"],
            width=bar_width,
            color=VISIBILITY_COLOR[key[1]],
            edgecolor="none",
        )
        axis_b.errorbar(
            x,
            cell["estimate"],
            yerr=[
                [cell["estimate"] - cell["lower"]],
                [cell["upper"] - cell["estimate"]],
            ],
            fmt="none",
            ecolor="0.15",
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
        )
    axis_b.set_xticks([0.11, 1.11])
    axis_b.set_xticklabels(["Human authority", "AI authority"])
    axis_b.set_ylabel("Wrong answers left unverified at $L{=}20$ (%)")
    axis_b.set_ylim(0, 80)
    axis_b.set_xlim(-0.25, 1.47)
    axis_b.grid(axis="y", color="0.85", linewidth=0.5)
    axis_b.set_axisbelow(True)
    axis_b.text(
        0.02,
        0.96,
        "(b)",
        transform=axis_b.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )

    for axis in axes:
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    legend = [
        Patch(facecolor=VISIBILITY_COLOR["hidden"], edgecolor="none", label="Hidden"),
        Patch(facecolor=VISIBILITY_COLOR["visible"], edgecolor="none", label="Visible"),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=OWNER_MARKER["human"],
            linestyle=OWNER_LINE["human"],
            markersize=4.5,
            label="Human authority",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=OWNER_MARKER["ai_system"],
            linestyle=OWNER_LINE["ai_system"],
            markersize=4.5,
            label="AI authority",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.7,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.30, wspace=0.28)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)

    for key, cell in unsafe.items():
        print(
            f"{key[0]:10} {key[1]:8} "
            f"{cell['estimate']:.2f} [{cell['lower']:.2f}, {cell['upper']:.2f}]"
        )


if __name__ == "__main__":
    main()
