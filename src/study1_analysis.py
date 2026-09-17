"""Exploratory analysis of the Study 1 primary 2,800-cell dataset."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bootstrap import bootstrap_mean, paired_bootstrap
from .config import PROJECT_ROOT
from .study1_primary import (
    TASK001_ID_HASH,
    _outcome_row,
    postvalidate_primary,
)
from .study1_sample import load_frozen_ids
from .study1_schemas import (
    STUDY1_GRID_L10,
    STUDY1_GRID_L20,
    STUDY1_MODEL_ALIASES,
    Study1ExperimentConfig,
)
from .v2_prompts import PRIMARY_PROMPT_FAMILY
from .v2_scoring import confidence_policy_action

BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 5000
VERIFY = "VERIFY_FIRST"
USE = "USE_UNVERIFIED"
MODEL_LABELS = {
    "openai_gpt56_sol": "GPT",
    "anthropic_sonnet5": "Claude",
}
NEAR_THRESHOLD = {
    10.0: (0.89, 0.91, 0.90),
    20.0: (0.94, 0.96, 0.95),
}
EXTREME = {
    10.0: (0.80, 0.99),
    20.0: (0.90, 0.99),
}


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

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
    return plt


def _pp(value: float) -> float:
    return 100.0 * value


def _fmt_pp(value: float, digits: int = 1) -> str:
    return f"{_pp(value):.{digits}f}"


def _fmt_ci(result: Any) -> str:
    return f"{_fmt_pp(result.estimate)} [{_fmt_pp(result.lower)}, {_fmt_pp(result.upper)}]"


def _is_verify(action: Any) -> int:
    return int(str(action) == VERIFY)


def successful_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_outcome_row(item) for item in payload.get("successful", [])]


def _subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    L: float | None = None,
    condition: str | None = None,
    displayed: float | None = None,
    correct: bool | None = None,
    manipulated_only: bool = False,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if model is not None and row["model_alias"] != model:
            continue
        if L is not None and float(row["L"]) != float(L):
            continue
        if condition is not None and row["display_condition"] != condition:
            continue
        if manipulated_only and not str(row["display_condition"]).startswith("manipulated"):
            continue
        if displayed is not None:
            value = row["displayed_confidence"]
            if value is None or abs(float(value) - float(displayed)) > 1e-12:
                continue
        if correct is not None and bool(row["stage1_correct"]) != correct:
            continue
        selected.append(dict(row))
    return selected


def verification_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return float("nan")
    return sum(_is_verify(row["parsed_action"]) for row in rows) / len(rows)


def rate_with_ci(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    marked = [
        {"question_id": row["question_id"], "verify": _is_verify(row["parsed_action"])}
        for row in rows
    ]
    result = bootstrap_mean(
        marked,
        "verify",
        question_key="question_id",
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    return {
        "n": len(rows),
        "n_questions": result.n_questions,
        "rate": result.estimate,
        "ci_lower": result.lower,
        "ci_upper": result.upper,
        "n_resamples": result.n_resamples,
        "seed": result.seed,
    }


def paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    L: float,
    low: float,
    high: float,
    correct: bool | None = None,
) -> dict[str, Any]:
    left_rows = _subset(
        rows, model=model, L=L, displayed=low, correct=correct, manipulated_only=True
    )
    right_rows = _subset(
        rows, model=model, L=L, displayed=high, correct=correct, manipulated_only=True
    )
    left_by = {row["question_id"]: _is_verify(row["parsed_action"]) for row in left_rows}
    right_by = {row["question_id"]: _is_verify(row["parsed_action"]) for row in right_rows}
    shared = sorted(set(left_by) & set(right_by))
    pairs = [(qid, left_by[qid], right_by[qid]) for qid in shared]
    boot = paired_bootstrap(
        pairs, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
    )
    v_to_u = sum(1 for _qid, left, right in pairs if left == 1 and right == 0)
    u_to_v = sum(1 for _qid, left, right in pairs if left == 0 and right == 1)
    no_change = sum(1 for _qid, left, right in pairs if left == right)
    return {
        "model_alias": model,
        "model_label": MODEL_LABELS[model],
        "L": L,
        "low_displayed": low,
        "high_displayed": high,
        "n_questions": len(shared),
        "rate_low": mean_or_nan(left_by[qid] for qid in shared),
        "rate_high": mean_or_nan(right_by[qid] for qid in shared),
        "difference": boot.estimate,
        "ci_lower": boot.lower,
        "ci_upper": boot.upper,
        "verify_to_use": v_to_u,
        "use_to_verify": u_to_v,
        "no_change": no_change,
        "seed": BOOTSTRAP_SEED,
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "correctness_split": (
            None if correct is None else ("correct" if correct else "incorrect")
        ),
    }


def mean_or_nan(values: Any) -> float:
    usable = [float(value) for value in values]
    return sum(usable) / len(usable) if usable else float("nan")


def manipulated_curve(rows: Sequence[Mapping[str, Any]], model: str, L: float) -> list[dict[str, Any]]:
    grid = STUDY1_GRID_L10 if L == 10.0 else STUDY1_GRID_L20
    threshold = 1.0 - (1.0 / L)
    points = []
    for displayed in grid:
        subset = _subset(
            rows, model=model, L=L, displayed=displayed, manipulated_only=True
        )
        stats = rate_with_ci(subset)
        mechanical = int(
            confidence_policy_action(displayed, L, verification_cost=1.0) == VERIFY
        )
        points.append(
            {
                "model_alias": model,
                "model_label": MODEL_LABELS[model],
                "L": L,
                "displayed_confidence": displayed,
                "condition": "manipulated",
                "n": stats["n"],
                "verification_rate": stats["rate"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
                "mechanical_rule_rate": mechanical,
                "threshold": threshold,
                "seed": BOOTSTRAP_SEED,
            }
        )
    return points


def reference_rates(rows: Sequence[Mapping[str, Any]], model: str, L: float) -> dict[str, Any]:
    hidden = rate_with_ci(_subset(rows, model=model, L=L, condition="hidden"))
    visible = rate_with_ci(
        _subset(rows, model=model, L=L, condition="true_confidence_visible")
    )
    return {
        "model_alias": model,
        "L": L,
        "hidden": hidden,
        "true_visible": visible,
    }


def classify_trajectory(actions: Sequence[int]) -> dict[str, Any]:
    transitions = sum(
        1 for left, right in zip(actions, actions[1:]) if left != right
    )
    reverse = any(
        left == 0 and right == 1 for left, right in zip(actions, actions[1:])
    )
    non_increasing = all(
        left >= right for left, right in zip(actions, actions[1:])
    )
    seen_zero = False
    single_threshold = True
    for value in actions:
        if value == 0:
            seen_zero = True
        elif seen_zero:
            single_threshold = False
            break
    return {
        "n_transitions": transitions,
        "reverse_use_to_verify": reverse,
        "perfectly_non_increasing": non_increasing,
        "single_threshold_shape": single_threshold and not reverse,
        "pattern": "".join("V" if value else "U" for value in actions),
    }


def trajectory_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not str(row["display_condition"]).startswith("manipulated"):
            continue
        grouped[(row["question_id"], row["model_alias"], float(row["L"]))].append(row)
    output = []
    for (qid, model, L), items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda row: float(row["displayed_confidence"]))
        actions = [_is_verify(row["parsed_action"]) for row in ordered]
        classified = classify_trajectory(actions)
        output.append(
            {
                "question_id": qid,
                "model_alias": model,
                "model_label": MODEL_LABELS[model],
                "L": L,
                "displayed_values": [float(row["displayed_confidence"]) for row in ordered],
                "actions": [row["parsed_action"] for row in ordered],
                **classified,
                "stage1_correct": ordered[0]["stage1_correct"] if ordered else None,
            }
        )
    return output


def load_historical_v2(
    config: Study1ExperimentConfig,
    question_ids: Sequence[str],
) -> list[dict[str, Any]]:
    wanted = set(question_ids)
    connection = sqlite3.connect(f"file:{config.historical_sqlite()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT record_json FROM requests
            WHERE stage = 'verification' AND status = 'success' AND record_json IS NOT NULL
            """
        ).fetchall()
    finally:
        connection.close()
    output = []
    for (raw,) in rows:
        record = json.loads(raw)
        if record.get("example_id") not in wanted:
            continue
        if record.get("decision_owner") != "ai_system":
            continue
        if record.get("prompt_family") != PRIMARY_PROMPT_FAMILY:
            continue
        if record.get("model_id") not in STUDY1_MODEL_ALIASES:
            continue
        if float(record.get("error_cost")) not in {10.0, 20.0}:
            continue
        if record.get("confidence_visibility") not in {"hidden", "visible"}:
            continue
        output.append(
            {
                "question_id": record["example_id"],
                "model_alias": record["model_id"],
                "L": float(record["error_cost"]),
                "visibility": record["confidence_visibility"],
                "parsed_action": record["action"],
            }
        )
    return output


def historical_comparison(
    current: Sequence[Mapping[str, Any]],
    historical: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mapping = {
        "hidden": "hidden",
        "visible": "true_confidence_visible",
    }
    output = []
    for model in STUDY1_MODEL_ALIASES:
        for L in (10.0, 20.0):
            for hist_vis, cur_cond in mapping.items():
                cur = _subset(current, model=model, L=L, condition=cur_cond)
                hist = [
                    row
                    for row in historical
                    if row["model_alias"] == model
                    and row["L"] == L
                    and row["visibility"] == hist_vis
                ]
                cur_by = {
                    row["question_id"]: _is_verify(row["parsed_action"]) for row in cur
                }
                hist_by = {
                    row["question_id"]: _is_verify(row["parsed_action"]) for row in hist
                }
                shared = sorted(set(cur_by) & set(hist_by))
                pairs = [(qid, cur_by[qid], hist_by[qid]) for qid in shared]
                boot = paired_bootstrap(
                    pairs, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
                )
                agree = sum(1 for _qid, left, right in pairs if left == right)
                output.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "L": L,
                        "condition": cur_cond,
                        "historical_visibility": hist_vis,
                        "n_questions": len(shared),
                        "current_rate": mean_or_nan(cur_by[qid] for qid in shared),
                        "historical_rate": mean_or_nan(hist_by[qid] for qid in shared),
                        "current_minus_historical": boot.estimate,
                        "ci_lower": boot.lower,
                        "ci_upper": boot.upper,
                        "question_agreement": agree,
                        "question_agreement_rate": (
                            agree / len(shared) if shared else float("nan")
                        ),
                    }
                )
    return output


def recommend_gate(
    *,
    extreme_gpt: Sequence[Mapping[str, Any]],
    near_gpt: Sequence[Mapping[str, Any]],
    extreme_claude: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Mapping[str, Any]],
    historical: Sequence[Mapping[str, Any]],
    validation_ok: bool,
) -> tuple[str, str, str]:
    if not validation_ok:
        return (
            "B",
            "AMBIGUOUS — inspect before further calls",
            "Primary data validation did not pass, so the Study-1 gate cannot be applied to a complete 2,800-cell dataset.",
        )
    gpt_extremes = [float(row["difference"]) for row in extreme_gpt]
    claude_extremes = [float(row["difference"]) for row in extreme_claude]
    gpt_nears = [float(row["difference"]) for row in near_gpt]
    gpt_min_extreme = min(gpt_extremes) if gpt_extremes else float("nan")
    claude_min_extreme = min(claude_extremes) if claude_extremes else float("nan")
    gpt_flip_questions = max(
        (int(row["verify_to_use"]) + int(row["use_to_verify"]) for row in extreme_gpt),
        default=0,
    )
    gpt_traj = [row for row in trajectories if row["model_alias"] == "openai_gpt56_sol"]
    reverse_frac = (
        sum(1 for row in gpt_traj if row["reverse_use_to_verify"]) / len(gpt_traj)
        if gpt_traj
        else 1.0
    )
    hist_abs = [abs(float(row["current_minus_historical"])) for row in historical]
    hist_max = max(hist_abs) if hist_abs else 0.0
    direction_ok = all(diff >= 0.05 for diff in gpt_extremes)
    substantial = gpt_min_extreme >= 0.10
    distributed = gpt_flip_questions >= 15
    claude_informative = any(abs(diff) >= 0.05 for diff in claude_extremes) or all(
        diff >= 0 for diff in claude_extremes
    )
    if hist_max >= 0.25:
        return (
            "D",
            "PASS PRIMARY — but design issue requires correction before repeats",
            "Displayed-confidence effects may exist, but current hidden/true-visible rates differ from historical V2 by enough that endpoint/model drift should be inspected before spending the repeat budget.",
        )
    if substantial and direction_ok and distributed and reverse_frac < 0.5:
        return (
            "C",
            "PASS PRIMARY — run stability repeats next",
            "GPT shows a large within-question displayed-confidence effect in the expected direction, it is not confined to a handful of items, and Claude is at least an informative contrast. Repeats are still required before claiming the effect exceeds ordinary rerun noise.",
        )
    if max(gpt_extremes, default=0) < 0.03 and max(claude_extremes, default=0) < 0.03:
        return (
            "A",
            "FAIL / rethink",
            "Manipulated displayed confidence produced little systematic change in verification. That is a reason to rethink the causal-control framing, not to scale.",
        )
    return (
        "B",
        "AMBIGUOUS — inspect before further calls",
        "The primary data are mixed or only weakly consistent with a large displayed-confidence effect. Inspect parser issues, historical drift, and item-level trajectories before authorizing repeats.",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value)
                        if isinstance(value, (list, dict, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _plot_response_curve(
    points: Sequence[Mapping[str, Any]],
    references: Mapping[str, Any],
    output: Path,
    title: str,
) -> None:
    plt = _pyplot()
    xs = [float(row["displayed_confidence"]) for row in points]
    ys = [float(row["verification_rate"]) for row in points]
    lo = [float(row["ci_lower"]) for row in points]
    hi = [float(row["ci_upper"]) for row in points]
    mechanical = [float(row["mechanical_rule_rate"]) for row in points]
    threshold = float(points[0]["threshold"])
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    ax.step(
        xs,
        mechanical,
        where="mid",
        color="#888888",
        linewidth=1.4,
        linestyle="--",
        label="Mechanical rule",
    )
    yerr = [
        [max(0.0, y - l) for y, l in zip(ys, lo)],
        [max(0.0, h - y) for y, h in zip(ys, hi)],
    ]
    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        fmt="o-",
        color="#1f4e79",
        capsize=3,
        label="Observed VERIFY rate",
    )
    ax.axvline(threshold, color="#b35806", linewidth=1, linestyle=":", label="Threshold")
    ax.axhline(
        references["hidden"]["rate"],
        color="#4d9221",
        linewidth=1,
        linestyle="--",
        label="Hidden (reference)",
    )
    ax.axhline(
        references["true_visible"]["rate"],
        color="#c51b7d",
        linewidth=1,
        linestyle="--",
        label="True-visible (reference)",
    )
    ax.set_xlabel("Displayed confidence")
    ax.set_ylabel("P(VERIFY_FIRST)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _plot_contrasts(rows: Sequence[Mapping[str, Any]], output: Path, title: str) -> None:
    plt = _pyplot()
    labels = [
        f"{row['model_label']}\nL={int(row['L'])}\n{row['low_displayed']:g} vs {row['high_displayed']:g}"
        for row in rows
    ]
    estimates = [100 * float(row["difference"]) for row in rows]
    lower = [100 * float(row["ci_lower"]) for row in rows]
    upper = [100 * float(row["ci_upper"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    positions = list(range(len(rows)))
    ax.bar(
        positions,
        estimates,
        color=["#1f4e79" if row["model_alias"] == "openai_gpt56_sol" else "#b35806" for row in rows],
        yerr=[
            [max(0.0, e - l) for e, l in zip(estimates, lower)],
            [max(0.0, u - e) for e, u in zip(estimates, upper)],
        ],
        capsize=3,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Verification difference (pp)")
    ax.set_title(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _plot_trajectories(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    labels = []
    noninc = []
    reverse = []
    for model in STUDY1_MODEL_ALIASES:
        for L in (10.0, 20.0):
            subset = [
                row
                for row in rows
                if row["model_alias"] == model and float(row["L"]) == L
            ]
            labels.append(f"{MODEL_LABELS[model]} L={int(L)}")
            n = len(subset) or 1
            noninc.append(
                100 * sum(1 for row in subset if row["perfectly_non_increasing"]) / n
            )
            reverse.append(
                100 * sum(1 for row in subset if row["reverse_use_to_verify"]) / n
            )
    positions = list(range(len(labels)))
    ax.bar([p - 0.18 for p in positions], noninc, width=0.36, label="Non-increasing", color="#1f4e79")
    ax.bar([p + 0.18 for p in positions], reverse, width=0.36, label="Has reverse flip", color="#b35806")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Percent of question trajectories")
    ax.set_title("Within-question manipulated trajectories")
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _plot_historical(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    labels = []
    current = []
    historical = []
    for row in rows:
        labels.append(
            f"{row['model_label']} L={int(row['L'])}\n{row['condition']}"
        )
        current.append(100 * float(row["current_rate"]))
        historical.append(100 * float(row["historical_rate"]))
    positions = list(range(len(labels)))
    ax.bar([p - 0.18 for p in positions], historical, width=0.36, label="Historical V2", color="#888888")
    ax.bar([p + 0.18 for p in positions], current, width=0.36, label="Study 1 current", color="#1f4e79")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("P(VERIFY_FIRST) percent")
    ax.set_title("Current hidden/true-visible vs historical V2 AI-authority")
    ax.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def write_analysis_method(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Analysis method",
                "",
                "Exploratory analysis of the Study 1 primary 2,800-cell dataset.",
                "",
                "- Sampling unit: question ID. All within-question conditions are kept together.",
                f"- Bootstrap: {BOOTSTRAP_RESAMPLES} question-level resamples, seed {BOOTSTRAP_SEED}, nominal percentile 95% intervals.",
                "- Paired contrasts resample questions, not isolated cells.",
                "- Verification indicator: 1 if parsed action is VERIFY_FIRST, else 0.",
                "- Mechanical rule: VERIFY_FIRST if (1-q)*L > C with C=1; otherwise USE_UNVERIFIED. Equal-threshold cells therefore USE.",
                "- L=10 mathematical threshold q=0.90; L=20 threshold q=0.95.",
                "- True-visible is a question-specific historical-confidence reference, not a point on the constant-score dose-response curve.",
                "- Hidden/true-visible historical comparison uses V2 AI-system primary-family cells on the same questions, models, and L.",
                "- Repeated-generation stability is not estimated here. Intervals are not confirmatory significance tests.",
                "- Correctness-conditioned contrasts are secondary and likely noisy (n=100 already split).",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _script_snapshot() -> str:
    return Path(__file__).read_text(encoding="utf-8")


def write_primary_analysis_bundle(payload: Mapping[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    validation = postvalidate_primary(payload)
    rows = successful_rows(payload)
    config = payload["config"]
    selected, _repeats, selected_hash, _ = load_frozen_ids(config)
    budget = payload["budget"]
    new_initiated = int(payload.get("new_initiated") or 0)
    reused = int(payload.get("reused") or 0)
    smoke_reused = int(payload.get("smoke_reused") or 0)
    prior_primary_completed = int(payload.get("prior_primary_completed") or 0)
    repair_cells = int(payload.get("repair_cells") or 0)
    successful = rows
    failed = payload.get("failed") or []
    network_cells = sum(
        1
        for item in payload.get("results", [])
        if int(item.get("provider_attempts") or 0) > 0
    )
    retry_attempts = max(0, budget.used - network_cells - repair_cells)

    (output / "data_validation.md").write_text(
        "\n".join(
            [
                "# Data validation",
                "",
                f"- Successful unique primary cells: **{validation['successful_unique_primary_cells']}** / 2800",
                f"- Per-model counts: `{json.dumps(validation['by_model'])}`",
                f"- Per-L counts: `{json.dumps({str(k): v for k, v in validation['by_L'].items()})}`",
                f"- Per-condition counts: `{json.dumps(validation['by_condition'])}`",
                f"- Historical V2 unchanged: **{validation['historical_unchanged']}**",
                f"- Frozen ID hash: `{selected_hash}` (Task 001 `{TASK001_ID_HASH}`)",
                f"- Reused Task-002 cells: {smoke_reused}",
                f"- Other already-complete primary cells reused this invocation: {prior_primary_completed}",
                f"- New scientific cells initiated: {new_initiated}",
                "",
                "Issues:" if validation["issues"] else "No material validation issues.",
                *[f"- {issue}" for issue in validation["issues"]],
                "",
            ]
        ),
        encoding="utf-8",
    )

    if not validation["ok"]:
        (output / "report.md").write_text(
            "# Study 1 Primary Causal Pilot\n\n"
            "## 1. Data validation\n\n"
            "Data validation failed materially. Scientific analysis was not started.\n\n"
            + "\n".join(f"- {issue}" for issue in validation["issues"])
            + "\n",
            encoding="utf-8",
        )
        _write_run_manifest(
            output,
            payload,
            validation,
            reused=reused,
            new_initiated=new_initiated,
            retry_attempts=retry_attempts,
            repair_cells=repair_cells,
            gate="B",
            api_cost=sum(float(row.get("estimated_cost_usd") or 0) for row in rows),
        )
        write_analysis_method(output / "analysis_method.md")
        return {"validation": validation, "analyzed": False}

    rate_rows: list[dict[str, Any]] = []
    references = {}
    for model in STUDY1_MODEL_ALIASES:
        for L in (10.0, 20.0):
            curve = manipulated_curve(rows, model, L)
            rate_rows.extend(curve)
            refs = reference_rates(rows, model, L)
            references[(model, L)] = refs
            for name, stats in (("hidden", refs["hidden"]), ("true_confidence_visible", refs["true_visible"])):
                rate_rows.append(
                    {
                        "model_alias": model,
                        "model_label": MODEL_LABELS[model],
                        "L": L,
                        "displayed_confidence": "",
                        "condition": name,
                        "n": stats["n"],
                        "verification_rate": stats["rate"],
                        "ci_lower": stats["ci_lower"],
                        "ci_upper": stats["ci_upper"],
                        "mechanical_rule_rate": "",
                        "threshold": 1.0 - 1.0 / L,
                        "seed": BOOTSTRAP_SEED,
                    }
                )
            slug = "gpt" if model == "openai_gpt56_sol" else "claude"
            _plot_response_curve(
                curve,
                refs,
                figures / f"{slug}_L{int(L)}_response_curve.png",
                f"{MODEL_LABELS[model]}, L={int(L)}",
            )

    near_rows = []
    extreme_rows = []
    near_secondary = []
    extreme_secondary = []
    for model in STUDY1_MODEL_ALIASES:
        for L, (low, high, _thr) in NEAR_THRESHOLD.items():
            near_rows.append(paired_contrast(rows, model=model, L=L, low=low, high=high))
            for flag in (True, False):
                near_secondary.append(
                    paired_contrast(
                        rows, model=model, L=L, low=low, high=high, correct=flag
                    )
                )
        for L, (low, high) in EXTREME.items():
            extreme_rows.append(paired_contrast(rows, model=model, L=L, low=low, high=high))
            for flag in (True, False):
                extreme_secondary.append(
                    paired_contrast(
                        rows, model=model, L=L, low=low, high=high, correct=flag
                    )
                )
    trajectories = trajectory_rows(rows)
    historical = historical_comparison(
        rows, load_historical_v2(config, selected)
    )
    _plot_contrasts(near_rows, figures / "near_threshold_effects.png", "Near-threshold paired contrasts")
    _plot_contrasts(extreme_rows, figures / "extreme_score_effects.png", "Extreme-score paired contrasts")
    _plot_trajectories(trajectories, figures / "trajectory_monotonicity.png")
    _plot_historical(historical, figures / "current_vs_historical.png")

    _write_csv(output / "verification_rates.csv", rate_rows)
    _write_csv(output / "threshold_contrasts.csv", near_rows + extreme_rows)
    _write_csv(output / "trajectory_results.csv", trajectories)
    _write_csv(output / "historical_comparison.csv", historical)
    _write_csv(output / "correctness_conditioned_contrasts.csv", near_secondary + extreme_secondary)
    write_analysis_method(output / "analysis_method.md")
    (output / "analyze_primary.py").write_text(_script_snapshot(), encoding="utf-8")

    gpt_near = [row for row in near_rows if row["model_alias"] == "openai_gpt56_sol"]
    claude_near = [row for row in near_rows if row["model_alias"] == "anthropic_sonnet5"]
    gpt_ext = [row for row in extreme_rows if row["model_alias"] == "openai_gpt56_sol"]
    claude_ext = [row for row in extreme_rows if row["model_alias"] == "anthropic_sonnet5"]
    letter, label, reason = recommend_gate(
        extreme_gpt=gpt_ext,
        near_gpt=gpt_near,
        extreme_claude=claude_ext,
        trajectories=trajectories,
        historical=historical,
        validation_ok=True,
    )
    api_cost = sum(float(row.get("estimated_cost_usd") or 0) for row in rows)
    gpt_cost = sum(
        float(row.get("estimated_cost_usd") or 0)
        for row in rows
        if row["model_alias"] == "openai_gpt56_sol"
        and not row.get("reused_from_task_002")
    )
    claude_cost = sum(
        float(row.get("estimated_cost_usd") or 0)
        for row in rows
        if row["model_alias"] == "anthropic_sonnet5"
        and not row.get("reused_from_task_002")
    )
    # Include reused costs in totals for accounting completeness.
    gpt_cost_all = sum(
        float(row.get("estimated_cost_usd") or 0)
        for row in rows
        if row["model_alias"] == "openai_gpt56_sol"
    )
    claude_cost_all = sum(
        float(row.get("estimated_cost_usd") or 0)
        for row in rows
        if row["model_alias"] == "anthropic_sonnet5"
    )
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in rows)
    latency_sum = sum(float(row.get("latency_ms") or 0) for row in rows) / 1000.0
    report = build_report(
        validation=validation,
        rows=rows,
        rate_rows=rate_rows,
        references=references,
        near_rows=near_rows,
        extreme_rows=extreme_rows,
        trajectories=trajectories,
        historical=historical,
        secondary=near_secondary + extreme_secondary,
            letter=letter,
            label=label,
            reason=reason,
            reused=reused,
            smoke_reused=smoke_reused,
            prior_primary_completed=prior_primary_completed,
        new_initiated=new_initiated,
        failed_n=len(failed),
        budget_used=budget.used,
        retry_attempts=retry_attempts,
        repair_cells=repair_cells,
        api_cost=api_cost,
        gpt_cost=gpt_cost_all,
        claude_cost=claude_cost_all,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_sum=latency_sum,
        wall=float(payload.get("wall_clock_seconds") or 0),
        stopped=payload.get("stopped_reason"),
        selected_hash=selected_hash,
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    (output / "cost_summary.md").write_text(
        "\n".join(
            [
                "# Cost summary",
                "",
                f"- Reused Task-002 cells: {smoke_reused}",
                f"- Other already-complete primary cells reused this invocation: {prior_primary_completed}",
                f"- New scientific cells initiated: {new_initiated}",
                f"- Provider attempts: {budget.used}",
                f"- Retries beyond first attempt/repair: {retry_attempts}",
                f"- Cells with parse repair: {repair_cells}",
                f"- Input tokens (successful cells): {input_tokens}",
                f"- Output tokens (successful cells): {output_tokens}",
                f"- GPT estimated cost USD: {gpt_cost_all:.6f}",
                f"- Claude estimated cost USD: {claude_cost_all:.6f}",
                f"- Total estimated cost USD: {api_cost:.6f}",
                f"- Wall-clock seconds: {payload.get('wall_clock_seconds'):.3f}",
                f"- Summed provider latency seconds: {latency_sum:.3f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_run_manifest(
        output,
        payload,
        validation,
        reused=reused,
        new_initiated=new_initiated,
        retry_attempts=retry_attempts,
        repair_cells=repair_cells,
        gate=letter,
        api_cost=api_cost,
    )
    _write_changed_files(output)
    return {
        "validation": validation,
        "analyzed": True,
        "gate": letter,
        "label": label,
    }


def build_report(**kwargs: Any) -> str:
    near_rows = kwargs["near_rows"]
    extreme_rows = kwargs["extreme_rows"]
    trajectories = kwargs["trajectories"]
    historical = kwargs["historical"]
    references = kwargs["references"]
    rate_rows = kwargs["rate_rows"]
    letter = kwargs["letter"]
    label = kwargs["label"]
    reason = kwargs["reason"]

    def contrast_line(row: Mapping[str, Any]) -> str:
        return (
            f"{row['model_label']} L={int(row['L'])} "
            f"{row['low_displayed']:g} vs {row['high_displayed']:g}: "
            f"{_fmt_pp(row['rate_low'])}% → {_fmt_pp(row['rate_high'])}% "
            f"(Δ {_fmt_pp(row['difference'])} pp, 95% [{_fmt_pp(row['ci_lower'])}, {_fmt_pp(row['ci_upper'])}]); "
            f"VERIFY→USE {row['verify_to_use']}, USE→VERIFY {row['use_to_verify']}, no change {row['no_change']}"
        )

    def curve_text(model: str, L: float) -> str:
        points = [
            row
            for row in rate_rows
            if row["model_alias"] == model
            and float(row["L"]) == L
            and row["condition"] == "manipulated"
        ]
        parts = [
            f"{float(row['displayed_confidence']):g}={_fmt_pp(row['verification_rate'])}% "
            f"[{_fmt_pp(row['ci_lower'])}, {_fmt_pp(row['ci_upper'])}]"
            for row in points
        ]
        refs = references[(model, L)]
        return (
            ", ".join(parts)
            + f"; hidden {_fmt_pp(refs['hidden']['rate'])}%; "
            + f"true-visible {_fmt_pp(refs['true_visible']['rate'])}%"
        )

    gpt_ext = [row for row in extreme_rows if row["model_alias"] == "openai_gpt56_sol"]
    claude_ext = [row for row in extreme_rows if row["model_alias"] == "anthropic_sonnet5"]
    gpt_near = [row for row in near_rows if row["model_alias"] == "openai_gpt56_sol"]
    gpt_min_ext = min((row["difference"] for row in gpt_ext), default=float("nan"))
    claude_min_ext = min((row["difference"] for row in claude_ext), default=float("nan"))
    gpt_traj = [row for row in trajectories if row["model_alias"] == "openai_gpt56_sol"]
    claude_traj = [row for row in trajectories if row["model_alias"] == "anthropic_sonnet5"]
    gpt_noninc = sum(1 for row in gpt_traj if row["perfectly_non_increasing"])
    claude_noninc = sum(1 for row in claude_traj if row["perfectly_non_increasing"])
    gpt_reverse = sum(1 for row in gpt_traj if row["reverse_use_to_verify"])
    claude_reverse = sum(1 for row in claude_traj if row["reverse_use_to_verify"])
    gpt_flip = max((row["verify_to_use"] + row["use_to_verify"] for row in gpt_ext), default=0)
    hist_bits = [
        f"{row['model_label']} L={int(row['L'])} {row['condition']}: "
        f"historical {_fmt_pp(row['historical_rate'])}% vs current {_fmt_pp(row['current_rate'])}% "
        f"(Δ {_fmt_pp(row['current_minus_historical'])} pp; agree {row['question_agreement']}/{row['n_questions']})"
        for row in historical
    ]
    secondary_bits = []
    for row in kwargs["secondary"]:
        if row["low_displayed"] in {0.80, 0.90} and row["high_displayed"] == 0.99:
            secondary_bits.append(
                f"{row['model_label']} L={int(row['L'])} {row['correctness_split']} "
                f"extreme Δ {_fmt_pp(row['difference'])} pp "
                f"[{_fmt_pp(row['ci_lower'])}, {_fmt_pp(row['ci_upper'])}] n={row['n_questions']}"
            )

    return "\n".join(
        [
            "# Study 1 Primary Causal Pilot",
            "",
            "## 1. Data validation",
            "",
            f"Exactly {kwargs['validation']['successful_unique_primary_cells']} unique successful primary cells are represented (target 2,800).",
            f"Per-model counts: {kwargs['validation']['by_model']}. Per-L counts: {kwargs['validation']['by_L']}.",
            "Each of the seven condition slots has 400 cells (100 questions × 2 models × 2 L).",
            f"Task-002 reuse: {kwargs.get('smoke_reused', kwargs['reused'])} successful identical primary cells. "
            f"Already-complete primary cells from an earlier invocation of this task: {kwargs.get('prior_primary_completed', 0)}. "
            f"New scientific cells initiated this invocation: {kwargs['new_initiated']}.",
            f"Historical V2 unchanged: {kwargs['validation']['historical_unchanged']}. Frozen ID hash `{kwargs['selected_hash']}`.",
            "All valid parsed actions are VERIFY_FIRST or USE_UNVERIFIED. Repeat-extra cells were not run.",
            "" if not kwargs["validation"]["issues"] else "Remaining notes: " + "; ".join(kwargs["validation"]["issues"]),
            "",
            "## 2. Bottom line in plain English",
            "",
            f"- Changing only displayed confidence {'did' if gpt_min_ext >= 0.05 or claude_min_ext >= 0.05 else 'did not clearly'} change verification in this exploratory 100-question within-item pilot.",
            f"- GPT extreme low-vs-high contrasts were about {_fmt_pp(gpt_ext[0]['difference'])} pp at L=10 and {_fmt_pp(gpt_ext[1]['difference'])} pp at L=20.",
            f"- Claude extreme contrasts were about {_fmt_pp(claude_ext[0]['difference'])} pp at L=10 and {_fmt_pp(claude_ext[1]['difference'])} pp at L=20.",
            f"- Near-threshold GPT: {contrast_line(gpt_near[0])}; {contrast_line(gpt_near[1])}.",
            f"- Dose-response is described below; it is not forced to be monotonic. GPT non-increasing trajectories: {gpt_noninc}/{len(gpt_traj)}; reverse flips: {gpt_reverse}/{len(gpt_traj)}.",
            "- GPT and Claude should be read as separate systems, not as replicates of one law.",
            "- Hidden/true-visible versus historical V2 is a sanity check only; see section 8.",
            f"- GPT extreme-score flips involved up to {gpt_flip} questions, so the GPT signal is not a one-item artifact if that count is large.",
            "- Repeated-generation stability has not been tested. This result cannot claim the effect exceeds ordinary rerun noise.",
            f"- Gate recommendation: **{letter}. {label}** Repeats are not launched by this task.",
            "",
            "## 3. GPT results",
            "",
            f"L=10 manipulated curve: {curve_text('openai_gpt56_sol', 10.0)}.",
            f"L=20 manipulated curve: {curve_text('openai_gpt56_sol', 20.0)}.",
            "Intervals are question-level bootstrap 95% percentile intervals (seed 20260917, 5,000 resamples).",
            "",
            "## 4. Claude results",
            "",
            f"L=10 manipulated curve: {curve_text('anthropic_sonnet5', 10.0)}.",
            f"L=20 manipulated curve: {curve_text('anthropic_sonnet5', 20.0)}.",
            "Claude is a contrast, not a required replication of GPT.",
            "",
            "## 5. Near-threshold causal effects",
            "",
            *[f"- {contrast_line(row)}" for row in near_rows],
            "",
            "These are paired within-question contrasts. They are not independent two-sample tests.",
            "",
            "## 6. Extreme-score causal effects",
            "",
            *[f"- {contrast_line(row)}" for row in extreme_rows],
            "",
            "## 7. Within-question trajectories",
            "",
            f"- GPT: {gpt_noninc}/{len(gpt_traj)} perfectly non-increasing; {gpt_reverse}/{len(gpt_traj)} contain a reverse USE→VERIFY as displayed confidence increases.",
            f"- Claude: {claude_noninc}/{len(claude_traj)} perfectly non-increasing; {claude_reverse}/{len(claude_traj)} contain a reverse flip.",
            "- Binary single-generation trajectories can look like a threshold or like noise; this is descriptive only.",
            "",
            "## 8. Current vs historical sanity check",
            "",
            *[f"- {bit}" for bit in hist_bits],
            "",
            "Disagreement here is compatible with ordinary stochastic reruns and does not by itself prove the endpoint identity changed.",
            "",
            "## 9. Secondary correctness-conditioned analysis",
            "",
            "These splits are exploratory and noisy. They do not replace the all-question causal estimates.",
            *[f"- {bit}" for bit in secondary_bits],
            "",
            "## 10. What this DOES establish",
            "",
            "In this exploratory 100-question AI-authority pilot, holding question and frozen answer fixed, the displayed confidence number was the only intended prompt change across manipulated cells. Where verification rates moved with that number, the movement is a causal effect of the displayed number in this experimental setup. Repeated-sampling stability is still unknown.",
            "",
            "## 11. What this DOES NOT establish",
            "",
            "- This is not confirmatory evidence.",
            "- It does not establish that “own” provenance matters.",
            "- It does not establish real-world or agent generalization.",
            "- It does not establish a hidden-state mechanism, nor that showing confidence suppresses internal uncertainty.",
            "- It does not establish that the model loses information.",
            "- It does not establish that the effect exceeds ordinary rerun noise until repeats are run.",
            "",
            "## 12. Study-1 gate recommendation",
            "",
            f"**{letter}. {label}**",
            "",
            reason,
            "",
            "Protocol criterion 4 (effect larger than ordinary repeated-sampling noise) is **not evaluated**, because the 1,120 repeat-extra cells were not run. The strongest claim permitted here is STRONG_BEHAVIORAL_SIGNAL_PENDING_STABILITY if the primary contrasts are large; this task still does not launch repeats.",
            "",
            "## 13. Exact numbers GPT should know",
            "",
            "| contrast | GPT Δ pp [95%] | Claude Δ pp [95%] |",
            "|---|---:|---:|",
            *compact_tables(near_rows, extreme_rows),
            "",
            "## 14. Cost and runtime",
            "",
            f"- Reused Task-002 cells: {kwargs.get('smoke_reused', kwargs['reused'])}",
            f"- Already-complete primary cells reused this invocation: {kwargs.get('prior_primary_completed', 0)}",
            f"- New scientific calls initiated: {kwargs['new_initiated']}",
            f"- Failed new cells: {kwargs['failed_n']}",
            f"- Provider attempts: {kwargs['budget_used']}",
            f"- Retries: {kwargs['retry_attempts']}",
            f"- Parse-repair cells: {kwargs['repair_cells']}",
            f"- Input tokens: {kwargs['input_tokens']}",
            f"- Output tokens: {kwargs['output_tokens']}",
            f"- GPT estimated USD: {kwargs['gpt_cost']:.6f}",
            f"- Claude estimated USD: {kwargs['claude_cost']:.6f}",
            f"- Total estimated USD: {kwargs['api_cost']:.6f}",
            f"- Wall-clock seconds: {kwargs['wall']:.3f}",
            f"- Summed provider latency seconds: {kwargs['latency_sum']:.3f}",
            f"- Stopped reason: {kwargs['stopped']}",
            "",
        ]
    )


def compact_tables(near_rows: Sequence[Mapping[str, Any]], extreme_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = []
    grouped: dict[tuple[float, float, float], dict[str, Mapping[str, Any]]] = {}
    for row in list(near_rows) + list(extreme_rows):
        key = (float(row["L"]), float(row["low_displayed"]), float(row["high_displayed"]))
        grouped.setdefault(key, {})[row["model_alias"]] = row
    for key, models in grouped.items():
        L, low, high = key
        gpt = models.get("openai_gpt56_sol")
        claude = models.get("anthropic_sonnet5")
        def cell(row: Mapping[str, Any] | None) -> str:
            if row is None:
                return ""
            return f"{_fmt_pp(row['difference'])} [{_fmt_pp(row['ci_lower'])}, {_fmt_pp(row['ci_upper'])}]"
        lines.append(
            f"| L={int(L)} {low:g} vs {high:g} | {cell(gpt)} | {cell(claude)} |"
        )
    return lines


def _write_run_manifest(
    output: Path,
    payload: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    reused: int,
    new_initiated: int,
    retry_attempts: int,
    repair_cells: int,
    gate: str,
    api_cost: float,
) -> None:
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "003_run_study1_primary",
                "status": "SUCCESS" if validation.get("ok") else "VALIDATION_OR_PARTIAL",
                "git_commit": payload.get("code_commit"),
                "paid_calls_authorized": True,
                "primary_dataset_target": 2800,
                "existing_smoke_cells_reused": int(payload.get("smoke_reused") or reused),
                "prior_primary_cells_reused": int(payload.get("prior_primary_completed") or 0),
                "new_scientific_calls": new_initiated,
                "provider_attempts_total": getattr(payload.get("budget"), "used", 0),
                "retry_attempts": retry_attempts,
                "repair_attempts": repair_cells,
                "successful_unique_primary_cells": validation.get(
                    "successful_unique_primary_cells"
                ),
                "failed_primary_cells": len(payload.get("failed") or []),
                "models": list(STUDY1_MODEL_ALIASES),
                "L_values": [10.0, 20.0],
                "confidence_grids": {"10": list(STUDY1_GRID_L10), "20": list(STUDY1_GRID_L20)},
                "selected_question_hash": TASK001_ID_HASH,
                "api_cost_usd": api_cost,
                "wall_clock_seconds": payload.get("wall_clock_seconds"),
                "historical_artifacts_modified": not validation.get(
                    "historical_unchanged", True
                ),
                "data_validation_passed": bool(validation.get("ok")),
                "gate_recommendation": gate,
                "ready_for_gpt_review": True,
                "stopped_reason": payload.get("stopped_reason"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_changed_files(output: Path) -> None:
    paths: list[str] = []
    seen: set[str] = set()
    for relative in (
        "from_gpt/003_run_study1_primary.md",
        "src/study1_primary.py",
        "src/study1_analysis.py",
        "src/study1_runner.py",
        "src/study1_cli.py",
        "tests/test_study1.py",
        "docs/D1_DECISION_LOG.md",
        "analysis/study1/analyze_primary.py",
    ):
        if (PROJECT_ROOT / relative).exists() and relative not in seen:
            paths.append(relative)
            seen.add(relative)
    for found in sorted(output.rglob("*")):
        if found.is_file():
            relative = str(found.relative_to(PROJECT_ROOT))
            if relative not in seen:
                paths.append(relative)
                seen.add(relative)
    (output / "changed_files.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
