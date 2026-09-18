"""Task 008 orchestrator: zero-cost ranking-invariance audit + decision packet."""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .study1_analysis import _pyplot
from .task008_analysis import (
    common_curve_analysis,
    condition_auc_table,
    decide_bucket,
    difficulty_logit_interactions,
    latent_item_prediction,
    load_gemini_grok,
    load_qualitative_grid,
    nesting_table,
    prospective_matrix,
    roc_and_coverage_points,
    shared_vs_condition_sensitive,
    workshop_points,
)
from .task008_common import (
    FIGURES_DIR,
    MODEL_LABELS,
    PAPER_DIRECTION,
    RETURN_DIR,
    TASK_ID,
    VISIBLE_CONDITIONS,
    assert_008_write_target,
    json_dump,
    protected_fingerprints,
    sha256_file,
    write_csv,
)

SRC_FILES = (
    Path(__file__),
    Path(__file__).with_name("task008_common.py"),
    Path(__file__).with_name("task008_analysis.py"),
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _fmt_pp(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{100.0 * number:.{digits}f}"


def _ci_pp(row: Mapping[str, Any], key: str, lo: str, hi: str) -> str:
    return f"{_fmt_pp(row[key])} [{_fmt_pp(row[lo])}, {_fmt_pp(row[hi])}]"


def plot_roc(roc: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharex=True, sharey=True)
    markers = {"moderate": "o", "stronger": "s"}
    colors = {
        "displayed_0.70": "C0",
        "displayed_0.90": "C1",
        "displayed_0.99": "C2",
        "hidden": "0.35",
    }
    for ax, model in zip(axes, ("openai_gpt56_sol", "anthropic_sonnet5")):
        ax.plot([0, 1], [0, 1], color="0.75", lw=1, ls="--", label="random")
        for family in ("moderate", "stronger"):
            pts = [
                r
                for r in roc
                if r["model_alias"] == model
                and r["family"] == family
                and r.get("sample_role") == "primary_n100"
            ]
            pts = sorted(
                pts,
                key=lambda r: (
                    0 if r["score_condition"] == "hidden" else 1,
                    float(r["displayed_confidence"] or 0),
                ),
            )
            vis = [p for p in pts if p["score_condition"] in VISIBLE_CONDITIONS]
            ax.plot(
                [float(p["fpr"]) for p in vis],
                [float(p["tpr"]) for p in vis],
                color="0.4",
                lw=1,
                alpha=0.8,
            )
            for p in pts:
                ax.scatter(
                    float(p["fpr"]),
                    float(p["tpr"]),
                    marker=markers[family],
                    c=colors[p["score_condition"]],
                    s=42,
                    zorder=3,
                    label=f"{family} {p['score_condition']}",
                )
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("FPR = P(VERIFY | correct)")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("TPR = P(VERIFY | wrong)")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Qualitative N=100 ROC-space points by displayed score")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_risk_coverage(risk: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharex=True, sharey=True)
    markers = {"moderate": "o", "stronger": "s"}
    for ax, model in zip(axes, ("openai_gpt56_sol", "anthropic_sonnet5")):
        ax.plot([0, 1], [0, 1], color="0.75", lw=1, ls="--", label="random checking")
        for family in ("moderate", "stronger"):
            pts = [
                r
                for r in risk
                if r["model_alias"] == model
                and r["family"] == family
                and r.get("sample_role") == "primary_n100"
                and r["score_condition"] in VISIBLE_CONDITIONS
            ]
            pts = sorted(pts, key=lambda r: float(r["coverage"]))
            ax.plot(
                [float(p["coverage"]) for p in pts],
                [float(p["risk_caught"]) for p in pts],
                color="0.4",
                lw=1,
            )
            for p in pts:
                ax.scatter(
                    float(p["coverage"]),
                    float(p["risk_caught"]),
                    marker=markers[family],
                    s=42,
                    label=f"{family} {p['score_condition']}",
                )
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Coverage = VERIFY rate")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Errors caught = P(VERIFY | wrong)")
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Qualitative N=100 risk-coverage points (not matched-budget)")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_nesting(nest: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    labels = []
    rates = []
    lo = []
    hi = []
    for row in nest:
        if abs(float(row["from_score"]) - 0.70) > 1e-9 or abs(float(row["to_score"]) - 0.99) > 1e-9:
            continue
        labels.append(f"{row['model_label']}\n{row['family']}")
        rates.append(100.0 * float(row["reversal_rate"]))
        lo.append(100.0 * float(row["reversal_ci_lower"]))
        hi.append(100.0 * float(row["reversal_ci_upper"]))
    x = np.arange(len(labels))
    ax.bar(x, rates, color="0.35")
    ax.errorbar(
        x,
        rates,
        yerr=[np.array(rates) - np.array(lo), np.array(hi) - np.array(rates)],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("U→V reversal rate (pp)")
    ax.set_title("Non-nesting when displayed score rises 0.70 → 0.99")
    ax.axhline(5.0, color="0.6", ls="--", lw=1, label="5 pp material line")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_slopes(interact: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    scores = np.array([0.70, 0.90, 0.99])
    for ax, model in zip(axes, ("openai_gpt56_sol", "anthropic_sonnet5")):
        for family, color in (("moderate", "C0"), ("stronger", "C1")):
            row = next(
                r
                for r in interact
                if r["model_alias"] == model and r["family"] == family
            )
            y = np.array([row["slope_at_0.70"], row["slope_at_0.90"], row["slope_at_0.99"]])
            ylo = np.array(
                [
                    row["slope_at_0.70_ci_lower"],
                    row["slope_at_0.90_ci_lower"],
                    row["slope_at_0.99_ci_lower"],
                ]
            )
            yhi = np.array(
                [
                    row["slope_at_0.70_ci_upper"],
                    row["slope_at_0.90_ci_upper"],
                    row["slope_at_0.99_ci_upper"],
                ]
            )
            ax.plot(scores, y, marker="o", color=color, label=family)
            ax.fill_between(scores, ylo, yhi, color=color, alpha=0.15)
        ax.axhline(0, color="0.7", lw=1)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Displayed score")
        ax.set_xticks(scores)
    axes[0].set_ylabel("Logit slope on other-models-correct")
    axes[1].legend(frameon=False)
    fig.suptitle("Difficulty slope vs displayed score (negative = harder items verified more)")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_auc(auc: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, model in zip(axes, ("openai_gpt56_sol", "anthropic_sonnet5")):
        for family, color in (("moderate", "C0"), ("stronger", "C1")):
            pts = [
                r
                for r in auc
                if r["model_alias"] == model
                and r["family"] == family
                and r["score_condition"] in VISIBLE_CONDITIONS
            ]
            xs = [float(p["displayed_confidence"]) for p in pts]
            ys = [float(p["auroc_verify_vs_wrongness"]) for p in pts]
            lo = [float(p["auroc_ci_lower"]) for p in pts]
            hi = [float(p["auroc_ci_upper"]) for p in pts]
            ax.plot(xs, ys, marker="o", color=color, label=family)
            ax.vlines(xs, lo, hi, color=color)
        ax.axhline(0.5, color="0.7", lw=1)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Displayed score")
        ax.set_xticks([0.70, 0.90, 0.99])
    axes[0].set_ylabel("AUROC of VERIFY for target error")
    axes[1].legend(frameon=False)
    fig.suptitle("Binary-action discrimination by condition (coarse)")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _row(table: Sequence[Mapping[str, Any]], **kwargs: Any) -> Mapping[str, Any]:
    for row in table:
        if all(row.get(k) == v for k, v in kwargs.items()):
            return row
    raise KeyError(kwargs)


def write_validation(before: Mapping[str, Any], after: Mapping[str, Any], grid: Mapping[str, Any]) -> None:
    lines = [
        "# Task 008 validation",
        "",
        "Zero API calls. Read-only analysis. `paperDirection.txt` was not modified.",
        "",
        "## Grid",
        "",
        f"- Merged qualitative rows: {grid['n_rows']} (expected 1600)",
        f"- Questions: {grid['n_questions']} (expected 100)",
        f"- Models: {grid['n_models']}",
        f"- Families: {grid['n_families']}",
        f"- Conditions: {grid['n_conditions']}",
        f"- Complete: {grid['complete']}",
        f"- Difficulty leakage rule: {grid['leakage_rule']}",
        "",
        "## Protected fingerprints (before = after)",
        "",
    ]
    mismatches = []
    for key, rec in before.items():
        rec2 = after.get(key, {})
        same = rec.get("sha256") == rec2.get("sha256")
        lines.append(
            f"- `{key}`: {rec.get('sha256', 'missing')} unchanged={same}"
        )
        if not same:
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(f"protected fingerprint changed: {mismatches}")
    lines.extend(
        [
            "",
            "## Analysis choices",
            "",
            "- Statistical unit: question ID. Bootstrap 5,000, seed 20260917, percentile 95% CIs.",
            "- Unpenalized logistic for low-dimensional models; Rasch item intercepts use L2 C=10 because unpenalized item FE separates.",
            "- Difficulty is leave-one-target-out other-model Stage-1 correctness (0–3).",
            "- Gemini/Grok n=20 are descriptive only.",
            "- No historical sqlite writes. No confirmatory sample frozen.",
            "",
        ]
    )
    path = RETURN_DIR / "validation.md"
    assert_008_write_target(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_workshop(s1: Sequence[Mapping[str, Any]], ckpt: Sequence[Mapping[str, Any]]) -> None:
    gpt_vis = [
        r
        for r in ckpt
        if r["model_alias"] == "openai_gpt56_sol"
    ]
    claude_vis = [
        r
        for r in ckpt
        if r["model_alias"] == "anthropic_sonnet5"
    ]

    def mean_field(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        return float(np.mean([float(r[key]) for r in rows]))

    gpt_cov_shift = mean_field(gpt_vis, "coverage_shift_visible_minus_hidden")
    gpt_agree = mean_field(gpt_vis, "visible_vs_raw_threshold_agreement")
    gpt_rank = mean_field(gpt_vis, "hidden_minus_conf_catch")
    claude_cov_shift = mean_field(claude_vis, "coverage_shift_visible_minus_hidden")
    claude_rank = mean_field(claude_vis, "hidden_minus_conf_catch")

    gpt_s1 = [
        r
        for r in s1
        if r["model_alias"] == "openai_gpt56_sol" and str(r["display_condition"]).startswith("manipulated")
    ]
    lines = [
        "# Workshop / Checkpoint A reinterpretation",
        "",
        "Exploratory only. Not confirmatory. Distinguishes coverage, ranking, and calibration.",
        "",
        "## Self-correction paragraph (use only if retained)",
        "",
        (
            "The original workshop observation that *visible* confidence reduced GPT’s error-catching "
            "is primarily an operating-point result, not a demonstration that ranking quality collapsed. "
            f"On V2-B, showing GPT its number cut coverage by about {_fmt_pp(gpt_cov_shift)} pp on average "
            f"and agreed with the raw-confidence threshold on {_fmt_pp(gpt_agree)}% of answers. "
            "Visible GPT therefore slides left along the confidence risk-coverage curve. "
            f"At matched budget, hidden GPT’s ranking edge over confidence-only is only {_fmt_pp(gpt_rank)} pp "
            "and the pooled interval includes zero (Checkpoint A). "
            "Calibration still matters for *cost*: an overconfident scalar plus a faithful threshold "
            "leaves too many errors unchecked at high L. That is a control-variable problem, not a "
            "ranking-degradation result. Claude is different on ranking: hidden Claude beats confidence-only "
            f"at matched budget by about {_fmt_pp(claude_rank)} pp, while visibility *increases* Claude’s "
            f"coverage (mean shift {_fmt_pp(claude_cov_shift)} pp). Do not collapse these into one law."
        ),
        "",
        "## Coverage vs ranking vs calibration",
        "",
        "| model | mean coverage shift vis−hid (pp) | vis vs raw-threshold agree (pp) | hidden−conf catch at matched budget (pp) |",
        "|---|---:|---:|---:|",
        f"| GPT | {_fmt_pp(gpt_cov_shift)} | {_fmt_pp(gpt_agree)} | {_fmt_pp(gpt_rank)} |",
        f"| Claude | {_fmt_pp(claude_cov_shift)} | {_fmt_pp(mean_field(claude_vis, 'visible_vs_raw_threshold_agreement'))} | {_fmt_pp(claude_rank)} |",
        "",
        "Study 1 numeric GPT cells are near-saturated on each side of the instructed threshold, so "
        "errors-caught differences across manipulated scores are almost entirely coverage. "
        f"GPT manipulated condition-cells in `workshop_study1_roc_points.csv`: {len(gpt_s1)} "
        "(each is 100 questions).",
        "",
        "Do not force this paragraph if a later confirmatory study shows ranking collapse at matched coverage.",
        "",
    ]
    path = RETURN_DIR / "workshop_reinterpretation.md"
    assert_008_write_target(path)
    path.write_text("\n".join(lines), encoding="utf-8")
    write_csv(RETURN_DIR / "workshop_study1_roc_points.csv", s1)
    write_csv(RETURN_DIR / "workshop_checkpoint_a_points.csv", ckpt)


def write_power_notes(design: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Prospective power notes",
        "",
        "Not a substitute for a simulation on a frozen sample. Question is the unit.",
        "",
        "## Detectable effects at N=300/500/700",
        "",
        "SE of a VERIFY rate near 0.5 is about 0.029 / 0.022 / 0.019. A 10 pp coverage shift is easy. "
        "A 5 pp reversal rate has SE ≈ 0.013 / 0.010 / 0.008, so N=500 can reject 5 pp if the truth is 0 "
        "and can distinguish 0 vs 8–10 pp more reliably than 0 vs 3 pp.",
        "",
        "Binary-action AUROC CIs on N=100 were already wide. N=500 tightens them, but one generation "
        "per cell still leaves item propensities noisy. That is why repeats exist.",
        "",
        "## Why more operating points help World I",
        "",
        "Three visible scores cannot strongly test a common ROC. P2 (five displayed/true-q/hidden points) "
        "is the better invariance-test grid. P1 is cheaper.",
        "",
        "## Repeats",
        "",
        "100 items × 3 generations estimates per-item flip rates and whether orderings are stable. "
        "150×3 is better for World R (need per-condition propensities). Task 004 already showed GPT "
        "identical-prompt flips ≈ 3% and Claude ≈ 3–4% under numeric prompts.",
        "",
        "## q2",
        "",
        "Task 005/005C: q2 is highly correlated with q1 and does not absorb the hidden residual after "
        "difficulty. Default prospective protocol omits q2.",
        "",
        "## Stakes",
        "",
        "One family. Stronger wording mainly shifts GPT coverage toward saturation at 0.70 (92%). "
        "That is a worse invariance test. Freeze moderate unless GPT review demands a wording robustness arm.",
        "",
        f"Design matrix rows: {len(design)}.",
        "",
        "Do not launch from this file.",
        "",
    ]
    path = RETURN_DIR / "prospective_power_notes.md"
    assert_008_write_target(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_packet(decision: Mapping[str, Any], interact: Sequence[Mapping[str, Any]], latent: Sequence[Mapping[str, Any]]) -> None:
    gpt = decision["per_model"]["openai_gpt56_sol"]
    claude = decision["per_model"]["anthropic_sonnet5"]
    lines = [
        "# Final narrative decision packet",
        "",
        "Ignore how we got here. This packet is about the strongest FINAL paper the current exploratory evidence can support.",
        "",
        f"**Current decision bucket:** `{decision['decision_bucket']}`",
        f"**Favored world:** {decision['favored_world']} ({decision['favor_strength']})",
        "",
        f"GPT exploratory class: `{gpt['label']}` (invariance votes {gpt['invariance_votes']}, reshape votes {gpt['reshape_votes']}; mean held-out u AUROC {_fmt(gpt['mean_u_auc'])}).",
        f"Claude exploratory class: `{claude['label']}` (invariance votes {claude['invariance_votes']}, reshape votes {claude['reshape_votes']}; mean held-out u AUROC {_fmt(claude['mean_u_auc'])}).",
        "",
        "## WORLD 1 — Ranking invariance",
        "",
        "**Strongest honest narrative.** Counterfactual confidence is a coverage knob. It moves how many answers the model verifies. It does not, on current evidence, rewrite a stable item ordering in a way that survives logit-scale and held-out-propensity tests. Probability-scale hard/easy gaps can still change as a sigmoid artifact.",
        "",
        "**Exact novelty claim.** Displayed confidence causally sets a verification operating point, including without explicit L/C arithmetic, while item ranking remains approximately invariant across that operating point.",
        "",
        "**Contributions (max 4).**",
        "1. Causal coverage control from a displayed confidence scalar, including qualitative stakes.",
        "2. Decomposition of selective oversight into operating point vs item ranking.",
        "3. Evidence that nested VERIFY sets and transferable item propensity are compatible with a threshold shift.",
        "4. Reinterpretation of the workshop visible-confidence catch drop as coverage/calibration, not ranking collapse.",
        "",
        "**Figures 1–4.** Fig 1: qualitative VERIFY vs displayed score. Fig 2: ROC / risk-coverage overlay. Fig 3: nesting/reversals. Fig 4: matched-budget workshop curve with visible points sliding along it.",
        "",
        "**Practical implication.** If you want more checking, lower the displayed score (or raise stakes). Do not expect that knob to retarget *which* items get checked, beyond the mechanical effect of a different threshold on a fixed ranking.",
        "",
        "**Closest prior-work distinction.** Not calibration-only, not “models follow expected-cost arithmetic,” not hidden-state suppression. The instrument is an externalized score used as a policy input.",
        "",
        "**Prospective result needed.** On a fresh sample, coverage must move while (a) reversals stay near rerun noise, (b) a shared item propensity predicts held-out scores, (c) condition-sensitive models add little, (d) matched-coverage catch stays similar.",
        "",
        "**Likely second-task requirement.** Repeated generations on a subset to show the latent ranking is stable, not a one-draw artifact. Optional: a cheap routing evaluation at matched budget.",
        "",
        "## WORLD 2 — Ranking reshaping",
        "",
        "**Strongest honest narrative.** The displayed score does not only ration verification. At non-saturated operating points it changes which items are treated as worth checking, so risk-coverage points leave a common ranking curve.",
        "",
        "**Exact novelty claim.** Counterfactual confidence feedback is an item-prioritization intervention, not merely a coverage control.",
        "",
        "**Contributions (max 4).**",
        "1. Causal displayed-score control without requiring L/C arithmetic.",
        "2. Identification of ranking/prioritization change separate from coverage.",
        "3. Logit-scale score × difficulty interaction and/or non-nested reversals as the statistical signature.",
        "4. Error-catching consequences at matched coverage, not raw catch at different budgets.",
        "",
        "**Figures 1–4.** Fig 1: score-response. Fig 2: ROC points off a common curve. Fig 3: switch-set / reversals. Fig 4: matched-coverage catch gap vs a shared-ranking baseline.",
        "",
        "**Practical implication.** Feeding back a confidence number can retarget oversight, so a dashboard that only sets a checking rate is incomplete.",
        "",
        "**Closest prior-work distinction.** Stronger than a threshold-shift paper. Must beat the link-function and saturation critiques.",
        "",
        "**Prospective result needed.** Material condition-sensitive improvement and/or matched-coverage catch change at identifiable (non-saturated) points, predeclared.",
        "",
        "**Likely second-task requirement.** More operating points plus item-level repeats so per-condition propensities exist.",
        "",
        "## WORLD 3 — Mixed / model-specific",
        "",
        "**Strongest honest narrative.** GPT and Claude do not share one ranking law. GPT looks closer to a coverage/threshold machine. Claude’s remaining checks may concentrate differently, but that pattern must be separated from saturation and probability-scale artifacts. Heterogeneity is a finding only if predeclared and replicated; it is not automatically a contribution.",
        "",
        "**Exact novelty claim.** Selective-oversight control by displayed confidence is real, but ranking invariance is model-dependent.",
        "",
        "**Contributions (max 4).**",
        "1. Causal coverage control without explicit arithmetic.",
        "2. A predeclared GPT vs Claude contrast on ranking invariance, not a post-hoc split.",
        "3. Difficulty as an evaluation covariate, not a production feature.",
        "4. Workshop coverage-vs-ranking split, which already differs by model.",
        "",
        "**Figures 1–4.** Same geometry, faceted by model, with a predeclared “same law / different law” test.",
        "",
        "**Practical implication.** Do not ship a universal “confidence slider” story. GPT-like systems may be coverage-controllable; Claude-like systems need a ranking check.",
        "",
        "**Closest prior-work distinction.** Cross-model policy difference, not a hidden-state story and not a four-model fishing expedition.",
        "",
        "**Prospective result needed.** Predeclare the GPT vs Claude invariance tests before seeing the fresh sample. Gemini/Grok are secondary.",
        "",
        "**Likely second-task requirement.** Enough Claude non-saturated points, and repeats, so “sharpening” cannot hide in a ceiling.",
        "",
        "## What existing evidence favors",
        "",
        f"- Bucket: `{decision['decision_bucket']}`",
        f"- Strength: `{decision['favor_strength']}`",
        "- This is exploratory N=100 with one generation per cell. It cannot freeze World 1 as a law.",
        "",
    ]
    path = RETURN_DIR / "final_narrative_decision_packet.md"
    assert_008_write_target(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    *,
    grid: Mapping[str, Any],
    roc: Sequence[Mapping[str, Any]],
    nest: Sequence[Mapping[str, Any]],
    shared: Sequence[Mapping[str, Any]],
    rich: Sequence[Mapping[str, Any]],
    interact: Sequence[Mapping[str, Any]],
    auc: Sequence[Mapping[str, Any]],
    common: Sequence[Mapping[str, Any]],
    latent: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    fingerprints: Mapping[str, Any],
) -> None:
    def roc_line(model: str, family: str, cond: str) -> str:
        row = _row(roc, model_alias=model, family=family, score_condition=cond, sample_role="primary_n100")
        return (
            f"VERIFY {_fmt_pp(row['verify_rate'])}% [{_fmt_pp(row['verify_rate_ci_lower'])}, {_fmt_pp(row['verify_rate_ci_upper'])}]; "
            f"FPR {_fmt_pp(row['fpr'])}%; TPR/catch {_fmt_pp(row['tpr'])}%; sat={row['saturation']}"
        )

    claude_mod = _row(interact, model_alias="anthropic_sonnet5", family="moderate")
    gpt_mod = _row(interact, model_alias="openai_gpt56_sol", family="moderate")
    lines = [
        "# Task 008 — Ranking-invariance audit",
        "",
        "Exploratory only. Not confirmatory. `paperDirection.txt` was not modified. API calls: **0**.",
        "",
        f"**Decision bucket:** `{decision['decision_bucket']}`",
        f"**Favored final-paper world:** {decision['favored_world']} (`{decision['favor_strength']}`)",
        "",
        "READY_FOR_GPT_REVIEW = YES",
        "",
        "## 1. Plain-English bottom line",
        "",
        (
            "Displayed confidence is a strong coverage control on the N=100 qualitative grid. "
            "The adversarial hypothesis is that ranking/discrimination stays put while the checking rate moves. "
            "Nesting is mostly consistent with a threshold shift (reversals are rare). "
            "A shared item propensity estimated from two visible scores predicts the third better than an intercept. "
            "Claude’s probability-scale “sharpening” is not automatically a logit-scale ranking rewrite: "
            f"moderate Claude γ={_fmt(claude_mod['gamma_score_x_difficulty'])} "
            f"[{_fmt(claude_mod['gamma_ci_lower'])}, {_fmt(claude_mod['gamma_ci_upper'])}], "
            f"CV Δlog-loss={_fmt(claude_mod['cv_delta_logloss'])}. "
            f"GPT moderate γ={_fmt(gpt_mod['gamma_score_x_difficulty'])} "
            f"[{_fmt(gpt_mod['gamma_ci_lower'])}, {_fmt(gpt_mod['gamma_ci_upper'])}], "
            f"CV Δlog-loss={_fmt(gpt_mod['cv_delta_logloss'])}. "
            "Three binary ROC points cannot prove a unique common ranking. "
            "The workshop GPT catch drop remains best read as an operating-point/calibration effect."
        ),
        "",
        "## 2. Validation",
        "",
        f"- Grid complete: {grid['complete']}; rows={grid['n_rows']}; questions={grid['n_questions']}",
        f"- Difficulty: {grid['leakage_rule']}",
        f"- paperDirection sha256: {fingerprints['paperDirection']['sha256']}",
        f"- V2 sha256: {fingerprints['v2']['sha256']}",
        f"- Study 1 sha256: {fingerprints['study1']['sha256']}",
        "- See `validation.md`.",
        "",
        "## 3. ROC / risk-coverage geometry",
        "",
    ]
    for model in ("openai_gpt56_sol", "anthropic_sonnet5"):
        lines.append(f"### {MODEL_LABELS[model]}")
        for family in ("moderate", "stronger"):
            lines.append(f"- {family} hidden: {roc_line(model, family, 'hidden')}")
            for cond in VISIBLE_CONDITIONS:
                lines.append(f"- {family} {cond}: {roc_line(model, family, cond)}")
        lines.append("")
    lines.extend(
        [
            "Figures: `figures/roc_space_gpt_claude.png`, `figures/risk_coverage_gpt_claude.png`.",
            "",
            "Common-curve geometry (three points; formal test weak):",
            "",
        ]
    )
    for row in common:
        lines.append(
            f"- {row['model_label']} {row['family']}: monotone FPR={row['monotone_fpr_as_score_rises']}, "
            f"monotone TPR={row['monotone_tpr_as_score_rises']}, concave={row['piecewise_roc_concave']}, "
            f"identifiable_points={row['n_identifiable_points']}/3, adjacent UV={row['adjacent_uv_count']}"
        )
    lines.extend(
        [
            "",
            "## 4. Nesting / reversals",
            "",
            "As score rises, nested VERIFY sets require U→V = 0. Nesting supports thresholding; it does not prove invariance.",
            "",
        ]
    )
    for row in nest:
        lines.append(
            f"- {row['model_label']} {row['family']} {row['from_score']}→{row['to_score']}: "
            f"VV={row['n_VV']} VU={row['n_VU']} UU={row['n_UU']} UV={row['n_UV']}; "
            f"reversal {_fmt_pp(row['reversal_rate'])} pp [{_fmt_pp(row['reversal_ci_lower'])}, {_fmt_pp(row['reversal_ci_upper'])}]"
        )
    lines.extend(
        [
            "",
            "## 5. Shared-ranking vs condition-sensitive model",
            "",
            "Shared: `logit P(VERIFY)=α_condition + u_item` (L2 item intercepts). "
            "Richer adds difficulty × score. Transferable comparison is grouped-CV on *new questions*, where u_item cannot be reused.",
            "",
        ]
    )
    for s, r in zip(shared, rich):
        lines.append(
            f"- {s['model_label']} {s['family']}: CV intercept {_fmt(s['cv_logloss_condition_intercept'])}; "
            f"CV +difficulty {_fmt(s['cv_logloss_condition_plus_difficulty'])}; "
            f"CV +interaction {_fmt(r['cv_logloss_interaction'])}; "
            f"Δ vs difficulty {_fmt(r['cv_delta_vs_difficulty_only'])}; "
            f"material_cv={r['material_cv_improvement']}"
        )
    lines.extend(
        [
            "",
            "## 6. Logit-scale score × difficulty interaction",
            "",
            "Primary difficulty = other-models-correct (0–3). Score centered at 0.90. Ridge C=1 on standardized covariates for coefficients; unpenalized grouped-CV for materiality.",
            "",
        ]
    )
    for row in interact:
        lines.append(
            f"- {row['model_label']} {row['family']}: β(0.90)={_fmt(row['beta_at_ref'])}; "
            f"γ={_fmt(row['gamma_score_x_difficulty'])} [{_fmt(row['gamma_ci_lower'])}, {_fmt(row['gamma_ci_upper'])}]; "
            f"slopes 0.70/0.90/0.99 = {_fmt(row['slope_at_0.70'])} / {_fmt(row['slope_at_0.90'])} / {_fmt(row['slope_at_0.99'])}; "
            f"CV Δlog-loss={_fmt(row['cv_delta_logloss'])}; material={row['material_logit_interaction']}; "
            f"saturated_cell_present={row['saturated_operating_point_present']}"
        )
    lines.extend(
        [
            "",
            "## 7. Condition-specific discrimination",
            "",
            "AUROC of the binary VERIFY bit for target wrongness. Coarse by construction.",
            "",
        ]
    )
    for row in auc:
        if row["score_condition"] not in VISIBLE_CONDITIONS:
            continue
        lines.append(
            f"- {row['model_label']} {row['family']} {row['score_condition']}: "
            f"AUROC {_fmt(row['auroc_verify_vs_wrongness'])} [{_fmt(row['auroc_ci_lower'])}, {_fmt(row['auroc_ci_upper'])}] "
            f"sat={row['saturation']}"
        )
    lines.extend(
        [
            "",
            "## 8. Cross-condition latent-item prediction",
            "",
        ]
    )
    for row in latent:
        if row.get("status") != "OK":
            lines.append(f"- {row.get('model_alias')} {row.get('family')}: {row.get('status')}")
            continue
        lines.append(
            f"- {row['model_label']} {row['family']} hold {row['held_out_condition']}: "
            f"u AUROC {_fmt(row['auroc_u_for_heldout_verify'])} [{_fmt(row['auroc_u_ci_lower'])}, {_fmt(row['auroc_u_ci_upper'])}]; "
            f"OOF LL intercept {_fmt(row['oof_logloss_intercept'])} / difficulty {_fmt(row['oof_logloss_difficulty'])} / "
            f"shared {_fmt(row['oof_logloss_shared_propensity'])} / rich {_fmt(row['oof_logloss_condition_sensitive'])}; "
            f"rich−shared+diff {_fmt(row.get('delta_rich_minus_shared_plus_difficulty', row.get('delta_rich_minus_shared')))}"
        )
    lines.extend(
        [
            "",
            "## 9. Does Claude “sharpening” survive latent-scale correction?",
            "",
            (
                "Task 007’s Claude hard−easy *probability* gap can widen as coverage falls even with a stable logit slope. "
                f"On the logit scale, moderate Claude γ CI is [{_fmt(claude_mod['gamma_ci_lower'])}, {_fmt(claude_mod['gamma_ci_upper'])}] "
                f"and grouped-CV interaction Δlog-loss is {_fmt(claude_mod['cv_delta_logloss'])} "
                f"(material_logit_interaction={claude_mod['material_logit_interaction']}). "
                "High-coverage 0.70 cells are saturated, so probability-scale sharpening there is partly a ceiling. "
                "Coefficient materiality now requires CV log-loss gain; a gamma CI excluding 0 is not enough under saturation."
            ),
            "",
            "## 10. Workshop error-catching reinterpretation",
            "",
            "See `workshop_reinterpretation.md`. GPT visibility mostly reduces coverage and tracks the raw-q threshold; "
            "matched-budget ranking advantage is uncertain. Calibration explains high-L cost. Claude hidden ranking is the exception.",
            "",
            "## 11. Current decision bucket",
            "",
            f"`{decision['decision_bucket']}`",
            "",
            (
                f"- GPT: class={decision['per_model']['openai_gpt56_sol']['label']}; "
                f"invariance_votes={decision['per_model']['openai_gpt56_sol']['invariance_votes']}; "
                f"reshape_votes={decision['per_model']['openai_gpt56_sol']['reshape_votes']}; "
                f"max_reversal={_fmt_pp(decision['per_model']['openai_gpt56_sol']['max_reversal'])} pp; "
                f"mean u AUROC={_fmt(decision['per_model']['openai_gpt56_sol']['mean_u_auc'])}"
            ),
            (
                f"- Claude: class={decision['per_model']['anthropic_sonnet5']['label']}; "
                f"invariance_votes={decision['per_model']['anthropic_sonnet5']['invariance_votes']}; "
                f"reshape_votes={decision['per_model']['anthropic_sonnet5']['reshape_votes']}; "
                f"max_reversal={_fmt_pp(decision['per_model']['anthropic_sonnet5']['max_reversal'])} pp; "
                f"mean u AUROC={_fmt(decision['per_model']['anthropic_sonnet5']['mean_u_auc'])}"
            ),
            "",
            "## 12. Final-paper narrative favored by current evidence",
            "",
            f"{decision['favored_world']} at strength `{decision['favor_strength']}`. Full prose in `final_narrative_decision_packet.md`.",
            "",
            "## 13. What the prospective study must prove or falsify",
            "",
            "- World 1: coverage moves; reversals stay near rerun noise; shared u_i predicts new scores; condition-sensitive models add <0.01 log-loss; matched-coverage catch is stable.",
            "- World 2: at identifiable operating points, ranking/discrimination changes enough to beat those tests.",
            "- World 3: the GPT vs Claude split is predeclared and replicates on fresh items.",
            "",
            "## 14. Recommended protocol shape (DO NOT EXECUTE)",
            "",
            "- One stakes family (moderate).",
            "- Prefer P2 if the goal is to test invariance (more operating points); P1 if budget-bound.",
            "- Primary GPT+Claude at N=500; Gemini/Grok on 200 of the same items.",
            "- Fresh Stage 1 + q1 mandatory. Drop q2.",
            "- Repeats: 100 items × 3 generations/cell.",
            "- Do not freeze a sample in this task.",
            "- Exact counts: `prospective_design_matrix.csv`.",
            "",
            "## 15. READY_FOR_GPT_REVIEW",
            "",
            "READY_FOR_GPT_REVIEW = YES",
            "",
            f"Generated {datetime.now(UTC).isoformat()}",
            "",
        ]
    )
    path = RETURN_DIR / "report.md"
    assert_008_write_target(path)
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_scripts() -> None:
    for src in SRC_FILES:
        dest = RETURN_DIR / src.name
        assert_008_write_target(dest)
        shutil.copy2(src, dest)


def list_changed() -> None:
    files = sorted(
        p.relative_to(RETURN_DIR.parent.parent).as_posix()
        for p in RETURN_DIR.rglob("*")
        if p.is_file()
    )
    files.extend(f"src/{p.name}" for p in SRC_FILES)
    path = RETURN_DIR / "changed_files.txt"
    assert_008_write_target(path)
    path.write_text("\n".join(files) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    before = protected_fingerprints()
    grid_rows, grid_val = load_qualitative_grid()
    gg = load_gemini_grok()
    for row in gg:
        row["is_wrong"] = 1 - int(row["stage1_correct"])
    roc, risk = roc_and_coverage_points(grid_rows, sample_role="primary_n100")
    roc_gg, risk_gg = roc_and_coverage_points(gg, sample_role="descriptive_n20")
    nest = nesting_table(grid_rows)
    interact = difficulty_logit_interactions(grid_rows)
    shared, rich = shared_vs_condition_sensitive(grid_rows)
    auc = condition_auc_table(grid_rows)
    common = common_curve_analysis(roc, nest)
    latent = latent_item_prediction(grid_rows)
    s1, ckpt = workshop_points()
    design = prospective_matrix()
    decision = decide_bucket(nest, interact, shared, rich, latent, auc)

    write_csv(RETURN_DIR / "roc_condition_points.csv", roc + roc_gg)
    write_csv(RETURN_DIR / "risk_coverage_condition_points.csv", risk + risk_gg)
    write_csv(RETURN_DIR / "nesting_reversals.csv", nest)
    write_csv(RETURN_DIR / "shared_ranking_model_results.csv", shared)
    write_csv(RETURN_DIR / "condition_sensitive_model_results.csv", rich)
    write_csv(RETURN_DIR / "difficulty_logit_interactions.csv", interact)
    write_csv(RETURN_DIR / "condition_auc.csv", auc)
    write_csv(RETURN_DIR / "common_curve_analysis.csv", common)
    write_csv(RETURN_DIR / "latent_item_prediction.csv", latent)
    write_csv(RETURN_DIR / "prospective_design_matrix.csv", design)
    json_dump(RETURN_DIR / "decision_summary.json", decision)

    plot_roc(roc, FIGURES_DIR / "roc_space_gpt_claude")
    plot_risk_coverage(risk, FIGURES_DIR / "risk_coverage_gpt_claude")
    plot_nesting(nest, FIGURES_DIR / "nesting_reversals")
    plot_slopes(interact, FIGURES_DIR / "logit_difficulty_slopes")
    plot_auc(auc, FIGURES_DIR / "condition_auc")

    write_workshop(s1, ckpt)
    write_power_notes(design)
    write_packet(decision, interact, latent)
    write_report(
        grid=grid_val,
        roc=roc,
        nest=nest,
        shared=shared,
        rich=rich,
        interact=interact,
        auc=auc,
        common=common,
        latent=latent,
        decision=decision,
        fingerprints=before,
    )
    copy_scripts()
    after = protected_fingerprints()
    write_validation(before, after, grid_val)
    list_changed()
    return decision


def main() -> None:
    decision = run()
    print("READY_FOR_GPT_REVIEW = YES")
    print(decision["decision_bucket"], decision["favored_world"], decision["favor_strength"])


if __name__ == "__main__":
    main()
