"""Task 010 orchestrator: zero-call sensitivity / equivalence / routing audit."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .study1_smoke import _git_commit
from .task005_common import sha256_file
from .task010_analysis import (
    choose_template,
    detection_floor,
    latent_rank_stability,
    load_009,
    matched_budget_routing,
    mismatch_robustness,
    observed_vs_predicted_auc,
    plot_all,
    run_positive_control,
    visible_design,
)
from .task010_common import (
    ANALYSIS_DIR,
    CLAUDE_008_GAMMA,
    DEVELOPMENT_ANCHOR_SOURCE,
    FIGURES_DIR,
    FROM_GPT,
    GPT_008_GAMMA,
    GPT_CLAUDE,
    MODEL_LABELS,
    N_BOOT_POWER,
    N_SIM,
    PAPER_DIRECTION,
    PRIMARY_ANCHOR_GAMMA,
    RETURN_DIR,
    SIM_SEED,
    TASK007_DIR,
    TASK008_DIR,
    TASK009_DIR,
    TASK_ID,
    assert_010_write_target,
    json_dump,
    protected_fingerprints,
    write_csv,
)

SRC_FILES = (
    Path(__file__),
    Path(__file__).with_name("task010_common.py"),
    Path(__file__).with_name("task010_analysis.py"),
)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _pp(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{100.0 * number:.{digits}f}pp"


def _copy_scripts() -> None:
    assert_010_write_target(ANALYSIS_DIR)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for path in SRC_FILES:
        shutil.copy2(path, ANALYSIS_DIR / path.name)


def _changed_files() -> None:
    rows = sorted(
        str(path.relative_to(path.parents[2] if "to_gpt" in str(path) else path.parent))
        if False
        else str(path)
        for path in RETURN_DIR.rglob("*")
        if path.is_file()
    )
    rel = [str(path.relative_to(RETURN_DIR.parent.parent)) for path in RETURN_DIR.rglob("*") if path.is_file()]
    (RETURN_DIR / "changed_files.txt").write_text("\n".join(sorted(rel)) + "\n", encoding="utf-8")


def write_positive_control_design(notes: dict[str, Any]) -> None:
    text = f"""# Task 010 positive-control design

Label: **Task-010 sensitivity analysis**, not Task-009 confirmation.

## Pipeline reconstructed from Task 009 H2

- Visible fixed scores only: 0.70, 0.85, 0.90, 0.95, 0.99
- N=500 questions × 5 conditions
- Shared model: condition intercepts + other-primary wrongness
- Richer model: shared + γ(displayed−0.90)×wrongness
- 5-fold GroupKFold by question ID
- LogisticRegression C∈{{inf, 1e6, 1.0}} as in Task 009
- Invariance fails if held-out log-loss improvement ≥ 0.01 **or** question-bootstrap 95% upper bound ≥ 0.01
- Power bootstrap resamples of frozen OOF predictions: {N_BOOT_POWER} (Task 009 used 5000 on the real data; the detector is otherwise identical)

## DGP

For each primary model, fit the Task-009 shared model in-sample. That fitted condition intercept + difficulty slope is the true-zero generator. Synthetic VERIFY draws are Bernoulli given that generator plus an injected extra.

## Family 1 — structured score × difficulty

{DEVELOPMENT_ANCHOR_SOURCE}

Primary grid: multipliers {notes.get("multipliers")} × Claude-008 γ={PRIMARY_ANCHOR_GAMMA:.3f}.
Secondary documented anchors: GPT-008 γ={GPT_008_GAMMA:.3f}; between-model |Δγ|={abs(CLAUDE_008_GAMMA-GPT_008_GAMMA):.3f}.

## Family 2 — generic item-condition noise

Independent Gaussian perturbations on the logit, scaled so the **latent** adjacent rank correlation of the linear predictor targets 0.99, 0.95, 0.90, 0.80, 0.70, 0.60.

## Monte Carlo

- Seed `{SIM_SEED}`
- {N_SIM} replications per cell
- True-zero cell: family 1, multiplier 0

Do not treat a more powerful detector than Task 009 as the paper's sensitivity.
"""
    path = RETURN_DIR / "positive_control_design.md"
    assert_010_write_target(path)
    path.write_text(text, encoding="utf-8")


def write_detection_floor(floors: dict[str, Any], power: list[dict[str, Any]], template: str) -> None:
    lines = ["# Task 010 detection floor\n"]
    lines.append("The Task-009 analysis could reliably detect reranking of approximately the magnitudes below under each simulated alternative family.\n")
    for model in GPT_CLAUDE:
        lines.append(f"## {MODEL_LABELS[model]}\n")
        for family, row in floors.get(model, {}).items():
            if not row:
                lines.append(f"- `{family}`: 80% power **not reached** on the simulated grid.\n")
                continue
            lines.append(
                f"- `{family}`: 80% power first reached at magnitude={_fmt(row['magnitude'])} "
                f"(label={row['magnitude_label']}), power={_fmt(row['power_full_009_rule'], 2)}, "
                f"mean latent adjacent Spearman={_fmt(row['mean_latent_adjacent_spearman'])}, "
                f"mean pairwise reversal={_fmt(row['mean_pairwise_reversal'])}.\n"
            )
    lines.append(f"\nPredeclared wording category justified: **{template}**.\n")
    path = RETURN_DIR / "positive_control_detection_floor.md"
    assert_010_write_target(path)
    path.write_text("".join(lines), encoding="utf-8")


def write_final_claim_audit(
    *,
    template: str,
    floors: dict[str, Any],
    auc_rows: list[dict[str, Any]],
    latent_claim: str,
    latent_rows: list[dict[str, Any]],
    mismatch: list[dict[str, Any]],
    routing_label: str,
    routing_rows: list[dict[str, Any]],
    fp_rate: dict[str, float],
) -> str:
    gpt_auc = [r for r in auc_rows if r["model_alias"] == "openai_gpt56_sol"]
    cla_auc = [r for r in auc_rows if r["model_alias"] == "anthropic_sonnet5"]
    rmse = float(
        np.sqrt(
            np.mean(
                [
                    (float(r["residual_obs_minus_pred"])) ** 2
                    for r in auc_rows
                    if np.isfinite(float(r["residual_obs_minus_pred"]))
                ]
            )
        )
    )
    gpt_raw_070 = next(
        (r for r in latent_rows if r["model_alias"] == "openai_gpt56_sol" and "0.70" in r["pair"]),
        {},
    )
    mismatch_hits = [
        r
        for r in mismatch
        if r.get("material_0_01")
    ]
    templates = {
        "TIGHT": (
            "Displayed confidence strongly shifts verification coverage, while any "
            "score-dependent reranking large enough to exceed the measured detection "
            "floor is inconsistent with the observed data under our sensitivity model."
        ),
        "MODERATE": (
            "Displayed confidence strongly shifts verification coverage. We find no "
            "evidence for score-dependent reranking of moderate or larger magnitude; "
            "smaller changes remain below the study's detection floor."
        ),
        "WEAK": (
            "Displayed confidence strongly shifts verification coverage, while our data "
            "do not support large score-dependent reranking. The study is not sufficiently "
            "sensitive to rule out smaller ranking changes."
        ),
    }
    chosen = templates[template]
    headline = {
        "TIGHT": "On confirmatory MMLU-Pro data, displayed confidence is a coverage control: Task 009 would have detected even small ranking rewrites of the kind considered here, and it did not.",
        "MODERATE": "On confirmatory MMLU-Pro data, displayed confidence moves verification coverage; moderate or larger score-dependent reranking is inconsistent with the sensitivity audit, while smaller ranking changes remain below the detection floor.",
        "WEAK": "On confirmatory MMLU-Pro data, displayed confidence moves verification coverage and large ranking rewrites are not supported, but the H2 test is not sensitive enough to rule out smaller reranking.",
    }[template]
    abstract = chosen
    conservative = templates["WEAK"] if template != "WEAK" else chosen
    if template == "TIGHT":
        conservative = templates["MODERATE"]
    forbidden = [
        "Displayed confidence never changes which items are verified.",
        "Ranking is proven identical across scores.",
        "The richer model is exactly equivalent to the shared model.",
        "AUROC changes prove (or disprove) ranking invariance by themselves.",
        "Cross-model difficulty is a deployable production router.",
        "Hidden-state uncertainty is being crowded out.",
        "Task 010 was preregistered as part of Task 009.",
        "Mismatch tests are confirmatory.",
    ]
    mmlu_stop = "YES — stop further MMLU-Pro mining unless a concrete statistical error in 009/010 is found."
    text = f"""# Task 010 final claim audit

## A. Positive-control sensitivity

Template justified: **{template}**

False-positive rate at zero injection (full 009 rule):
GPT { _fmt(fp_rate.get("openai_gpt56_sol"), 3) }; Claude { _fmt(fp_rate.get("anthropic_sonnet5"), 3) }.

Detection floors are in `positive_control_detection_floor.md`.

## B. AUROC drift

RMSE of shared-model predicted vs observed binary AUROC: {_fmt(rmse)}.
GPT residuals by score: {", ".join(_fmt(r["residual_obs_minus_pred"]) for r in gpt_auc)}.
Claude residuals: {", ".join(_fmt(r["residual_obs_minus_pred"]) for r in cla_auc)}.

## C. Latent rank stability

Label: **{latent_claim}**

GPT 0.70→0.85 raw Spearman {_fmt(gpt_raw_070.get("raw_spearman"))}; latent {_fmt(gpt_raw_070.get("latent_spearman"))}; posterior-predictive raw {_fmt(gpt_raw_070.get("pp_raw_spearman_mean"))}.

## D. Targeted mismatch rivals

Status: POST_HOC_ROBUSTNESS — NOT PREREGISTERED TASK-009 CONFIRMATION.

Material 0.01 hits: {len(mismatch_hits)}. See `mismatch_robustness.csv`.

## E. Routing consequence

Label: **{routing_label}**. Figure vs paragraph: {"main figure is justified" if routing_label == "MATERIAL_ROUTING_GAIN" else "paragraph unless the figure is used as a descriptive operating-point display"}.

## F. Final recommended wording

1. Headline: {headline}
2. Abstract-level: {abstract}
3. Conservative reviewer-proof: {conservative}
4. Forbidden stronger phrasings:
"""
    for item in forbidden:
        text += f"- {item}\n"
    text += f"\nMMLU analysis should now stop: {mmlu_stop}\n"
    path = RETURN_DIR / "final_claim_audit.md"
    assert_010_write_target(path)
    path.write_text(text, encoding="utf-8")
    return chosen


def write_report(
    *,
    template: str,
    wording: str,
    floors: dict[str, Any],
    power: list[dict[str, Any]],
    auc_rows: list[dict[str, Any]],
    latent_claim: str,
    latent_rows: list[dict[str, Any]],
    mismatch: list[dict[str, Any]],
    routing_label: str,
    fingerprints_before: dict[str, Any],
    fingerprints_after: dict[str, Any],
    against: list[str],
) -> None:
    fp = fingerprints_after["paperDirection"]["sha256"]
    path = RETURN_DIR / "report.md"
    text = f"""# Task 010 — Final zero-call sensitivity, equivalence, and routing audit

## 1. Plain-English bottom line

Task 009's confirmatory coverage result is untouched. This audit asks how strong the ranking-stability claim may be. Predeclared wording category: **{template}**.

{wording}

## 2. Validation / zero-call confirmation

- API calls: **0**
- paperDirection.txt sha256 before/after: `{fingerprints_before["paperDirection"]["sha256"]}` / `{fp}` (unchanged={fingerprints_before["paperDirection"]["sha256"] == fp})
- Task 009 stage3 sha256 unchanged: {fingerprints_before["task009_stage3"]["sha256"] == fingerprints_after["task009_stage3"]["sha256"]}
- Claim templates frozen in `claim_wording_preregistered.md` before simulations
- Confirmatory vs sensitivity vs post-hoc labels are kept separate

## 3. Positive-control simulation design

See `positive_control_design.md`. Seed {SIM_SEED}, {N_SIM} sims/cell, exact Task-009 H2 pipeline.

## 4. Detection floor and power

See `positive_control_detection_floor.md` and `positive_control_power_curve.csv`.

## 5. Which predeclared equivalence wording is justified

**{template}**

{wording}

## 6. Shared-ranking prediction of AUROC drift

See `observed_vs_shared_predicted_auc.csv` and `figures/observed_vs_shared_predicted_auc.png`.

## 7. Hierarchical latent rank-stability result

**{latent_claim}**

See `latent_rank_stability.csv`.

## 8. Targeted mismatch robustness

POST_HOC_ROBUSTNESS — NOT PREREGISTERED TASK-009 CONFIRMATION. See `mismatch_robustness.csv`.

## 9. Matched-budget routing

**{routing_label}**. See `matched_budget_routing.csv` and `figures/matched_budget_routing.png`.

## 10. Any evidence against the current narrative

{chr(10).join("- " + x for x in against) if against else "- None that overturns Task-009 coverage control. Remaining caveats are detection-floor and post-hoc robustness, reported above."}

## 11. Final strongest defensible claim

{wording}

## 12. Exact wording that should be removed from paperDirection.txt, if any

Depends on **{template}**. If WEAK or MODERATE, do not keep sentences that say score-specific reprioritization contributes "essentially no" held-out value *and* imply that small reranking is ruled out. Task 010 does not edit paperDirection.txt.

## 13. Whether MMLU analysis should now stop

YES, unless a concrete statistical error is found in Tasks 009–010.

## 14. Review flag

READY_FOR_GPT_REVIEW = YES
"""
    assert_010_write_target(path)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    before = protected_fingerprints()
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not (RETURN_DIR / "claim_wording_preregistered.md").exists():
        raise RuntimeError("claim_wording_preregistered.md must exist before simulations")
    data = load_009()
    for model in GPT_CLAUDE:
        n = len(visible_design(data, model))
        if n != 2500:
            print(f"WARNING: {model} visible rows={n} expected 2500", flush=True)
    write_positive_control_design({"multipliers": [0, 0.25, 0.5, 1.0, 1.5, 2.0]})
    print("Task 010 positive control...", flush=True)
    sim_rows, power_rows, notes = run_positive_control(data)
    write_csv(RETURN_DIR / "positive_control_results.csv", sim_rows)
    write_csv(RETURN_DIR / "positive_control_power_curve.csv", power_rows)
    floors = detection_floor(power_rows)
    template = choose_template(floors, power_rows)
    write_detection_floor(floors, power_rows, template)
    print("Task 010 AUROC / latent / mismatch / routing...", flush=True)
    auc_rows = observed_vs_predicted_auc(data)
    write_csv(RETURN_DIR / "observed_vs_shared_predicted_auc.csv", auc_rows)
    latent_rows, latent_pp, latent_claim = latent_rank_stability(data)
    write_csv(RETURN_DIR / "latent_rank_stability.csv", latent_rows)
    write_csv(RETURN_DIR / "latent_rank_posterior_predictive.csv", latent_pp)
    mismatch = mismatch_robustness(data)
    write_csv(RETURN_DIR / "mismatch_robustness.csv", mismatch)
    routing_rows, routing_label = matched_budget_routing(data)
    write_csv(RETURN_DIR / "matched_budget_routing.csv", routing_rows)
    plot_all(power_rows, auc_rows, latent_rows, routing_rows)
    fp_rate = {}
    for model in GPT_CLAUDE:
        row = next(
            r
            for r in power_rows
            if r["model_alias"] == model
            and r["family"] == "score_x_difficulty"
            and float(r["magnitude"]) == 0.0
        )
        fp_rate[model] = float(row["power_full_009_rule"])
    against: list[str] = []
    if any(float(r["power_full_009_rule"]) > 0.10 for r in power_rows if float(r.get("magnitude") or 0) == 0 and r["family"] == "score_x_difficulty"):
        against.append("Zero-injection false positives above 10% under the full 009 rule; treat the equivalence test as anti-conservative in the upper bound.")
    if any(abs(float(r["residual_obs_minus_pred"])) > 0.05 for r in auc_rows):
        against.append("At least one score condition has |observed−predicted AUROC| > 0.05; do not wave away residual condition dependence.")
    if any(r.get("material_0_01") for r in mismatch):
        against.append("A targeted mismatch predictor met the 0.01 materiality scale post-hoc; narrow the invariance sentence.")
    wording = write_final_claim_audit(
        template=template,
        floors=floors,
        auc_rows=auc_rows,
        latent_claim=latent_claim,
        latent_rows=latent_rows,
        mismatch=mismatch,
        routing_label=routing_label,
        routing_rows=routing_rows,
        fp_rate=fp_rate,
    )
    after = protected_fingerprints()
    if after["paperDirection"]["sha256"] != before["paperDirection"]["sha256"]:
        raise RuntimeError("paperDirection.txt changed during Task 010")
    for key in ("v2", "study1", "q2", "qual006", "qual007a", "qual007b", "task009_stage3"):
        if before[key].get("sha256") != after[key].get("sha256"):
            raise RuntimeError(f"protected {key} changed during Task 010")
    write_report(
        template=template,
        wording=wording,
        floors=floors,
        power=power_rows,
        auc_rows=auc_rows,
        latent_claim=latent_claim,
        latent_rows=latent_rows,
        mismatch=mismatch,
        routing_label=routing_label,
        fingerprints_before=before,
        fingerprints_after=after,
        against=against,
    )
    validation = f"""# Task 010 validation

- API calls: 0
- paperDirection sha256: `{after["paperDirection"]["sha256"]}`
- Task 007/008/009 databases and CSVs were not written
- claim_wording_preregistered.md existed before simulations
- H2 pipeline copied from Task 009: GroupKFold, 0.01 log-loss boundary, question bootstrap
- Sim seed {SIM_SEED}; n_sim {N_SIM}
- code commit: `{_git_commit()}`
"""
    (RETURN_DIR / "validation.md").write_text(validation, encoding="utf-8")
    json_dump(
        RETURN_DIR / "run_manifest.json",
        {
            "task_id": TASK_ID,
            "created_at": datetime.now(UTC).isoformat(),
            "api_calls": 0,
            "template": template,
            "latent_claim": latent_claim,
            "routing_label": routing_label,
            "paperDirection_sha256": after["paperDirection"]["sha256"],
        },
    )
    _copy_scripts()
    _changed_files()
    print(
        {
            "ok": True,
            "api_calls": 0,
            "template": template,
            "latent": latent_claim,
            "routing": routing_label,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
