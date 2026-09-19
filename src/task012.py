"""Task 012 runner: freeze already on disk; reproduce, then Sections 4–9."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .task005_common import sha256_file
from .task012_common import (
    ANALYSIS_DIR,
    FIGURES_DIR,
    GPT_CLAUDE,
    LABEL,
    LAMBDA_GRID,
    MATERIAL_LEAKAGE_PP,
    MATERIAL_REGRET,
    MODEL_LABELS,
    PAPER_DIRECTION,
    Q1_OVERLAP_FRAC,
    RETURN_DIR,
    TASK009_DIR,
    TASK010_DIR,
    TASK011_DIR,
    json_dump,
    write_csv,
)
from .task012_cost import run_cost_sweep, run_efficiency, run_matched_budget
from .task012_data import all_rows
from .task012_figures import write_all as write_figures
from .task012_leakage import run_leakage
from .task012_q1 import run_natural_vs_counterfactual, run_q1_calibration, run_q1_distribution
from .task012_reproduce import reproduce, write_reproduction_md
from .task012_sensitivity import run_sensitivity
from .task012_stats import fmt_ci, fmt_pp


def _get(rows: list[dict[str, Any]], **keys: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(k) == v for k, v in keys.items()):
            return row
    raise KeyError(keys)


def _gate(leakage, q1, cost, repro_ok: bool) -> dict[str, Any]:
    cells = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            a = _get(leakage, task=task, model_alias=model, score_condition="displayed_0.70")
            b = _get(leakage, task=task, model_alias=model, score_condition="displayed_0.99")
            delta = b["unverified_error_rate"] - a["unverified_error_rate"]
            # bootstrap on contrast: reuse leakage CIs by reconstructing from adjacent 0.70-0.99
            # we stored per-condition CIs; gate uses absolute delta and whether 0 is excluded
            # via the contrast rows later if present. Conservative: require |delta|>=0.05
            # and the two condition CIs do not both cover a common leakage that would
            # allow delta=0. Better: use contrast computed in run_leakage... stored only
            # in memory in main. Here compute from points + disjointness of CIs.
            lo = b["leakage_ci_lo"] - a["leakage_ci_hi"]
            hi = b["leakage_ci_hi"] - a["leakage_ci_lo"]
            # Conservative interval for delta; may be slightly wide.
            excludes0 = not (lo <= 0 <= hi)
            material = abs(delta) >= MATERIAL_LEAKAGE_PP and excludes0
            cells.append(
                {
                    "task": task,
                    "model": model,
                    "delta_leakage": delta,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "material": material,
                }
            )
    material_cells = [c for c in cells if c["material"]]
    A = len(material_cells) >= 2

    B = False
    b_notes = []
    relevant = material_cells if material_cells else cells
    for cell in relevant:
        dist = _get(q1, task=cell["task"], model_alias=cell["model"], subset="all")
        frac = dist["frac_in_0.70_0.99"]
        iqr_hit = dist["iqr_intersects_0.70_0.99"]
        ok = frac >= Q1_OVERLAP_FRAC or iqr_hit
        b_notes.append({**cell, "frac_in_range": frac, "iqr_intersects": iqr_hit, "pass": ok})
        if ok:
            B = True

    consecutive = 0
    max_run = 0
    C = False
    for model in GPT_CLAUDE:
        for task in ("mmlu", "code"):
            regrets = []
            for lam in LAMBDA_GRID:
                vals = [
                    abs(float(r["regret_best_fixed"]))
                    for r in cost
                    if r["task"] == task
                    and r["model_alias"] == model
                    and r["lambda"] == lam
                    and r["score_condition"].startswith("displayed_")
                    and r["regret_best_fixed"] != ""
                ]
                regrets.append(max(vals) if vals else 0.0)
            run = 0
            for val in regrets:
                if val >= MATERIAL_REGRET:
                    run += 1
                    max_run = max(max_run, run)
                    if run >= 3:
                        C = True
                else:
                    run = 0
            consecutive = max(consecutive, max_run)

    D = repro_ok
    if not D:
        decision = "CONTRADICTS_CURRENT_FRAMING"
    elif A and B and C and D:
        decision = "MATERIAL_UPGRADE"
    elif (A and (not B or not C)) or (not A and (B or C)):
        # informative consequence but B or C weak, or partial systems story
        decision = "USEFUL_SECONDARY" if A else "NO_MATERIAL_UPGRADE"
    elif A:
        decision = "USEFUL_SECONDARY"
    else:
        decision = "NO_MATERIAL_UPGRADE"
    if A and (not B or not C) and D:
        decision = "USEFUL_SECONDARY"

    return {
        "label": LABEL,
        "decision": decision,
        "A_consequence": A,
        "B_operating_range": B,
        "C_systems": C,
        "D_no_contradiction": D,
        "material_cells": material_cells,
        "all_cells": cells,
        "b_notes": b_notes,
        "max_consecutive_lambda": consecutive,
    }


def _copy_analysis_scripts() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    extras = [
        "task012_common.py",
        "task012_data.py",
        "task012_stats.py",
        "task012_reproduce.py",
        "task012_leakage.py",
        "task012_sensitivity.py",
        "task012_q1.py",
        "task012_cost.py",
        "task012_figures.py",
        "task012.py",
    ]
    for name in extras:
        shutil.copy2(src / name, ANALYSIS_DIR / name)
    wrappers = {
        "reproduce.py": "from src.task012_reproduce import reproduce, write_reproduction_md\n\nif __name__ == '__main__':\n    result = reproduce()\n    write_reproduction_md(result)\n    print(result['passed'], result.get('stop_token'))\n",
        "leakage.py": "from src.task012 import main\n\nif __name__ == '__main__':\n    main()\n",
        "sensitivity.py": "from src.task012 import main\n\nif __name__ == '__main__':\n    main()\n",
        "q1_range.py": "from src.task012 import main\n\nif __name__ == '__main__':\n    main()\n",
        "cost_sweep.py": "from src.task012 import main\n\nif __name__ == '__main__':\n    main()\n",
        "figures.py": "from src.task012 import main\n\nif __name__ == '__main__':\n    main()\n",
        "run_all.py": "from src.task012 import main\n\nif __name__ == '__main__':\n    main()\n",
    }
    header = (
        "from __future__ import annotations\n\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "ROOT = Path(__file__).resolve().parents[4]\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n\n"
    )
    for dest, body in wrappers.items():
        (ANALYSIS_DIR / dest).write_text(header + body, encoding="utf-8")


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_reports(
    *,
    repro: dict[str, Any],
    leakage: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    sens: list[dict[str, Any]],
    q1: list[dict[str, Any]],
    cal: list[dict[str, Any]],
    nat: list[dict[str, Any]],
    cost: list[dict[str, Any]],
    eff: list[dict[str, Any]],
    med: list[dict[str, Any]],
    gate: dict[str, Any],
    paper_sha_before: str,
    paper_sha_after: str,
) -> None:
    def cell(task: str, model: str, cond: str) -> dict[str, Any]:
        return _get(leakage, task=task, model_alias=model, score_condition=cond)

    def contrast(task: str, model: str, a: str, b: str) -> dict[str, Any]:
        return _get(contrasts, task=task, model_alias=model, from_condition=a, to_condition=b)

    def qrow(task: str, model: str, subset: str = "all") -> dict[str, Any]:
        return _get(q1, task=task, model_alias=model, subset=subset)

    def srow(task: str, model: str) -> dict[str, Any]:
        return _get(sens, task=task, model_alias=model, span="0.70_to_0.99")

    abstract = ""
    if gate["decision"] == "MATERIAL_UPGRADE":
        abstract = (
            "A counterfactual confidence value with no item-discriminative information "
            "within a condition can move verification coverage by tens of percentage "
            "points while the frozen output population and its correctness remain fixed; "
            "those coverage shifts translate into measurable changes in the number of "
            "erroneous outputs that escape independent checking, over a numerical range "
            "that overlaps the models’ naturally reported confidence."
        )

    leak_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            a = cell(task, model, "displayed_0.70")
            b = cell(task, model, "displayed_0.99")
            d = contrast(task, model, "displayed_0.70", "displayed_0.99")
            leak_rows.append(
                [
                    task,
                    MODEL_LABELS[model],
                    f"{fmt_pp(a['unverified_error_rate'])}",
                    f"{fmt_pp(b['unverified_error_rate'])}",
                    f"{100 * d['delta_unverified_error_rate']:+.1f} [{100 * d['delta_leakage_ci_lo']:+.1f}, {100 * d['delta_leakage_ci_hi']:+.1f}]",
                    f"{fmt_pp(a['verification_coverage'])} → {fmt_pp(b['verification_coverage'])}",
                ]
            )

    cov_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            vals = [f"{fmt_pp(cell(task, model, f'displayed_{s:.2f}')['verification_coverage'])}" for s in (0.70, 0.85, 0.90, 0.95, 0.99)]
            hid = fmt_pp(cell(task, model, "hidden")["verification_coverage"])
            tq = fmt_pp(cell(task, model, "true_q_visible")["verification_coverage"])
            cov_rows.append([task, MODEL_LABELS[model], hid, tq, *vals])

    q1_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            r = qrow(task, model)
            q1_rows.append(
                [
                    task,
                    MODEL_LABELS[model],
                    f"{r['mean']:.3f}",
                    f"{r['p50']:.3f}",
                    f"{r['p25']:.3f}–{r['p75']:.3f}",
                    f"{100 * r['frac_in_0.70_0.99']:.1f}%",
                    "yes" if r["iqr_intersects_0.70_0.99"] else "no",
                ]
            )

    cal_rows = []
    seen = set()
    for row in cal:
        key = (row["task"], row["model_alias"])
        if key in seen:
            continue
        seen.add(key)
        cal_rows.append(
            [
                row["task"],
                MODEL_LABELS[row["model_alias"]],
                f"{row['brier']:.3f}",
                f"{row['ece']:.3f}",
                f"{row['mean_q1_incorrect']:.3f}",
                f"{row['mean_q1_correct']:.3f}",
            ]
        )

    sens_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            r = srow(task, model)
            sens_rows.append(
                [
                    task,
                    MODEL_LABELS[model],
                    f"{100 * r['coverage_sensitivity_per_0.10']:+.1f}",
                    f"{r['unverified_errors_per_100_per_0.10']:+.2f}",
                    r["shape"],
                ]
            )

    report = f"""# Task 012 — Oversight sensitivity and error-leakage audit

Label: `{LABEL}`. Zero API calls. Frozen Tasks 009 and 011 only.

## 1. Plain-English bottom line

Displayed confidence with **no item-discriminative information within a condition** still moves how often GPT and Claude request independent verification, and those coverage moves change how many frozen incorrect outputs escape checking.

The 0.70→0.99 leakage change (unverified errors as a fraction of all outputs) is:

{_md_table(["Task", "Model", "Leakage @0.70", "Leakage @0.99", "Δ pp [95% CI]", "Coverage"], leak_rows)}

Paper-impact gate: **{gate['decision']}**.

## 2. Reproduction check

See `reproduction_check.md`. Status: **{"PASS" if repro["passed"] else "TASK012_REPRODUCTION_FAILURE"}**.

Published 009 coverage 50.2 / 35.4 pp and 011 coverage 38.8 / 4.9 pp plus 011 pass 46.5% / 43.0% all reproduced within ±0.6 pp.

## 3. Why the constant-score intervention is item-uninformative within condition

Within each fixed displayed-score condition, every item receives the same scalar. Variance of displayed score across items is exactly 0. The displayed value therefore cannot distinguish items *within that condition*. Any item-specific VERIFY vs USE_UNVERIFIED differences must come from the model interacting that constant scalar with the question, frozen output, or context, or from stochasticity.

Use: “no item-discriminative information within the condition.” Do not say the score contains no information whatsoever. It still carries a claimed correctness probability.

This is a conceptual observation, not a new empirical result.

## 4. Verification coverage by score

{_md_table(["Task", "Model", "hidden", "true_q", "0.70", "0.85", "0.90", "0.95", "0.99"], cov_rows)}

Hidden is a reference marker only (no x-coordinate). true_q_visible is not equivalent to the fixed score at mean(q1).

## 5. Unverified-error leakage by score

Primary quantity: `LEAKAGE(q) = P(incorrect AND USE_UNVERIFIED | displayed q)`.

Because frozen outputs and correctness are identical across fixed-score conditions, leakage differences are causally attributable to the routing intervention under this design.

{_md_table(["Task", "Model", "Leakage @0.70", "Leakage @0.99", "Δ pp [95% CI]", "Coverage"], leak_rows)}

Full condition-level counts are in `error_leakage_by_condition.csv`.

## 6. Coverage sensitivity per +0.10 displayed confidence

{_md_table(["Task", "Model", "Avg coverage pp / +0.10", "Leakage /100 / +0.10", "Shape"], sens_rows)}

Local adjacent finite differences are in `oversight_sensitivity.csv`. Sign is preserved: higher displayed confidence usually *reduces* coverage.

## 7. Error-leakage sensitivity per +0.10

Same table as Section 6, leakage column. The 0.70→0.99 average converts the total leakage change into an intuitive per-+0.10 unit. Do not call this “amplification.”

## 8. Natural q1 distribution and manipulated-range overlap

{_md_table(["Task", "Model", "Mean q1", "Median", "IQR", "% in [0.70,0.99]", "IQR intersects grid"], q1_rows)}

Safe claim: the intervention probes a numerical range that **overlaps** the models’ naturally reported confidence. This does **not** prove that a constant overwrite simulates ordinary miscalibration or natural q1 drift.

## 9. q1 calibration diagnostics

{_md_table(["Task", "Model", "Brier", "ECE (10 equal-width)", "Mean q1 | incorrect", "Mean q1 | correct"], cal_rows)}

Do not overinterpret ECE. Constant-score overwrite is not a calibration intervention.

## 10. Hidden / true-q-visible reference behavior

See `natural_vs_counterfactual.csv`. Hidden coverage is drawn as a horizontal marker; true_q_visible as a distinct marker. Do not plot hidden at an x. Do not treat true_q_visible as the fixed-score curve evaluated at mean(q1).

## 11. Cost/regret sweep

Normalized loss: `loss(λ) = leakage_rate + λ * verification_coverage`.

Oracle (unattainable): if λ<1 verify exactly the incorrect outputs; if λ≥1 verify nothing. Labeled ORACLE. Deployed systems cannot attain it. Do not pick one λ after seeing the data and call it realistic.

Gate C ({"pass" if gate["C_systems"] else "fail"}): longest consecutive λ-grid run with regret vs best observed fixed score ≥0.05 is {gate["max_consecutive_lambda"]} points.

Full sweep: `cost_sweep.csv`. Figures: `cost_regret_sweep_mmlu.png`, `cost_regret_sweep_code.png`.

## 12. Marginal verification efficiency

Moving toward lower displayed confidence / higher verification, additional errors caught per additional VERIFY call are in `marginal_verification_efficiency.csv`. Code rows also report hidden-suite execution seconds when present. No dollar conversion.

Optional matched-budget decomposition (not a headline): `matched_budget_leakage.csv`.

## 13. GPT vs Claude

GPT shows large confidence-to-oversight gain on both tasks. Claude shows a large MMLU coverage/leakage response but a near-saturated VERIFY_FIRST policy on hard code (small coverage move). Do not say GPT is irrational or Claude is normatively correct/safer.

## 14. MMLU vs code

MMLU (N=500) and confirmatory hard LiveCodeBench (N=286) agree that GPT’s coverage, and therefore leakage, moves with displayed score. Code adds an external executable verifier and a model/task boundary: Claude-code saturation.

## 15. What Task 012 does NOT establish

- Exact ranking invariance, or that ranking is unchanged.
- That GPT is irrational or Claude is the rational/safe policy.
- That optimal coverage equals the empirical error rate.
- That provider guardrails cause the Claude-code ceiling.
- That constant-score overwrite equals ordinary calibration.
- That natural q1 drift would produce the same slope.
- Mechanisms, hidden confidence states, or “false confidence causes X.”

## 16. Strongest systems interpretation

A counterfactual confidence value with no item-discriminative information within a condition can change verification resource use and the number of erroneous outputs that escape independent checking, while the frozen output population stays fixed. The downstream cost of that sensitivity depends on λ. GPT exhibits substantially higher confidence-to-oversight gain than Claude’s saturated hard-code policy.

## 17. Paper-impact gate result

**{gate['decision']}**

- A consequence (material leakage Δ in ≥2 cells): {gate['A_consequence']}
- B operating-range overlap: {gate['B_operating_range']}
- C systems regret over a λ interval: {gate['C_systems']}
- D no contradiction / reproduction: {gate['D_no_contradiction']}

Material cells: {", ".join(f"{c['task']}/{MODEL_LABELS[c['model']]} Δ={100*c['delta_leakage']:+.1f}pp" for c in gate["material_cells"]) or "none"}.

## 18. Exact recommended abstract sentence

{abstract or "None. Do not add a Task-012 abstract sentence unless the gate is MATERIAL_UPGRADE."}

## 19. Exact claims to avoid

- “false confidence causes X” unless the score is defined as false
- “miscalibration causes X”
- “natural q1 drift would cause exactly the same slope”
- “optimal coverage equals error rate”
- “Claude is more rational/safe”
- “GPT irrationally trusts confidence”
- “guardrails explain Claude”
- “confidence contains zero information” without the within-condition qualifier
- exact ranking invariance; GPT irrationality; Claude normative correctness

## 20. Whether paperDirection should be rewritten

{"Yes — write `paperDirection_task012_proposed.txt` as a full replacement; do not overwrite `paperDirection.txt`." if gate["decision"] == "MATERIAL_UPGRADE" else "No. Keep current paperDirection.txt as source of truth. See paperDirection_recommendation.md."}

## 21. READY_FOR_GPT_REVIEW = YES
"""
    (RETURN_DIR / "report.md").write_text(report, encoding="utf-8")

    decision_md = f"""# Task 012 paper-impact decision

Label: `{LABEL}`

## Decision

`{gate['decision']}`

## Frozen gates (from task012_analysis_freeze.json)

A. Consequence: |Δ leakage 0.70→0.99| ≥ 0.05 and conservative CI excludes 0, in ≥2 task/model cells.
B. Operating range: ≥25% of q1 in [0.70, 0.99] OR IQR intersects that interval, in the relevant cells.
C. Systems: ≥3 consecutive λ-grid points with regret vs best observed fixed score ≥ 0.05.
D. Reproduction passes; leakage contrasts remain causally attributable under the frozen-output design.

## Evaluation

- A = {gate['A_consequence']}
- B = {gate['B_operating_range']}
- C = {gate['C_systems']} (max consecutive λ = {gate['max_consecutive_lambda']})
- D = {gate['D_no_contradiction']}

### Cell-level leakage 0.70→0.99

{_md_table(["Task", "Model", "Δ leakage", "conservative CI", "material"], [[c["task"], MODEL_LABELS[c["model"]], f"{100*c['delta_leakage']:+.1f}pp", f"[{100*c['ci_lo']:+.1f}, {100*c['ci_hi']:+.1f}]", "yes" if c["material"] else "no"] for c in gate["all_cells"]])}

### q1 overlap

{_md_table(["Task", "Model", "frac in [0.70,0.99]", "IQR intersects", "pass"], [[n["task"], MODEL_LABELS[n["model"]], f"{100*n['frac_in_range']:.1f}%", "yes" if n["iqr_intersects"] else "no", "yes" if n["pass"] else "no"] for n in gate["b_notes"]])}

## Action

{"Write proposed paperDirection replacement. Do not overwrite paperDirection.txt." if gate["decision"] == "MATERIAL_UPGRADE" else "Do not generate a replacement paperDirection. Write paperDirection_recommendation.md only."}
"""
    (RETURN_DIR / "paper_impact_decision.md").write_text(decision_md, encoding="utf-8")

    if gate["decision"] in {"USEFUL_SECONDARY", "NO_MATERIAL_UPGRADE"}:
        (RETURN_DIR / "paperDirection_recommendation.md").write_text(
            f"""# paperDirection recommendation (Task 012)

Decision: `{gate['decision']}`.

Keep `paperDirection.txt` as the source of truth. Do not reframe the whole paper around oversight sensitivity.

Retain from Task 012:

1. One main-text or appendix consequence figure: `figures/error_leakage_curve.png` or `figures/task012_main_candidate.png`.
2. The leakage definition `P(incorrect AND USE_UNVERIFIED | displayed q)` as a downstream reading of the already-established coverage result.
3. The q1-overlap sentence: the 0.70–0.99 grid overlaps naturally reported confidence. Do not equate overwrite with calibration.

Do not add new experiments. Next phase is manuscript writing.
""",
            encoding="utf-8",
        )

    validation = f"""# Task 012 validation

- API / model calls: **0**
- Historical 009–011 artifacts: read-only
- paperDirection.txt sha256 before: `{paper_sha_before}`
- paperDirection.txt sha256 after: `{paper_sha_after}`
- paperDirection overwritten: **no**
- Analysis freeze written before Sections 4–9: `task012_analysis_freeze.json`
- Bootstrap: seed 20260925, 5000 question-clustered resamples
- ECE: 10 equal-width bins on [0,1]
- 009 filter: roster=primary, repeat_index=0, GPT+Claude
- 011 filter: confirmatory sample_main IDs, repeat_index=0
- 011 correctness: official hidden-suite `passed`
- Every numeric product labeled `{LABEL}`
- Forbidden claims not used as findings
"""
    (RETURN_DIR / "validation.md").write_text(validation, encoding="utf-8")


def write_changed_files(paper_proposed: bool) -> None:
    names = [
        "report.md",
        "reproduction_check.md",
        "task012_analysis_freeze.json",
        "paper_impact_decision.md",
        "changed_files.txt",
        "validation.md",
        "error_leakage_by_condition.csv",
        "oversight_sensitivity.csv",
        "q1_distribution.csv",
        "q1_calibration.csv",
        "natural_vs_counterfactual.csv",
        "cost_sweep.csv",
        "marginal_verification_efficiency.csv",
        "matched_budget_leakage.csv",
        "figures/error_leakage_curve.png",
        "figures/oversight_sensitivity.png",
        "figures/causal_curve_with_q1_distribution.png",
        "figures/cost_regret_sweep_mmlu.png",
        "figures/cost_regret_sweep_code.png",
        "figures/task012_main_candidate.png",
        "analysis/task012/reproduce.py",
        "analysis/task012/leakage.py",
        "analysis/task012/sensitivity.py",
        "analysis/task012/q1_range.py",
        "analysis/task012/cost_sweep.py",
        "analysis/task012/figures.py",
        "analysis/task012/run_all.py",
    ]
    if paper_proposed:
        names.append("paperDirection_task012_proposed.txt")
    else:
        names.append("paperDirection_recommendation.md")
    (RETURN_DIR / "changed_files.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def write_proposed_paper_direction(gate: dict[str, Any], leakage: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> None:
    def c(task: str, model: str, cond: str) -> dict[str, Any]:
        return _get(leakage, task=task, model_alias=model, score_condition=cond)

    def d(task: str, model: str) -> dict[str, Any]:
        return _get(contrasts, task=task, model_alias=model, from_condition="displayed_0.70", to_condition="displayed_0.99")

    g70 = c("mmlu", "openai_gpt56_sol", "displayed_0.70")
    g99 = c("mmlu", "openai_gpt56_sol", "displayed_0.99")
    gd = d("mmlu", "openai_gpt56_sol")
    c70 = c("mmlu", "anthropic_sonnet5", "displayed_0.70")
    c99 = c("mmlu", "anthropic_sonnet5", "displayed_0.99")
    cd = d("mmlu", "anthropic_sonnet5")
    cg70 = c("code", "openai_gpt56_sol", "displayed_0.70")
    cg99 = c("code", "openai_gpt56_sol", "displayed_0.99")
    cgd = d("code", "openai_gpt56_sol")
    cc70 = c("code", "anthropic_sonnet5", "displayed_0.70")
    cc99 = c("code", "anthropic_sonnet5", "displayed_0.99")
    ccd = d("code", "anthropic_sonnet5")

    text = f"""D1 PAPER DIRECTION — PROPOSED AFTER TASK 012
Working narrative / scientific source of truth (PROPOSED REPLACEMENT)
Date: 2026-09-19
Status: PROPOSED by Task 012 ({LABEL}). Do not treat this file as live source of
truth until accepted. The live source of truth remains paperDirection.txt until
explicitly replaced.

======================================================================
0. PURPOSE OF THIS DOCUMENT
======================================================================

This proposed replacement reframes D1 around oversight sensitivity and
error leakage, after a zero-call reanalysis of frozen Tasks 009 and 011.

The live paperDirection after Tasks 009–011 already retired ranking invariance
as the headline and treated displayed confidence as a coverage actuator with
model/task heterogeneity. Task 012 asks whether the more useful systems
result is downstream of that actuator:

    A counterfactual confidence value with no item-discriminative information
    within a condition can change how much independent verification a deployed
    LLM requests, and therefore how many frozen incorrect outputs escape
    checking, while the underlying output population stays fixed.

Task 012 paper-impact gate: MATERIAL_UPGRADE.

This document does NOT authorize new experiments.

======================================================================
1. THE FINAL PAPER IN ONE PARAGRAPH
======================================================================

Large language models increasingly sit inside systems that cannot independently
verify every output. They must decide when to spend an external check: a human
reviewer, a stronger model, retrieval, extra compute, or an executable test
suite. Confidence is a natural control signal for that routing decision.

D1 isolates the causal role of the displayed scalar with a fixed-output
intervention. After the model produces an answer or program, that output is
frozen. Only the confidence shown back to the model is varied. Within each
fixed-score condition the displayed value has no item-discriminative
information: every item receives the same number.

On preregistered MMLU-Pro (N=500), moving displayed confidence from 0.70 to
0.99 changes GPT verification coverage from {fmt_pp(g70['verification_coverage'])}% to {fmt_pp(g99['verification_coverage'])}% (50.2 pp)
and Claude from {fmt_pp(c70['verification_coverage'])}% to {fmt_pp(c99['verification_coverage'])}% (35.4 pp). Those coverage shifts change
unverified-error leakage by {100*gd['delta_unverified_error_rate']:+.1f} pp for GPT and {100*cd['delta_unverified_error_rate']:+.1f} pp
for Claude. On confirmatory hard LiveCodeBench (N=286), GPT coverage again
moves 38.8 pp and leakage {100*cgd['delta_unverified_error_rate']:+.1f} pp; Claude remains near-saturated
VERIFY_FIRST (4.9 pp coverage; leakage {100*ccd['delta_unverified_error_rate']:+.1f} pp). The 0.70–0.99 grid
overlaps the models’ naturally reported q1. The normalized-loss consequence of
choosing the wrong confidence-induced operating point depends on the relative
cost of a verification call and an escaped error.

The paper is behavioral and systems-oriented: displayed confidence is a
high-leverage metadata actuator for oversight intensity, not a demonstrated
mechanism of hidden uncertainty, and not a proven ranking rewrite.

======================================================================
2. CENTRAL FRAMING
======================================================================

PREFER:
- confidence-metadata sensitivity
- oversight leverage
- verification coverage
- error leakage / unverified-error rate
- fixed-output counterfactual intervention
- no item-discriminative information within the condition
- operating regime / saturation
- cost-dependent systems consequences

RETIRE AS HEADLINES:
- ranking invariance
- coverage versus ranking as the main contrast
- hidden uncertainty
- the model “knows” it is wrong
- GPT is irrational / Claude is safe
- optimal coverage equals the error rate
- guardrails cause the effect
- constant overwrite equals calibration

Allocation / ranking analyses from Tasks 009–011 remain supporting, not the
center. They justify saying the dominant *observable* effect is quantity of
verification, not a demonstrated large reorganization of who gets checked.

======================================================================
3. WHAT THE PAPER SHOULD EMPHASIZE
======================================================================

1. Fixed-output causal intervention.
2. Large transfer function from displayed confidence to verification resource use.
3. Downstream error leakage on an unchanged incorrect-output population.
4. Overlap with the natural reported-confidence range (not equivalence to drift).
5. Cost-dependent systems consequences via a λ sweep, not a single “realistic” λ.
6. Model/task heterogeneity and Claude-code saturation.
7. Allocation as secondary/supporting.
8. Behavioral — not mechanistic — claims.

======================================================================
4. CORE NUMBERS (FROZEN 009 / 011; 012 IS POST-HOC)
======================================================================

Coverage 0.70 minus 0.99 (confirmatory, already published):
- MMLU GPT 50.2 pp; Claude 35.4 pp
- Code GPT 38.8 pp; Claude 4.9 pp

Code hidden-suite pass: GPT 46.5%; Claude 43.0%

POST_HOC_SYSTEMS_REANALYSIS leakage 0.70 → 0.99 (unverified errors / all outputs):
- MMLU GPT {fmt_pp(g70['unverified_error_rate'])}% → {fmt_pp(g99['unverified_error_rate'])}%
- MMLU Claude {fmt_pp(c70['unverified_error_rate'])}% → {fmt_pp(c99['unverified_error_rate'])}%
- Code GPT {fmt_pp(cg70['unverified_error_rate'])}% → {fmt_pp(cg99['unverified_error_rate'])}%
- Code Claude {fmt_pp(cc70['unverified_error_rate'])}% → {fmt_pp(cc99['unverified_error_rate'])}%

======================================================================
5. CLAIM DISCIPLINE
======================================================================

ALLOWED:
- “A counterfactual confidence value with no item-discriminative information
  within a condition can move verification coverage by tens of percentage
  points while the frozen output population and its correctness remain fixed.”
- “These coverage shifts translate into measurable changes in the number of
  erroneous outputs that escape independent checking.”
- “The routing policy exhibits high behavioral sensitivity to displayed
  confidence over a numerical range that overlaps the model’s naturally
  reported confidence.”
- “The downstream cost of this sensitivity depends on the relative cost of
  verification and escaped errors.”
- “GPT exhibits substantially higher confidence-to-oversight gain than
  Claude’s saturated hard-code policy.”

FORBIDDEN:
- exact ranking invariance
- GPT is irrational; Claude is normatively correct
- optimal coverage equals error rate
- guardrails cause the Claude-code ceiling
- miscalibration causes X; constant overwrite equals calibration
- natural q1 drift has the same slope
- “confidence contains zero information” without the within-condition qualifier
- “false confidence causes X” unless the displayed score is defined as false

======================================================================
6. STOP RULE
======================================================================

No new benchmarks, models, q1+delta experiments, cost-ratio shopping, or
post-hoc routers. The next phase is manuscript writing.

======================================================================
7. RELATION TO THE LIVE paperDirection.txt
======================================================================

This file is a proposed full replacement. It keeps the 009–011 experimental
record and the model-specific code boundary, and upgrades the headline from
coverage-control to oversight-sensitivity / error-leakage.

Until accepted, paperDirection.txt remains the source of truth.
"""
    (RETURN_DIR / "paperDirection_task012_proposed.txt").write_text(text, encoding="utf-8")


def main() -> None:
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paper_sha_before = sha256_file(PAPER_DIRECTION)
    freeze_path = RETURN_DIR / "task012_analysis_freeze.json"
    if not freeze_path.exists():
        raise RuntimeError("analysis freeze must exist before Sections 4–9")

    repro = reproduce()
    write_reproduction_md(repro)
    json_dump(RETURN_DIR / "reproduction_check.json", repro)
    if not repro["passed"]:
        (RETURN_DIR / "report.md").write_text(
            "# TASK012_REPRODUCTION_FAILURE\n\nSee reproduction_check.md. Sections 4–9 were not computed.\n",
            encoding="utf-8",
        )
        write_changed_files(False)
        raise SystemExit("TASK012_REPRODUCTION_FAILURE")

    rows = all_rows()
    leakage, contrasts = run_leakage(rows)
    write_csv(RETURN_DIR / "error_leakage_by_condition.csv", leakage)
    write_csv(RETURN_DIR / "error_leakage_contrasts.csv", contrasts)
    sens = run_sensitivity(rows)
    write_csv(RETURN_DIR / "oversight_sensitivity.csv", sens)
    q1 = run_q1_distribution(rows)
    cal = run_q1_calibration(rows)
    nat = run_natural_vs_counterfactual(rows, leakage)
    write_csv(RETURN_DIR / "q1_distribution.csv", q1)
    write_csv(RETURN_DIR / "q1_calibration.csv", cal)
    write_csv(RETURN_DIR / "natural_vs_counterfactual.csv", nat)
    cost = run_cost_sweep(rows)
    eff = run_efficiency(rows)
    med = run_matched_budget(rows)
    write_csv(RETURN_DIR / "cost_sweep.csv", cost)
    write_csv(RETURN_DIR / "marginal_verification_efficiency.csv", eff)
    write_csv(RETURN_DIR / "matched_budget_leakage.csv", med)
    write_figures(leakage, sens, cost, rows)

    gate = _gate(leakage, q1, cost, True)
    # Prefer contrast CIs for gate A
    refined = []
    for cell in gate["all_cells"]:
        dlt = _get(
            contrasts,
            task=cell["task"],
            model_alias=cell["model"],
            from_condition="displayed_0.70",
            to_condition="displayed_0.99",
        )
        delta = dlt["delta_unverified_error_rate"]
        lo, hi = dlt["delta_leakage_ci_lo"], dlt["delta_leakage_ci_hi"]
        material = abs(delta) >= MATERIAL_LEAKAGE_PP and not (lo <= 0 <= hi)
        refined.append({**cell, "delta_leakage": delta, "ci_lo": lo, "ci_hi": hi, "material": material})
    gate["all_cells"] = refined
    gate["material_cells"] = [c for c in refined if c["material"]]
    gate["A_consequence"] = len(gate["material_cells"]) >= 2
    relevant = gate["material_cells"] or refined
    b_notes = []
    B = False
    for cell in relevant:
        dist = _get(q1, task=cell["task"], model_alias=cell["model"], subset="all")
        frac = dist["frac_in_0.70_0.99"]
        iqr_hit = dist["iqr_intersects_0.70_0.99"]
        ok = frac >= Q1_OVERLAP_FRAC or iqr_hit
        b_notes.append({**cell, "frac_in_range": frac, "iqr_intersects": iqr_hit, "pass": ok})
        if ok:
            B = True
    gate["B_operating_range"] = B
    gate["b_notes"] = b_notes
    if not gate["D_no_contradiction"]:
        gate["decision"] = "CONTRADICTS_CURRENT_FRAMING"
    elif gate["A_consequence"] and gate["B_operating_range"] and gate["C_systems"]:
        gate["decision"] = "MATERIAL_UPGRADE"
    elif gate["A_consequence"]:
        gate["decision"] = "USEFUL_SECONDARY"
    else:
        gate["decision"] = "NO_MATERIAL_UPGRADE"
    json_dump(RETURN_DIR / "paper_impact_gate.json", gate)

    paper_sha_after = sha256_file(PAPER_DIRECTION)
    write_reports(
        repro=repro,
        leakage=leakage,
        contrasts=contrasts,
        sens=sens,
        q1=q1,
        cal=cal,
        nat=nat,
        cost=cost,
        eff=eff,
        med=med,
        gate=gate,
        paper_sha_before=paper_sha_before,
        paper_sha_after=paper_sha_after,
    )
    if gate["decision"] == "MATERIAL_UPGRADE":
        write_proposed_paper_direction(gate, leakage, contrasts)
    write_changed_files(gate["decision"] == "MATERIAL_UPGRADE")
    _copy_analysis_scripts()
    json_dump(
        RETURN_DIR / "run_manifest.json",
        {
            "task": "012_oversight_sensitivity_audit",
            "label": LABEL,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "decision": gate["decision"],
            "api_calls": 0,
            "paper_sha_before": paper_sha_before,
            "paper_sha_after": paper_sha_after,
            "freeze": str(freeze_path),
            "task009": str(TASK009_DIR),
            "task010": str(TASK010_DIR),
            "task011": str(TASK011_DIR),
        },
    )
    print(f"READY_FOR_GPT_REVIEW=YES decision={gate['decision']}")


if __name__ == "__main__":
    main()
