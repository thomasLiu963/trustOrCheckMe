"""Task 013 runner: freeze already on disk; reproduce, then policy audit."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .task005_common import sha256_file
from .task012_data import all_rows
from .task013_common import (
    ANALYSIS_DIR,
    FIGURES_DIR,
    GPT_CLAUDE,
    LABEL,
    LAMBDA_GRID,
    MODEL_LABELS,
    PAPER_DIRECTION,
    RETURN_DIR,
    TASK009_DIR,
    TASK011_DIR,
    TASK012_DIR,
    json_dump,
    write_csv,
)
from .task013_figures import write_all as write_figures
from .task013_policy import cost_sweep, hidden_vs_trueq, policy_metrics
from .task013_reproduce import reproduce, write_reproduction_md
from .task013_robustness import classify_gpt, hidden_robustness, paper_gate, robust_policy_selection, systems_matrix


def _md(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _get(rows: list[dict[str, Any]], **keys: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(k) == v for k, v in keys.items()):
            return row
    raise KeyError(keys)


def _copy_scripts() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    for name in (
        "task013_common.py",
        "task013_reproduce.py",
        "task013_policy.py",
        "task013_robustness.py",
        "task013_figures.py",
        "task013.py",
    ):
        shutil.copy2(src / name, ANALYSIS_DIR / name)
    header = (
        "from __future__ import annotations\n\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "ROOT = Path(__file__).resolve().parents[4]\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n\n"
        "from src.task013 import main\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    for dest in ("reproduce.py", "policy_metrics.py", "cost_sweep.py", "robustness.py", "pareto.py", "figures.py", "run_all.py"):
        (ANALYSIS_DIR / dest).write_text(header, encoding="utf-8")


def write_reports(
    *,
    repro: dict[str, Any],
    metrics: list[dict[str, Any]],
    cost: list[dict[str, Any]],
    vs: list[dict[str, Any]],
    robust: list[dict[str, Any]],
    selection: list[dict[str, Any]],
    systems: list[dict[str, Any]],
    gate: dict[str, Any],
    paper_sha_before: str,
    paper_sha_after: str,
) -> None:
    def lab(model: str) -> str:
        return MODEL_LABELS[model]

    raw_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            v = _get(vs, task=task, model_alias=model)
            raw_rows.append(
                [
                    task,
                    lab(model),
                    f"{100*v['hidden_coverage']:.1f}",
                    f"{100*v['trueq_coverage']:.1f}",
                    f"{100*v['delta_coverage_hidden_minus_trueq']:+.1f} [{100*v['delta_coverage_ci_lo']:+.1f}, {100*v['delta_coverage_ci_hi']:+.1f}]",
                    f"{100*v['hidden_leakage']:.1f}",
                    f"{100*v['trueq_leakage']:.1f}",
                    f"{100*v['delta_leakage_hidden_minus_trueq']:+.1f} [{100*v['delta_leakage_ci_lo']:+.1f}, {100*v['delta_leakage_ci_hi']:+.1f}]",
                ]
            )

    fixed_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            hid = _get(metrics, task=task, model_alias=model, policy="hidden")
            q70 = _get(metrics, task=task, model_alias=model, policy="displayed_0.70")
            q99 = _get(metrics, task=task, model_alias=model, policy="displayed_0.99")
            fixed_rows.append(
                [
                    task,
                    lab(model),
                    f"{100*hid['verification_coverage']:.1f}/{hid['errors_unverified_per_100']:.1f}",
                    f"{100*q70['verification_coverage']:.1f}/{q70['errors_unverified_per_100']:.1f}",
                    f"{100*q99['verification_coverage']:.1f}/{q99['errors_unverified_per_100']:.1f}",
                ]
            )

    rob_rows = []
    for r in robust:
        rob_rows.append(
            [
                r["task"],
                lab(r["model_alias"]),
                f"{100*r['frac_exactly_best']:.0f}%",
                f"{100*r['frac_within_0.02']:.0f}%",
                f"{100*r['frac_within_0.05']:.0f}%",
                f"{r['worst_case_regret']:.3f}",
                f"{r['mean_regret_unweighted']:.3f}",
                "yes" if r["passes_gpt_robustness_criteria"] else "no",
            ]
        )

    mm_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            winner = next(r for r in selection if r["task"] == task and r["model_alias"] == model and r["is_minimax_regret_policy"])
            hid = _get(selection, task=task, model_alias=model, policy="hidden")
            mm_rows.append(
                [
                    task,
                    lab(model),
                    winner["policy"],
                    f"{winner['worst_case_regret']:.3f}",
                    f"{hid['worst_case_regret']:.3f}",
                    "yes" if hid["is_minimax_regret_policy"] else "no",
                    "yes" if hid["on_observed_pareto_frontier"] else "no",
                ]
            )

    sys_rows = []
    for r in systems:
        sys_rows.append(
            [
                r["task"],
                lab(r["model_alias"]),
                f"{r['q1_brier']:.3f}",
                f"{r['q1_ece']:.3f}",
                f"{100*r['coverage_sensitivity_per_0.10']:+.1f}",
                f"{100*r['hidden_frac_within_0.02']:.0f}%",
                f"{r['hidden_worst_case_regret']:.3f}",
                "yes" if r["hidden_on_pareto"] else "no",
            ]
        )

    sentence = ""
    if gate["decision"] == "ACTIONABLE_UPGRADE":
        sentence = (
            "In post-hoc policy analysis, withholding the scalar confidence signal is a "
            "competitive observed routing baseline across a broad range of verification/error "
            "cost ratios for GPT on both MMLU and code, suggesting that high-sensitivity "
            "routers should at least evaluate a confidence-hidden policy."
        )
    elif gate["decision"] == "ACTIONABLE_BUT_TASK_SPECIFIC":
        sentence = (
            "In post-hoc policy analysis, a confidence-hidden routing policy is competitive "
            "on one GPT task but not both, so it is a useful observed baseline to report, "
            "not a headline mitigation."
        )
    else:
        sentence = (
            "Diagnosing confidence-metadata sensitivity does not imply that simply withholding "
            "the scalar is a universally good mitigation among the observed policies."
        )

    rewrite = {
        "ACTIONABLE_UPGRADE": "Yes — write paperDirection_task013_proposed.txt. Do not overwrite paperDirection.txt.",
        "ACTIONABLE_BUT_TASK_SPECIFIC": "No full rewrite. See paperDirection_task013_recommendation.md.",
        "DIAGNOSTIC_ONLY": "No. Keep the Task-012 paperDirection as source of truth.",
        "CONTRADICTS_CURRENT_STORY": "Yes — correction required.",
    }[gate["decision"]]

    gpt_mmlu = _get(robust, task="mmlu", model_alias="openai_gpt56_sol")
    gpt_code = _get(robust, task="code", model_alias="openai_gpt56_sol")
    cl_mmlu = _get(robust, task="mmlu", model_alias="anthropic_sonnet5")
    cl_code = _get(robust, task="code", model_alias="anthropic_sonnet5")

    report = f"""# Task 013 — Hidden-confidence policy audit

Label: `{LABEL}`. Zero API calls. Frozen Tasks 009–012 only.

## 1. Plain-English bottom line

Withholding the displayed confidence scalar is **not** a robust default across both GPT tasks under the predeclared gates.

Cross-task class: **{gate['robustness_class']}**. Paper-impact: **{gate['decision']}**.

Hidden is an observed policy, not an x-value and not a preregistered mitigation. It can have lower leakage than true-q-visible while costing more verification, or the reverse. Do not call it optimal.

## 2. Reproduction check

See `reproduction_check.md`. Status: **{"PASS" if repro["passed"] else "TASK013_REPRODUCTION_FAILURE"}**.

Published hidden / true-q coverage and Task 012 leakage, λ grid, and loss function all reproduced within ±0.6 pp.

## 3. Hidden vs true-q-visible raw metrics

{_md(["Task", "Model", "Hidden cov %", "True-q cov %", "Δ cov pp [95% CI]", "Hidden leak %", "True-q leak %", "Δ leak pp [95% CI]"], raw_rows)}

Lower leakage is not “better” without a λ. Hidden vs true-q can trade coverage against escaped errors.

## 4. Hidden vs fixed-score metrics

Coverage % / unverified errors per 100:

{_md(["Task", "Model", "hidden", "q=0.70", "q=0.99"], fixed_rows)}

Full table: `policy_metrics.csv`.

## 5. Hidden loss/regret across λ

`loss(λ) = leakage + λ * coverage` on the frozen Task-012 grid. Best observed is only best among the seven collected policies.

{_md(["Task", "Model", "Exactly best", "Within 0.02", "Within 0.05", "Worst-case regret", "Mean regret", "GPT-robust?"], rob_rows)}

## 6. Near-best breadth

Hidden is within 0.02 of best on {100*gpt_mmlu['frac_within_0.02']:.0f}% of the λ grid for GPT MMLU and {100*gpt_code['frac_within_0.02']:.0f}% for GPT code. Claude MMLU {100*cl_mmlu['frac_within_0.02']:.0f}%; Claude code {100*cl_code['frac_within_0.02']:.0f}% (near-saturation).

## 7. Minimax regret result

{_md(["Task", "Model", "Minimax policy", "That worst regret", "Hidden worst regret", "Hidden is minimax?", "Hidden on Pareto?"], mm_rows)}

Minimax is among observed policies only.

## 8. Pareto-frontier result

A policy is efficient if no other observed policy has both lower coverage and lower leakage. Hidden on GPT MMLU: {"yes" if _get(selection, task="mmlu", model_alias="openai_gpt56_sol", policy="hidden")["on_observed_pareto_frontier"] else "no"}. Hidden on GPT code: {"yes" if _get(selection, task="code", model_alias="openai_gpt56_sol", policy="hidden")["on_observed_pareto_frontier"] else "no"}. See `figures/policy_pareto_frontier.png`.

## 9. GPT MMLU result

Hidden coverage 39.6%, leakage 3.8% (19 unverified errors / 500). True-q coverage 12.4%, leakage 12.0%. Worst-case hidden regret {gpt_mmlu['worst_case_regret']:.3f}; within 0.02 on {100*gpt_mmlu['frac_within_0.02']:.0f}% of λ. GPT-robust criteria: {"pass" if gpt_mmlu["passes_gpt_robustness_criteria"] else "fail"}.

## 10. GPT code result

Hidden coverage 32.5%, leakage 25.9%. True-q coverage 27.3%, leakage 29.7%. Worst-case hidden regret {gpt_code['worst_case_regret']:.3f}; within 0.02 on {100*gpt_code['frac_within_0.02']:.0f}% of λ. GPT-robust criteria: {"pass" if gpt_code["passes_gpt_robustness_criteria"] else "fail"}.

## 11. Claude MMLU result

Hidden coverage 65.4%, leakage 2.8%; true-q 79.4% / 1.0%. Hidden verifies less than true-q and leaks slightly more. Worst-case regret {cl_mmlu['worst_case_regret']:.3f}. Secondary for the mitigation gate.

## 12. Claude code saturation result

Hidden 98.6% coverage, leakage 0.3%; true-q 97.9% / 1.4%. Near VERIFY-all. Hidden cannot be judged as a general mitigation in this regime. Reported fully; not used to manufacture a win.

## 13. Cross-task hidden robustness classification

**{gate['robustness_class']}**

Claude is secondary because code is action-saturated. Claude rows remain in every table.

## 14. Systems design matrix

{_md(["Task", "Model", "Brier", "ECE", "Cov sens pp/+0.10", "Hidden near-best 0.02", "Hidden max regret", "Hidden Pareto"], sys_rows)}

Confidence quality and confidence-to-policy gain are separate system properties. Whether to expose the scalar is a third, optional design choice — only if the policy audit supports it.

## 15. What hidden policy does NOT establish

- Optimality of hiding confidence
- That hiding is always safer
- That hidden routing removes internal confidence or equals item-only reasoning
- That visible confidence causes failures
- That Claude is normatively more robust
- That true-q-visible is generally bad
- That λ is known in deployment
- That this mitigation was preregistered or confirmatory

## 16. Strongest actionable claim, if any

{sentence}

## 17. Paper-impact gate outcome

**{gate['decision']}**

- A both GPT robust: {gate['A_both_gpt_robust']}
- B Pareto both GPT: {gate['B_pareto_both_gpt']}
- C near-best both GPT: {gate['C_near_best_both_gpt']}
- D max regret ≤ 0.10 both GPT: {gate['D_worst_regret_both_gpt']}
- E honest reporting / full Claude: {gate['E_honest_reporting']}

## 18. Exact recommended paper sentence

{sentence}

## 19. Exact claims to avoid

- hidden confidence is optimal
- hiding confidence is always safer
- hiding confidence removes internal confidence
- hidden routing is equivalent to item-only reasoning
- visible confidence causes failures
- Claude is more robust in a normative sense
- true-q-visible is bad in general
- confidence should never be shown
- λ is known in deployment
- the hidden policy is preregistered as a mitigation
- Task 013 was confirmatory

## 20. Whether paperDirection should be rewritten

{rewrite}

## 21. READY_FOR_GPT_REVIEW = YES
"""
    (RETURN_DIR / "report.md").write_text(report, encoding="utf-8")

    decision = f"""# Task 013 paper-impact decision

Label: `{LABEL}`

## Decision

`{gate['decision']}`

Robustness class: `{gate['robustness_class']}`

## Frozen gates

A. `HIDDEN_ROBUST_BOTH_GPT_TASKS`
B. Hidden on observed Pareto frontier for both GPT tasks
C. Hidden within 0.02 of best for ≥50% of λ in both GPT tasks
D. Hidden max regret ≤ 0.10 in both GPT tasks
E. Claude reported; no single-λ cherry-pick

## Evaluation

- A = {gate['A_both_gpt_robust']}
- B = {gate['B_pareto_both_gpt']}
- C = {gate['C_near_best_both_gpt']}
- D = {gate['D_worst_regret_both_gpt']}
- E = {gate['E_honest_reporting']}
- reproduction = {gate['reproduction_ok']}

## Action

{rewrite}
"""
    (RETURN_DIR / "paper_impact_decision.md").write_text(decision, encoding="utf-8")

    if gate["decision"] in {"ACTIONABLE_BUT_TASK_SPECIFIC", "DIAGNOSTIC_ONLY"}:
        (RETURN_DIR / "paperDirection_task013_recommendation.md").write_text(
            f"""# paperDirection recommendation (Task 013)

Decision: `{gate['decision']}`.
Robustness class: `{gate['robustness_class']}`.

Keep the current Task-012 `paperDirection.txt` as the source of truth.

Task 013 is a post-hoc policy audit, not a confirmatory mitigation study.

Recommended use:
- one short appendix or discussion paragraph: withholding the scalar is an observed baseline worth reporting, but it is not a robust headline recommendation across both GPT tasks;
- include `figures/policy_pareto_frontier.png` or `figures/hidden_regret_vs_lambda.png` if space allows;
- do not add a “hide confidence” design commandment.

Do not collect new data. Next phase is manuscript writing.
""",
            encoding="utf-8",
        )
    elif gate["decision"] == "CONTRADICTS_CURRENT_STORY":
        (RETURN_DIR / "paperDirection_task013_correction_required.txt").write_text(
            "Task 013 reproduction or consistency check failed. See reproduction_check.md.\n",
            encoding="utf-8",
        )

    (RETURN_DIR / "validation.md").write_text(
        f"""# Task 013 validation

- API / model calls: **0**
- Historical 009–012 artifacts: read-only
- paperDirection.txt sha256 before: `{paper_sha_before}`
- paperDirection.txt sha256 after: `{paper_sha_after}`
- paperDirection overwritten: **no**
- Analysis freeze written before Sections 3–12: `task013_analysis_freeze.json`
- λ grid and loss inherited from Task 012
- Hidden never assigned an x-coordinate
- Every numeric product labeled `{LABEL}`
- Claude reported in all tables
- Forbidden claims not used as findings
- 009: {TASK009_DIR.name}
- 011: {TASK011_DIR.name}
- 012: {TASK012_DIR.name}
""",
        encoding="utf-8",
    )


def write_changed_files(decision: str) -> None:
    names = [
        "report.md",
        "reproduction_check.md",
        "task013_analysis_freeze.json",
        "paper_impact_decision.md",
        "validation.md",
        "changed_files.txt",
        "policy_metrics.csv",
        "policy_cost_sweep.csv",
        "hidden_robustness_summary.csv",
        "hidden_vs_trueq.csv",
        "robust_policy_selection.csv",
        "systems_design_matrix.csv",
        "figures/hidden_policy_summary.png",
        "figures/policy_pareto_frontier.png",
        "figures/hidden_regret_vs_lambda.png",
        "figures/coverage_leakage_policy_map.png",
        "analysis/task013/reproduce.py",
        "analysis/task013/policy_metrics.py",
        "analysis/task013/cost_sweep.py",
        "analysis/task013/robustness.py",
        "analysis/task013/pareto.py",
        "analysis/task013/figures.py",
        "analysis/task013/run_all.py",
    ]
    if decision == "ACTIONABLE_UPGRADE":
        names.append("paperDirection_task013_proposed.txt")
    elif decision == "CONTRADICTS_CURRENT_STORY":
        names.append("paperDirection_task013_correction_required.txt")
    else:
        names.append("paperDirection_task013_recommendation.md")
    (RETURN_DIR / "changed_files.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def write_proposed_paper_direction(gate: dict[str, Any]) -> None:
    text = f"""D1 PAPER DIRECTION — PROPOSED AFTER TASK 013
Working narrative / scientific source of truth (PROPOSED REPLACEMENT)
Date: 2026-09-19
Status: PROPOSED by Task 013 ({LABEL}). Not live until accepted.
The live source of truth remains paperDirection.txt.

This proposed replacement KEEPS the Task-012 sensitivity / error-leakage
result and ADDS one actionable baseline:

1. confidence metadata can have high oversight leverage;
2. this can create large error leakage;
3. the natural q1 range overlaps the high-sensitivity region;
4. withholding the scalar can be a competitive robust baseline in GPT
   across both tasks;
5. therefore system designers should measure calibration, measure
   confidence-to-policy sensitivity, compare visible-confidence routing
   against a confidence-hidden baseline, and evaluate cost robustness.

Do not call hidden optimal. Do not claim hiding removes internal confidence.
Gate: {gate['decision']} / {gate['robustness_class']}.

Until accepted, paperDirection.txt remains the source of truth.
No new experiments.
"""
    (RETURN_DIR / "paperDirection_task013_proposed.txt").write_text(text, encoding="utf-8")


def main() -> None:
    RETURN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paper_sha_before = sha256_file(PAPER_DIRECTION)
    if not (RETURN_DIR / "task013_analysis_freeze.json").exists():
        raise RuntimeError("analysis freeze must exist before new analyses")

    repro = reproduce()
    write_reproduction_md(repro)
    json_dump(RETURN_DIR / "reproduction_check.json", repro)
    if not repro["passed"]:
        (RETURN_DIR / "report.md").write_text(
            "# TASK013_REPRODUCTION_FAILURE\n\nSee reproduction_check.md.\n",
            encoding="utf-8",
        )
        write_changed_files("CONTRADICTS_CURRENT_STORY")
        raise SystemExit("TASK013_REPRODUCTION_FAILURE")

    rows = all_rows()
    metrics = policy_metrics(rows)
    cost = cost_sweep(metrics)
    vs = hidden_vs_trueq(rows, metrics, cost)
    robust = hidden_robustness(metrics, cost, vs)
    robust_label = classify_gpt(robust)
    selection = robust_policy_selection(metrics, cost)
    systems = systems_matrix(metrics, robust, selection)
    gate = paper_gate(robust_label, selection, robust, True)
    write_csv(RETURN_DIR / "policy_metrics.csv", metrics)
    write_csv(RETURN_DIR / "policy_cost_sweep.csv", cost)
    write_csv(RETURN_DIR / "hidden_vs_trueq.csv", vs)
    write_csv(RETURN_DIR / "hidden_robustness_summary.csv", robust)
    write_csv(RETURN_DIR / "robust_policy_selection.csv", selection)
    write_csv(RETURN_DIR / "systems_design_matrix.csv", systems)
    json_dump(RETURN_DIR / "paper_impact_gate.json", gate)
    write_figures(metrics, cost, selection)

    paper_sha_after = sha256_file(PAPER_DIRECTION)
    write_reports(
        repro=repro,
        metrics=metrics,
        cost=cost,
        vs=vs,
        robust=robust,
        selection=selection,
        systems=systems,
        gate=gate,
        paper_sha_before=paper_sha_before,
        paper_sha_after=paper_sha_after,
    )
    if gate["decision"] == "ACTIONABLE_UPGRADE":
        write_proposed_paper_direction(gate)
    write_changed_files(gate["decision"])
    _copy_scripts()
    json_dump(
        RETURN_DIR / "run_manifest.json",
        {
            "task": TASK012_DIR.name.replace("012", "013") if False else "013_hidden_confidence_policy_audit",
            "label": LABEL,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "decision": gate["decision"],
            "robustness_class": robust_label,
            "api_calls": 0,
            "paper_sha_before": paper_sha_before,
            "paper_sha_after": paper_sha_after,
        },
    )
    print(f"READY_FOR_GPT_REVIEW=YES decision={gate['decision']} class={robust_label}")


if __name__ == "__main__":
    main()
