"""Post-call Task 014 analysis. Do not run until Stage-3 completes."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .plotting import _pyplot, _save
from .task005_common import sha256_file
from .task012_common import LAMBDA_GRID
from .task012_stats import bootstrap_pair, coverage, ece, leakage
from .task014_common import (
    ANALYSIS_DIR,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DELTAS,
    FIGURES_DIR,
    GPT_CLAUDE,
    LABEL,
    MODEL_LABELS,
    PAPER_DIRECTION,
    RETURN_DIR,
    TASK012_DIR,
    WORLD_A,
    WORLD_B,
    WORLD_C,
    WORLD_D,
    WORLD_E,
    json_dump,
    load_csv,
    write_csv,
)
from .task014_data import FrozenItem, load_frozen_items
from .task014_transform import transform_q


class _Row:
    def __init__(self, question_id: str, verify: bool, incorrect: bool):
        self.question_id = question_id
        self.verify = verify
        self.incorrect = incorrect


def _cell(rows: list[dict[str, Any]], task: str, model: str, delta: float) -> list[_Row]:
    out = []
    for r in rows:
        if r["task"] == task and r["model_alias"] == model and abs(float(r["delta"]) - delta) < 1e-12:
            if str(r.get("parsed_action") or "") not in {"VERIFY_FIRST", "USE_UNVERIFIED"}:
                continue
            out.append(
                _Row(
                    str(r["question_id"]),
                    str(r.get("verify")).lower() in {"1", "true", "yes"} or r.get("parsed_action") == "VERIFY_FIRST",
                    str(r.get("incorrect")).lower() in {"1", "true", "yes"},
                )
            )
    return out


def _constant_effect() -> dict[tuple[str, str], dict[str, float]]:
    leak = load_csv(TASK012_DIR / "error_leakage_by_condition.csv")
    out = {}
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            a = next(r for r in leak if r["task"] == task and r["model_alias"] == model and r["score_condition"] == "displayed_0.70")
            b = next(r for r in leak if r["task"] == task and r["model_alias"] == model and r["score_condition"] == "displayed_0.99")
            out[(task, model)] = {
                "d_cov": float(b["verification_coverage"]) - float(a["verification_coverage"]),
                "d_leak": float(b["unverified_error_rate"]) - float(a["unverified_error_rate"]),
            }
    return out


def classify_world(effects: list[dict[str, Any]]) -> str:
    gpt = {r["task"]: r for r in effects if r["model_alias"] == "openai_gpt56_sol"}
    m, c = gpt["mmlu"], gpt["code"]

    def gpt_a(row: dict[str, Any], leak_floor: float) -> bool:
        return (
            row["delta_leakage"] >= leak_floor
            and row["delta_leakage_ci_lo"] > 0
            and abs(row["delta_coverage"]) >= 0.15
        )

    a_ok = gpt_a(m, 0.05) and gpt_a(c, 0.10)
    ret_ok = (m["retention_leakage"] >= 0.50 and c["retention_leakage"] >= 0.50) or (
        m["retention_coverage"] >= 0.50 and c["retention_coverage"] >= 0.50
    )
    if a_ok and ret_ok:
        return WORLD_A
    both_pos = m["delta_leakage_ci_lo"] > 0 and c["delta_leakage_ci_lo"] > 0
    if a_ok and not ret_ok:
        return WORLD_B
    if both_pos and not a_ok:
        return WORLD_B
    c_m = m["retention_leakage"] < 0.25 or m["delta_leakage"] < 0.05
    c_c = c["retention_leakage"] < 0.25 or c["delta_leakage"] < 0.10
    if c_m and c_c:
        return WORLD_C
    if (gpt_a(m, 0.05) or (m["delta_leakage_ci_lo"] > 0)) != (gpt_a(c, 0.10) or (c["delta_leakage_ci_lo"] > 0)):
        return WORLD_D
    return WORLD_D


def analyze_all() -> None:
    from .checkpointing import CheckpointStore
    from .task014_common import NEW_DELTAS, SQLITE_PATH
    from .task014_run import request_key, reused_delta0_record

    items = load_frozen_items()
    store = CheckpointStore(SQLITE_PATH)
    rows: list[dict[str, Any]] = []
    missing = 0
    cost = 0.0
    for item in items:
        rows.append(reused_delta0_record(item))
        for delta in NEW_DELTAS:
            rec = store.get_record(request_key(item, delta)) or {}
            if not rec:
                missing += 1
                continue
            cost += float(rec.get("estimated_cost_usd") or 0)
            rows.append(
                {
                    "task": item.task,
                    "question_id": item.question_id,
                    "model_alias": item.model_alias,
                    "delta": delta,
                    "q1": item.q1,
                    "q_delta": rec.get("q_delta"),
                    "parsed_action": rec.get("parsed_action"),
                    "verify": rec.get("verify"),
                    "incorrect": item.incorrect,
                    "reused_delta0": False,
                    "estimated_cost_usd": rec.get("estimated_cost_usd"),
                    "request_key": rec.get("request_key"),
                    "input_tokens": rec.get("input_tokens"),
                    "output_tokens": rec.get("output_tokens"),
                }
            )
    write_csv(RETURN_DIR / "stage3_offset_results.csv", rows)
    if missing:
        (RETURN_DIR / "deviations.md").write_text(
            f"# Task 014 deviations\n\nMissing nonzero-delta records: {missing}. WORLD E if material.\n",
            encoding="utf-8",
        )
        if missing > 50:
            _write_world_e(missing, cost)
            return

    const = _constant_effect()
    cov_rows = []
    leak_rows = []
    effects = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            by_d = {d: _cell(rows, task, model, d) for d in DELTAS}
            for d, cell in by_d.items():
                cov_rows.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "delta": d,
                        "n": len(cell),
                        "verification_coverage": coverage(cell),
                        "leakage_rate": leakage(cell),
                    }
                )
                leak_rows.append(cov_rows[-1])
            lo, hi = by_d[-1.5], by_d[1.5]
            d_cov = bootstrap_pair(lo, hi, coverage, seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_RESAMPLES)
            d_leak = bootstrap_pair(lo, hi, leakage, seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_RESAMPLES)
            c = const[(task, model)]
            ret_l = abs(d_leak.point) / abs(c["d_leak"]) if c["d_leak"] else float("nan")
            ret_c = abs(d_cov.point) / abs(c["d_cov"]) if c["d_cov"] else float("nan")
            effects.append(
                {
                    "label": LABEL,
                    "task": task,
                    "model_alias": model,
                    "delta_coverage": d_cov.point,
                    "delta_coverage_ci_lo": d_cov.lo,
                    "delta_coverage_ci_hi": d_cov.hi,
                    "delta_leakage": d_leak.point,
                    "delta_leakage_ci_lo": d_leak.lo,
                    "delta_leakage_ci_hi": d_leak.hi,
                    "constant_delta_coverage": c["d_cov"],
                    "constant_delta_leakage": c["d_leak"],
                    "retention_coverage": ret_c,
                    "retention_leakage": ret_l,
                    "coverage_minus_1.5": coverage(lo),
                    "coverage_plus_1.5": coverage(hi),
                    "leakage_minus_1.5": leakage(lo),
                    "leakage_plus_1.5": leakage(hi),
                }
            )
    write_csv(RETURN_DIR / "coverage_by_delta.csv", cov_rows)
    write_csv(RETURN_DIR / "leakage_by_delta.csv", leak_rows)
    write_csv(RETURN_DIR / "effect_retention.csv", effects)
    write_csv(RETURN_DIR / "cross_task_summary.csv", effects)

    cal = []
    item_q = {(it.task, it.question_id, it.model_alias): it for it in items}
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            subset = [it for it in items if it.task == task and it.model_alias == model]
            y = [not it.incorrect for it in subset]
            for d in DELTAS:
                qd = [transform_q(it.q1, d) for it in subset]
                ece_v, _ = ece(qd, y)
                cal.append(
                    {
                        "task": task,
                        "model_alias": model,
                        "delta": d,
                        "mean_q_correct": sum(q for q, ok in zip(qd, y) if ok) / max(1, sum(y)),
                        "mean_q_incorrect": sum(q for q, ok in zip(qd, y) if not ok) / max(1, len(y) - sum(y)),
                        "ece": ece_v,
                    }
                )

    cost_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            for d in DELTAS:
                cell = _cell(rows, task, model, d)
                leak = leakage(cell)
                cov = coverage(cell)
                for lam in LAMBDA_GRID:
                    cost_rows.append(
                        {
                            "task": task,
                            "model_alias": model,
                            "delta": d,
                            "lambda": lam,
                            "normalized_loss": leak + lam * cov,
                            "leakage_rate": leak,
                            "verification_coverage": cov,
                        }
                    )
    write_csv(RETURN_DIR / "cost_sweep.csv", cost_rows)

    eff = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            a = {r.question_id: r for r in _cell(rows, task, model, 1.5)}
            b = {r.question_id: r for r in _cell(rows, task, model, -1.5)}
            extra_v = extra_c = 0
            for qid, hi in b.items():
                lo = a.get(qid)
                if lo is None:
                    continue
                extra_v += int(hi.verify) - int(lo.verify)
                extra_c += int(hi.incorrect and hi.verify) - int(lo.incorrect and lo.verify)
            eff.append(
                {
                    "task": task,
                    "model_alias": model,
                    "from_delta": 1.5,
                    "to_delta": -1.5,
                    "additional_verify_calls": extra_v,
                    "additional_errors_caught": extra_c,
                    "errors_caught_per_added_verify": extra_c / extra_v if extra_v else float("nan"),
                }
            )
    write_csv(RETURN_DIR / "marginal_efficiency.csv", eff)

    world = WORLD_E if missing > 50 else classify_world(effects)
    write_figures(cov_rows, effects)
    write_packet(effects, cov_rows, world, cost, missing, items)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    for name in (
        "task014_common.py",
        "task014_transform.py",
        "task014_data.py",
        "task014_audit.py",
        "task014_preregister.py",
        "task014_run.py",
        "task014_analyze.py",
        "task014.py",
    ):
        shutil.copy2(src / name, ANALYSIS_DIR / name)
    print(f"READY_FOR_GPT_REVIEW=YES world={world} missing={missing} cost={cost:.2f}")


def write_figures(cov_rows: list[dict[str, Any]], effects: list[dict[str, Any]]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    xs = [0.2, 0.4, 0.6, 0.8]
    q1 = [0.35, 0.55, 0.7, 0.88]
    for i, d in enumerate((-1.5, 0.0, 1.5)):
        from .task014_transform import transform_q as tq
        ys = [tq(q, d) for q in q1]
        ax.plot(xs, ys, "-o", label=f"δ={d}")
    ax.set_xticks(xs, ["item A", "item B", "item C", "item D"])
    ax.set_ylabel("Displayed confidence")
    ax.set_title("Rank-preserving logit shift (ordering A<B<C<D kept)")
    ax.legend(frameon=False)
    _save(fig, FIGURES_DIR / "rank_preserving_intervention")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), constrained_layout=True)
    colors = {"openai_gpt56_sol": "#1f4e79", "anthropic_sonnet5": "#b85c38"}
    for col, task in enumerate(("mmlu", "code")):
        for row_i, metric in enumerate(("verification_coverage", "leakage_rate")):
            ax = axes[row_i, col]
            for model, color in colors.items():
                xs, ys = [], []
                for d in DELTAS:
                    rec = next(r for r in cov_rows if r["task"] == task and r["model_alias"] == model and r["delta"] == d)
                    xs.append(d)
                    ys.append(100 * rec[metric])
                ax.plot(xs, ys, "-o", color=color, label=MODEL_LABELS[model])
            ax.set_xlabel("δ")
            ax.set_ylabel("Coverage %" if metric == "verification_coverage" else "Leakage %")
            ax.set_title(("MMLU-Pro" if task == "mmlu" else "LiveCodeBench") + (" coverage" if row_i == 0 else " leakage"))
            if row_i == 0:
                ax.legend(frameon=False)
    fig.suptitle("Rank-preserving offset curves")
    _save(fig, FIGURES_DIR / "coverage_leakage_by_delta")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    labels = ["GPT MMLU", "Claude MMLU", "GPT code", "Claude code"]
    cov_off = [100 * r["delta_coverage"] for r in effects]
    cov_con = [100 * r["constant_delta_coverage"] for r in effects]
    leak_off = [100 * r["delta_leakage"] for r in effects]
    leak_con = [100 * r["constant_delta_leakage"] for r in effects]
    import numpy as np
    x = np.arange(4)
    axes[0].bar(x - 0.18, cov_con, 0.35, label="constant 0.70→0.99")
    axes[0].bar(x + 0.18, cov_off, 0.35, label="offset +1.5−(−1.5)")
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_ylabel("Δ coverage pp")
    axes[0].legend(frameon=False)
    axes[1].bar(x - 0.18, leak_con, 0.35, label="constant")
    axes[1].bar(x + 0.18, leak_off, 0.35, label="offset")
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("Δ leakage pp")
    fig.suptitle("Constant-score vs rank-preserving endpoint effect")
    _save(fig, FIGURES_DIR / "constant_vs_offset_effect")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), constrained_layout=True)
    items = load_frozen_items()
    for ax, task, model, title in (
        (axes[0, 0], "mmlu", "openai_gpt56_sol", "MMLU GPT"),
        (axes[0, 1], "mmlu", "anthropic_sonnet5", "MMLU Claude"),
        (axes[1, 0], "code", "openai_gpt56_sol", "Code GPT"),
        (axes[1, 1], "code", "anthropic_sonnet5", "Code Claude"),
    ):
        q1 = [it.q1 for it in items if it.task == task and it.model_alias == model]
        ax.hist(q1, bins=20, alpha=0.45, label="q1")
        from .task014_transform import transform_q as tq
        ax.hist([tq(q, 1.5) for q in q1], bins=20, alpha=0.45, label="δ=+1.5")
        ax.hist([tq(q, -1.5) for q in q1], bins=20, alpha=0.45, label="δ=-1.5")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("q1 vs transformed distributions")
    _save(fig, FIGURES_DIR / "transformed_q_distributions")
    plt.close(fig)
    shutil.copy2(FIGURES_DIR / "coverage_leakage_by_delta.png", FIGURES_DIR / "task014_main_candidate.png")


def _write_world_e(missing: int, cost: float) -> None:
    (RETURN_DIR / "report.md").write_text(
        f"# TASK014 TECHNICAL_OR_VERSION_FAILURE\n\nMissing records: {missing}. Cost so far ${cost:.2f}.\n\nREADY_FOR_GPT_REVIEW = YES\n",
        encoding="utf-8",
    )


def write_packet(effects, cov_rows, world, cost, missing, items) -> None:
    def e(task, model):
        return next(r for r in effects if r["task"] == task and r["model_alias"] == model)

    sha = sha256_file(PAPER_DIRECTION)
    gpt_m, gpt_c = e("mmlu", "openai_gpt56_sol"), e("code", "openai_gpt56_sol")
    cl_m, cl_c = e("mmlu", "anthropic_sonnet5"), e("code", "anthropic_sonnet5")

    def line(r):
        return (
            f"coverage {100*r['coverage_minus_1.5']:.1f}% → {100*r['coverage_plus_1.5']:.1f}% "
            f"(Δ {100*r['delta_coverage']:+.1f} [{100*r['delta_coverage_ci_lo']:+.1f}, {100*r['delta_coverage_ci_hi']:+.1f}]); "
            f"leakage {100*r['leakage_minus_1.5']:.1f}% → {100*r['leakage_plus_1.5']:.1f}% "
            f"(Δ {100*r['delta_leakage']:+.1f} [{100*r['delta_leakage_ci_lo']:+.1f}, {100*r['delta_leakage_ci_hi']:+.1f}]); "
            f"retention leak {r['retention_leakage']:.2f}, cov {r['retention_coverage']:.2f}"
        )

    if world == WORLD_A:
        claim = (
            "Systematically shifting the level of an item-informative confidence signal can substantially "
            "change verification coverage and error leakage even when the signal's item ordering is preserved."
        )
        integration = "MAJOR_UPGRADE"
        rewrite = "Yes — write paperDirection_task014_proposed.txt. Do not overwrite paperDirection.txt."
    elif world == WORLD_B:
        claim = "Item-level q1 information attenuates but does not remove confidence-level sensitivity."
        integration = "MODERATE_UPGRADE"
        rewrite = "Yes — write paperDirection_task014_proposed.txt as an update, not a replacement of the 012 leakage story."
    elif world == WORLD_C:
        claim = "Preserving q1's item-relative structure substantially attenuates the constant-score failure mode."
        integration = "SCOPE_CORRECTION"
        rewrite = "Yes — write paperDirection_task014_correction.txt."
    elif world == WORLD_E:
        claim = "No scientific interpretation."
        integration = "NO_INTERPRETATION"
        rewrite = "No scientific rewrite until the technical issue is resolved."
    else:
        claim = "GPT MMLU and GPT code fall into different buckets; do not force a universal claim."
        integration = "TASK_DEPENDENT"
        rewrite = "Yes — write paperDirection_task014_proposed.txt as a heterogeneity note."

    report = f"""# Task 014 — Rank-preserving confidence-level shift

## 1. Plain-English bottom line

World: **{world}**. {claim}

## 2. Exact model endpoints

- GPT: `gpt-5.6-sol`
- Claude: `claude-sonnet-5`
Same requested endpoints as Tasks 009/011.

## 3. Preregistration/freeze integrity

See `preregistration.md` and `analysis_freeze.json`. Freeze timestamp precedes new routing-outcome inspection.

## 4. δ=0 reuse decision

**REUSE_TRUE_Q_VISIBLE**. All 1572 δ=0 prompts were byte-identical to existing `true_q_visible` prompts.

## 5. Transformation audit

See `transformation_audit.md`. Spearman(q1, q_delta) = 1.0. Zero order reversals after `.12g` rendering. Six exact-zero q1 values (code) stay at 0 for every delta.

## 6. Order-preservation audit

Pairwise order preserved. New ties only from rendering/endpoints, quantified in `transformation_audit.csv`.

## 7. Prompt-integrity audit

Prompts match the 009/011 q-visible templates except the displayed number. Models were not told the score was transformed.

## 8. API calls / retries / cost

- Planned new scientific calls: 6288
- Missing nonzero-delta records: {missing}
- Estimated new-call USD: {cost:.4f}

## 9. Coverage by delta

See `coverage_by_delta.csv`.

## 10. Leakage by delta

See `leakage_by_delta.csv`.

## 11. GPT MMLU primary result

{line(gpt_m)}

## 12. GPT code primary result

{line(gpt_c)}

## 13. Claude MMLU result

{line(cl_m)}

## 14. Claude code saturation result

{line(cl_c)}

A small Claude-code effect is not evidence against the GPT result. Do not call Claude robust or invoke guardrails.

## 15. Constant-vs-offset effect retention

See `effect_retention.csv`.

## 16. Calibration/discrimination diagnostics

The monotone transform changes level/calibration and preserves q1 item ordering/AUROC up to ties. It does not preserve “all information.”

## 17. Cost/regret sweep

`loss(λ) = leakage + λ * coverage` on the Task-012 λ grid. See `cost_sweep.csv`. Do not cherry-pick λ.

## 18. Marginal verification efficiency

See `marginal_efficiency.csv`. Moving from δ=+1.5 toward δ=−1.5.

## 19. WORLD classification

**{world}**

## 20. Evidence against the current Task-012 framing

Task 012 remains the constant-score stress test. Task 014 tests whether level sensitivity survives when q1 ordering is preserved. World {world} determines whether 012 is an extreme case, a general level effect, or a scope correction.

## 21. Strongest defensible paper claim

{claim}

## 22. Claims that must NOT be used

- ordinary production calibration drift necessarily causes identical effects
- transformed scores are naturally generated
- ranking is perfectly informative
- q1 contains all relevant uncertainty
- Claude is normatively robust
- guardrails cause saturation
- hide-confidence is optimal (Task 013 already rejected that as a headline)

## 23. Paper integration recommendation

**{integration}**

## 24. Whether paperDirection should be rewritten

{rewrite}

## 25. Are any further experiments scientifically justified?

No. Task 014 is the final scientific experiment. Next: write the paper.

## 26. READY_FOR_GPT_REVIEW = YES
"""
    (RETURN_DIR / "report.md").write_text(report, encoding="utf-8")
    (RETURN_DIR / "cost_report.md").write_text(
        f"# Task 014 cost\n\nEstimated new-call USD: {cost:.4f}\nMissing records: {missing}\n",
        encoding="utf-8",
    )
    if not (RETURN_DIR / "deviations.md").exists():
        (RETURN_DIR / "deviations.md").write_text("# Task 014 deviations\n\nNone material.\n", encoding="utf-8")
    (RETURN_DIR / "validation.md").write_text(
        f"""# Task 014 validation

- paperDirection sha256 after run: `{sha}`
- paperDirection overwritten: no
- 009/011 artifacts: read-only
- δ=0 reused from true_q_visible: yes
- Hidden tests rerun: no
- Endpoints requested: gpt-5.6-sol / claude-sonnet-5
""",
        encoding="utf-8",
    )
    (RETURN_DIR / "paper_integration_packet.md").write_text(
        f"""# Task 014 paper integration

World: `{world}`
Integration: `{integration}`

{claim}

{rewrite}
""",
        encoding="utf-8",
    )
    names = [
        "report.md",
        "preregistration.md",
        "freeze_manifest.json",
        "analysis_freeze.json",
        "delta0_reuse_check.md",
        "transformation_audit.md",
        "cost_report.md",
        "deviations.md",
        "validation.md",
        "changed_files.txt",
        "paper_integration_packet.md",
        "transformation_audit.csv",
        "stage3_offset_conditions.csv",
        "coverage_by_delta.csv",
        "leakage_by_delta.csv",
        "effect_retention.csv",
        "cost_sweep.csv",
        "marginal_efficiency.csv",
        "cross_task_summary.csv",
    ]
    (RETURN_DIR / "changed_files.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    json_dump(RETURN_DIR / "run_manifest.json", {"world": world, "cost_usd": cost, "missing": missing, "finished_at": datetime.now(UTC).isoformat()})

    if integration == "MAJOR_UPGRADE" or integration == "MODERATE_UPGRADE" or integration == "TASK_DEPENDENT":
        (RETURN_DIR / "paperDirection_task014_proposed.txt").write_text(
            f"D1 PAPER DIRECTION — PROPOSED AFTER TASK 014\nWorld: {world}\n{claim}\nDo not overwrite paperDirection.txt until accepted.\n",
            encoding="utf-8",
        )
    elif integration == "SCOPE_CORRECTION":
        (RETURN_DIR / "paperDirection_task014_correction.txt").write_text(
            f"D1 SCOPE CORRECTION AFTER TASK 014\nWorld: {world}\n{claim}\n",
            encoding="utf-8",
        )
