"""Pre-call transformation and δ=0 reuse audit. No Stage-3 outcomes."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

from .task012_stats import brier, ece, percentiles
from .task014_common import (
    DELTAS,
    GPT_CLAUDE,
    LABEL,
    RETURN_DIR,
    write_csv,
)
from .task014_data import FrozenItem, load_frozen_items, offset_prompt
from .task014_transform import render_q, transform_q


def _pairwise_order_preserved(q1: list[float], qd: list[float]) -> float:
    n = len(q1)
    if n < 2:
        return float("nan")
    agree = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            d1 = q1[i] - q1[j]
            d2 = qd[i] - qd[j]
            if d1 == 0:
                continue
            total += 1
            if d1 * d2 > 0:
                agree += 1
            elif d2 == 0:
                continue
    return agree / total if total else 1.0


def run_transform_audit(items: list[FrozenItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            subset = [it for it in items if it.task == task and it.model_alias == model]
            q1 = [it.q1 for it in subset]
            y = [0.0 if it.incorrect else 1.0 for it in subset]
            n0 = sum(q == 0.0 for q in q1)
            n1 = sum(q == 1.0 for q in q1)
            for delta in DELTAS:
                qd = [transform_q(q, delta) for q in q1]
                rendered = [render_q(v) for v in qd]
                rendered_f = [float(s) for s in rendered]
                q1_r = [render_q(q) for q in q1]
                new_ties = 0
                q1_pairs = 0
                for i in range(len(q1)):
                    for j in range(i + 1, len(q1)):
                        if q1[i] != q1[j]:
                            q1_pairs += 1
                            if rendered_f[i] == rendered_f[j]:
                                new_ties += 1
                reversals = 0
                for i in range(len(q1)):
                    for j in range(i + 1, len(q1)):
                        if q1[i] < q1[j] and rendered_f[i] > rendered_f[j]:
                            reversals += 1
                        if q1[i] > q1[j] and rendered_f[i] < rendered_f[j]:
                            reversals += 1
                rho = spearmanr(q1, rendered_f).statistic
                tau = kendalltau(q1, rendered_f).statistic
                try:
                    auc = float(roc_auc_score(y, rendered_f))
                    auc_q1 = float(roc_auc_score(y, q1))
                except ValueError:
                    auc = auc_q1 = float("nan")
                ece_v, _ = ece(rendered_f, [bool(v) for v in y])
                pct = percentiles(qd)
                rows.append(
                    {
                        "label": LABEL,
                        "task": task,
                        "model_alias": model,
                        "delta": delta,
                        "n": len(subset),
                        "n_q1_exact_0": n0,
                        "n_q1_exact_1": n1,
                        "mean": pct["mean"],
                        "median": pct["p50"],
                        "iqr": pct["iqr"],
                        "p05": pct["p05"],
                        "p25": pct["p25"],
                        "p50": pct["p50"],
                        "p75": pct["p75"],
                        "p95": pct["p95"],
                        "min": min(qd),
                        "max": max(qd),
                        "frac_ge_0.90": sum(v >= 0.90 for v in qd) / len(qd),
                        "frac_ge_0.95": sum(v >= 0.95 for v in qd) / len(qd),
                        "frac_ge_0.99": sum(v >= 0.99 for v in qd) / len(qd),
                        "frac_le_0.70": sum(v <= 0.70 for v in qd) / len(qd),
                        "spearman_q1": float(rho),
                        "kendall_tau_q1": float(tau),
                        "pairwise_order_preserved": _pairwise_order_preserved(q1, rendered_f),
                        "n_new_ties_from_render": new_ties,
                        "frac_new_ties_from_render": new_ties / q1_pairs if q1_pairs else 0.0,
                        "n_order_reversals": reversals,
                        "brier": brier(rendered_f, [bool(v) for v in y]),
                        "ece": ece_v,
                        "auroc_q_delta": auc,
                        "auroc_q1": auc_q1,
                        "n_rendered_distinct": len(set(rendered)),
                        "n_q1_rendered_distinct": len(set(q1_r)),
                    }
                )
    return rows


def delta0_reuse(items: list[FrozenItem]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    n = 0
    identical = 0
    for item in items:
        n += 1
        built = offset_prompt(item, 0.0)
        if built == item.true_q_prompt:
            identical += 1
        else:
            if len(mismatches) < 8:
                mismatches.append(
                    {
                        "task": item.task,
                        "question_id": item.question_id,
                        "model_alias": item.model_alias,
                        "q1": item.q1,
                        "built_len": len(built),
                        "trueq_len": len(item.true_q_prompt),
                    }
                )
    reuse = identical == n and n > 0
    return {
        "n_items": n,
        "n_byte_identical": identical,
        "reuse_delta0": reuse,
        "decision": "REUSE_TRUE_Q_VISIBLE" if reuse else "RERUN_DELTA0",
        "mismatches": mismatches,
    }


def write_audit_md(audit: list[dict[str, Any]], reuse: dict[str, Any]) -> None:
    reversals = sum(int(r["n_order_reversals"]) for r in audit)
    lines = [
        "# Task 014 transformation audit",
        "",
        "Computed from frozen q1/correctness only. No new Stage-3 outcomes were inspected.",
        "",
        f"Exact q1 endpoints 0 or 1: {sum(r['n_q1_exact_0'] + r['n_q1_exact_1'] for r in audit if r['delta']==0)} item-model rows.",
        f"Order reversals after `.12g` rendering: **{reversals}**.",
        "",
        "| Task | Model | δ | mean | p50 | ≥0.90 | ≤0.70 | Spearman | reversals | AUROC | Brier | ECE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in audit:
        lines.append(
            f"| {r['task']} | {r['model_alias']} | {r['delta']} | {r['mean']:.3f} | {r['median']:.3f} | "
            f"{100*r['frac_ge_0.90']:.1f}% | {100*r['frac_le_0.70']:.1f}% | {r['spearman_q1']:.6f} | "
            f"{r['n_order_reversals']} | {r['auroc_q_delta']:.3f} | {r['brier']:.3f} | {r['ece']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## δ=0 reuse",
            "",
            f"Items compared: {reuse['n_items']}",
            f"Byte-identical prompts: {reuse['n_byte_identical']}",
            f"Decision: **{reuse['decision']}**",
            "",
        ]
    )
    if reversals:
        lines.append("STOP: rendering introduced order reversals. Fix formatting before scientific calls.")
    (RETURN_DIR / "transformation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reuse_md = [
        "# Task 014 δ=0 reuse check",
        "",
        "Compared Task-014 δ=0 rendered Stage-3 prompts to existing Task-009/011 `true_q_visible` prompts.",
        "",
        f"Decision: **{reuse['decision']}**",
        "",
        f"- items: {reuse['n_items']}",
        f"- byte-identical: {reuse['n_byte_identical']}",
        "",
        "Decision was made from prompt identity only, not from routing outcomes.",
        "",
    ]
    if reuse["mismatches"]:
        reuse_md.append("Sample mismatches:")
        for row in reuse["mismatches"]:
            reuse_md.append(f"- {row}")
    (RETURN_DIR / "delta0_reuse_check.md").write_text("\n".join(reuse_md) + "\n", encoding="utf-8")
