#!/usr/bin/env python3
"""Regenerate D1 headline tables and figures from the public CSVs only.

No model calls. No benchmark text. No hidden tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from analysis.constants import (
    ALL_CONDS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED_012,
    BOOTSTRAP_SEED_014,
    BOOTSTRAP_SEED_015,
    DATA_DIR,
    DELTAS,
    FIGURES_DIR,
    FIXED_CONDS,
    GPT_CLAUDE,
    HEADLINES,
    LAMBDA_GRID,
    MANIFEST_DIR,
    REPRO_TOL_LOSS,
    REPRO_TOL_PP,
    REPRO_TOL_RATE,
)
from analysis.figures import (
    fixed_score_coverage_leakage,
    loss_spread,
    rank_preserving_response,
    unmanipulated_escape_callout,
)
from analysis.io_csv import filter_offset, filter_routing, load_offset, load_routing, write_csv
from analysis.metrics import (
    bootstrap_pair,
    bootstrap_stat,
    coverage,
    escape_rate,
    error_rate,
    leakage,
    loss,
)

ROOT = Path(__file__).resolve().parent


def _pp(x: float) -> float:
    return 100.0 * x


def _check(name: str, got: float, target: float, tol: float, failures: list[str]) -> None:
    if abs(got - target) > tol:
        failures.append(f"{name}: got {got:.6f}, expected {target:.6f} ± {tol}")


def main() -> int:
    routing = load_routing(DATA_DIR / "item_level_routing.csv")
    offset = load_offset(DATA_DIR / "item_level_offset.csv")
    if not routing or not offset:
        raise SystemExit("missing released CSVs")

    failures: list[str] = []
    summary: dict[str, object] = {}

    # --- Task 009 / 011 coverage and pass rates ---
    coverage_rows = []
    for task, seed_key in (("mmlu", "009"), ("code", "011")):
        for model in GPT_CLAUDE:
            low = filter_routing(routing, task=task, model=model, condition="displayed_0.70")
            high = filter_routing(routing, task=task, model=model, condition="displayed_0.99")
            # 009/011 headline is coverage(0.70) - coverage(0.99)
            delta = bootstrap_pair(high, low, coverage, seed=BOOTSTRAP_SEED_012, n_boot=BOOTSTRAP_RESAMPLES)
            unique = {}
            for row in low:
                unique[row.question_id] = row
            pass_rate = 1.0 - error_rate(list(unique.values()))
            coverage_rows.append(
                {
                    "task": task,
                    "model_alias": model,
                    "coverage_0.70": coverage(low),
                    "coverage_0.99": coverage(high),
                    "coverage_delta_pp": _pp(delta.point),
                    "coverage_delta_ci_lo_pp": _pp(delta.lo),
                    "coverage_delta_ci_hi_pp": _pp(delta.hi),
                    "pass_rate": pass_rate,
                    "n": len(low),
                }
            )
            label = "gpt" if model == "openai_gpt56_sol" else "claude"
            _check(
                f"{seed_key}_{label}_coverage_pp",
                _pp(delta.point),
                HEADLINES[f"{seed_key}_{label}_coverage_pp"],
                REPRO_TOL_PP,
                failures,
            )
            if task == "code":
                _check(
                    f"011_{label}_pass_pct",
                    100 * pass_rate,
                    HEADLINES[f"011_{label}_pass_pct"],
                    REPRO_TOL_PP,
                    failures,
                )
    write_csv(DATA_DIR / "derived_coverage_effects.csv", coverage_rows)
    summary["coverage_effects"] = coverage_rows

    # --- Task 012 leakage 0.70 -> 0.99 ---
    leakage_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            for cond in ALL_CONDS:
                cell = filter_routing(routing, task=task, model=model, condition=cond)
                leakage_rows.append(
                    {
                        "task": task,
                        "model_alias": model,
                        "score_condition": cond,
                        "n": len(cell),
                        "verification_coverage": coverage(cell),
                        "leakage_rate": leakage(cell),
                        "error_rate": error_rate(cell),
                        "conditional_escape_rate": escape_rate(cell),
                    }
                )
    write_csv(DATA_DIR / "derived_condition_rates.csv", leakage_rows)

    leak_contrasts = []
    for task, key in (("mmlu", "012_gpt_mmlu_leakage_pp"), ("code", "012_gpt_code_leakage_pp")):
        model = "openai_gpt56_sol"
        a = filter_routing(routing, task=task, model=model, condition="displayed_0.70")
        b = filter_routing(routing, task=task, model=model, condition="displayed_0.99")
        d = bootstrap_pair(a, b, leakage, seed=BOOTSTRAP_SEED_012, n_boot=BOOTSTRAP_RESAMPLES)
        leak_contrasts.append(
            {
                "task": task,
                "model_alias": model,
                "leakage_0.70": leakage(a),
                "leakage_0.99": leakage(b),
                "delta_leakage_pp": _pp(d.point),
                "ci_lo_pp": _pp(d.lo),
                "ci_hi_pp": _pp(d.hi),
            }
        )
        _check(key, _pp(d.point), HEADLINES[key], REPRO_TOL_PP, failures)
    write_csv(DATA_DIR / "derived_leakage_contrasts.csv", leak_contrasts)
    summary["leakage_contrasts"] = leak_contrasts

    # --- Task 012 cost sweep ---
    cost_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            stats = {
                cond: (
                    leakage(filter_routing(routing, task=task, model=model, condition=cond)),
                    coverage(filter_routing(routing, task=task, model=model, condition=cond)),
                )
                for cond in ALL_CONDS
            }
            for lam in LAMBDA_GRID:
                fixed_losses = {c: loss(*stats[c], lam) for c in FIXED_CONDS}
                best_fixed = min(fixed_losses.values())
                for cond, (leak, cov) in stats.items():
                    cost_rows.append(
                        {
                            "task": task,
                            "model_alias": model,
                            "score_condition": cond,
                            "lambda": lam,
                            "leakage_rate": leak,
                            "verification_coverage": cov,
                            "normalized_loss": loss(leak, cov, lam),
                            "best_fixed_loss": best_fixed,
                            "regret_best_fixed": loss(leak, cov, lam) - best_fixed if cond in FIXED_CONDS else "",
                        }
                    )
    write_csv(DATA_DIR / "derived_cost_sweep.csv", cost_rows)
    gpt_mmlu_099 = next(
        r
        for r in cost_rows
        if r["task"] == "mmlu"
        and r["model_alias"] == "openai_gpt56_sol"
        and r["score_condition"] == "displayed_0.99"
        and abs(float(r["lambda"]) - 0.001) < 1e-15
    )
    _check("012_cost_gpt_mmlu_099_l001", float(gpt_mmlu_099["normalized_loss"]), 0.138082, REPRO_TOL_LOSS, failures)

    # --- Task 014 coverage/leakage by delta and retention ---
    offset_rates = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            for delta in DELTAS:
                cell = filter_offset(offset, task=task, model=model, delta=delta)
                offset_rates.append(
                    {
                        "task": task,
                        "model_alias": model,
                        "delta": delta,
                        "n": len(cell),
                        "verification_coverage": coverage(cell),
                        "leakage_rate": leakage(cell),
                    }
                )
    write_csv(DATA_DIR / "derived_offset_rates.csv", offset_rates)

    retention = []
    for task, cov_key, leak_key, ret_c_key, ret_l_key in (
        ("mmlu", "014_gpt_mmlu_coverage_pp", "014_gpt_mmlu_leakage_pp", "014_gpt_mmlu_retention_coverage", "014_gpt_mmlu_retention_leakage"),
        ("code", "014_gpt_code_coverage_pp", "014_gpt_code_leakage_pp", "014_gpt_code_retention_coverage", "014_gpt_code_retention_leakage"),
    ):
        model = "openai_gpt56_sol"
        lo = filter_offset(offset, task=task, model=model, delta=-1.5)
        hi = filter_offset(offset, task=task, model=model, delta=1.5)
        d_cov = bootstrap_pair(lo, hi, coverage, seed=BOOTSTRAP_SEED_014, n_boot=BOOTSTRAP_RESAMPLES)
        d_leak = bootstrap_pair(lo, hi, leakage, seed=BOOTSTRAP_SEED_014, n_boot=BOOTSTRAP_RESAMPLES)
        const_cov = coverage(filter_routing(routing, task=task, model=model, condition="displayed_0.99")) - coverage(
            filter_routing(routing, task=task, model=model, condition="displayed_0.70")
        )
        const_leak = leakage(filter_routing(routing, task=task, model=model, condition="displayed_0.99")) - leakage(
            filter_routing(routing, task=task, model=model, condition="displayed_0.70")
        )
        ret_c = d_cov.point / const_cov if const_cov else float("nan")
        ret_l = d_leak.point / const_leak if const_leak else float("nan")
        retention.append(
            {
                "task": task,
                "model_alias": model,
                "delta_coverage_pp": _pp(d_cov.point),
                "delta_leakage_pp": _pp(d_leak.point),
                "retention_coverage": ret_c,
                "retention_leakage": ret_l,
            }
        )
        _check(cov_key, _pp(d_cov.point), HEADLINES[cov_key], REPRO_TOL_PP, failures)
        _check(leak_key, _pp(d_leak.point), HEADLINES[leak_key], REPRO_TOL_PP, failures)
        _check(ret_c_key, ret_c, HEADLINES[ret_c_key], REPRO_TOL_RATE, failures)
        _check(ret_l_key, ret_l, HEADLINES[ret_l_key], REPRO_TOL_RATE, failures)

    claude_m = "anthropic_sonnet5"
    lo = filter_offset(offset, task="mmlu", model=claude_m, delta=-1.5)
    hi = filter_offset(offset, task="mmlu", model=claude_m, delta=1.5)
    _check("014_claude_mmlu_coverage_pp", _pp(coverage(hi) - coverage(lo)), HEADLINES["014_claude_mmlu_coverage_pp"], REPRO_TOL_PP, failures)
    _check("014_claude_mmlu_leakage_pp", _pp(leakage(hi) - leakage(lo)), HEADLINES["014_claude_mmlu_leakage_pp"], REPRO_TOL_PP, failures)
    write_csv(DATA_DIR / "derived_retention.csv", retention)
    summary["retention"] = retention

    # --- Task 015 escape + loss spread ---
    escape_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            cell = filter_routing(routing, task=task, model=model, condition="true_q_visible")
            ci = bootstrap_stat(cell, escape_rate, seed=BOOTSTRAP_SEED_015, n_boot=BOOTSTRAP_RESAMPLES)
            escape_rows.append(
                {
                    "task": task,
                    "model_alias": model,
                    "n": len(cell),
                    "error_rate": error_rate(cell),
                    "leakage_rate": leakage(cell),
                    "conditional_escape_rate": ci.point,
                    "ci_lo": ci.lo,
                    "ci_hi": ci.hi,
                }
            )
    write_csv(DATA_DIR / "derived_unmanipulated_escape.csv", escape_rows)
    gpt_code_esc = next(r for r in escape_rows if r["task"] == "code" and r["model_alias"] == "openai_gpt56_sol")
    _check("015_gpt_code_escape", gpt_code_esc["conditional_escape_rate"], HEADLINES["015_gpt_code_escape"], REPRO_TOL_RATE, failures)
    _check("015_gpt_code_escape_ci_lo", gpt_code_esc["ci_lo"], HEADLINES["015_gpt_code_escape_ci_lo"], 0.02, failures)
    _check("015_gpt_code_escape_ci_hi", gpt_code_esc["ci_hi"], HEADLINES["015_gpt_code_escape_ci_hi"], 0.02, failures)

    spread_rows = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            values = []
            for lam in LAMBDA_GRID:
                losses = []
                for delta in DELTAS:
                    cell = filter_offset(offset, task=task, model=model, delta=delta)
                    losses.append(loss(leakage(cell), coverage(cell), lam))
                values.append(max(losses) - min(losses))
            rec = {
                "task": task,
                "model_alias": model,
                "max_spread": max(values),
                "mean_spread": sum(values) / len(values),
                "min_spread": min(values),
            }
            spread_rows.append(rec)
            key = {
                ("mmlu", "openai_gpt56_sol"): "015_gpt_mmlu_max_spread",
                ("code", "openai_gpt56_sol"): "015_gpt_code_max_spread",
                ("mmlu", "anthropic_sonnet5"): "015_claude_mmlu_max_spread",
                ("code", "anthropic_sonnet5"): "015_claude_code_max_spread",
            }[(task, model)]
            _check(key, rec["max_spread"], HEADLINES[key], REPRO_TOL_LOSS, failures)
    write_csv(DATA_DIR / "derived_loss_spread.csv", spread_rows)
    summary["loss_spread"] = spread_rows
    summary["escape"] = escape_rows

    fixed_score_coverage_leakage(routing)
    rank_preserving_response(offset)
    loss_spread(offset)
    unmanipulated_escape_callout(escape_rows)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "reproduction_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"Wrote derived tables under {DATA_DIR}")
    print(f"Wrote figures under {FIGURES_DIR}")
    if failures:
        print("REPRODUCTION FAILURES:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("All headline assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
