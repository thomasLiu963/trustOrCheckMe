"""Section 2: reproduce hidden/true_q coverage and Task 012 leakage/cost."""

from __future__ import annotations

from typing import Any

from .task012_data import all_rows, filter_rows
from .task012_stats import coverage, leakage
from .task013_common import (
    FIXED_CONDS,
    GPT_CLAUDE,
    LABEL,
    LAMBDA_GRID,
    REPRO_TOL_PP,
    RETURN_DIR,
    TARGET_COVERAGE,
    TASK012_DIR,
    load_csv,
)


def _pp(value: float) -> float:
    return 100.0 * value


def reproduce() -> dict[str, Any]:
    rows = all_rows()
    checks: list[dict[str, Any]] = []
    for (task, model, cond), target in TARGET_COVERAGE.items():
        got = _pp(coverage(filter_rows(rows, task=task, model=model, condition=cond)))
        checks.append(
            {
                "source": "published_009_011",
                "task": task,
                "model": model,
                "metric": f"{cond}_coverage",
                "target": target,
                "reproduced": round(got, 3),
                "ok": abs(got - target) <= REPRO_TOL_PP,
            }
        )

    prior = load_csv(TASK012_DIR / "error_leakage_by_condition.csv")
    by_prior = {(r["task"], r["model_alias"], r["score_condition"]): r for r in prior}
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            for cond in ("hidden", "true_q_visible") + FIXED_CONDS:
                subset = filter_rows(rows, task=task, model=model, condition=cond)
                prev = by_prior[(task, model, cond)]
                got_l = leakage(subset)
                tgt_l = float(prev["unverified_error_rate"])
                got_c = coverage(subset)
                tgt_c = float(prev["verification_coverage"])
                checks.append(
                    {
                        "source": "task012",
                        "task": task,
                        "model": model,
                        "metric": f"{cond}_leakage",
                        "target": round(100 * tgt_l, 3),
                        "reproduced": round(100 * got_l, 3),
                        "ok": abs(_pp(got_l) - _pp(tgt_l)) <= REPRO_TOL_PP,
                    }
                )
                checks.append(
                    {
                        "source": "task012",
                        "task": task,
                        "model": model,
                        "metric": f"{cond}_coverage",
                        "target": round(100 * tgt_c, 3),
                        "reproduced": round(100 * got_c, 3),
                        "ok": abs(_pp(got_c) - _pp(tgt_c)) <= REPRO_TOL_PP,
                    }
                )

    prior_cost = load_csv(TASK012_DIR / "cost_sweep.csv")
    grid_ok = list(LAMBDA_GRID) == [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    fn_ok = all(
        abs(float(r["normalized_loss"]) - (float(r["leakage_rate"]) + float(r["lambda"]) * float(r["verification_coverage"])))
        < 1e-9
        for r in prior_cost
        if r["score_condition"] in FIXED_CONDS
    )
    checks.append(
        {
            "source": "task012",
            "task": "both",
            "model": "shared",
            "metric": "lambda_grid",
            "target": "012_grid",
            "reproduced": ",".join(str(x) for x in LAMBDA_GRID),
            "ok": grid_ok,
        }
    )
    checks.append(
        {
            "source": "task012",
            "task": "both",
            "model": "shared",
            "metric": "loss_function",
            "target": "leakage + lambda * coverage",
            "reproduced": "leakage + lambda * coverage",
            "ok": fn_ok,
        }
    )

    passed = all(c["ok"] for c in checks)
    return {
        "label": LABEL,
        "passed": passed,
        "stop_token": None if passed else "TASK013_REPRODUCTION_FAILURE",
        "tolerance_pp": REPRO_TOL_PP,
        "checks": checks,
    }


def write_reproduction_md(result: dict[str, Any]) -> None:
    lines = [
        "# Task 013 reproduction check",
        "",
        f"Label: `{LABEL}`. Zero API calls. Frozen 009–012 artifacts only.",
        "",
        f"Status: **{'PASS' if result['passed'] else 'TASK013_REPRODUCTION_FAILURE'}**",
        "",
        f"Rounding tolerance: ±{result['tolerance_pp']} percentage points.",
        "",
        "| Source | Task | Model | Metric | Target | Reproduced | OK |",
        "|---|---|---|---|---|---|:---:|",
    ]
    for c in result["checks"]:
        lines.append(
            f"| {c['source']} | {c['task']} | {c['model']} | {c['metric']} | {c['target']} | {c['reproduced']} | {'yes' if c['ok'] else 'NO'} |"
        )
    lines.extend(["", "Filters match Task 012: 009 primary repeat 0; 011 confirmatory IDs only.", ""])
    if not result["passed"]:
        lines.append("STOP. Do not compute Sections 3–12 until this discrepancy is resolved.")
    (RETURN_DIR / "reproduction_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
