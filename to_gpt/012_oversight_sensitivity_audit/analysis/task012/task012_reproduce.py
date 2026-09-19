"""Section 3: reproduce published 009 / 011 numbers before new analysis."""

from __future__ import annotations

from typing import Any

from .task012_common import REPRO_TOL_PP, RETURN_DIR, TARGET_009, TARGET_011
from .task012_data import filter_rows, load_009_rows, load_011_rows
from .task012_stats import coverage, error_rate


def _pp(value: float) -> float:
    return 100.0 * value


def reproduce() -> dict[str, Any]:
    rows_009 = load_009_rows()
    rows_011 = load_011_rows()
    checks: list[dict[str, Any]] = []

    for model, key, target, rows, cond_a, cond_b in (
        ("openai_gpt56_sol", "coverage_0.70_0.99", TARGET_009[("openai_gpt56_sol", "coverage_0.70_0.99")], rows_009, "displayed_0.70", "displayed_0.99"),
        ("anthropic_sonnet5", "coverage_0.70_0.99", TARGET_009[("anthropic_sonnet5", "coverage_0.70_0.99")], rows_009, "displayed_0.70", "displayed_0.99"),
    ):
        a = coverage(filter_rows(rows, model=model, condition=cond_a))
        b = coverage(filter_rows(rows, model=model, condition=cond_b))
        got = _pp(a - b)
        ok = abs(got - target) <= REPRO_TOL_PP
        checks.append(
            {
                "task": "mmlu",
                "model": model,
                "metric": key,
                "target": target,
                "reproduced": round(got, 3),
                "n_a": len(filter_rows(rows, model=model, condition=cond_a)),
                "n_b": len(filter_rows(rows, model=model, condition=cond_b)),
                "ok": ok,
            }
        )

    for model, key, target in (
        ("openai_gpt56_sol", "coverage_0.70_0.99", TARGET_011[("openai_gpt56_sol", "coverage_0.70_0.99")]),
        ("anthropic_sonnet5", "coverage_0.70_0.99", TARGET_011[("anthropic_sonnet5", "coverage_0.70_0.99")]),
    ):
        a = coverage(filter_rows(rows_011, model=model, condition="displayed_0.70"))
        b = coverage(filter_rows(rows_011, model=model, condition="displayed_0.99"))
        got = _pp(a - b)
        checks.append(
            {
                "task": "code",
                "model": model,
                "metric": key,
                "target": target,
                "reproduced": round(got, 3),
                "n_a": len(filter_rows(rows_011, model=model, condition="displayed_0.70")),
                "n_b": len(filter_rows(rows_011, model=model, condition="displayed_0.99")),
                "ok": abs(got - target) <= REPRO_TOL_PP,
            }
        )

    for model, target in (
        ("openai_gpt56_sol", TARGET_011[("openai_gpt56_sol", "pass_rate")]),
        ("anthropic_sonnet5", TARGET_011[("anthropic_sonnet5", "pass_rate")]),
    ):
        cond_rows = filter_rows(rows_011, model=model, condition="displayed_0.70")
        got = _pp(1.0 - error_rate(cond_rows))
        checks.append(
            {
                "task": "code",
                "model": model,
                "metric": "pass_rate",
                "target": target,
                "reproduced": round(got, 3),
                "n_a": len(cond_rows),
                "n_b": len(cond_rows),
                "ok": abs(got - target) <= REPRO_TOL_PP,
            }
        )

    passed = all(c["ok"] for c in checks)
    return {
        "label": "POST_HOC_SYSTEMS_REANALYSIS",
        "passed": passed,
        "stop_token": None if passed else "TASK012_REPRODUCTION_FAILURE",
        "tolerance_pp": REPRO_TOL_PP,
        "checks": checks,
        "n_009": len(rows_009),
        "n_011": len(rows_011),
    }


def write_reproduction_md(result: dict[str, Any]) -> None:
    lines = [
        "# Task 012 reproduction check",
        "",
        "Label: `POST_HOC_SYSTEMS_REANALYSIS`. Zero API calls. Frozen 009/011 artifacts only.",
        "",
        f"Status: **{'PASS' if result['passed'] else 'TASK012_REPRODUCTION_FAILURE'}**",
        "",
        f"Rounding tolerance: ±{result['tolerance_pp']} percentage points.",
        "",
        "| Task | Model | Metric | Published | Reproduced | N | OK |",
        "|---|---|---|---:|---:|---:|:---:|",
    ]
    for c in result["checks"]:
        lines.append(
            f"| {c['task']} | {c['model']} | {c['metric']} | {c['target']} | {c['reproduced']} | {c['n_a']} | {'yes' if c['ok'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "MMLU filter: `roster=primary`, `repeat_index=0`, GPT+Claude.",
            "Code filter: confirmatory `sample_main.csv` IDs only, `repeat_index=0`.",
            "Code pass rate: official hidden-suite `passed` on those IDs.",
            "",
        ]
    )
    if not result["passed"]:
        lines.append("STOP. Do not compute Sections 4–9 until this discrepancy is resolved.")
    (RETURN_DIR / "reproduction_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
