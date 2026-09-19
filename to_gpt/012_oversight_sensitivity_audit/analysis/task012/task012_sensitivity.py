"""Section 5: finite-difference coverage and leakage sensitivity."""

from __future__ import annotations

from typing import Any

from .task012_common import ADJACENT, FIXED_SCORES, GPT_CLAUDE, LABEL, write_csv
from .task012_data import ItemRow, filter_rows
from .task012_stats import bootstrap_pair, coverage, leakage


def _per_010(delta: float, dq: float) -> float:
    if dq == 0:
        return float("nan")
    return delta / dq * 0.10


def classify_shape(coverages: list[float]) -> str:
    if not coverages:
        return "unknown"
    adj = [coverages[i + 1] - coverages[i] for i in range(len(coverages) - 1)]
    span = abs(coverages[-1] - coverages[0])
    if (all(c > 0.90 for c in coverages) or all(c < 0.10 for c in coverages)) and max(abs(x) for x in adj) < 0.05:
        return "saturated"
    if span > 0 and max(abs(x) for x in adj) >= 0.60 * span:
        return "threshold-like"
    signs = [x for x in adj if abs(x) > 1e-12]
    if signs and all(x < 0 for x in signs):
        return "monotonic"
    if signs and all(x > 0 for x in signs):
        return "monotonic"
    return "mixed"


def run_sensitivity(rows: list[ItemRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in ("mmlu", "code"):
        for model in GPT_CLAUDE:
            covs: list[float] = []
            local: list[dict[str, Any]] = []
            for qa, qb in ADJACENT:
                a = filter_rows(rows, task=task, model=model, condition=f"displayed_{qa:.2f}")
                b = filter_rows(rows, task=task, model=model, condition=f"displayed_{qb:.2f}")
                d_cov = bootstrap_pair(a, b, coverage)
                d_leak = bootstrap_pair(a, b, leakage)
                dq = qb - qa
                row = {
                    "label": LABEL,
                    "task": task,
                    "model_alias": model,
                    "from_score": qa,
                    "to_score": qb,
                    "span": "adjacent",
                    "coverage_sensitivity_per_0.10": _per_010(d_cov.point, dq),
                    "coverage_sens_ci_lo": _per_010(d_cov.lo, dq),
                    "coverage_sens_ci_hi": _per_010(d_cov.hi, dq),
                    "leakage_sensitivity_per_0.10": _per_010(d_leak.point, dq),
                    "leakage_sens_ci_lo": _per_010(d_leak.lo, dq),
                    "leakage_sens_ci_hi": _per_010(d_leak.hi, dq),
                    "unverified_errors_per_100_per_0.10": 100.0 * _per_010(d_leak.point, dq),
                    "delta_coverage": d_cov.point,
                    "delta_leakage": d_leak.point,
                    "bootstrap_n": d_leak.n_boot,
                    "shape": "",
                    "is_max_local_coverage": False,
                    "is_max_local_leakage": False,
                }
                local.append(row)
                if not covs:
                    covs.append(coverage(a))
                covs.append(coverage(b))
            a70 = filter_rows(rows, task=task, model=model, condition="displayed_0.70")
            a99 = filter_rows(rows, task=task, model=model, condition="displayed_0.99")
            d_cov = bootstrap_pair(a70, a99, coverage)
            d_leak = bootstrap_pair(a70, a99, leakage)
            avg = {
                "label": LABEL,
                "task": task,
                "model_alias": model,
                "from_score": 0.70,
                "to_score": 0.99,
                "span": "0.70_to_0.99",
                "coverage_sensitivity_per_0.10": _per_010(d_cov.point, 0.29),
                "coverage_sens_ci_lo": _per_010(d_cov.lo, 0.29),
                "coverage_sens_ci_hi": _per_010(d_cov.hi, 0.29),
                "leakage_sensitivity_per_0.10": _per_010(d_leak.point, 0.29),
                "leakage_sens_ci_lo": _per_010(d_leak.lo, 0.29),
                "leakage_sens_ci_hi": _per_010(d_leak.hi, 0.29),
                "unverified_errors_per_100_per_0.10": 100.0 * _per_010(d_leak.point, 0.29),
                "delta_coverage": d_cov.point,
                "delta_leakage": d_leak.point,
                "bootstrap_n": d_leak.n_boot,
                "shape": classify_shape(covs),
                "is_max_local_coverage": False,
                "is_max_local_leakage": False,
            }
            if local:
                max_c = max(local, key=lambda r: abs(r["coverage_sensitivity_per_0.10"]))
                max_l = max(local, key=lambda r: abs(r["leakage_sensitivity_per_0.10"]))
                max_c["is_max_local_coverage"] = True
                max_l["is_max_local_leakage"] = True
            for row in local:
                row["shape"] = avg["shape"]
            out.extend(local)
            out.append(avg)
    return out


def write_sensitivity(path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)
