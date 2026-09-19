"""Rates, bootstrap CIs, and calibration for Task 012."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .task012_common import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, ECE_BINS
from .task012_data import ItemRow


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float
    n_boot: int


def mean_bool(rows: Sequence[ItemRow], fn: Callable[[ItemRow], bool]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([fn(r) for r in rows]))


def coverage(rows: Sequence[ItemRow]) -> float:
    return mean_bool(rows, lambda r: r.verify)


def leakage(rows: Sequence[ItemRow]) -> float:
    return mean_bool(rows, lambda r: r.incorrect and not r.verify)


def error_rate(rows: Sequence[ItemRow]) -> float:
    return mean_bool(rows, lambda r: r.incorrect)


def error_catch(rows: Sequence[ItemRow]) -> float:
    bad = [r for r in rows if r.incorrect]
    if not bad:
        return float("nan")
    return mean_bool(bad, lambda r: r.verify)


def unverified_error_share(rows: Sequence[ItemRow]) -> float:
    bad = [r for r in rows if r.incorrect]
    if not bad:
        return float("nan")
    return mean_bool(bad, lambda r: not r.verify)


def verify_error_precision(rows: Sequence[ItemRow]) -> float:
    verified = [r for r in rows if r.verify]
    if not verified:
        return float("nan")
    return mean_bool(verified, lambda r: r.incorrect)


def by_question(rows: Sequence[ItemRow]) -> dict[str, list[ItemRow]]:
    out: dict[str, list[ItemRow]] = {}
    for row in rows:
        out.setdefault(row.question_id, []).append(row)
    return out


def bootstrap_stat(
    rows: Sequence[ItemRow],
    stat: Callable[[list[ItemRow]], float],
    *,
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = BOOTSTRAP_RESAMPLES,
) -> CI:
    grouped = by_question(rows)
    ids = np.array(sorted(grouped), dtype=object)
    if ids.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    point = float(stat(list(rows)))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = ids[rng.integers(0, ids.size, size=ids.size)]
        resampled = [row for qid in sample for row in grouped[str(qid)]]
        draws[i] = stat(resampled)
    return CI(point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), n_boot)


def bootstrap_pair(
    rows_a: Sequence[ItemRow],
    rows_b: Sequence[ItemRow],
    stat: Callable[[list[ItemRow]], float],
    *,
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = BOOTSTRAP_RESAMPLES,
) -> CI:
    ga = by_question(rows_a)
    gb = by_question(rows_b)
    ids = np.array(sorted(set(ga) & set(gb)), dtype=object)
    if ids.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    point = float(stat(list(rows_b)) - stat(list(rows_a)))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = ids[rng.integers(0, ids.size, size=ids.size)]
        a = [row for qid in sample for row in ga[str(qid)]]
        b = [row for qid in sample for row in gb[str(qid)]]
        draws[i] = stat(b) - stat(a)
    return CI(point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), n_boot)


def percentiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {k: float("nan") for k in ("p05", "p10", "p25", "p50", "p75", "p90", "p95", "mean", "sd", "iqr")}
    qs = np.quantile(arr, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p05": float(qs[0]),
        "p10": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p90": float(qs[5]),
        "p95": float(qs[6]),
        "iqr": float(qs[4] - qs[2]),
    }


def brier(q1: Sequence[float], correct: Sequence[bool]) -> float:
    q = np.asarray(q1, dtype=float)
    y = np.asarray(correct, dtype=float)
    return float(np.mean((q - y) ** 2))


def ece(q1: Sequence[float], correct: Sequence[bool], n_bins: int = ECE_BINS) -> tuple[float, list[dict[str, float]]]:
    q = np.asarray(q1, dtype=float)
    y = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table: list[dict[str, float]] = []
    total = float(q.size)
    acc = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (q >= lo) & (q < hi) if i < n_bins - 1 else (q >= lo) & (q <= hi)
        n = int(mask.sum())
        if n == 0:
            table.append(
                {
                    "bin": i,
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": 0,
                    "mean_q1": float("nan"),
                    "emp_acc": float("nan"),
                    "abs_gap": float("nan"),
                }
            )
            continue
        mean_q = float(q[mask].mean())
        emp = float(y[mask].mean())
        gap = abs(mean_q - emp)
        acc += (n / total) * gap
        table.append(
            {
                "bin": i,
                "lo": float(lo),
                "hi": float(hi),
                "n": n,
                "mean_q1": mean_q,
                "emp_acc": emp,
                "abs_gap": gap,
            }
        )
    return acc, table


def fmt_pp(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}"


def fmt_ci(ci: CI, scale: float = 100.0, digits: int = 1) -> str:
    return f"{scale * ci.point:.{digits}f} [{scale * ci.lo:.{digits}f}, {scale * ci.hi:.{digits}f}]"
