"""Coverage, leakage, escape, loss, and item-clustered bootstrap."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float
    n_boot: int


def coverage(rows: Sequence[Any]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([bool(r.verify) for r in rows]))


def leakage(rows: Sequence[Any]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([bool(r.incorrect) and not bool(r.verify) for r in rows]))


def error_rate(rows: Sequence[Any]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([bool(r.incorrect) for r in rows]))


def escape_rate(rows: Sequence[Any]) -> float:
    bad = [r for r in rows if r.incorrect]
    if not bad:
        return float("nan")
    return float(np.mean([not bool(r.verify) for r in bad]))


def precision(rows: Sequence[Any]) -> float:
    verified = [r for r in rows if r.verify]
    if not verified:
        return float("nan")
    return float(np.mean([bool(r.incorrect) for r in verified]))


def loss(leak: float, cov: float, lam: float) -> float:
    return leak + lam * cov


def _by_question(rows: Sequence[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for row in rows:
        out.setdefault(row.question_id, []).append(row)
    return out


def bootstrap_stat(
    rows: Sequence[Any],
    stat: Callable[[list[Any]], float],
    *,
    seed: int,
    n_boot: int,
) -> CI:
    grouped = _by_question(rows)
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
    rows_a: Sequence[Any],
    rows_b: Sequence[Any],
    stat: Callable[[list[Any]], float],
    *,
    seed: int,
    n_boot: int,
) -> CI:
    ga = _by_question(rows_a)
    gb = _by_question(rows_b)
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
