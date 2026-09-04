"""Deterministic question-level and paired bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    seed: int
    n_questions: int
    n_valid_resamples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: Iterable[Any]) -> float:
    usable = [number for value in values if (number := _finite(value)) is not None]
    return sum(usable) / len(usable) if usable else float("nan")


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def percentile_interval(
    values: Iterable[Any], confidence_level: float = 0.95
) -> tuple[float, float]:
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    usable = sorted(
        number for value in values if (number := _finite(value)) is not None
    )
    alpha = (1.0 - confidence_level) / 2.0
    return _quantile(usable, alpha), _quantile(usable, 1.0 - alpha)


def question_level_bootstrap(
    records: Iterable[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    question_key: str | Callable[[T], Any] = "example_id",
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 2025,
) -> BootstrapResult:
    """Resample question clusters, preserving all rows belonging to a question."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    def get_key(record: T) -> Any:
        if callable(question_key):
            return question_key(record)
        if not isinstance(record, Mapping):
            raise TypeError("String question_key requires mapping records")
        return record.get(question_key)

    rows = list(records)
    clusters: dict[Any, list[T]] = defaultdict(list)
    for row in rows:
        identifier = get_key(row)
        if identifier is not None:
            clusters[identifier].append(row)
    identifiers = sorted(clusters, key=repr)
    estimate = _finite(statistic(rows))
    if not identifiers:
        return BootstrapResult(
            estimate if estimate is not None else float("nan"),
            float("nan"),
            float("nan"),
            confidence_level,
            n_resamples,
            seed,
            0,
            0,
        )

    rng = random.Random(seed)
    sampled_statistics: list[float] = []
    for _ in range(n_resamples):
        sampled_rows: list[T] = []
        for identifier in rng.choices(identifiers, k=len(identifiers)):
            sampled_rows.extend(clusters[identifier])
        value = _finite(statistic(sampled_rows))
        if value is not None:
            sampled_statistics.append(value)
    lower, upper = percentile_interval(sampled_statistics, confidence_level)
    return BootstrapResult(
        estimate if estimate is not None else float("nan"),
        lower,
        upper,
        confidence_level,
        n_resamples,
        seed,
        len(identifiers),
        len(sampled_statistics),
    )


def bootstrap_mean(
    records: Iterable[Mapping[str, Any]],
    value_key: str,
    *,
    question_key: str = "example_id",
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 2025,
) -> BootstrapResult:
    """Question-level CI for a row-level mean."""
    return question_level_bootstrap(
        records,
        lambda sample: mean(row.get(value_key) for row in sample),
        question_key=question_key,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )


def paired_bootstrap(
    pairs: Iterable[tuple[Any, Any, Any]],
    *,
    difference: Callable[[Any, Any], float] | None = None,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 2025,
) -> BootstrapResult:
    """Bootstrap paired differences supplied as (question_id, left, right).

    Multiple observations for a question are averaged within that question
    before resampling, so the independent sampling unit remains the question.
    The estimate is ``left - right``.
    """
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if difference is None:
        difference = lambda left, right: float(left) - float(right)
    by_question: dict[Any, list[float]] = defaultdict(list)
    for identifier, left, right in pairs:
        if identifier is None:
            continue
        try:
            value = _finite(difference(left, right))
        except (TypeError, ValueError):
            value = None
        if value is not None:
            by_question[identifier].append(value)
    question_differences = {
        identifier: mean(values) for identifier, values in by_question.items()
    }
    identifiers = sorted(question_differences, key=repr)
    estimate = mean(question_differences.values())
    if not identifiers:
        return BootstrapResult(
            estimate,
            float("nan"),
            float("nan"),
            confidence_level,
            n_resamples,
            seed,
            0,
            0,
        )

    rng = random.Random(seed)
    replicates = [
        mean(
            question_differences[identifier]
            for identifier in rng.choices(identifiers, k=len(identifiers))
        )
        for _ in range(n_resamples)
    ]
    lower, upper = percentile_interval(replicates, confidence_level)
    return BootstrapResult(
        estimate,
        lower,
        upper,
        confidence_level,
        n_resamples,
        seed,
        len(identifiers),
        len(replicates),
    )


def paired_policy_bootstrap(
    records: Iterable[Mapping[str, Any]],
    *,
    left_policy: str,
    right_policy: str,
    value_key: str = "realized_cost",
    policy_key: str = "policy",
    question_key: str = "example_id",
    stratum_keys: Sequence[str] = ("model_id", "error_cost"),
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 2025,
) -> BootstrapResult:
    """Pair policy outcomes on question and requested strata, then bootstrap."""
    matched: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(dict)
    for row in records:
        policy = row.get(policy_key)
        if policy not in {left_policy, right_policy}:
            continue
        key = (row.get(question_key), *(row.get(name) for name in stratum_keys))
        matched[key][str(policy)] = row.get(value_key)
    pairs = [
        (key[0], values[left_policy], values[right_policy])
        for key, values in matched.items()
        if left_policy in values and right_policy in values
    ]
    return paired_bootstrap(
        pairs,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
