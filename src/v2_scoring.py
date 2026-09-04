"""Pure scoring and factor validation for V2 verification decisions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import Any

USE_UNVERIFIED = "USE_UNVERIFIED"
VERIFY_FIRST = "VERIFY_FIRST"
VALID_V2_ACTIONS = frozenset({USE_UNVERIFIED, VERIFY_FIRST})
V2_ERROR_COSTS = (2.0, 5.0, 10.0, 20.0)


def normalize_v2_action(action: Any) -> str:
    value = getattr(action, "value", action)
    if not isinstance(value, str):
        raise TypeError("V2 action must be a string")
    normalized = value.strip().upper()
    if normalized not in VALID_V2_ACTIONS:
        raise ValueError(
            f"Invalid V2 action {action!r}; expected USE_UNVERIFIED or VERIFY_FIRST"
        )
    return normalized


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def confidence_policy_action(
    probability_correct: float,
    error_cost: float,
    verification_cost: float = 1.0,
) -> str:
    q = _finite(probability_correct, "probability_correct")
    loss = _finite(error_cost, "error_cost")
    cost = _finite(verification_cost, "verification_cost")
    if not 0 <= q <= 1 or loss < 0 or cost < 0:
        raise ValueError("probability and costs are outside valid ranges")
    return VERIFY_FIRST if (1 - q) * loss > cost else USE_UNVERIFIED


def score_verification_decision(
    *,
    is_correct: bool,
    action: Any,
    probability_correct: float,
    error_cost: float,
    verification_cost: float = 1.0,
    calibrated_probability: float | None = None,
) -> dict[str, Any]:
    normalized = normalize_v2_action(action)
    verified = normalized == VERIFY_FIRST
    realized = (
        verification_cost
        if verified
        else (0.0 if bool(is_correct) else float(error_cost))
    )
    oracle = 0.0 if bool(is_correct) else float(verification_cost)
    raw_action = confidence_policy_action(
        probability_correct, error_cost, verification_cost
    )
    calibrated_action = confidence_policy_action(
        (
            probability_correct
            if calibrated_probability is None
            else calibrated_probability
        ),
        error_cost,
        verification_cost,
    )
    return {
        "action": normalized,
        "used_unverified": not verified,
        "verified_first": verified,
        "verify": verified,
        "unsafe_unverified_use": not verified and not bool(is_correct),
        "unnecessary_verification": verified and bool(is_correct),
        "realized_cost": realized,
        "oracle_cost": oracle,
        "regret": realized - oracle,
        "raw_confidence_baseline_action": raw_action,
        "calibrated_confidence_baseline_action": calibrated_action,
        "direct_vs_raw_confidence_disagreement": normalized != raw_action,
        "direct_vs_calibrated_confidence_disagreement": (
            normalized != calibrated_action
        ),
    }


def validate_factor_completeness(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_error_costs: Sequence[float] = V2_ERROR_COSTS,
) -> list[dict[str, Any]]:
    """Return incomplete/duplicate primary-factor groups."""
    grouped: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        key = (row.get("example_id"), row.get("model_id"), row.get("prompt_family"))
        grouped[key].append(row)
    expected = {
        (owner, visibility, float(cost))
        for owner in ("human", "ai_system")
        for visibility in ("hidden", "visible")
        for cost in expected_error_costs
    }
    issues: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        observed = [
            (
                row.get("decision_owner"),
                row.get("confidence_visibility"),
                float(row.get("error_cost")),
            )
            for row in rows
        ]
        observed_set = set(observed)
        if observed_set != expected or len(observed) != len(expected):
            issues.append(
                {
                    "example_id": key[0],
                    "model_id": key[1],
                    "prompt_family": key[2],
                    "observed_count": len(observed),
                    "missing_cells": sorted(expected - observed_set),
                    "duplicate_count": len(observed) - len(observed_set),
                }
            )
    return issues


def v2_monotonicity(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Score monotonicity within every matched owner/confidence sequence."""
    keys = (
        "example_id",
        "model_id",
        "decision_owner",
        "confidence_visibility",
        "prompt_family",
    )
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        by_cost = {
            float(row["error_cost"]): normalize_v2_action(row["action"])
            for row in rows
        }
        costs = sorted(by_cost)
        values = [by_cost[cost] == VERIFY_FIRST for cost in costs]
        reversals = sum(left and not right for left, right in pairwise(values))
        output.append(
            {
                **dict(zip(keys, key)),
                "observed_stakes": costs,
                "complete": tuple(costs) == V2_ERROR_COSTS,
                "any_violation": bool(reversals),
                "adjacent_verify_to_use_count": reversals,
            }
        )
    return output
