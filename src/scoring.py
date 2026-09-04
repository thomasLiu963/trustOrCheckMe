"""Pure decision rules and scoring for the trust-or-verify experiment."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import Any

VERIFY = "VERIFY"
RELY = "RELY"
VALID_ACTIONS = frozenset({RELY, VERIFY})
VERIFICATION_COST = 1.0
ERROR_COSTS = (2.0, 5.0, 10.0, 20.0)


def normalize_action(action: Any) -> str:
    """Return a canonical action, rejecting values outside RELY/VERIFY."""
    if not isinstance(action, str):
        raise TypeError(f"Action must be a string, got {type(action).__name__}")
    normalized = action.strip().upper()
    if normalized not in VALID_ACTIONS:
        raise ValueError(f"Invalid action {action!r}; expected RELY or VERIFY")
    return normalized


def _probability(value: Any, name: str = "q") -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in [0, 1]") from exc
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return probability


def _nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def confidence_policy_action(
    q: float,
    error_cost: float,
    verification_cost: float = VERIFICATION_COST,
) -> str:
    """Apply the frozen baseline, with equality assigned to RELY.

    VERIFY iff ``(1 - q) * L > C``. This exact strict inequality makes the
    boundary deterministic and consistent across raw and calibrated policies.
    """
    probability = _probability(q)
    loss = _nonnegative(error_cost, "error_cost")
    cost = _nonnegative(verification_cost, "verification_cost")
    return VERIFY if (1.0 - probability) * loss > cost else RELY


def raw_confidence_action(
    q: float,
    error_cost: float,
    verification_cost: float = VERIFICATION_COST,
) -> str:
    return confidence_policy_action(q, error_cost, verification_cost)


def calibrated_confidence_action(
    calibrated_q: float,
    error_cost: float,
    verification_cost: float = VERIFICATION_COST,
) -> str:
    return confidence_policy_action(calibrated_q, error_cost, verification_cost)


def oracle_action(is_correct: bool) -> str:
    """Return the non-deployable action chosen with outcome knowledge."""
    return RELY if bool(is_correct) else VERIFY


def realized_cost(
    is_correct: bool,
    action: str,
    error_cost: float,
    verification_cost: float = VERIFICATION_COST,
) -> float:
    """Score one correct/wrong × RELY/VERIFY decision."""
    normalized = normalize_action(action)
    loss = _nonnegative(error_cost, "error_cost")
    cost = _nonnegative(verification_cost, "verification_cost")
    if normalized == VERIFY:
        return cost
    return 0.0 if bool(is_correct) else loss


def oracle_cost(
    is_correct: bool,
    verification_cost: float = VERIFICATION_COST,
) -> float:
    return (
        0.0
        if bool(is_correct)
        else _nonnegative(verification_cost, "verification_cost")
    )


def regret(
    is_correct: bool,
    action: str,
    error_cost: float,
    verification_cost: float = VERIFICATION_COST,
) -> float:
    return realized_cost(
        is_correct, action, error_cost, verification_cost
    ) - oracle_cost(is_correct, verification_cost)


def unsafe_reliance(is_correct: bool, action: str) -> bool:
    return not bool(is_correct) and normalize_action(action) == RELY


def unnecessary_verification(is_correct: bool, action: str) -> bool:
    return bool(is_correct) and normalize_action(action) == VERIFY


def disagreement_direction(direct_action: str, baseline_action: str) -> str:
    """Classify direct advice relative to a confidence-derived baseline."""
    direct = normalize_action(direct_action)
    baseline = normalize_action(baseline_action)
    if direct == baseline:
        return "agree"
    if direct == RELY:
        return "more_trusting"
    return "more_cautious"


def score_decision(
    *,
    is_correct: bool,
    action: str,
    error_cost: float,
    verification_cost: float = VERIFICATION_COST,
    raw_probability: float | None = None,
    calibrated_probability: float | None = None,
) -> dict[str, Any]:
    """Return all pure per-decision outcomes and available disagreements."""
    normalized = normalize_action(action)
    result: dict[str, Any] = {
        "action": normalized,
        "is_correct": bool(is_correct),
        "verify": normalized == VERIFY,
        "unsafe_reliance": unsafe_reliance(is_correct, normalized),
        "unnecessary_verification": unnecessary_verification(is_correct, normalized),
        "realized_cost": realized_cost(
            is_correct, normalized, error_cost, verification_cost
        ),
        "oracle_action": oracle_action(is_correct),
        "oracle_cost": oracle_cost(is_correct, verification_cost),
    }
    result["regret"] = result["realized_cost"] - result["oracle_cost"]

    for prefix, probability in (
        ("raw", raw_probability),
        ("calibrated", calibrated_probability),
    ):
        if probability is None:
            result[f"{prefix}_confidence_baseline_action"] = None
            result[f"direct_vs_{prefix}_confidence_disagreement"] = None
            result[f"direct_vs_{prefix}_confidence_direction"] = None
            continue
        baseline = confidence_policy_action(probability, error_cost, verification_cost)
        direction = disagreement_direction(normalized, baseline)
        result[f"{prefix}_confidence_baseline_action"] = baseline
        result[f"direct_vs_{prefix}_confidence_disagreement"] = direction != "agree"
        result[f"direct_vs_{prefix}_confidence_direction"] = direction
    # Preserve the schema's historical unqualified name for the raw policy.
    result["confidence_baseline_action"] = result["raw_confidence_baseline_action"]
    return result


def score_policy(
    *,
    is_correct: bool,
    action: str,
    error_cost: float,
    verification_cost: float = VERIFICATION_COST,
) -> dict[str, Any]:
    """Score any direct or baseline policy with the same cost model."""
    normalized = normalize_action(action)
    cost = realized_cost(is_correct, normalized, error_cost, verification_cost)
    optimum = oracle_cost(is_correct, verification_cost)
    return {
        "action": normalized,
        "verify": normalized == VERIFY,
        "unsafe_reliance": unsafe_reliance(is_correct, normalized),
        "unnecessary_verification": unnecessary_verification(is_correct, normalized),
        "realized_cost": cost,
        "oracle_cost": optimum,
        "regret": cost - optimum,
    }


def monotonicity(
    actions: Sequence[str],
    error_costs: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Score one action sequence after ordering it by increasing error cost."""
    if error_costs is not None and len(actions) != len(error_costs):
        raise ValueError("actions and error_costs must have equal lengths")
    if error_costs is None:
        ordered = [normalize_action(action) for action in actions]
        ordered_costs: list[float] | None = None
    else:
        pairs = sorted(
            (
                (_nonnegative(cost, "error_cost"), normalize_action(action))
                for cost, action in zip(error_costs, actions)
            ),
            key=lambda pair: pair[0],
        )
        ordered_costs = [pair[0] for pair in pairs]
        ordered = [pair[1] for pair in pairs]

    adjacent_reversals = sum(
        left == VERIFY and right == RELY for left, right in pairwise(ordered)
    )
    return {
        "actions": ordered,
        "error_costs": ordered_costs,
        "any_violation": adjacent_reversals > 0,
        "adjacent_verify_to_rely_count": adjacent_reversals,
        "complete": len(ordered) > 0,
    }


def monotonicity_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = ("model_id", "example_id"),
    action_key: str = "action",
    error_cost_key: str = "error_cost",
    expected_error_costs: Sequence[float] = ERROR_COSTS,
) -> list[dict[str, Any]]:
    """Compute per-example monotonicity while retaining partial sequences."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(name) for name in group_keys)
        if all(value is not None for value in key):
            grouped.setdefault(key, []).append(record)

    expected = tuple(float(value) for value in expected_error_costs)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: repr(item[0])):
        by_cost: dict[float, str] = {}
        for record in group:
            try:
                cost = _nonnegative(record.get(error_cost_key), error_cost_key)
                by_cost[cost] = normalize_action(record.get(action_key))
            except (TypeError, ValueError):
                continue
        costs = sorted(by_cost)
        result = monotonicity([by_cost[cost] for cost in costs], costs)
        result["observed_stakes"] = costs
        result["missing_stakes"] = [cost for cost in expected if cost not in by_cost]
        result["complete"] = not result["missing_stakes"]
        result.update(dict(zip(group_keys, key)))
        output.append(result)
    return output
