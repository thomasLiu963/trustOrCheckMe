import pytest

from src.scoring import (
    RELY,
    VERIFY,
    confidence_policy_action,
    oracle_cost,
    realized_cost,
    regret,
    score_decision,
)


@pytest.mark.parametrize(
    ("q", "loss", "expected"),
    [
        (0.72, 10, VERIFY),
        (0.95, 10, RELY),
        (0.50, 2, RELY),
        (0.49, 2, VERIFY),
    ],
)
def test_cost_policy_boundary(q: float, loss: float, expected: str) -> None:
    assert confidence_policy_action(q, loss, 1) == expected


@pytest.mark.parametrize(
    ("correct", "action", "loss", "cost", "oracle", "expected_regret"),
    [
        (True, RELY, 10, 0, 0, 0),
        (True, VERIFY, 10, 1, 0, 1),
        (False, RELY, 10, 10, 1, 9),
        (False, VERIFY, 10, 1, 1, 0),
    ],
)
def test_all_decision_outcomes(
    correct: bool,
    action: str,
    loss: float,
    cost: float,
    oracle: float,
    expected_regret: float,
) -> None:
    assert realized_cost(correct, action, loss) == cost
    assert oracle_cost(correct) == oracle
    assert regret(correct, action, loss) == expected_regret


def test_directional_disagreement() -> None:
    row = score_decision(
        is_correct=False,
        action=RELY,
        error_cost=10,
        raw_probability=0.72,
        calibrated_probability=0.96,
    )
    assert row["direct_vs_raw_confidence_direction"] == "more_trusting"
    assert row["direct_vs_calibrated_confidence_direction"] == "agree"
