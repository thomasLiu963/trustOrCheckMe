import pytest

from src.scoring import monotonicity


@pytest.mark.parametrize(
    "actions",
    [
        ["RELY", "RELY", "RELY", "RELY"],
        ["RELY", "RELY", "VERIFY", "VERIFY"],
        ["RELY", "VERIFY", "VERIFY", "VERIFY"],
        ["VERIFY", "VERIFY", "VERIFY", "VERIFY"],
    ],
)
def test_monotone_sequences(actions: list[str]) -> None:
    result = monotonicity(actions, [2, 5, 10, 20])
    assert result["any_violation"] is False
    assert result["adjacent_verify_to_rely_count"] == 0


@pytest.mark.parametrize(
    ("actions", "reversals"),
    [
        (["VERIFY", "RELY", "VERIFY", "VERIFY"], 1),
        (["RELY", "VERIFY", "RELY", "VERIFY"], 1),
        (["VERIFY", "RELY", "VERIFY", "RELY"], 2),
    ],
)
def test_violating_sequences(actions: list[str], reversals: int) -> None:
    result = monotonicity(actions, [2, 5, 10, 20])
    assert result["any_violation"] is True
    assert result["adjacent_verify_to_rely_count"] == reversals


def test_orders_by_loss() -> None:
    result = monotonicity(["VERIFY", "RELY", "VERIFY", "RELY"], [10, 2, 20, 5])
    assert result["actions"] == ["RELY", "RELY", "VERIFY", "VERIFY"]
    assert result["any_violation"] is False
