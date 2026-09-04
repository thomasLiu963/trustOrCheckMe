from src.calibration import (
    deterministic_model_splits,
    fit_per_model_calibrators,
)


def _records() -> list[dict]:
    return [
        {
            "model_id": model,
            "example_id": f"q-{index}",
            "probability_correct": index / 99,
            "is_correct": index >= 50,
        }
        for model in ("a", "b")
        for index in range(100)
    ]


def test_split_is_deterministic_disjoint_and_per_model() -> None:
    records = _records()
    first = deterministic_model_splits(
        records, seed=20260904, calibration_fraction=0.20
    )
    second = deterministic_model_splits(
        records, seed=20260904, calibration_fraction=0.20
    )
    assert first == second
    for split in first.values():
        calibration = set(split["calibration_ids"])
        evaluation = set(split["evaluation_ids"])
        assert len(calibration) == 20
        assert len(evaluation) == 80
        assert calibration.isdisjoint(evaluation)


def test_calibrator_never_uses_evaluation_labels() -> None:
    records = _records()
    splits = deterministic_model_splits(
        records, seed=20260904, calibration_fraction=0.20
    )
    _, metadata = fit_per_model_calibrators(records, splits)
    for model, details in metadata.items():
        assert set(details["fit_ids"]) == set(splits[model]["calibration_ids"])
        assert not set(details["fit_ids"]) & set(splits[model]["evaluation_ids"])
        assert details["evaluation_labels_used"] is False
