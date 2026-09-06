"""Leakage-safe deterministic calibration utilities.

The implementation is dependency-light: isotonic regression is fitted with
the pool-adjacent-violators algorithm and predicts with clipped linear
interpolation, including constant and empty-calibration-set fallbacks.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _unit_interval(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return number


def _stable_key(seed: int, model_id: str, example_id: str) -> str:
    material = f"{int(seed)}\0{model_id}\0{example_id}".encode()
    return hashlib.sha256(material).hexdigest()


def deterministic_model_splits(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int = 2025,
    calibration_fraction: float = 0.20,
    model_key: str = "model_id",
    id_key: str = "example_id",
) -> dict[str, dict[str, Any]]:
    """Create an exact, deterministic per-model calibration/evaluation split.

    For two or more IDs, both partitions are nonempty. A singleton is retained
    for evaluation and therefore never used to fit a calibrator.
    """
    fraction = _unit_interval(calibration_fraction, "calibration_fraction")
    ids_by_model: dict[str, set[str]] = defaultdict(set)
    for record in records:
        model = record.get(model_key)
        example = record.get(id_key)
        if model is not None and example is not None:
            ids_by_model[str(model)].add(str(example))

    output: dict[str, dict[str, Any]] = {}
    for model, identifiers in sorted(ids_by_model.items()):
        ranked = sorted(
            identifiers, key=lambda item: (_stable_key(seed, model, item), item)
        )
        n = len(ranked)
        if n < 2 or fraction == 0:
            n_calibration = 0
        elif fraction == 1:
            n_calibration = n - 1
        else:
            n_calibration = min(n - 1, max(1, round(n * fraction)))
        calibration_ids = sorted(ranked[:n_calibration])
        evaluation_ids = sorted(ranked[n_calibration:])
        output[model] = {
            "model_id": model,
            "seed": int(seed),
            "calibration_fraction_requested": fraction,
            "calibration_fraction_realized": (n_calibration / n if n else float("nan")),
            "calibration_ids": calibration_ids,
            "evaluation_ids": evaluation_ids,
            "n_total": n,
            "n_calibration": n_calibration,
            "n_evaluation": n - n_calibration,
        }
    return output


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Serializable monotone probability calibrator."""

    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]
    n_samples: int
    fallback: str | None = None

    def predict_one(self, probability: float) -> float:
        q = _unit_interval(probability, "probability")
        if not self.x_thresholds:
            return q
        if len(self.x_thresholds) == 1:
            return self.y_thresholds[0]
        if q <= self.x_thresholds[0]:
            return self.y_thresholds[0]
        if q >= self.x_thresholds[-1]:
            return self.y_thresholds[-1]
        upper = bisect.bisect_right(self.x_thresholds, q)
        lower = upper - 1
        x0, x1 = self.x_thresholds[lower], self.x_thresholds[upper]
        y0, y1 = self.y_thresholds[lower], self.y_thresholds[upper]
        if x1 == x0:
            return y1
        weight = (q - x0) / (x1 - x0)
        return min(1.0, max(0.0, y0 + weight * (y1 - y0)))

    def predict(self, probabilities: Iterable[float]) -> list[float]:
        return [self.predict_one(value) for value in probabilities]

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata["x_thresholds"] = list(self.x_thresholds)
        metadata["y_thresholds"] = list(self.y_thresholds)
        metadata["method"] = "isotonic_pava_clipped_linear_interpolation"
        return metadata

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> IsotonicCalibrator:
        return cls(
            tuple(float(value) for value in metadata.get("x_thresholds", ())),
            tuple(float(value) for value in metadata.get("y_thresholds", ())),
            int(metadata.get("n_samples", 0)),
            metadata.get("fallback"),
        )


def fit_isotonic(
    probabilities: Sequence[float],
    outcomes: Sequence[bool | int | float],
) -> IsotonicCalibrator:
    """Fit isotonic regression using only the explicitly supplied samples."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal lengths")
    if not probabilities:
        return IsotonicCalibrator((), (), 0, "identity_no_calibration_samples")

    samples = sorted(
        (
            _unit_interval(probability, "probability"),
            _unit_interval(outcome, "outcome"),
        )
        for probability, outcome in zip(probabilities, outcomes)
    )

    # Collapse tied x values before PAVA so their labels receive equal weight.
    grouped: list[tuple[float, float, int]] = []
    for x_value, outcome in samples:
        if grouped and grouped[-1][0] == x_value:
            x, total, count = grouped[-1]
            grouped[-1] = (x, total + outcome, count + 1)
        else:
            grouped.append((x_value, outcome, 1))

    # Blocks are [start_index, end_index, weighted_sum, weight].
    blocks: list[list[float | int]] = []
    for index, (_, total, count) in enumerate(grouped):
        blocks.append([index, index, total, count])
        while len(blocks) >= 2:
            previous, current = blocks[-2], blocks[-1]
            previous_mean = float(previous[2]) / int(previous[3])
            current_mean = float(current[2]) / int(current[3])
            if previous_mean <= current_mean:
                break
            blocks[-2:] = [
                [
                    int(previous[0]),
                    int(current[1]),
                    float(previous[2]) + float(current[2]),
                    int(previous[3]) + int(current[3]),
                ]
            ]

    fitted = [0.0] * len(grouped)
    for start, end, total, count in blocks:
        mean = float(total) / int(count)
        for index in range(int(start), int(end) + 1):
            fitted[index] = mean

    x_thresholds = tuple(item[0] for item in grouped)
    y_thresholds = tuple(fitted)
    fallback = "constant_single_x" if len(x_thresholds) == 1 else None
    return IsotonicCalibrator(x_thresholds, y_thresholds, len(probabilities), fallback)


def fit_per_model_calibrators(
    records: Iterable[Mapping[str, Any]],
    splits: Mapping[str, Mapping[str, Any]],
    *,
    model_key: str = "model_id",
    id_key: str = "example_id",
    probability_key: str = "probability_correct",
    outcome_key: str = "is_correct",
) -> tuple[dict[str, IsotonicCalibrator], dict[str, dict[str, Any]]]:
    """Fit each model solely on IDs assigned to its calibration partition."""
    rows = list(records)
    calibrators: dict[str, IsotonicCalibrator] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for model, split in sorted(splits.items()):
        allowed = {str(value) for value in split.get("calibration_ids", ())}
        probabilities: list[float] = []
        outcomes: list[float] = []
        used_ids: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if str(row.get(model_key)) != model:
                continue
            identifier = str(row.get(id_key))
            if identifier not in allowed or identifier in seen:
                continue
            try:
                probabilities.append(
                    _unit_interval(row.get(probability_key), probability_key)
                )
                outcomes.append(_unit_interval(row.get(outcome_key), outcome_key))
            except ValueError:
                continue
            used_ids.append(identifier)
            seen.add(identifier)
        calibrator = fit_isotonic(probabilities, outcomes)
        calibrators[model] = calibrator
        model_metadata = calibrator.to_metadata()
        model_metadata.update(
            {
                "model_id": model,
                "fit_ids": sorted(used_ids),
                "fit_partition": "calibration",
                "evaluation_labels_used": False,
            }
        )
        metadata[model] = model_metadata
    return calibrators, metadata


def assign_calibrated_probabilities(
    records: Iterable[Mapping[str, Any]],
    splits: Mapping[str, Mapping[str, Any]],
    calibrators: Mapping[str, IsotonicCalibrator],
    *,
    evaluation_only: bool = True,
    model_key: str = "model_id",
    id_key: str = "example_id",
    probability_key: str = "probability_correct",
) -> list[dict[str, Any]]:
    """Return copies with partition and calibrated probabilities attached."""
    output: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        model = str(row.get(model_key))
        identifier = str(row.get(id_key))
        split = splits.get(model, {})
        calibration_ids = {str(value) for value in split.get("calibration_ids", ())}
        evaluation_ids = {str(value) for value in split.get("evaluation_ids", ())}
        if identifier in calibration_ids:
            partition = "calibration"
        elif identifier in evaluation_ids:
            partition = "evaluation"
        else:
            partition = "unassigned"
        row["calibration_partition"] = partition
        row["calibrated_probability"] = None
        calibrator = calibrators.get(model)
        if (
            (not evaluation_only or partition == "evaluation")
            and calibrator is not None
            and calibrator.n_samples > 0
        ):
            try:
                row["calibrated_probability"] = calibrator.predict_one(
                    row.get(probability_key)
                )
            except ValueError:
                pass
        output.append(row)
    return output


def persist_calibration_artifacts(
    output_directory: str | Path,
    splits: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, Path]:
    """Persist split IDs and fitted metadata as stable JSON artifacts."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    split_path = directory / "split_ids.json"
    metadata_path = directory / "calibrator_metadata.json"
    split_path.write_text(
        json.dumps(splits, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return split_path, metadata_path


def brier_score(
    probabilities: Iterable[float], outcomes: Iterable[bool | int | float]
) -> float:
    pairs = [
        (
            _unit_interval(probability, "probability"),
            _unit_interval(outcome, "outcome"),
        )
        for probability, outcome in zip(probabilities, outcomes)
    ]
    if not pairs:
        return float("nan")
    return sum((probability - outcome) ** 2 for probability, outcome in pairs) / len(
        pairs
    )


def expected_calibration_error(
    probabilities: Iterable[float],
    outcomes: Iterable[bool | int | float],
    *,
    n_bins: int = 10,
) -> float:
    """Fixed-width ECE; empty bins contribute zero."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for probability, outcome in zip(probabilities, outcomes):
        q = _unit_interval(probability, "probability")
        y = _unit_interval(outcome, "outcome")
        index = min(n_bins - 1, int(q * n_bins))
        bins[index].append((q, y))
    total = sum(len(items) for items in bins)
    if total == 0:
        return float("nan")
    ece = 0.0
    for items in bins:
        if not items:
            continue
        mean_probability = sum(item[0] for item in items) / len(items)
        mean_outcome = sum(item[1] for item in items) / len(items)
        ece += len(items) / total * abs(mean_probability - mean_outcome)
    return ece
