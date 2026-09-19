"""CSV loaders for the public D1 tables."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _b(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class RoutingRow:
    task: str
    question_id: str
    model_alias: str
    score_condition: str
    displayed_confidence: float | None
    q1: float | None
    verify: bool
    incorrect: bool


@dataclass(frozen=True)
class OffsetRow:
    task: str
    question_id: str
    model_alias: str
    delta: float
    q1: float | None
    q_delta: float | None
    verify: bool
    incorrect: bool


def load_routing(path: Path) -> list[RoutingRow]:
    out = []
    for raw in load_csv(path):
        out.append(
            RoutingRow(
                task=str(raw["task"]),
                question_id=str(raw["question_id"]),
                model_alias=str(raw["model_alias"]),
                score_condition=str(raw["score_condition"]),
                displayed_confidence=_f(raw.get("displayed_confidence")),
                q1=_f(raw.get("q1")),
                verify=_b(raw.get("verify")),
                incorrect=_b(raw.get("incorrect")),
            )
        )
    return out


def load_offset(path: Path) -> list[OffsetRow]:
    out = []
    for raw in load_csv(path):
        out.append(
            OffsetRow(
                task=str(raw["task"]),
                question_id=str(raw["question_id"]),
                model_alias=str(raw["model_alias"]),
                delta=float(raw["delta"]),
                q1=_f(raw.get("q1")),
                q_delta=_f(raw.get("q_delta")),
                verify=_b(raw.get("verify")),
                incorrect=_b(raw.get("incorrect")),
            )
        )
    return out


def filter_routing(
    rows: list[RoutingRow],
    *,
    task: str | None = None,
    model: str | None = None,
    condition: str | None = None,
) -> list[RoutingRow]:
    out = rows
    if task is not None:
        out = [r for r in out if r.task == task]
    if model is not None:
        out = [r for r in out if r.model_alias == model]
    if condition is not None:
        out = [r for r in out if r.score_condition == condition]
    return out


def filter_offset(
    rows: list[OffsetRow],
    *,
    task: str | None = None,
    model: str | None = None,
    delta: float | None = None,
) -> list[OffsetRow]:
    out = rows
    if task is not None:
        out = [r for r in out if r.task == task]
    if model is not None:
        out = [r for r in out if r.model_alias == model]
    if delta is not None:
        out = [r for r in out if abs(r.delta - delta) < 1e-12]
    return out
