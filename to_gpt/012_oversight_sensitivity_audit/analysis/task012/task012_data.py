"""Read-only loaders for frozen Task 009 / 011 tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .task012_common import (
    ALL_CONDS,
    FIXED_CONDS,
    GPT_CLAUDE,
    TASK009_DIR,
    TASK011_DIR,
    load_csv,
)


@dataclass(frozen=True)
class ItemRow:
    task: str
    question_id: str
    model_alias: str
    score_condition: str
    displayed_confidence: float | None
    verify: bool
    incorrect: bool
    q1: float | None
    hidden_marker: bool
    true_q_visible: bool


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _b(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_009_rows() -> list[ItemRow]:
    stage3 = load_csv(TASK009_DIR / "stage3_results.csv")
    q1_rows = load_csv(TASK009_DIR / "stage1_q1_results.csv")
    q1_by = {(r["question_id"], r["model_alias"]): _f(r.get("q1")) for r in q1_rows}
    correct_by = {(r["question_id"], r["model_alias"]): _b(r.get("stage1_correct")) for r in q1_rows}
    out: list[ItemRow] = []
    for row in stage3:
        if row.get("roster") != "primary":
            continue
        if int(row.get("repeat_index") or 0) != 0:
            continue
        model = row["model_alias"]
        if model not in GPT_CLAUDE:
            continue
        cond = row["score_condition"]
        if cond not in ALL_CONDS:
            continue
        qid = row["question_id"]
        key = (qid, model)
        correct = correct_by[key]
        out.append(
            ItemRow(
                task="mmlu",
                question_id=qid,
                model_alias=model,
                score_condition=cond,
                displayed_confidence=_f(row.get("displayed_confidence")),
                verify=_b(row.get("verify")),
                incorrect=not correct,
                q1=q1_by.get(key),
                hidden_marker=cond == "hidden",
                true_q_visible=cond == "true_q_visible",
            )
        )
    return out


def load_011_rows() -> list[ItemRow]:
    confirmatory = {r["question_id"] for r in load_csv(TASK011_DIR / "sample_main.csv")}
    stage3 = load_csv(TASK011_DIR / "stage3_results.csv")
    q1_rows = load_csv(TASK011_DIR / "q1_results.csv")
    hidden = load_csv(TASK011_DIR / "hidden_test_results.csv")
    q1_by = {
        (r["question_id"], r["model_alias"]): _f(r.get("probability_correct") or r.get("q1"))
        for r in q1_rows
    }
    passed_by = {(r["question_id"], r["model_alias"]): _b(r.get("passed")) for r in hidden}
    out: list[ItemRow] = []
    for row in stage3:
        if int(row.get("repeat_index") or 0) != 0:
            continue
        qid = row["question_id"]
        if qid not in confirmatory:
            continue
        model = row["model_alias"]
        if model not in GPT_CLAUDE:
            continue
        cond = row["score_condition"]
        if cond not in ALL_CONDS:
            continue
        key = (qid, model)
        passed = passed_by.get(key)
        if passed is None:
            passed = _b(row.get("stage1_correct"))
        out.append(
            ItemRow(
                task="code",
                question_id=qid,
                model_alias=model,
                score_condition=cond,
                displayed_confidence=_f(row.get("displayed_confidence")),
                verify=_b(row.get("verify")),
                incorrect=not passed,
                q1=q1_by.get(key),
                hidden_marker=cond == "hidden",
                true_q_visible=cond == "true_q_visible",
            )
        )
    return out


def all_rows() -> list[ItemRow]:
    return load_009_rows() + load_011_rows()


def filter_rows(
    rows: list[ItemRow],
    *,
    task: str | None = None,
    model: str | None = None,
    condition: str | None = None,
    fixed_only: bool = False,
) -> list[ItemRow]:
    out = rows
    if task is not None:
        out = [r for r in out if r.task == task]
    if model is not None:
        out = [r for r in out if r.model_alias == model]
    if condition is not None:
        out = [r for r in out if r.score_condition == condition]
    if fixed_only:
        out = [r for r in out if r.score_condition in FIXED_CONDS]
    return out
