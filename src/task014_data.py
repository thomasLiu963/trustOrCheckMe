"""Load frozen 009/011 items for Task 014. Read-only on historical artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .checkpointing import CheckpointStore
from .schemas import BenchmarkExample
from .task006_prompts import build_qualitative_prompt
from .task011_common import DATASET_NAME, SQLITE_PATH as TASK011_SQLITE
from .task011_prompts import build_stage3_prompt
from .task014_common import (
    EXAMPLES_009,
    GPT_CLAUDE,
    PROBLEMS_011,
    TASK009_DIR,
    TASK011_DIR,
    load_csv,
)
from .task014_transform import transform_q


@dataclass(frozen=True)
class FrozenItem:
    task: str
    question_id: str
    model_alias: str
    q1: float
    incorrect: bool
    frozen_output: str
    question: str
    choices: dict[str, str] | None
    true_q_prompt: str
    true_q_action: str
    true_q_verify: bool


def _009_examples() -> dict[str, BenchmarkExample]:
    out: dict[str, BenchmarkExample] = {}
    for line in EXAMPLES_009.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ex = BenchmarkExample.model_validate_json(line)
            out[ex.example_id] = ex
    return out


def _011_problems(ids: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with PROBLEMS_011.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = str(item["question_id"])
            if qid in ids and qid not in out:
                out[qid] = {
                    "question_id": qid,
                    "question_content": item.get("question_content") or "",
                    "starter_code": item.get("starter_code") or "",
                }
            if len(out) == len(ids):
                break
    missing = ids - set(out)
    if missing:
        raise RuntimeError(f"missing 011 problems: {sorted(missing)[:8]}")
    return out


def _011_code(ids: set[str]) -> dict[tuple[str, str], str]:
    store = CheckpointStore(TASK011_SQLITE)
    out: dict[tuple[str, str], str] = {}
    for qid in ids:
        for alias in GPT_CLAUDE:
            rec = store.find_success(
                stage="code",
                dataset=DATASET_NAME,
                example_id=qid,
                model_alias=alias,
            )
            if rec is None:
                raise RuntimeError(f"missing frozen code {qid} {alias}")
            out[(qid, alias)] = str(rec.get("frozen_code") or "")
    return out


def _true_q_actions(task: str) -> dict[tuple[str, str], tuple[str, bool]]:
    path = (TASK009_DIR if task == "mmlu" else TASK011_DIR) / "stage3_results.csv"
    out: dict[tuple[str, str], tuple[str, bool]] = {}
    for row in load_csv(path):
        if row.get("score_condition") != "true_q_visible":
            continue
        if int(row.get("repeat_index") or 0) != 0:
            continue
        if task == "mmlu" and row.get("roster") not in {None, "", "primary"}:
            continue
        out[(row["question_id"], row["model_alias"])] = (
            str(row.get("parsed_action") or ""),
            str(row.get("verify")).strip().lower() in {"1", "true", "yes"},
        )
    return out


def load_frozen_items() -> list[FrozenItem]:
    examples = _009_examples()
    q1_009 = {(r["question_id"], r["model_alias"]): r for r in load_csv(TASK009_DIR / "stage1_q1_results.csv")}
    actions_009 = _true_q_actions("mmlu")
    items: list[FrozenItem] = []
    for (qid, alias), meta in q1_009.items():
        if alias not in GPT_CLAUDE:
            continue
        if qid not in examples:
            continue
        ex = examples[qid]
        q1 = float(meta["q1"])
        prompt = build_qualitative_prompt(
            question=ex.question,
            choices=ex.choices,
            frozen_answer=str(meta["frozen_answer"]),
            family="moderate",
            displayed_confidence=q1,
        )
        action, verify = actions_009[(qid, alias)]
        items.append(
            FrozenItem(
                task="mmlu",
                question_id=qid,
                model_alias=alias,
                q1=q1,
                incorrect=not str(meta["stage1_correct"]).strip() in {"1", "true", "True"},
                frozen_output=str(meta["frozen_answer"]),
                question=ex.question,
                choices=dict(ex.choices),
                true_q_prompt=prompt,
                true_q_action=action,
                true_q_verify=verify,
            )
        )

    code_ids = {r["question_id"] for r in load_csv(TASK011_DIR / "sample_main.csv")}
    problems = _011_problems(code_ids)
    codes = _011_code(code_ids)
    q1_011 = {(r["question_id"], r["model_alias"]): r for r in load_csv(TASK011_DIR / "q1_results.csv")}
    hidden = {(r["question_id"], r["model_alias"]): r for r in load_csv(TASK011_DIR / "hidden_test_results.csv")}
    actions_011 = _true_q_actions("code")
    for qid in sorted(code_ids):
        for alias in GPT_CLAUDE:
            meta = q1_011[(qid, alias)]
            q1 = float(meta.get("probability_correct") or meta.get("q1"))
            code = codes[(qid, alias)]
            prompt = build_stage3_prompt(
                question=problems[qid]["question_content"],
                frozen_code=code,
                displayed_confidence=q1,
            )
            action, verify = actions_011[(qid, alias)]
            passed = str(hidden[(qid, alias)].get("passed")).strip() in {"1", "true", "True"}
            items.append(
                FrozenItem(
                    task="code",
                    question_id=qid,
                    model_alias=alias,
                    q1=q1,
                    incorrect=not passed,
                    frozen_output=code,
                    question=problems[qid]["question_content"],
                    choices=None,
                    true_q_prompt=prompt,
                    true_q_action=action,
                    true_q_verify=verify,
                )
            )
    return items


def offset_prompt(item: FrozenItem, delta: float) -> str:
    displayed = transform_q(item.q1, delta)
    if item.task == "mmlu":
        assert item.choices is not None
        return build_qualitative_prompt(
            question=item.question,
            choices=item.choices,
            frozen_answer=item.frozen_output,
            family="moderate",
            displayed_confidence=displayed,
        )
    return build_stage3_prompt(
        question=item.question,
        frozen_code=item.frozen_output,
        displayed_confidence=displayed,
    )
