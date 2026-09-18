"""Task 007 paths, caps, and isolation guards.

Never writes historical V2, Study 1, q2, Task 005/005B/005C/006 data, or paperDirection.txt.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .study1_sample import (
    _load_historical_stage12_on_connection,
    _open_historical_readonly,
    hash_id_list,
    load_frozen_ids,
)
from .study1_schemas import load_study1_config
from .task005_common import Q2_SQLITE, STUDY1_SQLITE, V2_SQLITE, file_fingerprint, sha256_file
from .task006_common import EXPECTED_REPEAT_IDS, QUAL_SQLITE, REPEAT_HASH
from .task006_prompts import PROMPT_FAMILY_MODERATE, PROMPT_FAMILY_STRONGER
from .config import load_models_config

TASK_ID = "007_full_qualitative_four_model_decision_packet"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
LANE_A_DIR = RETURN_DIR / "lane_A_full_qualitative"
LANE_B_DIR = RETURN_DIR / "lane_B_gemini_grok"
LANE_C_DIR = RETURN_DIR / "lane_C_score_vs_difficulty"
LANE_D_DIR = RETURN_DIR / "lane_D_prospective_prep"
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"
TASK001_ID_HASH = "badd6938e5ded12c9dd62733426e1db26d9843bb6a2321a4e4c9eb7e3547fe94"
PROMPT_VERSION = "task006_qualitative_pilot_v1"
PROMPT_HASH_MODERATE = "ec90530c644038dcee86accfc578c501533fdb69d5ea23caeae23dcc1d06dbad"
PROMPT_HASH_STRONGER = "b758935fecdd49ad7a6a0a491d9c71abee35f90981a7030d90ac59a18dca9f3b"

LANE_A_SQLITE = (
    PROJECT_ROOT / "results" / "study1_qualitative_full80" / "qualitative_full80.sqlite3"
)
LANE_B_SQLITE = (
    PROJECT_ROOT
    / "results"
    / "study1_qualitative_gemini_grok20"
    / "qualitative_gemini_grok20.sqlite3"
)
LANE_A_EXPERIMENT = "study1_qualitative_full80"
LANE_B_EXPERIMENT = "study1_qualitative_gemini_grok20"
LANE_A_RUN_ID = "study1-qualitative-007a"
LANE_B_RUN_ID = "study1-qualitative-007b"

LANE_A_CAP = 1280
LANE_B_CAP = 320
TOTAL_CAP = 1600
LANE_A_PROVIDER_CAP = 1664
LANE_B_PROVIDER_CAP = 416
CONCURRENCY = 4
MAX_PARSE_REPAIRS = 1
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 5000

GPT_CLAUDE = ("openai_gpt56_sol", "anthropic_sonnet5")
GEMINI_GROK = ("google_gemini38_flash", "xai_grok420_nonreasoning")
ALL_V2_MODELS = GPT_CLAUDE + GEMINI_GROK
GPT_ENDPOINT = "gpt-5.6-sol"
CLAUDE_ENDPOINT = "claude-sonnet-5"
GEMINI_ENDPOINT = "gemini-3.8-flash"
GROK_ENDPOINT = "grok-4.20-0309-non-reasoning"
OTHER_BY_TARGET = {
    "openai_gpt56_sol": (
        "anthropic_sonnet5",
        "google_gemini38_flash",
        "xai_grok420_nonreasoning",
    ),
    "anthropic_sonnet5": (
        "openai_gpt56_sol",
        "google_gemini38_flash",
        "xai_grok420_nonreasoning",
    ),
    "google_gemini38_flash": (
        "openai_gpt56_sol",
        "anthropic_sonnet5",
        "xai_grok420_nonreasoning",
    ),
    "xai_grok420_nonreasoning": (
        "openai_gpt56_sol",
        "anthropic_sonnet5",
        "google_gemini38_flash",
    ),
}
CORRECT_KEY = {
    "openai_gpt56_sol": "gpt_correct",
    "anthropic_sonnet5": "claude_correct",
    "google_gemini38_flash": "gemini_correct",
    "xai_grok420_nonreasoning": "grok_correct",
}


def remaining_80_ids() -> tuple[list[str], list[str], str, str, str]:
    config = load_study1_config()
    selected, repeats, selected_hash, repeat_hash = load_frozen_ids(config)
    if selected_hash != TASK001_ID_HASH:
        raise RuntimeError(f"Study-1 100 hash {selected_hash} != {TASK001_ID_HASH}")
    if repeat_hash != REPEAT_HASH:
        raise RuntimeError(f"20-q hash {repeat_hash} != {REPEAT_HASH}")
    if tuple(repeats) != EXPECTED_REPEAT_IDS:
        raise RuntimeError("frozen 20 IDs do not match Task 006")
    remaining = [qid for qid in selected if qid not in set(repeats)]
    if len(remaining) != 80:
        raise RuntimeError(f"expected 80 remaining IDs, got {len(remaining)}")
    if set(remaining) & set(repeats):
        raise RuntimeError("remaining 80 overlaps Task-006 20")
    if set(remaining) | set(repeats) != set(selected):
        raise RuntimeError("80+20 does not reconstruct the frozen 100")
    return selected, remaining, hash_id_list(remaining), selected_hash, repeat_hash


def load_historical_aliases(
    question_ids: Sequence[str], aliases: Sequence[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    config = load_study1_config()
    models = load_models_config(config.resolve_path(config.models_config_path))
    sqlite_path = config.historical_sqlite()
    bundle: dict[tuple[str, str], dict[str, Any]] = {}
    connection = _open_historical_readonly(sqlite_path)
    try:
        for question_id in question_ids:
            for alias in aliases:
                spec = models.models[alias]
                bundle[(question_id, alias)] = _load_historical_stage12_on_connection(
                    connection,
                    example_id=question_id,
                    model_alias=alias,
                    requested_model_id=spec.api_model,
                )
    finally:
        connection.close()
    return bundle


def assert_007_write_target(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {
        V2_SQLITE.resolve(),
        STUDY1_SQLITE.resolve(),
        Q2_SQLITE.resolve(),
        QUAL_SQLITE.resolve(),
        PAPER_DIRECTION.resolve(),
    }
    if resolved in forbidden:
        raise RuntimeError(f"Task 007 refuses to write protected path: {resolved}")
    if resolved.name == "v2.sqlite3":
        raise RuntimeError("Task 007 refuses any checkpoint named v2.sqlite3")
    blocked_parents = (
        "005_parallel_diagnostic_sprint",
        "005b_difficulty_control",
        "005c_hidden_residual_after_difficulty",
        "006_qualitative_pilot_parallel_prep",
    )
    text = str(resolved)
    if any(name in text for name in blocked_parents) and "007_" not in text:
        raise RuntimeError(f"Task 007 refuses to write prior-task path: {resolved}")


def protected_fingerprints() -> dict[str, Any]:
    extra = {
        "task005_report": PROJECT_ROOT / "to_gpt" / "005_parallel_diagnostic_sprint" / "report.md",
        "task005b_report": PROJECT_ROOT / "to_gpt" / "005b_difficulty_control" / "report.md",
        "task005c_report": PROJECT_ROOT
        / "to_gpt"
        / "005c_hidden_residual_after_difficulty"
        / "report.md",
        "task006_report": PROJECT_ROOT / "to_gpt" / "006_qualitative_pilot_parallel_prep" / "report.md",
        "task006_sqlite": QUAL_SQLITE,
    }
    out = {
        "v2": file_fingerprint(V2_SQLITE),
        "study1": file_fingerprint(STUDY1_SQLITE),
        "q2": file_fingerprint(Q2_SQLITE) if Q2_SQLITE.exists() else {"exists": False},
        "paperDirection": {"path": str(PAPER_DIRECTION), "sha256": sha256_file(PAPER_DIRECTION)},
    }
    for name, path in extra.items():
        if path.exists():
            out[name] = file_fingerprint(path)
        else:
            out[name] = {"exists": False, "path": str(path)}
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value)
                        if isinstance(value, (list, dict, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_qual_row(row: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    displayed = row.get("displayed_confidence")
    if displayed in (None, "", "None"):
        displayed_val: float | None = None
    else:
        displayed_val = float(displayed)
    stage1 = row.get("stage1_correct")
    if isinstance(stage1, bool):
        stage1_val = stage1
    else:
        stage1_val = str(stage1).strip().lower() in {"1", "true", "yes"}
    return {
        "source": source,
        "question_id": row["question_id"],
        "model_alias": row["model_alias"],
        "model_endpoint": row.get("model_endpoint"),
        "family": row["family"],
        "score_condition": row["score_condition"],
        "displayed_confidence": displayed_val,
        "displayed_token": row.get("displayed_token") or None,
        "frozen_answer": row.get("frozen_answer"),
        "stage1_correct": stage1_val,
        "parsed_action": row.get("parsed_action"),
        "verify": int(row["verify"]),
        "raw_response": row.get("raw_response"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "estimated_cost_usd": float(row.get("estimated_cost_usd") or 0),
        "prompt_hash": row.get("prompt_hash"),
        "request_key": row.get("request_key"),
    }


__all__ = [
    "ALL_V2_MODELS",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CLAUDE_ENDPOINT",
    "CONCURRENCY",
    "CORRECT_KEY",
    "EXPECTED_REPEAT_IDS",
    "GEMINI_ENDPOINT",
    "GEMINI_GROK",
    "GPT_CLAUDE",
    "GPT_ENDPOINT",
    "GROK_ENDPOINT",
    "LANE_A_CAP",
    "LANE_A_DIR",
    "LANE_A_EXPERIMENT",
    "LANE_A_PROVIDER_CAP",
    "LANE_A_RUN_ID",
    "LANE_A_SQLITE",
    "LANE_B_CAP",
    "LANE_B_DIR",
    "LANE_B_EXPERIMENT",
    "LANE_B_PROVIDER_CAP",
    "LANE_B_RUN_ID",
    "LANE_B_SQLITE",
    "LANE_C_DIR",
    "LANE_D_DIR",
    "MAX_PARSE_REPAIRS",
    "OTHER_BY_TARGET",
    "PAPER_DIRECTION",
    "PROMPT_FAMILY_MODERATE",
    "PROMPT_FAMILY_STRONGER",
    "PROMPT_HASH_MODERATE",
    "PROMPT_HASH_STRONGER",
    "PROMPT_VERSION",
    "QUAL_SQLITE",
    "REPEAT_HASH",
    "RETURN_DIR",
    "TASK001_ID_HASH",
    "TASK_ID",
    "TOTAL_CAP",
    "assert_007_write_target",
    "coerce_qual_row",
    "load_csv",
    "load_historical_aliases",
    "protected_fingerprints",
    "remaining_80_ids",
    "sha256_file",
    "write_csv",
]
