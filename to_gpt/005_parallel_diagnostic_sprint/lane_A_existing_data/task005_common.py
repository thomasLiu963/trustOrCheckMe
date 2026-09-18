"""Shared Task 005 paths and read-only helpers.

Never opens historical V2 or Study 1 with a writable CheckpointStore.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

TASK_ID = "005_parallel_diagnostic_sprint"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
LANE_A_DIR = RETURN_DIR / "lane_A_existing_data"
LANE_B_DIR = RETURN_DIR / "lane_B_q2"
LANE_C_DIR = RETURN_DIR / "lane_C_reasoning"
LANE_D_DIR = RETURN_DIR / "lane_D_qualitative_prep"
Q2_SQLITE = PROJECT_ROOT / "results" / "study2_q2_diagnostic" / "q2.sqlite3"
REASONING_SQLITE = PROJECT_ROOT / "results" / "study1_reasoning_diagnostic" / "reasoning.sqlite3"
V2_SQLITE = PROJECT_ROOT / "results" / "v2" / "raw" / "v2.sqlite3"
STUDY1_SQLITE = PROJECT_ROOT / "results" / "study1_causal_pilot" / "study1.sqlite3"
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"

BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 5000
Q2_SCIENTIFIC_CAP = 1000
Q2_PROVIDER_ATTEMPT_CAP = 1300
Q2_RUN_ID = "study2-q2-005"
Q2_EXPERIMENT_VERSION = "study2_q2_diagnostic"
PRIMARY_FAMILY = "v2_owner_match_v1"
SATURATION_LABEL = "ACTION_SATURATED__ITEM_DISCRIMINATION_NOT_IDENTIFIABLE"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_json_records(
    path: Path, *, stage: str | None = None, status: str = "success"
) -> list[dict[str, Any]]:
    connection = open_readonly(path)
    try:
        if stage is None:
            rows = connection.execute(
                """
                SELECT record_json, stake_json, example_id, model_alias, status
                FROM requests
                WHERE status = ? AND record_json IS NOT NULL
                """,
                (status,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT record_json, stake_json, example_id, model_alias, status
                FROM requests
                WHERE stage = ? AND status = ? AND record_json IS NOT NULL
                """,
                (stage, status),
            ).fetchall()
    finally:
        connection.close()
    output: list[dict[str, Any]] = []
    for row in rows:
        record = json.loads(row["record_json"])
        stake = json.loads(row["stake_json"]) if row["stake_json"] else {}
        record.setdefault("example_id", row["example_id"])
        record.setdefault("model_alias", row["model_alias"])
        record["_stake"] = stake
        output.append(record)
    return output


def assert_not_historical_write_target(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {V2_SQLITE.resolve(), STUDY1_SQLITE.resolve()}
    if resolved in forbidden:
        raise RuntimeError(f"refusing to write to protected sqlite: {resolved}")
    if resolved.name == "v2.sqlite3":
        raise RuntimeError("refusing any checkpoint named v2.sqlite3")
    if resolved == PAPER_DIRECTION.resolve():
        raise RuntimeError("refusing to write paperDirection.txt")
