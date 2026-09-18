"""Shared Task 006 paths and isolation guards.

Never writes historical V2, Study 1, q2, Task 005, Task 005B, or paperDirection.txt.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .task005_common import (
    Q2_SQLITE,
    STUDY1_SQLITE,
    V2_SQLITE,
    file_fingerprint,
    load_json_records,
    open_readonly,
    sha256_file,
)

TASK_ID = "006_qualitative_pilot_parallel_prep"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
LANE_A_DIR = RETURN_DIR / "lane_A_qualitative_pilot"
LANE_B_DIR = RETURN_DIR / "lane_B_routing"
LANE_C_DIR = RETURN_DIR / "lane_C_fresh_pool"
LANE_D_DIR = RETURN_DIR / "lane_D_model_feasibility"
QUAL_SQLITE = (
    PROJECT_ROOT / "results" / "study1_qualitative_pilot" / "qualitative_pilot.sqlite3"
)
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"
TASK005_RETURN = PROJECT_ROOT / "to_gpt" / "005_parallel_diagnostic_sprint"
TASK005B_RETURN = PROJECT_ROOT / "to_gpt" / "005b_difficulty_control"
TASK005B_SRC = PROJECT_ROOT / "src" / "task005b_difficulty.py"

BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 5000
SCIENTIFIC_CAP = 320
PROVIDER_ATTEMPT_CAP = 400
CONCURRENCY = 4
MAX_PARSE_REPAIRS = 1
RUN_ID = "study1-qualitative-006"
EXPERIMENT_VERSION = "study1_qualitative_pilot"
PROMPT_VERSION = "task006_qualitative_pilot_v1"
REPEAT_HASH = "45e06fe9daf40dd1cbd9cbf91b49cef09c5d924df17f2cf2bf2f203090c26d38"
GPT_ENDPOINT = "gpt-5.6-sol"
CLAUDE_ENDPOINT = "claude-sonnet-5"
CANONICAL_HIDDEN = {
    "decision_owner": "ai_system",
    "L": 10.0,
    "confidence_visibility": "hidden",
    "rationale": (
        "Predeclared to match Study 1's AI-system / L=10 hidden cell. "
        "Not selected by inspecting correctness or catch rates."
    ),
}

EXPECTED_REPEAT_IDS = (
    "mmlu_pro:test:1731",
    "mmlu_pro:test:6452",
    "mmlu_pro:test:429",
    "mmlu_pro:test:3333",
    "mmlu_pro:test:774",
    "mmlu_pro:test:4568",
    "mmlu_pro:test:9646",
    "mmlu_pro:test:11802",
    "mmlu_pro:test:10497",
    "mmlu_pro:test:2776",
    "mmlu_pro:test:2849",
    "mmlu_pro:test:1065",
    "mmlu_pro:test:4290",
    "mmlu_pro:test:3888",
    "mmlu_pro:test:471",
    "mmlu_pro:test:11138",
    "mmlu_pro:test:10164",
    "mmlu_pro:test:4773",
    "mmlu_pro:test:8662",
    "mmlu_pro:test:10842",
)

SCORE_VALUES = (0.70, 0.90, 0.99)
SCORE_CONDITIONS = ("hidden", "displayed_0.70", "displayed_0.90", "displayed_0.99")
STAKES_FAMILIES = ("moderate", "stronger")
SATURATION_LABEL = "ITEM_ROUTING_NOT_IDENTIFIABLE_DUE_TO_ACTION_SATURATION"


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_006_write_target(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {
        V2_SQLITE.resolve(),
        STUDY1_SQLITE.resolve(),
        Q2_SQLITE.resolve(),
        PAPER_DIRECTION.resolve(),
        TASK005B_SRC.resolve(),
    }
    if resolved in forbidden:
        raise RuntimeError(f"Task 006 refuses to write protected path: {resolved}")
    if resolved.name == "v2.sqlite3":
        raise RuntimeError("Task 006 refuses any checkpoint named v2.sqlite3")
    if "005b_difficulty_control" in str(resolved):
        raise RuntimeError("Task 006 refuses to write Task 005B outputs")
    if resolved == TASK005_RETURN.resolve() or TASK005_RETURN.resolve() in resolved.parents:
        raise RuntimeError("Task 006 refuses to modify Task 005 outputs")


def protected_fingerprints() -> dict[str, Any]:
    return {
        "v2": file_fingerprint(V2_SQLITE),
        "study1": file_fingerprint(STUDY1_SQLITE),
        "q2": file_fingerprint(Q2_SQLITE) if Q2_SQLITE.exists() else {"exists": False},
        "paperDirection": {
            "path": str(PAPER_DIRECTION),
            "sha256": sha256_file(PAPER_DIRECTION),
        },
    }


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CANONICAL_HIDDEN",
    "CLAUDE_ENDPOINT",
    "CONCURRENCY",
    "EXPERIMENT_VERSION",
    "EXPECTED_REPEAT_IDS",
    "GPT_ENDPOINT",
    "LANE_A_DIR",
    "LANE_B_DIR",
    "LANE_C_DIR",
    "LANE_D_DIR",
    "MAX_PARSE_REPAIRS",
    "PAPER_DIRECTION",
    "PROMPT_VERSION",
    "PROVIDER_ATTEMPT_CAP",
    "Q2_SQLITE",
    "QUAL_SQLITE",
    "REPEAT_HASH",
    "RETURN_DIR",
    "RUN_ID",
    "SATURATION_LABEL",
    "SCIENTIFIC_CAP",
    "SCORE_CONDITIONS",
    "SCORE_VALUES",
    "STAKES_FAMILIES",
    "STUDY1_SQLITE",
    "TASK005B_RETURN",
    "TASK_ID",
    "V2_SQLITE",
    "assert_006_write_target",
    "file_fingerprint",
    "load_json_records",
    "open_readonly",
    "prompt_sha256",
    "protected_fingerprints",
    "sha256_file",
]
