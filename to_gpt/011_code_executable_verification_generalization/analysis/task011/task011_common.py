"""Task 011 paths and isolation. Do not write paperDirection or Tasks 007–010 data."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .task005_common import Q2_SQLITE, STUDY1_SQLITE, V2_SQLITE, file_fingerprint, sha256_file
from .task006_common import QUAL_SQLITE
from .task007_common import LANE_A_SQLITE, LANE_B_SQLITE
from .task009_common import SQLITE_PATH as TASK009_SQLITE
from .task010_common import RETURN_DIR as TASK010_DIR

TASK_ID = "011_code_executable_verification_generalization"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
FIGURES_DIR = RETURN_DIR / "figures"
ANALYSIS_DIR = RETURN_DIR / "analysis" / "task011"
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"
FROM_GPT = PROJECT_ROOT / "from_gpt" / f"{TASK_ID}.md"
SQLITE_PATH = PROJECT_ROOT / "results" / "study011_code_generalization" / "code.sqlite3"
SAMPLE_DIR = PROJECT_ROOT / "results" / "study011_code_generalization" / "sample"
CACHE_DIR = PROJECT_ROOT / ".cache"
LCB_REPO = CACHE_DIR / "LiveCodeBench"
LCB_DATA = CACHE_DIR / "livecodebench_data"

LCB_REPO_URL = "https://github.com/LiveCodeBench/LiveCodeBench"
LCB_HF_LITE = "livecodebench/code_generation_lite"
LCB_RELEASE = "release_v6"
LCB_COMMIT_PIN = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
DATASET_FILES = (
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
)

GPT_CLAUDE = ("openai_gpt56_sol", "anthropic_sonnet5")
MODEL_LABELS = {
    "openai_gpt56_sol": "GPT",
    "anthropic_sonnet5": "Claude",
}
GPT_ENDPOINT = "gpt-5.6-sol"
CLAUDE_ENDPOINT = "claude-sonnet-5"
ENDPOINTS = {
    "openai_gpt56_sol": GPT_ENDPOINT,
    "anthropic_sonnet5": CLAUDE_ENDPOINT,
}

FAMILY = "moderate"
PROMPT_VERSION = "task011_code_moderate_v1"
PROMPT_FAMILY = "qualitative_stakes_moderate_code_v1"
EXPERIMENT_VERSION = "study011_code_generalization_v1"
RUN_ID = "study011-code"
DATASET_NAME = "livecodebench_code_generation_lite"

PILOT_N = 40
PILOT_SEED = 20260921
SECOND_PILOT_SEED = 20260924
MAIN_N = 500
MAIN_SEED = 20260922
REPEAT_N = 100
REPEAT_SEED = 20260923
N_REPEAT_GENERATIONS = 3
N_REPEAT_EXTRAS = 2
MIN_ELIGIBLE = 300
MIN_HARD_ONLY = 250
TARGET_ELIGIBLE = 500

PILOT_CONDITIONS = ("displayed_0.70", "displayed_0.90", "displayed_0.99")
SCORE_CONDITIONS = (
    "hidden",
    "true_q_visible",
    "displayed_0.70",
    "displayed_0.85",
    "displayed_0.90",
    "displayed_0.95",
    "displayed_0.99",
)
VISIBLE_FIXED = (0.70, 0.85, 0.90, 0.95, 0.99)
VISIBLE_FIXED_CONDITIONS = tuple(f"displayed_{score:.2f}" for score in VISIBLE_FIXED)
ADJACENT_PAIRS = tuple(
    zip(VISIBLE_FIXED_CONDITIONS[:-1], VISIBLE_FIXED_CONDITIONS[1:])
)
H1_LOW = "displayed_0.70"
H1_HIGH = "displayed_0.99"

CODE_MAX_OUTPUT_TOKENS = 4096
STAGE3_MAX_OUTPUT_TOKENS = 64
CONCURRENCY_CODE = 24
CONCURRENCY_ROUTE = 24
CONCURRENCY_PER_PROVIDER = 12
EVAL_WORKERS = 16
MAX_TRANSIENT_RETRIES = 3
MAX_PARSE_REPAIRS = 1
RETRY_FAILED_PASSES = 1
EVAL_TIMEOUT_SEC = 6
EVAL_MEMORY_BYTES = 4 * 1024 * 1024 * 1024

PLANNED_PILOT = PILOT_N * len(GPT_CLAUDE) * (2 + len(PILOT_CONDITIONS))
PLANNED_STAGE12 = MAIN_N * len(GPT_CLAUDE) * 2
PLANNED_STAGE3 = MAIN_N * len(GPT_CLAUDE) * len(SCORE_CONDITIONS)
PLANNED_REPEAT_EXTRAS = REPEAT_N * len(GPT_CLAUDE) * len(VISIBLE_FIXED_CONDITIONS) * N_REPEAT_EXTRAS
PLANNED_MAIN = PLANNED_STAGE12 + PLANNED_STAGE3 + PLANNED_REPEAT_EXTRAS
# First excluded pilot was all-hard from a sampler bug; a second excluded mixed pilot is allowed.
SCIENTIFIC_CAP = (2 * PLANNED_PILOT) + PLANNED_MAIN
PROVIDER_ATTEMPT_CAP = SCIENTIFIC_CAP + 1600

BOOTSTRAP_SEED = 20260922
BOOTSTRAP_RESAMPLES = 5000
CV_FOLDS = 5
MATERIAL_LOGLOSS = 0.01
MATCHED_BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
REFERENCE_SCORE = 0.90
EPS = 1e-6

COVERAGE_LARGE_PP = 0.20
COVERAGE_MODERATE_PP = 0.10
ROUTING_GAIN_PP = 0.05

# Official LiveCodeBench ERRATA.md identifiers (broken / interactive / multi-output).
ERRATA_QUESTION_IDS = frozenset(
    {
        "abc311_c",
        "abc326_d",
        "abc327_b",
        "abc333_e",
        "abc343_e",
        "abc362_c",
        "find-words-containing-character",
        "find-the-peaks",
        "generate-binary-strings-without-adjacent-zeros",
        "arc185_c",
        "abc343_a",
        "abc337_e",
        "abc355_e",
        "abc350_c",
        "apply-operations-to-make-string-empty",
        "most-frequent-ids",
        "arc189_a",
    }
)

FORBIDDEN_SQLITE = (
    V2_SQLITE,
    STUDY1_SQLITE,
    Q2_SQLITE,
    QUAL_SQLITE,
    LANE_A_SQLITE,
    LANE_B_SQLITE,
    TASK009_SQLITE,
)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    assert_011_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    assert_011_write_target(path)
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


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_011_write_target(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {item.resolve() for item in FORBIDDEN_SQLITE}
    forbidden.add(PAPER_DIRECTION.resolve())
    forbidden.add(TASK010_DIR.resolve())
    if resolved in forbidden or resolved.name == "paperDirection.txt":
        raise RuntimeError(f"Task 011 refuses to write protected path: {resolved}")
    allowed = (
        RETURN_DIR.resolve(),
        SQLITE_PATH.parent.resolve(),
        (PROJECT_ROOT / "src").resolve(),
        FROM_GPT.parent.resolve(),
        CACHE_DIR.resolve(),
    )
    if resolved == FROM_GPT.resolve():
        return
    if any(resolved == root or root in resolved.parents for root in allowed):
        if "to_gpt/" in str(resolved) and TASK_ID not in str(resolved):
            raise RuntimeError(f"Task 011 refuses prior to_gpt outputs: {resolved}")
        return
    raise RuntimeError(f"Task 011 write target is outside allowed roots: {resolved}")


def protected_fingerprints() -> dict[str, Any]:
    return {
        "paperDirection": {
            "path": str(PAPER_DIRECTION),
            "sha256": sha256_file(PAPER_DIRECTION) if PAPER_DIRECTION.exists() else None,
        },
        "v2": file_fingerprint(V2_SQLITE) if V2_SQLITE.exists() else {"exists": False},
        "study1": file_fingerprint(STUDY1_SQLITE) if STUDY1_SQLITE.exists() else {"exists": False},
        "task009": file_fingerprint(TASK009_SQLITE) if TASK009_SQLITE.exists() else {"exists": False},
        "task010_report": file_fingerprint(TASK010_DIR / "report.md")
        if (TASK010_DIR / "report.md").exists()
        else {"exists": False},
    }
