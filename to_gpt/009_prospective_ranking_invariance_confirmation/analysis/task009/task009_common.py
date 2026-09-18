"""Task 009 paths, freeze constants, and isolation guards.

Never writes historical V2, Study 1, q2, Tasks 003–008 data, or paperDirection.txt.
"""

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

TASK_ID = "009_prospective_ranking_invariance_confirmation"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
FIGURES_DIR = RETURN_DIR / "figures"
ANALYSIS_DIR = RETURN_DIR / "analysis" / "task009"
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"
SQLITE_PATH = PROJECT_ROOT / "results" / "study009_prospective" / "prospective.sqlite3"
SAMPLE_DIR = PROJECT_ROOT / "results" / "study009_prospective" / "sample"
FROM_GPT = PROJECT_ROOT / "from_gpt" / f"{TASK_ID}.md"

POOL_HASH = "53b11137658de739537a41032d92aada7f50742ef07e9615da10e070ab37c4fa"
PRIMARY_SEED = 20260918
SECONDARY_SEED = 20260918
REPEAT_SEED = 20260919
PRIMARY_N = 500
SECONDARY_N = 200
REPEAT_N = 100
N_REPEAT_GENERATIONS = 3
N_REPEAT_EXTRAS = 2

FAMILY = "moderate"
PROMPT_VERSION = "task006_qualitative_pilot_v1"
PROMPT_FAMILY = "qualitative_stakes_moderate_v1"
PROMPT_HASH_MODERATE = "ec90530c644038dcee86accfc578c501533fdb69d5ea23caeae23dcc1d06dbad"
EXPERIMENT_VERSION = "study009_prospective_v1"
RUN_ID = "study009-prospective"
PILOT_OR_CONFIRMATORY = "confirmatory"

GPT_CLAUDE = ("openai_gpt56_sol", "anthropic_sonnet5")
GEMINI_GROK = ("google_gemini38_flash", "xai_grok420_nonreasoning")
ALL_MODELS = GPT_CLAUDE + GEMINI_GROK
MODEL_LABELS = {
    "openai_gpt56_sol": "GPT",
    "anthropic_sonnet5": "Claude",
    "google_gemini38_flash": "Gemini",
    "xai_grok420_nonreasoning": "Grok",
}
GPT_ENDPOINT = "gpt-5.6-sol"
CLAUDE_ENDPOINT = "claude-sonnet-5"
GEMINI_ENDPOINT = "gemini-3.8-flash"
GROK_ENDPOINT = "grok-4.20-0309-non-reasoning"
ENDPOINTS = {
    "openai_gpt56_sol": GPT_ENDPOINT,
    "anthropic_sonnet5": CLAUDE_ENDPOINT,
    "google_gemini38_flash": GEMINI_ENDPOINT,
    "xai_grok420_nonreasoning": GROK_ENDPOINT,
}

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
ADJACENT_PAIRS = (
    ("displayed_0.70", "displayed_0.85"),
    ("displayed_0.85", "displayed_0.90"),
    ("displayed_0.90", "displayed_0.95"),
    ("displayed_0.95", "displayed_0.99"),
)
H1_LOW = "displayed_0.70"
H1_HIGH = "displayed_0.99"
H1_FLOORS = {"openai_gpt56_sol": 0.20, "anthropic_sonnet5": 0.15}

PLANNED_PRIMARY_STAGE12 = PRIMARY_N * len(GPT_CLAUDE) * 2
PLANNED_PRIMARY_STAGE3 = PRIMARY_N * len(GPT_CLAUDE) * len(SCORE_CONDITIONS)
PLANNED_REPEAT_EXTRAS = REPEAT_N * len(GPT_CLAUDE) * len(SCORE_CONDITIONS) * N_REPEAT_EXTRAS
PLANNED_PRIMARY_TOTAL = PLANNED_PRIMARY_STAGE12 + PLANNED_PRIMARY_STAGE3 + PLANNED_REPEAT_EXTRAS
PLANNED_SECONDARY_STAGE12 = SECONDARY_N * len(GEMINI_GROK) * 2
PLANNED_SECONDARY_STAGE3 = SECONDARY_N * len(GEMINI_GROK) * len(SCORE_CONDITIONS)
PLANNED_SECONDARY_TOTAL = PLANNED_SECONDARY_STAGE12 + PLANNED_SECONDARY_STAGE3
SCIENTIFIC_CAP = PLANNED_PRIMARY_TOTAL + PLANNED_SECONDARY_TOTAL
PROVIDER_ATTEMPT_CAP = 20020
CONCURRENCY = 20
MAX_TRANSIENT_RETRIES = 3
MAX_PARSE_REPAIRS = 1
RETRY_FAILED_PASSES = 1
FAILURE_INCONCLUSIVE_FRAC = 0.05

BOOTSTRAP_SEED = 20260918
BOOTSTRAP_RESAMPLES = 5000
CV_FOLDS = 5
MATERIAL_LOGLOSS = 0.01
MATERIAL_AUROC = 0.05
SATURATION_LOW = 0.05
SATURATION_HIGH = 0.95
SATURATION_LABEL = "ITEM_ROUTING_NOT_IDENTIFIABLE_DUE_TO_ACTION_SATURATION"
MATCHED_BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
REFERENCE_SCORE = 0.90
EPS = 1e-6

FORBIDDEN_SQLITE = (
    V2_SQLITE,
    STUDY1_SQLITE,
    Q2_SQLITE,
    QUAL_SQLITE,
    LANE_A_SQLITE,
    LANE_B_SQLITE,
)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


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


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_009_write_target(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {item.resolve() for item in FORBIDDEN_SQLITE}
    forbidden.add(PAPER_DIRECTION.resolve())
    if resolved in forbidden:
        raise RuntimeError(f"Task 009 refuses to write protected path: {resolved}")
    if resolved.name == "v2.sqlite3" or resolved.name == "paperDirection.txt":
        raise RuntimeError(f"Task 009 refuses to write {resolved.name}")
    allowed_roots = (
        RETURN_DIR.resolve(),
        SQLITE_PATH.parent.resolve(),
        (PROJECT_ROOT / "src").resolve(),
        FROM_GPT.parent.resolve(),
    )
    if resolved == FROM_GPT.resolve():
        return
    if any(resolved == root or root in resolved.parents for root in allowed_roots):
        if "to_gpt/00" in str(resolved) and TASK_ID not in str(resolved):
            if resolved != FROM_GPT.resolve():
                raise RuntimeError(f"Task 009 refuses prior to_gpt outputs: {resolved}")
        return
    raise RuntimeError(f"Task 009 write target is outside allowed roots: {resolved}")


def protected_fingerprints() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "v2": file_fingerprint(V2_SQLITE) if V2_SQLITE.exists() else {"exists": False},
        "study1": file_fingerprint(STUDY1_SQLITE) if STUDY1_SQLITE.exists() else {"exists": False},
        "q2": file_fingerprint(Q2_SQLITE) if Q2_SQLITE.exists() else {"exists": False},
        "qual006": file_fingerprint(QUAL_SQLITE) if QUAL_SQLITE.exists() else {"exists": False},
        "qual007a": file_fingerprint(LANE_A_SQLITE) if LANE_A_SQLITE.exists() else {"exists": False},
        "qual007b": file_fingerprint(LANE_B_SQLITE) if LANE_B_SQLITE.exists() else {"exists": False},
        "paperDirection": {
            "path": str(PAPER_DIRECTION),
            "sha256": sha256_file(PAPER_DIRECTION) if PAPER_DIRECTION.exists() else None,
        },
    }
    return payload


def freeze_paths() -> dict[str, Path]:
    return {
        "preregistration": RETURN_DIR / "preregistration.md",
        "freeze_manifest": RETURN_DIR / "freeze_manifest.json",
        "sample_500": RETURN_DIR / "sample_500.csv",
        "secondary_200": RETURN_DIR / "secondary_200.csv",
        "repeat_100": RETURN_DIR / "repeat_100.csv",
        "sample_500_local": SAMPLE_DIR / "sample_500.csv",
        "secondary_200_local": SAMPLE_DIR / "secondary_200.csv",
        "repeat_100_local": SAMPLE_DIR / "repeat_100.csv",
        "examples_500": SAMPLE_DIR / "examples_500.jsonl",
        "examples_200": SAMPLE_DIR / "examples_200.jsonl",
        "examples_100": SAMPLE_DIR / "examples_100.jsonl",
        "freeze_local": SAMPLE_DIR / "freeze_manifest.json",
    }


def frozen() -> bool:
    paths = freeze_paths()
    return paths["freeze_manifest"].exists() and paths["sample_500"].exists()
