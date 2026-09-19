"""Task 012 paths and frozen analysis constants. Zero API calls. Read-only on 009–011."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .task005_common import sha256_file

TASK_ID = "012_oversight_sensitivity_audit"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
FIGURES_DIR = RETURN_DIR / "figures"
ANALYSIS_DIR = RETURN_DIR / "analysis" / "task012"
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"
FROM_GPT = PROJECT_ROOT / "from_gpt" / f"{TASK_ID}.md"

TASK009_DIR = PROJECT_ROOT / "to_gpt" / "009_prospective_ranking_invariance_confirmation"
TASK010_DIR = PROJECT_ROOT / "to_gpt" / "010_final_sensitivity_equivalence_audit"
TASK011_DIR = PROJECT_ROOT / "to_gpt" / "011_code_executable_verification_generalization"

GPT_CLAUDE = ("openai_gpt56_sol", "anthropic_sonnet5")
MODEL_LABELS = {"openai_gpt56_sol": "GPT", "anthropic_sonnet5": "Claude"}
FIXED_SCORES = (0.70, 0.85, 0.90, 0.95, 0.99)
FIXED_CONDS = tuple(f"displayed_{score:.2f}" for score in FIXED_SCORES)
ADJACENT = tuple(zip(FIXED_SCORES[:-1], FIXED_SCORES[1:]))
ALL_CONDS = ("hidden", "true_q_visible") + FIXED_CONDS

BOOTSTRAP_SEED = 20260925
BOOTSTRAP_RESAMPLES = 5000
ECE_BINS = 10
LAMBDA_GRID = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    1.0,
)
REPRO_TOL_PP = 0.6
MATERIAL_LEAKAGE_PP = 0.05
MATERIAL_REGRET = 0.05
Q1_OVERLAP_FRAC = 0.25
LABEL = "POST_HOC_SYSTEMS_REANALYSIS"

TARGET_009 = {
    ("openai_gpt56_sol", "coverage_0.70_0.99"): 50.2,
    ("anthropic_sonnet5", "coverage_0.70_0.99"): 35.4,
}
TARGET_011 = {
    ("openai_gpt56_sol", "coverage_0.70_0.99"): 38.8,
    ("anthropic_sonnet5", "coverage_0.70_0.99"): 4.9,
    ("openai_gpt56_sol", "pass_rate"): 46.5,
    ("anthropic_sonnet5", "pass_rate"): 43.0,
}


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
