"""Task 008 paths and isolation guards.

Read-only on Tasks 003–007, V2, Study 1, q2, and paperDirection.txt.
Zero API calls. Writes only under to_gpt/008_ranking_invariance_audit/.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .task005_common import (
    Q2_SQLITE,
    STUDY1_SQLITE,
    V2_SQLITE,
    file_fingerprint,
    sha256_file,
)
from .task006_common import QUAL_SQLITE
from .task007_common import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    GPT_CLAUDE,
    LANE_A_SQLITE,
    LANE_B_SQLITE,
    PAPER_DIRECTION,
    coerce_qual_row,
    load_csv,
    write_csv,
)

TASK_ID = "008_ranking_invariance_audit"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
FIGURES_DIR = RETURN_DIR / "figures"
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"

MERGED100 = (
    PROJECT_ROOT
    / "to_gpt"
    / "007_full_qualitative_four_model_decision_packet"
    / "lane_A_full_qualitative"
    / "results_merged100.csv"
)
DIFFICULTY_FEATURES = (
    PROJECT_ROOT
    / "to_gpt"
    / "007_full_qualitative_four_model_decision_packet"
    / "lane_C_score_vs_difficulty"
    / "difficulty_features.csv"
)
GEMINI_GROK_RESULTS = (
    PROJECT_ROOT
    / "to_gpt"
    / "007_full_qualitative_four_model_decision_packet"
    / "lane_B_gemini_grok"
    / "results.csv"
)
STUDY1_PRIMARY = (
    PROJECT_ROOT / "to_gpt" / "003_study1_primary_results" / "primary_results.csv"
)
CHECKPOINT_A_CELLS = (
    PROJECT_ROOT / "to_gpt" / "checkpoint_A" / "checkpoint_A_cell_results.csv"
)

VISIBLE_CONDITIONS = ("displayed_0.70", "displayed_0.90", "displayed_0.99")
VISIBLE_SCORES = (0.70, 0.90, 0.99)
SCORE_PAIRS = ((0.70, 0.90), (0.90, 0.99), (0.70, 0.99))
STAKES_FAMILIES = ("moderate", "stronger")
REFERENCE_SCORE = 0.90
MATERIAL_LOGLOSS = 0.01
MATERIAL_AUROC = 0.05
MATERIAL_REVERSAL = 0.05
SATURATION_LOW = 0.10
SATURATION_HIGH = 0.90
MODEL_LABELS = {
    "openai_gpt56_sol": "GPT",
    "anthropic_sonnet5": "Claude",
    "google_gemini38_flash": "Gemini",
    "xai_grok420_nonreasoning": "Grok",
}


def assert_008_write_target(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {
        V2_SQLITE.resolve(),
        STUDY1_SQLITE.resolve(),
        Q2_SQLITE.resolve(),
        QUAL_SQLITE.resolve(),
        LANE_A_SQLITE.resolve(),
        LANE_B_SQLITE.resolve(),
        PAPER_DIRECTION.resolve(),
    }
    if resolved in forbidden:
        raise RuntimeError(f"Task 008 refuses to write protected path: {resolved}")
    if resolved.name in {"v2.sqlite3", "paperDirection.txt"}:
        raise RuntimeError(f"Task 008 refuses protected name: {resolved.name}")
    if RETURN_DIR.resolve() not in resolved.parents and resolved != RETURN_DIR.resolve():
        if resolved.suffix == ".py" and resolved.parent == PROJECT_ROOT / "src":
            return
        raise RuntimeError(f"Task 008 writes only under {RETURN_DIR} or src/: {resolved}")


def protected_fingerprints() -> dict[str, Any]:
    paths = {
        "v2": V2_SQLITE,
        "study1": STUDY1_SQLITE,
        "q2": Q2_SQLITE,
        "task006_sqlite": QUAL_SQLITE,
        "task007_full80_sqlite": LANE_A_SQLITE,
        "task007_gemini_grok_sqlite": LANE_B_SQLITE,
        "paperDirection": PAPER_DIRECTION,
        "merged100": MERGED100,
        "difficulty_features": DIFFICULTY_FEATURES,
        "task003_report": PROJECT_ROOT / "to_gpt" / "003_study1_primary_results" / "report.md",
        "task004_report": PROJECT_ROOT / "to_gpt" / "004_study1_stability_results" / "report.md",
        "task005b_report": PROJECT_ROOT / "to_gpt" / "005b_difficulty_control" / "report.md",
        "task005c_report": PROJECT_ROOT
        / "to_gpt"
        / "005c_hidden_residual_after_difficulty"
        / "report.md",
        "task007_report": PROJECT_ROOT
        / "to_gpt"
        / "007_full_qualitative_four_model_decision_packet"
        / "report.md",
        "task007_packet": PROJECT_ROOT
        / "to_gpt"
        / "007_full_qualitative_four_model_decision_packet"
        / "paper_direction_decision_packet.md",
        "checkpoint_a_report": PROJECT_ROOT / "to_gpt" / "checkpoint_A" / "checkpoint_A_report.md",
    }
    out: dict[str, Any] = {}
    for name, path in paths.items():
        if path.exists():
            out[name] = file_fingerprint(path)
        else:
            out[name] = {"exists": False, "path": str(path)}
    return out


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CHECKPOINT_A_CELLS",
    "DIFFICULTY_FEATURES",
    "FIGURES_DIR",
    "GEMINI_GROK_RESULTS",
    "GPT_CLAUDE",
    "MATERIAL_AUROC",
    "MATERIAL_LOGLOSS",
    "MATERIAL_REVERSAL",
    "MERGED100",
    "MODEL_LABELS",
    "PAPER_DIRECTION",
    "REFERENCE_SCORE",
    "RETURN_DIR",
    "SATURATION_HIGH",
    "SATURATION_LOW",
    "SCORE_PAIRS",
    "STAKES_FAMILIES",
    "STUDY1_PRIMARY",
    "TASK_ID",
    "VISIBLE_CONDITIONS",
    "VISIBLE_SCORES",
    "assert_008_write_target",
    "coerce_qual_row",
    "json_dump",
    "load_csv",
    "protected_fingerprints",
    "sha256_file",
    "write_csv",
]
