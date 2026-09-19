"""Task 010 paths and isolation. Zero API calls. Read-only on 007–009 and paperDirection."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .task005_common import Q2_SQLITE, STUDY1_SQLITE, V2_SQLITE, file_fingerprint, sha256_file
from .task006_common import QUAL_SQLITE
from .task007_common import LANE_A_SQLITE, LANE_B_SQLITE
from .task009_common import (
    ADJACENT_PAIRS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CV_FOLDS,
    EPS,
    GPT_CLAUDE,
    MATERIAL_LOGLOSS,
    MATCHED_BUDGETS,
    MODEL_LABELS,
    REFERENCE_SCORE,
    VISIBLE_FIXED,
    VISIBLE_FIXED_CONDITIONS,
)

TASK_ID = "010_final_sensitivity_equivalence_audit"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
FIGURES_DIR = RETURN_DIR / "figures"
ANALYSIS_DIR = RETURN_DIR / "analysis" / "task010"
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"
FROM_GPT = PROJECT_ROOT / "from_gpt" / f"{TASK_ID}.md"

TASK007_DIR = PROJECT_ROOT / "to_gpt" / "007_full_qualitative_four_model_decision_packet"
TASK008_DIR = PROJECT_ROOT / "to_gpt" / "008_ranking_invariance_audit"
TASK009_DIR = PROJECT_ROOT / "to_gpt" / "009_prospective_ranking_invariance_confirmation"

SIM_SEED = 20261010
N_SIM = 500
N_BOOT_POWER = 400
LATENT_BOOT = 1000
ROUTING_BOOT = 1000

# Frozen before inspecting Task-010 simulation/routing tables.
TIGHT_MIN_POWER = 0.80
SMALL_RANK_CORR = 0.90
SMALL_REVERSAL = 0.05
MODERATE_RANK_CORR = 0.80
MODERATE_REVERSAL = 0.10
LARGE_RANK_CORR = 0.70
LARGE_REVERSAL = 0.20

# Routing figure decision, frozen before Task-010 recomputation.
ROUTING_MATERIAL_PP = 0.10
ROUTING_MODEST_PP = 0.03
ROUTING_BUDGETS = (0.20, 0.30, 0.40)

DEVELOPMENT_ANCHOR_SOURCE = (
    "Task-008 moderate logit-scale score×difficulty gamma (C=1 ridge, difficulty = "
    "other-models-correct 0–3). Claude γ=5.157 is the primary development anchor "
    "because it was the largest development interaction people treated as possible "
    "reshaping; GPT γ=0.316 is the secondary own-model anchor; |Claude−GPT|=4.840 "
    "is the between-model anchor. Task-009 codes difficulty as other-primary "
    "wrongness in {0,1} and interacts (score-0.90)×wrongness. Injections use that "
    "009 coding; 008 γ is treated as the logit interaction magnitude on 009's "
    "binary wrongness (same units as a one-model easiness contrast, opposite sign "
    "to 008's other-correct coding). Primary 1C grid uses Claude 5.157."
)
CLAUDE_008_GAMMA = 5.15665762979811
GPT_008_GAMMA = 0.3164604552314793
BETWEEN_MODEL_GAMMA = abs(CLAUDE_008_GAMMA - GPT_008_GAMMA)
PRIMARY_ANCHOR_GAMMA = CLAUDE_008_GAMMA
INJECTION_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
FAMILY2_TARGET_RHO = (0.99, 0.95, 0.90, 0.80, 0.70, 0.60)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    assert_010_write_target(path)
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_dump(path: Path, payload: Any) -> None:
    assert_010_write_target(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_010_write_target(path: Path) -> None:
    resolved = path.resolve()
    forbidden = {
        V2_SQLITE.resolve(),
        STUDY1_SQLITE.resolve(),
        Q2_SQLITE.resolve(),
        QUAL_SQLITE.resolve(),
        LANE_A_SQLITE.resolve(),
        LANE_B_SQLITE.resolve(),
        PAPER_DIRECTION.resolve(),
        TASK007_DIR.resolve(),
        TASK008_DIR.resolve(),
        TASK009_DIR.resolve(),
    }
    if resolved in forbidden or resolved.name == "paperDirection.txt":
        raise RuntimeError(f"Task 010 refuses to write protected path: {resolved}")
    allowed = (
        RETURN_DIR.resolve(),
        (PROJECT_ROOT / "src").resolve(),
        FROM_GPT.parent.resolve(),
    )
    if resolved == FROM_GPT.resolve():
        return
    if any(resolved == root or root in resolved.parents for root in allowed):
        if "to_gpt/" in str(resolved) and TASK_ID not in str(resolved):
            raise RuntimeError(f"Task 010 refuses prior to_gpt outputs: {resolved}")
        return
    raise RuntimeError(f"Task 010 write target is outside allowed roots: {resolved}")


def protected_fingerprints() -> dict[str, Any]:
    paths = {
        "paperDirection": PAPER_DIRECTION,
        "v2": V2_SQLITE,
        "study1": STUDY1_SQLITE,
        "q2": Q2_SQLITE,
        "qual006": QUAL_SQLITE,
        "qual007a": LANE_A_SQLITE,
        "qual007b": LANE_B_SQLITE,
        "task009_stage3": TASK009_DIR / "stage3_results.csv",
        "task009_report": TASK009_DIR / "report.md",
    }
    out: dict[str, Any] = {}
    for name, path in paths.items():
        out[name] = file_fingerprint(path) if path.exists() else {"exists": False, "path": str(path)}
    return out
