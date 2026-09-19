"""Task 013 paths and frozen constants. Zero API. Read-only on 009–012."""

from __future__ import annotations

from .config import PROJECT_ROOT
from .task012_common import (
    ALL_CONDS,
    FIXED_CONDS,
    GPT_CLAUDE,
    LAMBDA_GRID,
    MODEL_LABELS,
    REPRO_TOL_PP,
    TASK009_DIR,
    TASK011_DIR,
    json_dump,
    load_csv,
    write_csv,
)

TASK_ID = "013_hidden_confidence_policy_audit"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
FIGURES_DIR = RETURN_DIR / "figures"
ANALYSIS_DIR = RETURN_DIR / "analysis" / "task013"
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"
TASK012_DIR = PROJECT_ROOT / "to_gpt" / "012_oversight_sensitivity_audit"

LABEL = "POST_HOC_POLICY_REANALYSIS"
BOOTSTRAP_SEED = 20260926
BOOTSTRAP_RESAMPLES = 5000
NEAR_BEST = (0.01, 0.02, 0.05)
POLICIES = ALL_CONDS

TARGET_COVERAGE = {
    ("mmlu", "openai_gpt56_sol", "hidden"): 39.6,
    ("mmlu", "openai_gpt56_sol", "true_q_visible"): 12.4,
    ("mmlu", "anthropic_sonnet5", "hidden"): 65.4,
    ("mmlu", "anthropic_sonnet5", "true_q_visible"): 79.4,
    ("code", "openai_gpt56_sol", "hidden"): 32.5,
    ("code", "openai_gpt56_sol", "true_q_visible"): 27.3,
    ("code", "anthropic_sonnet5", "hidden"): 98.6,
    ("code", "anthropic_sonnet5", "true_q_visible"): 97.9,
}

__all__ = [
    "ALL_CONDS",
    "ANALYSIS_DIR",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "FIGURES_DIR",
    "FIXED_CONDS",
    "GPT_CLAUDE",
    "LABEL",
    "LAMBDA_GRID",
    "MODEL_LABELS",
    "NEAR_BEST",
    "PAPER_DIRECTION",
    "POLICIES",
    "REPRO_TOL_PP",
    "RETURN_DIR",
    "TARGET_COVERAGE",
    "TASK009_DIR",
    "TASK011_DIR",
    "TASK012_DIR",
    "TASK_ID",
    "json_dump",
    "load_csv",
    "write_csv",
]
