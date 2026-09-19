"""Task 015 paths and frozen constants. Zero API. Read-only on 009–014."""

from __future__ import annotations

from .config import PROJECT_ROOT
from .task012_common import LAMBDA_GRID, json_dump, load_csv, write_csv

TASK_ID = "015_normative_robustness_audit"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
FIGURES_DIR = RETURN_DIR / "figures"
ANALYSIS_DIR = RETURN_DIR / "analysis" / "task015"
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"
TASK014_DIR = PROJECT_ROOT / "to_gpt" / "014_rank_preserving_confidence_shift"
TASK012_DIR = PROJECT_ROOT / "to_gpt" / "012_oversight_sensitivity_audit"

GPT_CLAUDE = ("openai_gpt56_sol", "anthropic_sonnet5")
MODEL_LABELS = {"openai_gpt56_sol": "GPT", "anthropic_sonnet5": "Claude"}
TASK_LABELS = {"mmlu": "MMLU-Pro", "code": "LiveCodeBench"}
DELTAS = (-1.5, -0.75, 0.0, 0.75, 1.5)
LABEL = "POST_HOC_NORMATIVE_REANALYSIS"
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_RESAMPLES = 5000
Q1_SEED = 20260928
Q1_FOLDS = 5
SPREAD_THRESHOLDS = (0.01, 0.02, 0.05, 0.10)
