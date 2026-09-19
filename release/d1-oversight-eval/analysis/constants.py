"""Frozen analysis constants. No API calls."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
FIGURES_DIR = PACKAGE_ROOT / "figures"
MANIFEST_DIR = DATA_DIR / "manifests"

GPT_CLAUDE = ("openai_gpt56_sol", "anthropic_sonnet5")
MODEL_LABELS = {"openai_gpt56_sol": "GPT", "anthropic_sonnet5": "Claude"}
TASK_LABELS = {"mmlu": "MMLU-Pro", "code": "LiveCodeBench hard"}
ENDPOINTS = {"openai_gpt56_sol": "gpt-5.6-sol", "anthropic_sonnet5": "claude-sonnet-5"}

FIXED_SCORES = (0.70, 0.85, 0.90, 0.95, 0.99)
FIXED_CONDS = tuple(f"displayed_{score:.2f}" for score in FIXED_SCORES)
ALL_CONDS = ("hidden", "true_q_visible") + FIXED_CONDS
DELTAS = (-1.5, -0.75, 0.0, 0.75, 1.5)

LAMBDA_GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)

BOOTSTRAP_SEED_012 = 20260925
BOOTSTRAP_SEED_014 = 20260927
BOOTSTRAP_SEED_015 = 20260928
BOOTSTRAP_RESAMPLES = 5000

REPRO_TOL_PP = 0.6
REPRO_TOL_RATE = 0.006
REPRO_TOL_LOSS = 0.002

# Reported headline values from frozen 009–015 reports (percentage points or rates).
HEADLINES = {
    "009_gpt_coverage_pp": 50.2,
    "009_claude_coverage_pp": 35.4,
    "011_gpt_coverage_pp": 38.8,
    "011_claude_coverage_pp": 4.9,
    "011_gpt_pass_pct": 46.5,
    "011_claude_pass_pct": 43.0,
    "012_gpt_mmlu_leakage_pp": 11.8,
    "012_gpt_code_leakage_pp": 22.0,
    "014_gpt_mmlu_coverage_pp": -16.2,
    "014_gpt_mmlu_leakage_pp": 6.0,
    "014_gpt_code_coverage_pp": -7.0,
    "014_gpt_code_leakage_pp": 4.5,
    "014_claude_mmlu_coverage_pp": -31.2,
    "014_claude_mmlu_leakage_pp": 3.4,
    "015_gpt_code_escape": 0.5555555555555556,
    "015_gpt_code_escape_ci_lo": 0.4723926380368098,
    "015_gpt_code_escape_ci_hi": 0.6346182634730538,
    "015_gpt_mmlu_max_spread": 0.102,
    "015_gpt_code_max_spread": 0.059363636363636396,
    "015_claude_mmlu_max_spread": 0.278,
    "015_claude_code_max_spread": 0.038461538461538436,
    "014_gpt_mmlu_retention_coverage": 0.32270916334661354,
    "014_gpt_mmlu_retention_leakage": 0.5084745762711864,
    "014_gpt_code_retention_coverage": 0.18018018018018014,
    "014_gpt_code_retention_leakage": 0.20634920634920645,
}
