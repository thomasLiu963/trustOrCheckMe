"""Task 014 paths and frozen constants. Isolated from 009–013 writes."""

from __future__ import annotations

from .config import PROJECT_ROOT
from .task012_common import LAMBDA_GRID, json_dump, load_csv, write_csv
from .task009_common import ENDPOINTS as ENDPOINTS_009
from .task011_common import ENDPOINTS as ENDPOINTS_011
from .task009_common import SQLITE_PATH as TASK009_SQLITE
from .task011_common import SQLITE_PATH as TASK011_SQLITE

TASK_ID = "014_rank_preserving_confidence_shift"
RETURN_DIR = PROJECT_ROOT / "to_gpt" / TASK_ID
FIGURES_DIR = RETURN_DIR / "figures"
ANALYSIS_DIR = RETURN_DIR / "analysis" / "task014"
SAMPLE_DIR = PROJECT_ROOT / "results" / "study014_offset" / "sample"
SQLITE_PATH = PROJECT_ROOT / "results" / "study014_offset" / "offset.sqlite3"
PAPER_DIRECTION = PROJECT_ROOT / "paperDirection.txt"
TASK009_DIR = PROJECT_ROOT / "to_gpt" / "009_prospective_ranking_invariance_confirmation"
TASK011_DIR = PROJECT_ROOT / "to_gpt" / "011_code_executable_verification_generalization"
TASK012_DIR = PROJECT_ROOT / "to_gpt" / "012_oversight_sensitivity_audit"
EXAMPLES_009 = PROJECT_ROOT / "results" / "study009_prospective" / "sample" / "examples_500.jsonl"
PROBLEMS_011 = PROJECT_ROOT / "results" / "study011_code_generalization" / "sample" / "main_problems.jsonl"

GPT_CLAUDE = ("openai_gpt56_sol", "anthropic_sonnet5")
MODEL_LABELS = {"openai_gpt56_sol": "GPT", "anthropic_sonnet5": "Claude"}
ENDPOINTS = {
    "openai_gpt56_sol": ENDPOINTS_009["openai_gpt56_sol"],
    "anthropic_sonnet5": ENDPOINTS_009["anthropic_sonnet5"],
}
assert ENDPOINTS == {
    "openai_gpt56_sol": ENDPOINTS_011["openai_gpt56_sol"],
    "anthropic_sonnet5": ENDPOINTS_011["anthropic_sonnet5"],
}

DELTAS = (-1.5, -0.75, 0.0, 0.75, 1.5)
NEW_DELTAS = (-1.5, -0.75, 0.75, 1.5)
DISPLAY_FORMAT = ".12g"
ROUNDING = "python format(number, '.12g')"
LABEL = "TASK014_RANK_PRESERVING_SHIFT"

EXPERIMENT_VERSION = "study014_offset_v1"
PROMPT_VERSION_009 = "task006_qualitative_pilot_v1"
PROMPT_FAMILY_009 = "qualitative_stakes_moderate_v1"
PROMPT_VERSION_011 = "task011_code_moderate_v1"
PROMPT_FAMILY_011 = "qualitative_stakes_moderate_code_v1"
RUN_ID = "study014-offset"
FAMILY = "moderate"

CONCURRENCY_ROUTE = 24
CONCURRENCY_PER_PROVIDER = 12
MAX_TRANSIENT_RETRIES = 3
MAX_PARSE_REPAIRS = 1
SCIENTIFIC_CAP = 7000
PROVIDER_ATTEMPT_CAP = 8600

BOOTSTRAP_SEED = 20260927
BOOTSTRAP_RESAMPLES = 5000
REPRO_TOL_PP = 0.6

WORLD_A = "LEVEL_SENSITIVITY_STRONGLY_GENERALIZES"
WORLD_B = "LEVEL_SENSITIVITY_PARTIAL"
WORLD_C = "ITEM_INFORMATION_LARGELY_PROTECTS"
WORLD_D = "MIXED_BY_TASK"
WORLD_E = "TECHNICAL_OR_VERSION_FAILURE"
