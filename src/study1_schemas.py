"""Study 1 schemas. Independent of frozen V2 VerificationDecisionRecord."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .config import PROJECT_ROOT, StrictConfigModel, _load_yaml
from .schemas import AnswerLabel, Probability, StrictModel, VerificationAction

DEFAULT_STUDY1_CONFIG = PROJECT_ROOT / "config" / "experiment_study1.yaml"

STUDY1_MODEL_ALIASES = ("openai_gpt56_sol", "anthropic_sonnet5")
STUDY1_ERROR_COSTS = (10.0, 20.0)
STUDY1_VERIFICATION_COST = 1.0
STUDY1_GRID_L10 = (0.80, 0.88, 0.89, 0.91, 0.99)
STUDY1_GRID_L20 = (0.90, 0.93, 0.94, 0.96, 0.99)
STUDY1_PRIMARY_CALLS = 2800
STUDY1_REPEAT_EXTRA_CALLS = 1120
STUDY1_TOTAL_CALLS = 3920
HISTORICAL_V2_SQLITE = "results/v2/raw/v2.sqlite3"


class DisplayCondition(str, Enum):
    HIDDEN = "hidden"
    TRUE_CONFIDENCE_VISIBLE = "true_confidence_visible"
    MANIPULATED_1 = "manipulated_1"
    MANIPULATED_2 = "manipulated_2"
    MANIPULATED_3 = "manipulated_3"
    MANIPULATED_4 = "manipulated_4"
    MANIPULATED_5 = "manipulated_5"


class DisplaySourceCondition(str, Enum):
    NONE = "none"
    REPORTED_CONFIDENCE = "reported_confidence"
    ASSIGNED_GRID = "assigned_grid"


class Study1Phase(str, Enum):
    PRIMARY = "primary"
    REPEATS = "repeats"


PRIMARY_DISPLAY_CONDITIONS = (
    DisplayCondition.HIDDEN,
    DisplayCondition.TRUE_CONFIDENCE_VISIBLE,
    DisplayCondition.MANIPULATED_1,
    DisplayCondition.MANIPULATED_2,
    DisplayCondition.MANIPULATED_3,
    DisplayCondition.MANIPULATED_4,
    DisplayCondition.MANIPULATED_5,
)

MANIPULATED_CONDITIONS = (
    DisplayCondition.MANIPULATED_1,
    DisplayCondition.MANIPULATED_2,
    DisplayCondition.MANIPULATED_3,
    DisplayCondition.MANIPULATED_4,
    DisplayCondition.MANIPULATED_5,
)

GRIDS_BY_L: dict[float, tuple[float, float, float, float, float]] = {
    10.0: STUDY1_GRID_L10,
    20.0: STUDY1_GRID_L20,
}


def display_source_for(condition: DisplayCondition) -> DisplaySourceCondition:
    if condition == DisplayCondition.HIDDEN:
        return DisplaySourceCondition.NONE
    if condition == DisplayCondition.TRUE_CONFIDENCE_VISIBLE:
        return DisplaySourceCondition.REPORTED_CONFIDENCE
    return DisplaySourceCondition.ASSIGNED_GRID


def grid_value(error_cost: float, condition: DisplayCondition) -> float:
    if condition not in MANIPULATED_CONDITIONS:
        raise ValueError(f"{condition} is not a manipulated condition")
    grid = GRIDS_BY_L[float(error_cost)]
    return grid[MANIPULATED_CONDITIONS.index(condition)]


class Study1DecisionRecord(StrictModel):
    """One Study 1 Stage-3 observation. Not a V2 VerificationDecisionRecord."""

    study_id: Literal["study1_causal_pilot"] = "study1_causal_pilot"
    experiment_version: Literal["study1"] = "study1"
    pilot_or_confirmatory: Literal["exploratory"] = "exploratory"
    phase: Study1Phase
    run_id: str = Field(min_length=1)
    request_key: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_alias: str = Field(min_length=1)
    model_endpoint: str = Field(min_length=1)
    returned_model_id: str | None = None
    provider: Literal["openai", "anthropic"]
    api_timestamp: str | None = None
    L: float
    C: float = STUDY1_VERIFICATION_COST
    reported_confidence: Probability
    displayed_confidence: Probability | None
    display_condition: DisplayCondition
    display_source_condition: DisplaySourceCondition
    frozen_answer: AnswerLabel
    stage1_correct: bool
    prompt_version: str = Field(min_length=1)
    prompt_family: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    selected_sample_hash: str = Field(min_length=1)
    code_commit: str | None = None
    repeat_index: int = Field(ge=0, le=2)
    attempt_number: int = Field(ge=1)
    raw_response: str | None = None
    parsed_action: VerificationAction | None = None
    parse_status: Literal["not_run", "success", "parse_failed", "refused"] = "not_run"
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    historical_answer_request_key: str = Field(min_length=1)
    historical_confidence_request_key: str = Field(min_length=1)
    model_settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        if value not in STUDY1_MODEL_ALIASES:
            raise ValueError(f"Study 1 model alias must be one of {STUDY1_MODEL_ALIASES}")
        return value

    @model_validator(mode="after")
    def validate_study1_design(self) -> Study1DecisionRecord:
        if self.C != STUDY1_VERIFICATION_COST:
            raise ValueError("Study 1 C must be 1")
        if self.L not in STUDY1_ERROR_COSTS:
            raise ValueError("Study 1 L must be 10 or 20")
        if self.display_condition == DisplayCondition.HIDDEN:
            if self.displayed_confidence is not None:
                raise ValueError("hidden condition must not store a displayed_confidence")
            if self.display_source_condition != DisplaySourceCondition.NONE:
                raise ValueError("hidden display_source_condition must be none")
        elif self.display_condition == DisplayCondition.TRUE_CONFIDENCE_VISIBLE:
            if self.displayed_confidence != self.reported_confidence:
                raise ValueError(
                    "true-visible displayed_confidence must equal reported_confidence"
                )
            if self.display_source_condition != DisplaySourceCondition.REPORTED_CONFIDENCE:
                raise ValueError(
                    "true-visible display_source_condition must be reported_confidence"
                )
        else:
            expected = grid_value(self.L, self.display_condition)
            if self.displayed_confidence != expected:
                raise ValueError(
                    "manipulated displayed_confidence must equal the frozen grid value"
                )
            if self.display_source_condition != DisplaySourceCondition.ASSIGNED_GRID:
                raise ValueError(
                    "manipulated display_source_condition must be assigned_grid"
                )
        if self.phase == Study1Phase.PRIMARY and self.repeat_index != 0:
            raise ValueError("primary phase rows must have repeat_index 0")
        if self.phase == Study1Phase.REPEATS and self.repeat_index not in {1, 2}:
            raise ValueError("repeat-extra rows must have repeat_index 1 or 2")
        return self


class Study1ExperimentConfig(StrictConfigModel):
    study_id: Literal["study1_causal_pilot"]
    experiment_version: Literal["study1"]
    pilot_or_confirmatory: Literal["exploratory"]
    status: Literal["DEVELOPMENT"]
    authority_condition: Literal["ai_system"]
    prompt_family: str
    prompt_version: str
    prompt_source: str
    seeds: dict[str, int]
    sample: dict[str, Any]
    models: list[str]
    historical: dict[str, Any]
    costs: dict[str, Any]
    displayed_confidence_grids: dict[float, list[float]]
    threshold_contrasts: dict[float, list[float]]
    conditions: list[str]
    phases: list[str]
    planned_calls: dict[str, int]
    model_inference: dict[str, Any]
    paid_call_safety: dict[str, Any]
    results: dict[str, Any]
    models_config_path: Path

    @field_validator("displayed_confidence_grids", "threshold_contrasts", mode="before")
    @classmethod
    def coerce_float_keys(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {float(key): item for key, item in value.items()}
        return value

    @model_validator(mode="after")
    def validate_frozen_pilot_design(self) -> Study1ExperimentConfig:
        if list(self.models) != list(STUDY1_MODEL_ALIASES):
            raise ValueError("Study 1 models must be GPT then Claude only")
        if self.costs.get("verification_cost") != STUDY1_VERIFICATION_COST:
            raise ValueError("Study 1 C must be 1")
        if tuple(self.costs.get("error_costs") or ()) != STUDY1_ERROR_COSTS:
            raise ValueError("Study 1 L values must be [10, 20]")
        if tuple(self.displayed_confidence_grids[10.0]) != STUDY1_GRID_L10:
            raise ValueError("L=10 grid is frozen")
        if tuple(self.displayed_confidence_grids[20.0]) != STUDY1_GRID_L20:
            raise ValueError("L=20 grid is frozen")
        if self.phases != ["primary", "repeats"]:
            raise ValueError("Study 1 phases must be primary and repeats only")
        if self.planned_calls.get("primary") != STUDY1_PRIMARY_CALLS:
            raise ValueError("planned primary calls must be 2800")
        if self.planned_calls.get("repeat_extra") != STUDY1_REPEAT_EXTRA_CALLS:
            raise ValueError("planned repeat-extra calls must be 1120")
        if self.sample.get("selection_uses_observed_results") is not False:
            raise ValueError("selection must not use observed results")
        return self

    def resolve_path(self, path: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
        resolved = Path(path)
        return resolved if resolved.is_absolute() else project_root / resolved

    def historical_sqlite(self) -> Path:
        return self.resolve_path(self.historical["v2_checkpoint_path"])

    def study1_sqlite(self) -> Path:
        return self.resolve_path(self.results["checkpoint_path"])


def load_study1_config(
    path: str | Path = DEFAULT_STUDY1_CONFIG,
) -> Study1ExperimentConfig:
    return Study1ExperimentConfig.model_validate(_load_yaml(Path(path)))
