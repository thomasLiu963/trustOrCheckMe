"""Validated YAML configuration for the frozen pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_CONFIG = PROJECT_ROOT / "config" / "experiment.yaml"
DEFAULT_V2_EXPERIMENT_CONFIG = PROJECT_ROOT / "config" / "experiment_v2.yaml"
DEFAULT_MODELS_CONFIG = PROJECT_ROOT / "config" / "models.yaml"


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetConfig(StrictConfigModel):
    name: Literal["mmlu_pro"]
    hf_repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: Literal["test"]
    pilot_size: int = Field(gt=0)
    main_size: int = Field(gt=0)
    stratify_by: Literal["category"]
    pilot_sample_path: Path
    pilot_manifest_path: Path
    cache_dir: Path

    @model_validator(mode="after")
    def validate_sizes(self) -> DatasetConfig:
        if self.main_size < self.pilot_size:
            raise ValueError("main_size must be at least pilot_size")
        return self


class CostsConfig(StrictConfigModel):
    verification_cost: float = Field(gt=0)
    error_costs: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_error_costs(self) -> CostsConfig:
        if any(cost <= 0 for cost in self.error_costs):
            raise ValueError("all error costs must be positive")
        if len(set(self.error_costs)) != len(self.error_costs):
            raise ValueError("error costs must be unique")
        if self.error_costs != sorted(self.error_costs):
            raise ValueError("error costs must be in ascending order")
        return self


class ConfidenceConfig(StrictConfigModel):
    min: float = Field(ge=0, le=1)
    max: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_range(self) -> ConfidenceConfig:
        if self.min >= self.max:
            raise ValueError("confidence min must be less than max")
        return self


class InferenceConfig(StrictConfigModel):
    no_tools: Literal[True]
    no_external_retrieval: Literal[True]
    concurrency: int = Field(default=4, gt=0)
    max_transient_retries: int = Field(default=3, ge=0)
    max_parse_repairs: int = Field(default=1, ge=0)


class ResultsConfig(StrictConfigModel):
    checkpoint_path: Path
    analysis_directory: Path
    paper_output_directory: Path


class BootstrapConfig(StrictConfigModel):
    n_resamples: int = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)


class CalibrationConfig(StrictConfigModel):
    method: Literal["isotonic"]
    calibration_fraction: float = Field(gt=0, lt=1)


class ExperimentConfig(StrictConfigModel):
    seed: int = Field(ge=0)
    dataset: DatasetConfig
    costs: CostsConfig
    confidence: ConfidenceConfig
    model_inference: InferenceConfig
    results: ResultsConfig
    bootstrap: BootstrapConfig
    calibration: CalibrationConfig

    def resolve_path(self, path: Path, project_root: Path = PROJECT_ROOT) -> Path:
        """Resolve a configured path relative to the repository root."""
        return path if path.is_absolute() else project_root / path


class V2DatasetConfig(StrictConfigModel):
    name: Literal["mmlu_pro"]
    hf_repo: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    split: Literal["test"]
    v2a_size: int = Field(gt=0)
    v2b_size: int = Field(gt=0)
    robustness_size: int = Field(gt=0)
    include_v1_sample: Literal[True]
    stratify_by: Literal["category"]
    v1_sample_path: Path
    v2a_sample_path: Path
    v2b_sample_path: Path
    robustness_sample_path: Path
    manifest_directory: Path
    cache_dir: Path

    @model_validator(mode="after")
    def validate_sizes(self) -> V2DatasetConfig:
        if self.v2b_size < self.v2a_size:
            raise ValueError("v2b_size must be at least v2a_size")
        if self.robustness_size > self.v2b_size:
            raise ValueError("robustness_size cannot exceed v2b_size")
        return self


class V2FactorsConfig(StrictConfigModel):
    decision_owners: list[Literal["human", "ai_system"]]
    confidence_visibility: list[Literal["hidden", "visible"]]

    @model_validator(mode="after")
    def validate_complete_factorial(self) -> V2FactorsConfig:
        if self.decision_owners != ["human", "ai_system"]:
            raise ValueError("V2 decision owners must be [human, ai_system]")
        if self.confidence_visibility != ["hidden", "visible"]:
            raise ValueError("V2 confidence states must be [hidden, visible]")
        return self


class V2PromptFamiliesConfig(StrictConfigModel):
    primary: str = Field(min_length=1)
    robustness: str = Field(min_length=1)


class V2RobustnessConfig(StrictConfigModel):
    enabled_by_default: Literal[False]
    subset_size: int = Field(gt=0)
    confidence_visibility: list[Literal["hidden"]]


class V2ResultsConfig(ResultsConfig):
    v1_checkpoint_path: Path


class V2ReuseConfig(StrictConfigModel):
    allow_v1_stage1: bool
    allow_v1_stage2: bool
    require_exact_model_config_match: Literal[True]
    require_exact_prompt_version_match: Literal[True]


class V2ExperimentConfig(StrictConfigModel):
    experiment_id: Literal["trust_check_v2"]
    experiment_version: Literal["v2"]
    seed: int = Field(ge=0)
    dataset: V2DatasetConfig
    costs: CostsConfig
    factors: V2FactorsConfig
    prompt_families: V2PromptFamiliesConfig
    robustness: V2RobustnessConfig
    model_inference: InferenceConfig
    results: V2ResultsConfig
    bootstrap: BootstrapConfig
    calibration: CalibrationConfig
    reuse: V2ReuseConfig

    def resolve_path(self, path: Path, project_root: Path = PROJECT_ROOT) -> Path:
        return path if path.is_absolute() else project_root / path

    @model_validator(mode="after")
    def validate_frozen_design(self) -> V2ExperimentConfig:
        if self.costs.verification_cost != 1.0:
            raise ValueError("V2 verification cost must be 1")
        if self.costs.error_costs != [2.0, 5.0, 10.0, 20.0]:
            raise ValueError("V2 error costs must be [2, 5, 10, 20]")
        if self.robustness.subset_size != self.dataset.robustness_size:
            raise ValueError("robustness subset sizes must agree")
        return self


class PricingConfig(StrictConfigModel):
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class ThinkingConfig(StrictConfigModel):
    type: Literal["disabled"]


class ModelSpec(StrictConfigModel):
    provider: Literal["openai", "anthropic", "google", "xai"]
    api_model: str = Field(min_length=1)
    api_style: Literal["responses", "messages", "google_genai"]
    max_output_tokens: int = Field(gt=0)
    pricing_per_million_tokens: PricingConfig
    reasoning_effort: Literal["none"] | None = None
    thinking: ThinkingConfig | None = None
    thinking_level: Literal["low"] | None = None

    @model_validator(mode="after")
    def validate_provider_settings(self) -> ModelSpec:
        if self.provider == "openai":
            if self.api_style != "responses" or self.reasoning_effort != "none":
                raise ValueError(
                    "OpenAI pilot models must use Responses with reasoning effort none"
                )
            if self.thinking is not None:
                raise ValueError("OpenAI models must not define Anthropic thinking")
            if self.thinking_level is not None:
                raise ValueError("OpenAI models must not define thinking_level")
        elif self.provider == "anthropic":
            if self.api_style != "messages" or self.thinking is None:
                raise ValueError(
                    "Anthropic pilot models must use Messages with thinking disabled"
                )
            if self.reasoning_effort is not None:
                raise ValueError("Anthropic models must not define reasoning_effort")
            if self.thinking_level is not None:
                raise ValueError("Anthropic models must not define thinking_level")
        elif self.provider == "google":
            if self.api_style != "google_genai" or self.thinking_level != "low":
                raise ValueError(
                    "Google V2 models must use Google GenAI with thinking level low"
                )
            if self.reasoning_effort is not None or self.thinking is not None:
                raise ValueError("Google models use only thinking_level")
        else:
            if self.api_style != "responses":
                raise ValueError("xAI V2 models must use the Responses API style")
            if any(
                value is not None
                for value in (self.reasoning_effort, self.thinking, self.thinking_level)
            ):
                raise ValueError("xAI non-reasoning models need no reasoning setting")
        return self


class ModelsConfig(StrictConfigModel):
    models: dict[str, ModelSpec] = Field(min_length=1)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return value


def load_experiment_config(
    path: str | Path = DEFAULT_EXPERIMENT_CONFIG,
) -> ExperimentConfig:
    """Load and validate experiment settings."""
    return ExperimentConfig.model_validate(_load_yaml(Path(path)))


def load_v2_experiment_config(
    path: str | Path = DEFAULT_V2_EXPERIMENT_CONFIG,
) -> V2ExperimentConfig:
    """Load and validate the frozen V2 factorial settings."""
    return V2ExperimentConfig.model_validate(_load_yaml(Path(path)))


def load_models_config(path: str | Path = DEFAULT_MODELS_CONFIG) -> ModelsConfig:
    """Load and validate model/provider settings."""
    return ModelsConfig.model_validate(_load_yaml(Path(path)))
