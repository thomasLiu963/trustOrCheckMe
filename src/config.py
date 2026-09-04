"""Validated YAML configuration for the frozen pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_CONFIG = PROJECT_ROOT / "config" / "experiment.yaml"
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


class PricingConfig(StrictConfigModel):
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class ThinkingConfig(StrictConfigModel):
    type: Literal["disabled"]


class ModelSpec(StrictConfigModel):
    provider: Literal["openai", "anthropic"]
    api_model: str = Field(min_length=1)
    api_style: Literal["responses", "messages"]
    max_output_tokens: int = Field(gt=0)
    pricing_per_million_tokens: PricingConfig
    reasoning_effort: Literal["none"] | None = None
    thinking: ThinkingConfig | None = None

    @model_validator(mode="after")
    def validate_provider_settings(self) -> ModelSpec:
        if self.provider == "openai":
            if self.api_style != "responses" or self.reasoning_effort != "none":
                raise ValueError(
                    "OpenAI pilot models must use Responses with reasoning effort none"
                )
            if self.thinking is not None:
                raise ValueError("OpenAI models must not define Anthropic thinking")
        else:
            if self.api_style != "messages" or self.thinking is None:
                raise ValueError(
                    "Anthropic pilot models must use Messages with thinking disabled"
                )
            if self.reasoning_effort is not None:
                raise ValueError("Anthropic models must not define reasoning_effort")
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


def load_models_config(path: str | Path = DEFAULT_MODELS_CONFIG) -> ModelsConfig:
    """Load and validate model/provider settings."""
    return ModelsConfig.model_validate(_load_yaml(Path(path)))
