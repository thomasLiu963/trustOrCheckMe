"""Shared validated data contracts for all experimental stages."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

AnswerLabel = Annotated[str, StringConstraints(pattern=r"^[A-Z]$")]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


def utc_now() -> datetime:
    return datetime.now(UTC)


def _contains_forbidden_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(
            _contains_forbidden_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Recommendation(str, Enum):
    RELY = "RELY"
    VERIFY = "VERIFY"


class AnswerPayload(StrictModel):
    answer: AnswerLabel


class ConfidencePayload(StrictModel):
    probability_correct: Probability


class TrustPayload(StrictModel):
    recommendation: Recommendation


OutputPayload = AnswerPayload | ConfidencePayload | TrustPayload


class BenchmarkExample(StrictModel):
    example_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    choices: dict[AnswerLabel, str] = Field(min_length=2)
    correct_label: AnswerLabel
    split: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    source_index: int = Field(ge=0)

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, choices: dict[str, str]) -> dict[str, str]:
        if any(not text.strip() for text in choices.values()):
            raise ValueError("choice text must not be empty")
        expected = [chr(ord("A") + index) for index in range(len(choices))]
        if list(choices) != expected:
            raise ValueError("choice labels must be contiguous and ordered from A")
        return choices

    @model_validator(mode="after")
    def validate_correct_label(self) -> BenchmarkExample:
        if self.correct_label not in self.choices:
            raise ValueError("correct_label must identify one of the choices")
        return self


class AnswerRecord(StrictModel):
    run_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    answer_label: AnswerLabel
    is_correct: bool
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    choices: dict[AnswerLabel, str] = Field(min_length=2)
    correct_label: AnswerLabel
    raw_response: str
    prompt_version: str = Field(min_length=1)
    timestamp: AwareDatetime = Field(default_factory=utc_now)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider_metadata")
    @classmethod
    def exclude_chain_of_thought(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        if _contains_forbidden_key(metadata, "cot_content"):
            raise ValueError("cot_content must never be persisted")
        return metadata


class ConfidenceRecord(StrictModel):
    run_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    frozen_answer_label: AnswerLabel
    probability_correct: Probability
    raw_response: str
    prompt_version: str = Field(min_length=1)
    timestamp: AwareDatetime = Field(default_factory=utc_now)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class TrustDecisionRecord(StrictModel):
    run_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    frozen_answer_label: AnswerLabel
    verification_cost: NonNegativeFloat
    error_cost: NonNegativeFloat
    recommendation: Recommendation
    raw_response: str
    prompt_version: str = Field(min_length=1)
    timestamp: AwareDatetime = Field(default_factory=utc_now)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredDecisionRecord(StrictModel):
    run_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    verification_cost: NonNegativeFloat
    error_cost: NonNegativeFloat
    recommendation: Recommendation
    is_correct: bool
    unsafe_reliance: bool
    unnecessary_verification: bool
    realized_cost: NonNegativeFloat
    oracle_cost: NonNegativeFloat
    regret: NonNegativeFloat
    confidence_baseline_action: Recommendation
    calibrated_confidence_baseline_action: Recommendation
    direct_vs_raw_confidence_disagreement: bool
    direct_vs_calibrated_confidence_disagreement: bool

    @model_validator(mode="after")
    def validate_indicators(self) -> ScoredDecisionRecord:
        expected_unsafe = (
            self.recommendation == Recommendation.RELY and not self.is_correct
        )
        expected_unnecessary = (
            self.recommendation == Recommendation.VERIFY and self.is_correct
        )
        if self.unsafe_reliance != expected_unsafe:
            raise ValueError("unsafe_reliance is inconsistent with the decision")
        if self.unnecessary_verification != expected_unnecessary:
            raise ValueError(
                "unnecessary_verification is inconsistent with the decision"
            )
        if self.direct_vs_raw_confidence_disagreement != (
            self.recommendation != self.confidence_baseline_action
        ):
            raise ValueError("raw-confidence disagreement indicator is inconsistent")
        if self.direct_vs_calibrated_confidence_disagreement != (
            self.recommendation != self.calibrated_confidence_baseline_action
        ):
            raise ValueError(
                "calibrated-confidence disagreement indicator is inconsistent"
            )
        return self


class ModelResponse(StrictModel):
    model_alias: str = Field(min_length=1)
    provider: Literal["openai", "anthropic"]
    requested_model_id: str = Field(min_length=1)
    provider_model_id: str | None = None
    stage: Literal["answer", "confidence", "trust"]
    request_key: str = Field(min_length=1)
    status: Literal["received", "success", "failed"] = "received"
    raw_response: str
    parsed_response: OutputPayload | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_seconds: NonNegativeFloat
    finish_reason: str | None = None
    refused: bool | None = None
    error_type: str | None = None
    error_message: str | None = None
    timestamp: AwareDatetime = Field(default_factory=utc_now)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider_metadata")
    @classmethod
    def exclude_chain_of_thought(cls, metadata: dict[str, Any]) -> dict[str, Any]:
        if _contains_forbidden_key(metadata, "cot_content"):
            raise ValueError("cot_content must never be persisted")
        return metadata

    @model_validator(mode="after")
    def validate_status(self) -> ModelResponse:
        if self.status == "success" and self.parsed_response is None:
            raise ValueError("successful responses require a parsed payload")
        if self.status == "failed" and self.parsed_response is not None:
            raise ValueError("failed responses cannot contain a parsed payload")
        return self
