"""Versioned, provider-neutral prompts and strict output parsing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import ValidationError

from .schemas import AnswerPayload, ConfidencePayload, TrustPayload

Stage = Literal["stage_1", "stage_2", "stage_3"]

STAGE_1_PROMPT_VERSION = "stage_1_answer_v1"
STAGE_2_PROMPT_VERSION = "stage_2_confidence_v1"
STAGE_3_PROMPT_VERSION = "stage_3_direct_trust_v1"

PROMPT_VERSIONS: dict[Stage, str] = {
    "stage_1": STAGE_1_PROMPT_VERSION,
    "stage_2": STAGE_2_PROMPT_VERSION,
    "stage_3": STAGE_3_PROMPT_VERSION,
}

# Runtime-facing aliases keep stage naming explicit and stable.
ANSWER_PROMPT_VERSION = STAGE_1_PROMPT_VERSION
CONFIDENCE_PROMPT_VERSION = STAGE_2_PROMPT_VERSION
TRUST_PROMPT_VERSION = STAGE_3_PROMPT_VERSION

STAGE_1_TEMPLATE = """You are answering a multiple-choice question.

Question:
{question}

Choices:
{formatted_choices}

Return only valid JSON in exactly this form:
{{"answer":"A"}}

The answer must be one of the provided choice labels."""

STAGE_2_TEMPLATE = """You previously answered the multiple-choice question below.

Question:
{question}

Choices:
{formatted_choices}

Your frozen answer was:
{answer_label}

Do not change the answer and do not solve the question again.

Estimate the probability that this frozen answer is correct.
Return a number between 0 and 1.

Return only valid JSON in exactly this form:
{{"probability_correct":0.72}}"""

STAGE_3_TEMPLATE = """A human is deciding whether to rely on an AI-generated answer or independently verify it.

Question:
{question}

Choices:
{formatted_choices}

The AI's frozen answer is:
{answer_label}

The answer cannot be changed.

If the human chooses VERIFY, verification costs {verification_cost} unit(s) and then the human obtains the correct answer.

If the human chooses RELY:
- if the frozen answer is correct, the cost is 0;
- if the frozen answer is wrong, the cost is {error_cost} unit(s).

Recommend what the human should do.

Return only valid JSON in exactly one of these forms:
{{"recommendation":"RELY"}}
{{"recommendation":"VERIFY"}}"""

REPAIR_TEMPLATES: dict[Stage, str] = {
    "stage_1": (
        'Return only valid JSON in exactly this form: {"answer":"A"}. '
        "Replace A with one provided choice label. Do not add any other text."
    ),
    "stage_2": (
        "Return only valid JSON in exactly this form: "
        '{"probability_correct":0.72}. Replace 0.72 with a number from 0 to 1. '
        "Do not add any other text."
    ),
    "stage_3": (
        'Return only valid JSON as {"recommendation":"RELY"} or '
        '{"recommendation":"VERIFY"}. Do not add any other text.'
    ),
}


class OutputParseError(ValueError):
    """Raised when a model response violates the stage output contract."""


def format_choices(choices: Mapping[str, str] | Sequence[str]) -> str:
    """Format choices deterministically without changing their text."""
    if isinstance(choices, Mapping):
        items = list(choices.items())
    else:
        items = [
            (chr(ord("A") + index), choice) for index, choice in enumerate(choices)
        ]
    if len(items) < 2:
        raise ValueError("at least two choices are required")
    expected = [chr(ord("A") + index) for index in range(len(items))]
    labels = [label for label, _ in items]
    if labels != expected:
        raise ValueError("choice labels must be contiguous and ordered from A")
    if any(not text.strip() for _, text in items):
        raise ValueError("choice text must not be empty")
    return "\n".join(f"{label}. {text}" for label, text in items)


def build_stage_1_prompt(
    *, question: str, choices: Mapping[str, str] | Sequence[str]
) -> str:
    return STAGE_1_TEMPLATE.format(
        question=question, formatted_choices=format_choices(choices)
    )


def _choice_labels(choices: Mapping[str, str] | Sequence[str]) -> list[str]:
    if isinstance(choices, Mapping):
        return list(choices)
    return [chr(ord("A") + index) for index in range(len(choices))]


def _validate_frozen_answer(
    answer_label: str, choices: Mapping[str, str] | Sequence[str]
) -> None:
    if answer_label not in _choice_labels(choices):
        raise ValueError("frozen answer must be one of the provided choice labels")


def build_stage_2_prompt(
    *,
    question: str,
    choices: Mapping[str, str] | Sequence[str],
    answer_label: str,
) -> str:
    _validate_frozen_answer(answer_label, choices)
    return STAGE_2_TEMPLATE.format(
        question=question,
        formatted_choices=format_choices(choices),
        answer_label=answer_label,
    )


def _format_cost(cost: float) -> str:
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("costs must be finite and non-negative")
    return format(cost, "g")


def build_stage_3_prompt(
    *,
    question: str,
    choices: Mapping[str, str] | Sequence[str],
    answer_label: str,
    verification_cost: float,
    error_cost: float,
) -> str:
    """Build the direct-advice prompt; Stage-2 confidence is intentionally absent."""
    _validate_frozen_answer(answer_label, choices)
    return STAGE_3_TEMPLATE.format(
        question=question,
        formatted_choices=format_choices(choices),
        answer_label=answer_label,
        verification_cost=_format_cost(verification_cost),
        error_cost=_format_cost(error_cost),
    )


def parse_answer_payload(raw: str, allowed_labels: Sequence[str]) -> AnswerPayload:
    try:
        payload = AnswerPayload.model_validate_json(raw)
    except ValidationError as exc:
        raise OutputParseError(f"invalid Stage-1 output: {exc}") from exc
    if payload.answer not in allowed_labels:
        raise OutputParseError("Stage-1 answer is not one of the provided labels")
    return payload


def parse_confidence_payload(raw: str) -> ConfidencePayload:
    try:
        return ConfidencePayload.model_validate_json(raw)
    except ValidationError as exc:
        raise OutputParseError(f"invalid Stage-2 output: {exc}") from exc


def parse_trust_payload(raw: str) -> TrustPayload:
    try:
        return TrustPayload.model_validate_json(raw)
    except ValidationError as exc:
        raise OutputParseError(f"invalid Stage-3 output: {exc}") from exc


def build_repair_prompt(
    stage: Stage | Literal["answer", "confidence", "trust"],
    original_prompt: str = "",
    invalid_response: str = "",
    parse_error: str = "",
) -> str:
    """Return a bounded-retry instruction that changes only output formatting."""
    runtime_to_prompt_stage: dict[str, Stage] = {
        "answer": "stage_1",
        "confidence": "stage_2",
        "trust": "stage_3",
    }
    instruction = REPAIR_TEMPLATES[runtime_to_prompt_stage.get(stage, stage)]
    if not original_prompt:
        return instruction
    return (
        f"{original_prompt}\n\n"
        "Your previous response could not be parsed. Do not reconsider or change "
        "the substantive response; only repair its format.\n"
        f"Previous response:\n{invalid_response}\n\n{instruction}"
    )


# Common runtime interface.
build_answer_prompt = build_stage_1_prompt
build_confidence_prompt = build_stage_2_prompt
build_trust_prompt = build_stage_3_prompt


def parse_answer_response(
    raw_text: str, choices: Mapping[str, str] | Sequence[str]
) -> AnswerPayload:
    return parse_answer_payload(raw_text, _choice_labels(choices))


def parse_confidence_response(raw_text: str) -> ConfidencePayload:
    return parse_confidence_payload(raw_text)


def parse_trust_response(raw_text: str) -> TrustPayload:
    return parse_trust_payload(raw_text)
