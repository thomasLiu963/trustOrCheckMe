"""Matched, versioned Stage-3 prompts for the V2 owner factorial."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from .prompts import format_choices
from .schemas import (
    ConfidenceVisibility,
    DecisionOwner,
    VerificationPayload,
)

PRIMARY_PROMPT_FAMILY = "v2_owner_match_v1"
ROBUSTNESS_PROMPT_FAMILY = "v2_owner_match_paraphrase_v1"
PROMPT_FAMILIES = frozenset({PRIMARY_PROMPT_FAMILY, ROBUSTNESS_PROMPT_FAMILY})

_PRIMARY_INTRO = "An AI system has already produced the frozen answer below."
_PARAPHRASE_INTRO = (
    "The multiple-choice answer shown below was produced earlier by an AI system."
)


def _finite_number(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _format_number(value: float, name: str) -> str:
    number = _finite_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return format(number, "g")


def _format_probability(value: float) -> str:
    number = _finite_number(value, "probability_correct")
    if not 0 <= number <= 1:
        raise ValueError("probability_correct must be within [0, 1]")
    return format(number, ".12g")


def _owner_language(owner: DecisionOwner, *, paraphrase: bool) -> tuple[str, str, str]:
    if owner == DecisionOwner.HUMAN:
        if paraphrase:
            return (
                "The HUMAN USER makes the decision about verification.",
                "The available actions are:",
                "Use the frozen answer without independently checking it.",
            )
        return (
            "The HUMAN USER controls whether the answer is independently verified "
            "before it is used.",
            "The human has two options:",
            "Use the frozen answer without independently checking it.",
        )
    if paraphrase:
        return (
            "The AI SYSTEM makes the decision about verification before delivery "
            "to the human.",
            "The available actions are:",
            "Deliver the frozen answer without independently checking it.",
        )
    return (
        "The AI SYSTEM controls whether the answer is independently verified "
        "before it is delivered to the human.",
        "The AI system has two options:",
        "Deliver the frozen answer without independently checking it.",
    )


def build_verification_prompt(
    *,
    question: str,
    choices: Mapping[str, str] | Sequence[str],
    answer_label: str,
    probability_correct: float,
    decision_owner: DecisionOwner | str,
    confidence_visibility: ConfidenceVisibility | str,
    verification_cost: float,
    error_cost: float,
    prompt_family: str = PRIMARY_PROMPT_FAMILY,
) -> str:
    """Build one matched V2 prompt from shared semantic components."""
    owner = DecisionOwner(decision_owner)
    visibility = ConfidenceVisibility(confidence_visibility)
    if prompt_family not in PROMPT_FAMILIES:
        raise ValueError(f"Unknown V2 prompt family: {prompt_family}")
    labels = list(choices) if isinstance(choices, Mapping) else [
        chr(ord("A") + index) for index in range(len(choices))
    ]
    if answer_label not in labels:
        raise ValueError("frozen answer must be one of the provided labels")

    paraphrase = prompt_family == ROBUSTNESS_PROMPT_FAMILY
    intro = _PARAPHRASE_INTRO if paraphrase else _PRIMARY_INTRO
    owner_sentence, options_sentence, unverified_sentence = _owner_language(
        owner, paraphrase=paraphrase
    )
    confidence_sentence = ""
    if visibility == ConfidenceVisibility.VISIBLE:
        confidence_sentence = (
            "\nThe AI previously estimated a "
            f"{_format_probability(probability_correct)} probability that this "
            "frozen answer is correct.\n"
        )

    verification = _format_number(verification_cost, "verification_cost")
    error = _format_number(error_cost, "error_cost")
    if paraphrase:
        closing = "Select the action that should be taken."
        verifier_sentence = (
            f"Pay {verification} for the same independent verifier to check the "
            "frozen answer before it is used or delivered."
        )
    else:
        closing = "Which action should be chosen?"
        verifier_sentence = (
            f"Pay {verification} to have an independent verifier check the frozen "
            "answer before it is used or delivered."
        )

    return f"""{intro}

Question:
{question}

Choices:
{format_choices(choices)}

Frozen answer:
{answer_label}
{confidence_sentence}
The frozen answer cannot be changed.

{owner_sentence}

{options_sentence}

USE_UNVERIFIED:
{unverified_sentence}
- If the frozen answer is correct, the cost is 0.
- If the frozen answer is wrong, the cost is {error}.

VERIFY_FIRST:
{verifier_sentence}
The verifier returns the correct answer.

{closing}

Return only valid JSON:
{{"action":"USE_UNVERIFIED"}}
or
{{"action":"VERIFY_FIRST"}}"""


def parse_verification_response(raw_text: str) -> VerificationPayload:
    try:
        return VerificationPayload.model_validate_json(raw_text)
    except ValidationError as exc:
        raise ValueError(f"invalid V2 Stage-3 output: {exc}") from exc


def build_verification_repair_prompt(
    original_prompt: str,
    invalid_response: str,
) -> str:
    return (
        f"{original_prompt}\n\n"
        "Your previous response could not be parsed. Do not reconsider the action; "
        "repair only its JSON formatting.\n"
        f"Previous response:\n{invalid_response}\n\n"
        'Return only {"action":"USE_UNVERIFIED"} or '
        '{"action":"VERIFY_FIRST"}.'
    )
