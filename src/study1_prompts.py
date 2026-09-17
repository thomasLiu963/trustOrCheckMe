"""Study 1 Stage-3 prompts cloned from historical V2 AI-authority wording.

Historical source (do not rewrite the prose):
    src/v2_prompts.py
    function: build_verification_prompt
    family: PRIMARY_PROMPT_FAMILY = v2_owner_match_v1  (not the paraphrase family)
    owner branch: DecisionOwner.AI_SYSTEM, paraphrase=False
    lines 21, 69-73, 105-111, 122-126, 128-159

The only intended experimental change in manipulated conditions, relative to the
matching visible AI-authority template, is the numeric displayed confidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from .prompts import format_choices
from .schemas import VerificationPayload
from .v2_prompts import (
    PRIMARY_PROMPT_FAMILY,
    _format_number,
    _format_probability,
    build_verification_repair_prompt,
)

STUDY1_PROMPT_FAMILY = "study1_ai_authority_v2_wording_v1"
STUDY1_PROMPT_VERSION = "study1_causal_pilot_stage3_v1"
STUDY1_AUTHORITY = "ai_system"
HISTORICAL_PROMPT_SOURCE = (
    "src/v2_prompts.py:build_verification_prompt "
    "AI-system primary (v2_owner_match_v1, paraphrase=False)"
)

# Copied exactly from src/v2_prompts.py AI-system primary branch.
_PRIMARY_INTRO = "An AI system has already produced the frozen answer below."
_OWNER_SENTENCE = (
    "The AI SYSTEM controls whether the answer is independently verified "
    "before it is delivered to the human."
)
_OPTIONS_SENTENCE = "The AI system has two options:"
_UNVERIFIED_SENTENCE = (
    "Deliver the frozen answer without independently checking it."
)
_CLOSING = "Which action should be chosen?"
_CONFIDENCE_SENTENCE_PREFIX = "The AI previously estimated a "
_CONFIDENCE_SENTENCE_SUFFIX = " probability that this frozen answer is correct."


def format_displayed_probability(value: float) -> str:
    """Use the historical V2 probability formatter; do not pretty-print."""
    return _format_probability(value)


def confidence_sentence(displayed_confidence: float) -> str:
    return (
        "\n"
        + _CONFIDENCE_SENTENCE_PREFIX
        + format_displayed_probability(displayed_confidence)
        + _CONFIDENCE_SENTENCE_SUFFIX
        + "\n"
    )


def build_study1_verification_prompt(
    *,
    question: str,
    choices: Mapping[str, str] | Sequence[str],
    frozen_answer: str,
    verification_cost: float,
    error_cost: float,
    displayed_confidence: float | None,
) -> str:
    """Render one Study 1 Stage-3 prompt using historical AI-authority wording."""
    labels = list(choices) if isinstance(choices, Mapping) else [
        chr(ord("A") + index) for index in range(len(choices))
    ]
    if frozen_answer not in labels:
        raise ValueError("frozen answer must be one of the provided labels")
    if displayed_confidence is None:
        inserted_confidence = ""
    else:
        inserted_confidence = confidence_sentence(displayed_confidence)

    verification = _format_number(verification_cost, "verification_cost")
    error = _format_number(error_cost, "error_cost")
    verifier_sentence = (
        f"Pay {verification} to have an independent verifier check the frozen "
        "answer before it is used or delivered."
    )
    return f"""{_PRIMARY_INTRO}

Question:
{question}

Choices:
{format_choices(choices)}

Frozen answer:
{frozen_answer}
{inserted_confidence}
The frozen answer cannot be changed.

{_OWNER_SENTENCE}

{_OPTIONS_SENTENCE}

USE_UNVERIFIED:
{_UNVERIFIED_SENTENCE}
- If the frozen answer is correct, the cost is 0.
- If the frozen answer is wrong, the cost is {error}.

VERIFY_FIRST:
{verifier_sentence}
The verifier returns the correct answer.

{_CLOSING}

Return only valid JSON:
{{"action":"USE_UNVERIFIED"}}
or
{{"action":"VERIFY_FIRST"}}"""


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def parse_study1_verification_response(raw_text: str) -> VerificationPayload:
    try:
        return VerificationPayload.model_validate_json(raw_text)
    except ValidationError as exc:
        raise ValueError(f"invalid Study 1 Stage-3 output: {exc}") from exc


def build_study1_repair_prompt(original_prompt: str, invalid_response: str) -> str:
    """Reuse the historical fixed repair wording; never customize it."""
    return build_verification_repair_prompt(original_prompt, invalid_response)


def historical_visible_matches_v2_builder(
    *,
    question: str,
    choices: Mapping[str, str] | Sequence[str],
    frozen_answer: str,
    probability: float,
    verification_cost: float,
    error_cost: float,
) -> bool:
    """Sanity check: Study 1 true-visible must match V2 AI-visible primary."""
    from .v2_prompts import build_verification_prompt

    study1 = build_study1_verification_prompt(
        question=question,
        choices=choices,
        frozen_answer=frozen_answer,
        verification_cost=verification_cost,
        error_cost=error_cost,
        displayed_confidence=probability,
    )
    historical = build_verification_prompt(
        question=question,
        choices=choices,
        answer_label=frozen_answer,
        probability_correct=probability,
        decision_owner="ai_system",
        confidence_visibility="visible",
        verification_cost=verification_cost,
        error_cost=error_cost,
        prompt_family=PRIMARY_PROMPT_FAMILY,
    )
    return study1 == historical
