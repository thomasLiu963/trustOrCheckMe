import pytest

from src.prompts import (
    OutputParseError,
    build_stage_2_prompt,
    build_stage_3_prompt,
    parse_answer_response,
    parse_confidence_response,
    parse_trust_response,
)

CHOICES = {"A": "one", "B": "two", "C": "three"}


def test_valid_payloads() -> None:
    assert parse_answer_response('{"answer":"C"}', CHOICES).answer == "C"
    assert (
        parse_confidence_response('{"probability_correct":0.72}').probability_correct
        == 0.72
    )
    assert parse_trust_response('{"recommendation":"VERIFY"}').recommendation.value == (
        "VERIFY"
    )


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":"c"}',
        '{"answer":"D"}',
        '{"answer":"C","explanation":"because"}',
        '```json\n{"answer":"C"}\n```',
        "C",
    ],
)
def test_answer_parser_is_strict(raw: str) -> None:
    with pytest.raises(OutputParseError):
        parse_answer_response(raw, CHOICES)


@pytest.mark.parametrize("value", [-0.01, 1.01, "high", None])
def test_confidence_range_rejected(value: object) -> None:
    with pytest.raises(OutputParseError):
        parse_confidence_response(f'{{"probability_correct":{value!r}}}')


def test_stage_separation() -> None:
    confidence = build_stage_2_prompt(question="Q?", choices=CHOICES, answer_label="B")
    trust = build_stage_3_prompt(
        question="Q?",
        choices=CHOICES,
        answer_label="B",
        verification_cost=1,
        error_cost=20,
    )
    assert "error cost" not in confidence.lower()
    assert "verification costs" not in confidence.lower()
    assert "probability_correct" not in trust
    assert "0.72" not in trust
