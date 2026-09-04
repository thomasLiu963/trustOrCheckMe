import asyncio

import pytest

from src.config import load_models_config
from src.model_adapters import create_adapter


@pytest.fixture
def adapters():
    models = load_models_config().models
    return {alias: create_adapter(alias, config) for alias, config in models.items()}


def test_openai_frozen_payload(adapters) -> None:
    payload = (
        adapters["openai_gpt56_sol"]
        .prepare_request(stage="answer", prompt="test")
        .payload
    )
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["input"] == "test"
    assert payload["max_output_tokens"] == 64
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["required"] == ["answer"]


def test_anthropic_frozen_payload(adapters) -> None:
    payload = (
        adapters["anthropic_sonnet5"]
        .prepare_request(stage="trust", prompt="test")
        .payload
    )
    assert payload["model"] == "claude-sonnet-5"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 64
    assert payload["messages"] == [{"role": "user", "content": "test"}]
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert payload["output_config"]["format"]["schema"]["required"] == [
        "recommendation"
    ]
    assert not {"temperature", "top_p", "top_k", "tools"} & payload.keys()


def test_confidence_schema_uses_cross_provider_supported_subset(adapters) -> None:
    for adapter in adapters.values():
        payload = adapter.prepare_request(stage="confidence", prompt="test").payload
        schema = (
            payload["text"]["format"]["schema"]
            if adapter.provider == "openai"
            else payload["output_config"]["format"]["schema"]
        )
        probability = schema["properties"]["probability_correct"]
        assert probability == {"type": "number"}


def test_paid_calls_are_hard_gated(adapters) -> None:
    with pytest.raises(PermissionError):
        asyncio.run(
            adapters["openai_gpt56_sol"].generate(
                stage="answer",
                prompt="test",
                request_key="request",
                allow_paid=False,
            )
        )
