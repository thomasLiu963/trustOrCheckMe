import asyncio

import pytest

from src.config import load_models_config
from src.model_adapters import create_adapter


@pytest.fixture
def adapters():
    models = load_models_config().models
    return {alias: create_adapter(alias, config) for alias, config in models.items()}


def test_openai_frozen_payload(adapters) -> None:
    payload = adapters["openai_gpt56_sol"].prepare_request(prompt="test").payload
    assert payload == {
        "model": "gpt-5.6-sol",
        "reasoning": {"effort": "none"},
        "input": "test",
        "max_output_tokens": 64,
    }


def test_anthropic_frozen_payload(adapters) -> None:
    payload = adapters["anthropic_sonnet5"].prepare_request(prompt="test").payload
    assert payload == {
        "model": "claude-sonnet-5",
        "thinking": {"type": "disabled"},
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "test"}],
    }
    assert not {"temperature", "top_p", "top_k", "tools"} & payload.keys()


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
