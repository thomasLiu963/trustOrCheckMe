"""Async provider adapters for the frozen two-model pilot."""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import inspect
import os
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from .checkpointing import sanitize_payload
from .schemas import ModelResponse

AttemptCallback = Callable[[Mapping[str, Any]], Awaitable[None] | None]


@dataclasses.dataclass(frozen=True)
class PreparedRequest:
    provider: str
    api_style: str
    requested_model_id: str
    payload: dict[str, Any]

    def sanitized_payload(self) -> dict[str, Any]:
        return sanitize_payload(self.payload)


class AdapterError(RuntimeError):
    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


def output_schema(stage: str) -> dict[str, Any]:
    """Return the common provider-neutral JSON schema for one stage."""
    properties: dict[str, Any]
    required: list[str]
    if stage == "answer":
        properties = {
            "answer": {
                "type": "string",
                "enum": [chr(ord("A") + index) for index in range(10)],
            }
        }
        required = ["answer"]
    elif stage == "confidence":
        properties = {
            "probability_correct": {
                "type": "number",
            }
        }
        required = ["probability_correct"]
    elif stage == "trust":
        properties = {
            "recommendation": {
                "type": "string",
                "enum": ["RELY", "VERIFY"],
            }
        }
        required = ["recommendation"]
    elif stage == "verification":
        properties = {
            "action": {
                "type": "string",
                "enum": ["USE_UNVERIFIED", "VERIFY_FIRST"],
            }
        }
        required = ["action"]
    else:
        raise ValueError(f"Unknown experiment stage: {stage!r}")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _value(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _safe_error_message(error: BaseException) -> str:
    message = str(error)
    for variable in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
    ):
        secret = os.getenv(variable)
        if secret:
            message = message.replace(secret, "<redacted>")
    message = re.sub(
        r"(?i)(api[-_ ]?key|authorization|bearer)\s*[:=]\s*\S+",
        r"\1=<redacted>",
        message,
    )
    return message[:4000]


def _is_transient(error: BaseException) -> bool:
    status = getattr(error, "status_code", None)
    if status in {408, 409, 425, 429} or (isinstance(status, int) and status >= 500):
        return True
    name = type(error).__name__.lower()
    return any(
        marker in name
        for marker in (
            "timeout",
            "connection",
            "ratelimit",
            "internalserver",
            "serviceunavailable",
            "overloaded",
        )
    ) or isinstance(error, (TimeoutError, ConnectionError, asyncio.TimeoutError))


async def _notify(callback: AttemptCallback | None, event: Mapping[str, Any]) -> None:
    if callback is None:
        return
    result = callback(event)
    if inspect.isawaitable(result):
        await result


def _schema_instance(schema: type[Any], values: Mapping[str, Any]) -> Any:
    """Construct a Pydantic model or dataclass while tolerating field aliases."""
    fields = getattr(schema, "model_fields", None)
    if fields is None:
        fields = getattr(schema, "__fields__", None)
    if fields is None and dataclasses.is_dataclass(schema):
        fields = getattr(schema, "__dataclass_fields__", None)
    if fields is not None:
        accepted = set(fields)
        return schema(
            **{key: value for key, value in values.items() if key in accepted}
        )
    try:
        signature = inspect.signature(schema)
    except (TypeError, ValueError):
        return schema(**dict(values))
    accepted = set(signature.parameters)
    return schema(**{key: value for key, value in values.items() if key in accepted})


def _model_response(
    *,
    model_alias: str,
    provider: str,
    requested_model_id: str,
    returned_model_id: str | None,
    stage: str,
    request_key: str,
    raw_text: str,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None,
    latency_seconds: float,
    finish_reason: str | None,
    refusal: bool,
) -> ModelResponse:
    timestamp = datetime.now(UTC)
    values = {
        "model_alias": model_alias,
        "model_id": model_alias,
        "provider": provider,
        "requested_model_id": requested_model_id,
        "requested_api_model_id": requested_model_id,
        "api_model": requested_model_id,
        "returned_model_id": returned_model_id,
        "provider_model_id": returned_model_id,
        "stage": stage,
        "request_key": request_key,
        "raw_response": raw_text,
        "raw_text": raw_text,
        "text": raw_text,
        "parsed_response": None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": latency_seconds,
        "latency": latency_seconds,
        "finish_reason": finish_reason,
        "stop_reason": finish_reason,
        "refusal": refusal,
        "is_refusal": refusal,
        "timestamp": timestamp,
        "created_at": timestamp,
    }
    return _schema_instance(ModelResponse, values)


class ModelAdapter(abc.ABC):
    provider: str
    api_style: str

    def __init__(
        self,
        *,
        model_alias: str,
        api_model: str,
        max_output_tokens: int = 64,
        max_transient_retries: int = 3,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 30.0,
        client: Any = None,
    ) -> None:
        self.model_alias = model_alias
        self.api_model = api_model
        expected_max_tokens = 512 if self.provider == "google" else 64
        if int(max_output_tokens) != expected_max_tokens:
            raise ValueError(
                f"Frozen {self.provider} max_output_tokens must be "
                f"{expected_max_tokens}"
            )
        self.max_output_tokens = expected_max_tokens
        self.max_transient_retries = max(0, int(max_transient_retries))
        self.backoff_base_seconds = max(0.0, float(backoff_base_seconds))
        self.backoff_max_seconds = max(
            self.backoff_base_seconds, float(backoff_max_seconds)
        )
        self._client = client

    @abc.abstractmethod
    def prepare_request(self, *, stage: str, prompt: str) -> PreparedRequest:
        """Construct a key-free request that is safe to inspect in dry runs."""

    @abc.abstractmethod
    async def _send(self, payload: Mapping[str, Any]) -> Any:
        """Send one provider request."""

    @abc.abstractmethod
    def _decode(
        self,
        response: Any,
    ) -> tuple[str, str | None, int | None, int | None, int | None, str | None, bool]:
        """Decode provider response into common metadata."""

    async def generate(
        self,
        *,
        stage: str,
        prompt: str,
        request_key: str,
        allow_paid: bool = False,
        attempt_callback: AttemptCallback | None = None,
        attempt_kind: str = "generation",
    ) -> ModelResponse:
        if not allow_paid:
            raise PermissionError(
                "Paid model calls require explicit allow_paid=True; use "
                "prepare_request() for zero-cost construction."
            )
        prepared = self.prepare_request(stage=stage, prompt=prompt)
        payload = prepared.payload
        safe_payload = prepared.sanitized_payload()

        for transport_attempt in range(self.max_transient_retries + 1):
            started_at = datetime.now(UTC).isoformat()
            started = time.perf_counter()
            try:
                provider_response = await self._send(payload)
            except Exception as error:
                latency = time.perf_counter() - started
                transient = _is_transient(error)
                safe_message = _safe_error_message(error)
                await _notify(
                    attempt_callback,
                    {
                        "attempt_kind": attempt_kind,
                        "provider": self.provider,
                        "requested_model_id": self.api_model,
                        "started_at": started_at,
                        "finished_at": datetime.now(UTC).isoformat(),
                        "latency_seconds": latency,
                        "success": False,
                        "error": safe_message,
                        "sanitized_payload": safe_payload,
                    },
                )
                if not transient or transport_attempt >= self.max_transient_retries:
                    raise AdapterError(safe_message, transient=transient) from error
                delay = min(
                    self.backoff_max_seconds,
                    self.backoff_base_seconds * (2**transport_attempt),
                )
                await asyncio.sleep(random.uniform(0.5 * delay, 1.5 * delay))
                continue

            latency = time.perf_counter() - started
            (
                raw_text,
                returned_model_id,
                input_tokens,
                output_tokens,
                total_tokens,
                finish_reason,
                refusal,
            ) = self._decode(provider_response)
            return _model_response(
                model_alias=self.model_alias,
                provider=self.provider,
                requested_model_id=self.api_model,
                returned_model_id=returned_model_id,
                stage=stage,
                request_key=request_key,
                raw_text=raw_text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                finish_reason=finish_reason,
                refusal=refusal,
            )

        raise AssertionError("Retry loop terminated unexpectedly")


class OpenAIAdapter(ModelAdapter):
    provider = "openai"
    api_style = "responses"

    def prepare_request(self, *, stage: str, prompt: str) -> PreparedRequest:
        payload = {
            "model": self.api_model,
            "reasoning": {"effort": "none"},
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"{stage}_output",
                    "strict": True,
                    "schema": output_schema(stage),
                }
            },
        }
        return PreparedRequest(
            provider=self.provider,
            api_style=self.api_style,
            requested_model_id=self.api_model,
            payload=payload,
        )

    async def _send(self, payload: Mapping[str, Any]) -> Any:
        if self._client is None:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is required for paid OpenAI calls")
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise RuntimeError("Install the official 'openai' package") from error
            self._client = AsyncOpenAI(api_key=key)
        return await self._client.responses.create(**dict(payload))

    def _decode(
        self,
        response: Any,
    ) -> tuple[str, str | None, int | None, int | None, int | None, str | None, bool]:
        raw_text = getattr(response, "output_text", None) or ""
        refusal = False
        if not raw_text:
            text_parts: list[str] = []
            for item in getattr(response, "output", None) or []:
                for content in getattr(item, "content", None) or []:
                    content_type = getattr(content, "type", "")
                    if content_type in {"output_text", "text"}:
                        text_parts.append(getattr(content, "text", "") or "")
                    elif content_type == "refusal":
                        refusal = True
                        text_parts.append(getattr(content, "refusal", "") or "")
            raw_text = "".join(text_parts)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if (
            total_tokens is None
            and input_tokens is not None
            and output_tokens is not None
        ):
            total_tokens = input_tokens + output_tokens

        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete, "reason", None)
        finish_reason = incomplete_reason or status
        if status in {"refused", "content_filter"}:
            refusal = True
        return (
            str(raw_text),
            getattr(response, "model", None),
            input_tokens,
            output_tokens,
            total_tokens,
            finish_reason,
            refusal,
        )


class AnthropicAdapter(ModelAdapter):
    provider = "anthropic"
    api_style = "messages"

    def prepare_request(self, *, stage: str, prompt: str) -> PreparedRequest:
        payload = {
            "model": self.api_model,
            "thinking": {"type": "disabled"},
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": output_schema(stage),
                }
            },
        }
        return PreparedRequest(
            provider=self.provider,
            api_style=self.api_style,
            requested_model_id=self.api_model,
            payload=payload,
        )

    async def _send(self, payload: Mapping[str, Any]) -> Any:
        if self._client is None:
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is required for paid Anthropic calls"
                )
            try:
                from anthropic import AsyncAnthropic
            except ImportError as error:
                raise RuntimeError(
                    "Install the official 'anthropic' package"
                ) from error
            self._client = AsyncAnthropic(api_key=key)
        return await self._client.messages.create(**dict(payload))

    def _decode(
        self,
        response: Any,
    ) -> tuple[str, str | None, int | None, int | None, int | None, str | None, bool]:
        text_parts: list[str] = []
        refusal = False
        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif block_type in {"refusal", "redacted_thinking"}:
                refusal = True
                if block_type == "refusal":
                    text_parts.append(getattr(block, "refusal", "") or "")

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason in {"refusal", "content_filter"}:
            refusal = True
        return (
            "".join(text_parts),
            getattr(response, "model", None),
            input_tokens,
            output_tokens,
            total_tokens,
            stop_reason,
            refusal,
        )


class GoogleAdapter(ModelAdapter):
    provider = "google"
    api_style = "google_genai_vertex"

    def prepare_request(self, *, stage: str, prompt: str) -> PreparedRequest:
        payload = {
            "model": self.api_model,
            "contents": prompt,
            "config": {
                "max_output_tokens": self.max_output_tokens,
                "response_mime_type": "application/json",
                "response_json_schema": output_schema(stage),
                "thinking_config": {"thinking_level": "LOW"},
            },
        }
        return PreparedRequest(
            provider=self.provider,
            api_style=self.api_style,
            requested_model_id=self.api_model,
            payload=payload,
        )

    async def _send(self, payload: Mapping[str, Any]) -> Any:
        if self._client is None:
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
            if not project:
                raise RuntimeError(
                    "GOOGLE_CLOUD_PROJECT is required for Vertex AI calls"
                )
            try:
                from google import genai
            except ImportError as error:
                raise RuntimeError(
                    "Install the official 'google-genai' package"
                ) from error
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        return await self._client.aio.models.generate_content(**dict(payload))

    def _decode(
        self,
        response: Any,
    ) -> tuple[str, str | None, int | None, int | None, int | None, str | None, bool]:
        raw_text = str(getattr(response, "text", None) or "")
        usage = getattr(response, "usage_metadata", None)
        input_tokens = _value(usage, "prompt_token_count")
        candidate_tokens = _value(usage, "candidates_token_count")
        thought_tokens = _value(usage, "thoughts_token_count")
        output_tokens = (
            (candidate_tokens or 0) + (thought_tokens or 0)
            if candidate_tokens is not None or thought_tokens is not None
            else None
        )
        total_tokens = _value(usage, "total_token_count")
        candidates = getattr(response, "candidates", None) or []
        finish_reason = None
        refusal = False
        if candidates:
            finish_reason = str(_value(candidates[0], "finish_reason", "") or "")
            refusal = finish_reason.upper() in {
                "SAFETY",
                "BLOCKLIST",
                "PROHIBITED_CONTENT",
                "SPII",
            }
        return (
            raw_text,
            getattr(response, "model_version", None),
            input_tokens,
            output_tokens,
            total_tokens,
            finish_reason,
            refusal,
        )


class XAIAdapter(OpenAIAdapter):
    provider = "xai"
    api_style = "responses"

    def prepare_request(self, *, stage: str, prompt: str) -> PreparedRequest:
        payload = {
            "model": self.api_model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"{stage}_output",
                    "strict": True,
                    "schema": output_schema(stage),
                },
            },
        }
        return PreparedRequest(
            provider=self.provider,
            api_style=self.api_style,
            requested_model_id=self.api_model,
            payload=payload,
        )

    async def _send(self, payload: Mapping[str, Any]) -> Any:
        if self._client is None:
            key = os.getenv("XAI_API_KEY")
            if not key:
                raise RuntimeError("XAI_API_KEY is required for paid xAI calls")
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise RuntimeError("Install the official 'openai' package") from error
            self._client = AsyncOpenAI(api_key=key, base_url="https://api.x.ai/v1")
        return await self._client.responses.create(**dict(payload))


def create_adapter(
    model_alias: str,
    model_config: Any,
    *,
    max_transient_retries: int = 3,
    backoff_base_seconds: float = 1.0,
    backoff_max_seconds: float = 30.0,
    client: Any = None,
) -> ModelAdapter:
    provider = str(_value(model_config, "provider", "")).lower()
    api_model = str(
        _value(model_config, "api_model", _value(model_config, "model_id", ""))
    )
    max_tokens = int(_value(model_config, "max_output_tokens", 64))
    common = {
        "model_alias": model_alias,
        "api_model": api_model,
        "max_output_tokens": max_tokens,
        "max_transient_retries": max_transient_retries,
        "backoff_base_seconds": backoff_base_seconds,
        "backoff_max_seconds": backoff_max_seconds,
        "client": client,
    }
    if provider == "openai":
        if api_model != "gpt-5.6-sol":
            raise ValueError("Frozen OpenAI pilot model must be 'gpt-5.6-sol'")
        return OpenAIAdapter(**common)
    if provider == "anthropic":
        if api_model != "claude-sonnet-5":
            raise ValueError("Frozen Anthropic pilot model must be 'claude-sonnet-5'")
        return AnthropicAdapter(**common)
    if provider == "google":
        if api_model != "gemini-3.8-flash":
            raise ValueError("Frozen Google V2 model must be 'gemini-3.8-flash'")
        return GoogleAdapter(**common)
    if provider == "xai":
        if api_model != "grok-4.20-0309-non-reasoning":
            raise ValueError(
                "Frozen xAI V2 model must be 'grok-4.20-0309-non-reasoning'"
            )
        return XAIAdapter(**common)
    raise ValueError(f"Unsupported provider for model {model_alias!r}: {provider!r}")
