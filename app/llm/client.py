"""Groq (OpenAI-compatible) client: schema-constrained JSON, plain chat, and streaming chat."""
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings

T = TypeVar("T", bound=BaseModel)


@lru_cache
def client() -> openai.OpenAI:
    return openai.OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)


def _schema_miss(e: BaseException) -> bool:
    # Groq validates (rather than constrains) strict-schema output; a miss is a 400 worth retrying.
    return isinstance(e, openai.BadRequestError) and "json_validate_failed" in str(e)


class QuotaExhausted(RuntimeError):
    """A daily (not per-minute) provider limit: retrying in-process won't help."""

    def __init__(self, model: str, retry_in: str | None):
        self.model, self.retry_in = model, retry_in
        super().__init__(f"daily token quota for {model} reached" + (f"; retry in {retry_in}" if retry_in else ""))


def _per_day(e: BaseException) -> bool:
    return isinstance(e, openai.RateLimitError) and "per day" in str(e)


def _retryable(e: BaseException) -> bool:
    if _per_day(e):
        return False  # waiting minutes for a daily window just hangs the request
    return isinstance(e, (openai.RateLimitError, openai.APIConnectionError, ValidationError)) or _schema_miss(e)


def raise_if_quota(e: BaseException, model: str) -> None:
    if _per_day(e):
        m = re.search(r"try again in ([\dhms.]+)", str(e))
        raise QuotaExhausted(model, m.group(1).rstrip(".") if m else None) from e


_transient = retry(
    retry=retry_if_exception(_retryable),
    wait=wait_exponential(min=5, max=90),
    stop=stop_after_attempt(5),
    reraise=True,
)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@_transient
def structured(
    system: str, user: str, schema: type[T], model: str | None = None, max_tokens: int = 2500,
    usage: "Usage | None" = None,
) -> T:
    resp = client().chat.completions.create(
        model=model or settings.extraction_model,
        temperature=0,
        max_completion_tokens=max_tokens,
        reasoning_effort="low",
        messages=_messages(system, user),
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True},
        },
    )
    if usage is not None and resp.usage:
        usage.prompt_tokens, usage.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
    return schema.model_validate(json.loads(resp.choices[0].message.content))


@_transient
def chat(system: str, user: str, model: str, usage: Usage, max_tokens: int = 1500) -> str:
    resp = client().chat.completions.create(
        model=model, temperature=0, max_completion_tokens=max_tokens, reasoning_effort="low",
        messages=_messages(system, user),
    )
    if resp.usage:
        usage.prompt_tokens, usage.completion_tokens = resp.usage.prompt_tokens, resp.usage.completion_tokens
    return resp.choices[0].message.content or ""


@_transient
def _open_stream(system: str, user: str, model: str, max_tokens: int):
    return client().chat.completions.create(
        model=model, temperature=0, max_completion_tokens=max_tokens, reasoning_effort="low",
        stream=True, stream_options={"include_usage": True}, messages=_messages(system, user),
    )


def chat_stream(system: str, user: str, model: str, usage: Usage, max_tokens: int = 1500) -> Iterator[str]:
    """Yield answer text deltas; fills `usage` when the stream ends. Retries cover opening only."""
    for event in _open_stream(system, user, model, max_tokens):
        if getattr(event, "usage", None):
            usage.prompt_tokens, usage.completion_tokens = event.usage.prompt_tokens, event.usage.completion_tokens
        if event.choices and (delta := event.choices[0].delta.content):
            yield delta
