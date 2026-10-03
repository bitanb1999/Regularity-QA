"""Langfuse tracing for /ask.

Spans are created explicitly (`start_observation` + `.end()`) and passed down rather than relying
on OpenTelemetry's implicit "current span": the streaming endpoint runs as a generator that
Starlette steps through in a thread pool, where context doesn't reliably survive across yields.

Tracing is a no-op when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are unset, and an unreachable
Langfuse server only produces background export warnings; requests are never affected.
"""
import logging
from functools import lru_cache
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)

# USD per 1M tokens, Groq list prices (console.groq.com/docs/models, checked 2026-10-03).
# The free tier bills nothing; traces show the list-price equivalent so cost per query is visible.
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


def cost(model: str, prompt_tokens: int, completion_tokens: int) -> dict[str, float] | None:
    if model not in PRICES:
        return None
    pin, pout = PRICES[model]
    i, o = prompt_tokens * pin / 1e6, completion_tokens * pout / 1e6
    return {"input": i, "output": o, "total": i + o}


class _NoopSpan:
    """Stands in when tracing is off, so call sites never branch."""

    trace_id = None

    def start_observation(self, **_: Any) -> "_NoopSpan":
        return self

    def update(self, **_: Any) -> "_NoopSpan":
        return self

    def end(self, **_: Any) -> None:
        pass


NOOP = _NoopSpan()


@lru_cache
def _client():
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import get_client
        return get_client()
    except Exception as e:  # noqa: BLE001 - observability must never break requests
        log.warning("Langfuse disabled: %s", e)
        return None


def enabled() -> bool:
    return _client() is not None


def start_trace(name: str, input: Any, metadata: dict | None = None):
    """Root observation of a trace (its name becomes the trace name in Langfuse)."""
    client = _client()
    if client is None:
        return NOOP
    try:
        from langfuse import propagate_attributes
        with propagate_attributes(trace_name=name, tags=["api", settings.answer_model]):
            return client.start_observation(name=name, as_type="chain", input=input, metadata=metadata)
    except Exception as e:  # noqa: BLE001
        log.warning("Langfuse trace start failed: %s", e)
        return NOOP


def generation_update(span, model: str, prompt_tokens: int, completion_tokens: int, **kwargs: Any) -> None:
    span.update(
        model=model,
        usage_details={"input": prompt_tokens, "output": completion_tokens},
        cost_details=cost(model, prompt_tokens, completion_tokens),
        **kwargs,
    )


def flush() -> None:
    if (client := _client()) is not None:
        client.flush()
