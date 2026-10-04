"""OpenTelemetry tracing exported to Arize Phoenix (OpenInference conventions).

If PHOENIX_COLLECTOR_ENDPOINT is unset, the global tracer stays a no-op, so
tests and offline evaluation run without a collector.
"""

from __future__ import annotations

import json
import logging

from opentelemetry import trace

from .config import get_settings

log = logging.getLogger(__name__)
tracer = trace.get_tracer("regrag")
_configured = False


def setup_tracing() -> bool:
    global _configured
    settings = get_settings()
    if _configured or not settings.phoenix_collector_endpoint:
        return _configured
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from phoenix.otel import register

    provider = register(
        project_name=settings.phoenix_project,
        endpoint=f"{settings.phoenix_collector_endpoint.rstrip('/')}/v1/traces",
        batch=True,
        set_global_tracer_provider=True,
    )
    AnthropicInstrumentor().instrument(tracer_provider=provider)
    _configured = True
    log.info("Tracing to Phoenix at %s", settings.phoenix_collector_endpoint)
    return True


def set_documents(span, prefix: str, docs: list[tuple[str, float, str]]) -> None:
    """Record retrieved documents using OpenInference retrieval attributes."""
    for i, (doc_id, score, content) in enumerate(docs):
        span.set_attribute(f"{prefix}.{i}.document.id", doc_id)
        span.set_attribute(f"{prefix}.{i}.document.score", float(score))
        span.set_attribute(f"{prefix}.{i}.document.content", content[:2000])


def set_io(span, value_in, value_out=None) -> None:
    span.set_attribute("input.value", value_in if isinstance(value_in, str) else json.dumps(value_in))
    if value_out is not None:
        span.set_attribute(
            "output.value", value_out if isinstance(value_out, str) else json.dumps(value_out)
        )
