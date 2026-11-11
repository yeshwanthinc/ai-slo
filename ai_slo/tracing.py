"""OpenTelemetry tracing for the agent, exported to a local Phoenix server.

Span layout per task (one tree per task):

  invoke_agent                 gen_ai.operation.name=invoke_agent, openinference AGENT
  ├─ chat                      gen_ai.operation.name=chat,         openinference LLM   (one per model call)
  ├─ execute_tool              gen_ai.operation.name=execute_tool, openinference TOOL  (one per tool call)
  │  └─ memory.read / memory.write   (memory tools only)
  └─ ...

Attribute names come from ai_slo.attributes. If no provider is configured the
OpenTelemetry API is a no-op, so the agent runs the same without tracing.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode

from . import attributes as A

TRACER_NAME = "ai_slo.agent"
DEFAULT_PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def setup_tracing(
    endpoint: str = DEFAULT_PHOENIX_ENDPOINT,
    service_name: str = "ai-slo-demo",
    simple: bool = False,
) -> TracerProvider:
    """Install a global tracer provider that exports OTLP/HTTP to Phoenix.

    simple=True exports each span immediately (used by the CLI so short demo
    runs flush before exit).
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=endpoint)
    processor = SimpleSpanProcessor(exporter) if simple else BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return provider


def set_common_attrs(span: trace.Span, openinference_kind: str, operation: str) -> None:
    span.set_attribute(A.GEN_AI_OPERATION_NAME, operation)
    span.set_attribute("openinference.span.kind", openinference_kind)


def mark_error(span: trace.Span, message: str) -> None:
    span.set_status(Status(StatusCode.ERROR, message))


def set_messages(span: trace.Span, prefix: str, messages: list[dict]) -> None:
    """Write chat messages in OpenInference form, which Phoenix parses:
    <prefix>.<i>.message.role and <prefix>.<i>.message.content."""
    for i, m in enumerate(messages):
        span.set_attribute(f"{prefix}.{i}.message.role", str(m.get("role", "")))
        span.set_attribute(f"{prefix}.{i}.message.content", str(m.get("content") or ""))
