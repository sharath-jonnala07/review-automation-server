"""OpenTelemetry tracing configuration."""

from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import get_settings

_tracer: trace.Tracer | None = None


def init_tracing() -> None:
    """Initialize OpenTelemetry tracing."""
    settings = get_settings()
    if not settings.otlp_endpoint:
        return

    resource = Resource(
        attributes={
            SERVICE_NAME: settings.app_name,
            SERVICE_VERSION: settings.app_version,
        }
    )

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    global _tracer
    _tracer = trace.get_tracer(settings.app_name, settings.app_version)


def get_tracer() -> trace.Tracer:
    """Get the global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("pulse-agent", "0.1.0")
    return _tracer


@contextmanager
def start_span(name: str, **attributes: object) -> Generator[trace.Span, None, None]:
    """Context manager for starting a span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if isinstance(value, str | bool | int | float):
                span.set_attribute(key, value)
        yield span
