"""
Shared OpenTelemetry setup for distributed tracing
Based on: https://docs.langchain.com/langsmith/trace-with-opentelemetry
"""
import os
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def setup_otel(service_name: str = "default-service"):
    # Check if a provider already exists
    existing_provider = trace.get_tracer_provider()

    # Only create and set a new provider if one doesn't exist yet
    # or if it's the default ProxyTracerProvider
    if existing_provider.__class__.__name__ == "ProxyTracerProvider":
        print(f"[OTEL] Creating new TracerProvider for {service_name}", file=sys.stderr)
        # Create provider
        provider = TracerProvider()

        # OTLP exporter to LangSmith
        otlp_exporter = OTLPSpanExporter(
            endpoint="https://api.smith.langchain.com/otel/v1/traces",
            headers={
                "x-api-key": os.getenv("LANGSMITH_API_KEY"),
                "Langsmith-Project": os.getenv("LANGSMITH_PROJECT", "default")
            }
        )

        # Add exporter
        processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(processor)

        # Set global provider
        trace.set_tracer_provider(provider)
        print(f"[OTEL] ✓ TracerProvider configured for {service_name}", file=sys.stderr)
    else:
        print(f"[OTEL] Using existing TracerProvider: {existing_provider.__class__.__name__}", file=sys.stderr)
        provider = existing_provider

    # Return tracer
    return trace.get_tracer(service_name)