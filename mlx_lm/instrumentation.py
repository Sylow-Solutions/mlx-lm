# Copyright © 2023-2024 Apple Inc.

"""
OpenTelemetry instrumentation utilities for MLX server.
"""

import functools
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Union

import mlx.core as mx
import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode


# Initialize OpenTelemetry components
def init_telemetry(
    service_name: str = "mlx-service",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    resource_attributes: Optional[Dict[str, str]] = None,
) -> None:
    """Initialize OpenTelemetry telemetry with OTLP exporters."""

    # Default OTLP endpoint
    if otlp_endpoint is None:
        otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )

    # Build resource attributes
    default_attributes = {
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "production"),
        "cluster.name": os.getenv("CLUSTER_NAME", "nightshift"),
        "service.instance.id": os.getenv("HOSTNAME", "unknown"),
    }

    if resource_attributes:
        default_attributes.update(resource_attributes)

    resource = Resource.create(default_attributes)

    # Initialize tracing
    tracer_provider = TracerProvider(resource=resource)
    span_processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    )
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)

    # Initialize metrics
    metric_reader = PeriodicExportingMetricReader(
        exporter=OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True),
        export_interval_millis=10000,  # Export every 10 seconds
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Initialize structured logging with OpenTelemetry correlation
    LoggingInstrumentor().instrument(set_logging_format=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            add_trace_context,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def add_trace_context(logger, method_name, event_dict):
    """Add trace context to log entries."""
    span = trace.get_current_span()
    if span and span.is_recording():
        ctx = span.get_span_context()
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
        event_dict["trace_flags"] = format(ctx.trace_flags, "02x")
    return event_dict


# Tracer and meter instances
_tracer = None
_meter = None
_logger = None


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("mlx.service")
    return _tracer


def get_meter() -> metrics.Meter:
    """Get the global meter instance."""
    global _meter
    if _meter is None:
        _meter = metrics.get_meter("mlx.service")
    return _meter


def get_logger() -> structlog.BoundLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = structlog.get_logger("mlx.service")
    return _logger


# Metric definitions
class MLXMetrics:
    """Container for MLX-specific metrics."""

    def __init__(self):
        meter = get_meter()

        # Request metrics
        self.request_counter = meter.create_counter(
            "mlx_requests_total",
            description="Total number of requests",
            unit="1",
        )

        self.request_duration = meter.create_histogram(
            "mlx_request_duration_seconds",
            description="Request duration in seconds",
            unit="s",
        )

        self.active_requests = meter.create_up_down_counter(
            "mlx_active_requests",
            description="Number of active requests",
            unit="1",
        )

        # Model metrics
        self.model_load_duration = meter.create_histogram(
            "mlx_model_load_duration_seconds",
            description="Model loading duration in seconds",
            unit="s",
        )

        self.inference_duration = meter.create_histogram(
            "mlx_inference_duration_seconds",
            description="Inference duration in seconds",
            unit="s",
        )

        self.tokens_generated = meter.create_histogram(
            "mlx_tokens_generated",
            description="Number of tokens generated per request",
            unit="1",
        )

        self.tokens_per_second = meter.create_histogram(
            "mlx_tokens_per_second",
            description="Token generation rate",
            unit="1/s",
        )

        self.prompt_tokens = meter.create_histogram(
            "mlx_prompt_tokens",
            description="Number of prompt tokens per request",
            unit="1",
        )

        # Memory metrics
        self.memory_usage = meter.create_observable_gauge(
            "mlx_memory_usage_bytes",
            [self._observe_memory_usage],
            description="MLX memory usage in bytes",
            unit="By",
        )

        self.peak_memory = meter.create_histogram(
            "mlx_peak_memory_bytes",
            description="Peak memory usage during inference",
            unit="By",
        )

        # Cache metrics
        self.cache_hits = meter.create_counter(
            "mlx_cache_hits_total",
            description="Total number of prompt cache hits",
            unit="1",
        )

        self.cache_misses = meter.create_counter(
            "mlx_cache_misses_total",
            description="Total number of prompt cache misses",
            unit="1",
        )

        # Error metrics
        self.errors = meter.create_counter(
            "mlx_errors_total",
            description="Total number of errors",
            unit="1",
        )

        # Response size metrics
        self.request_size = meter.create_histogram(
            "mlx_request_size_bytes",
            description="Request payload size in bytes",
            unit="By",
        )

        self.response_size = meter.create_histogram(
            "mlx_response_size_bytes",
            description="Response payload size in bytes",
            unit="By",
        )

    def _observe_memory_usage(self, options: CallbackOptions) -> Observation:
        """Callback to observe current memory usage."""
        if mx.metal.is_available():
            # Get memory info from MLX
            memory_info = mx.metal.device_info()
            used = memory_info.get("peak_allocated_size", 0)
            return Observation(used)
        return Observation(0)


# Global metrics instance
_metrics = None


def get_metrics() -> MLXMetrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = MLXMetrics()
    return _metrics


# Decorator for tracing functions
def trace_method(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
) -> Callable:
    """Decorator to trace method execution."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = span_name or f"{func.__module__}.{func.__name__}"

            with tracer.start_as_current_span(name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function arguments as attributes
                if args:
                    span.set_attribute("args.count", len(args))
                if kwargs:
                    span.set_attribute("kwargs.keys", str(list(kwargs.keys())))

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return wrapper

    return decorator


# Context manager for tracing
@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL,
):
    """Context manager for creating a trace span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


# Timer context manager for metrics
@contextmanager
def timer(histogram: metrics.Histogram, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for timing operations."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        histogram.record(duration, attributes=attributes)


# Helper to log with trace context
def log_with_trace(
    level: str,
    message: str,
    **kwargs,
) -> None:
    """Log a message with trace context."""
    logger = get_logger()
    log_method = getattr(logger, level.lower())
    log_method(message, **kwargs)


# Error categorization
class ErrorCategory:
    """Error categories for tracking."""

    MODEL_LOAD_ERROR = "model_load_error"
    INFERENCE_ERROR = "inference_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_ERROR = "resource_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


def categorize_error(exception: Exception) -> str:
    """Categorize an exception into error types."""
    error_mapping = {
        FileNotFoundError: ErrorCategory.MODEL_LOAD_ERROR,
        ValueError: ErrorCategory.VALIDATION_ERROR,
        MemoryError: ErrorCategory.RESOURCE_ERROR,
        ConnectionError: ErrorCategory.NETWORK_ERROR,
        TimeoutError: ErrorCategory.NETWORK_ERROR,
    }

    for error_type, category in error_mapping.items():
        if isinstance(exception, error_type):
            return category

    # Check for specific error messages
    error_msg = str(exception).lower()
    if "model" in error_msg or "load" in error_msg:
        return ErrorCategory.MODEL_LOAD_ERROR
    elif "inference" in error_msg or "generate" in error_msg:
        return ErrorCategory.INFERENCE_ERROR
    elif "memory" in error_msg or "resource" in error_msg:
        return ErrorCategory.RESOURCE_ERROR

    return ErrorCategory.UNKNOWN_ERROR


# MLX-specific resource monitoring
def get_mlx_metrics() -> Dict[str, Any]:
    """Get current MLX resource metrics."""
    metrics_data = {
        "cpu_count": os.cpu_count(),
        "metal_available": mx.metal.is_available(),
    }

    if mx.metal.is_available():
        device_info = mx.metal.device_info()
        metrics_data.update(
            {
                "gpu_architecture": device_info.get("architecture", "unknown"),
                "gpu_memory_limit": device_info.get("memory_size", 0),
                "gpu_peak_allocated": device_info.get("peak_allocated_size", 0),
            }
        )

    return metrics_data
