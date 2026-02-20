"""
Prometheus monitoring metrics for the inference service.
Tracks request count, latency, and prediction distribution.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# Request counter - total requests by endpoint and method
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference service requests",
    ["method", "endpoint", "status_code"],
)

# Request latency histogram
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Prediction distribution gauge
PREDICTION_CAT_COUNT = Counter(
    "prediction_cat_total",
    "Total number of cat predictions",
)

PREDICTION_DOG_COUNT = Counter(
    "prediction_dog_total",
    "Total number of dog predictions",
)

# Model info gauge
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the model is loaded and ready (1=yes, 0=no)",
)


def get_metrics():
    """Generate latest Prometheus metrics."""
    return generate_latest()


def get_content_type():
    """Get the content type for Prometheus metrics."""
    return CONTENT_TYPE_LATEST
