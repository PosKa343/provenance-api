"""Prometheus metrics. Each process (API, worker) exposes its own."""

from prometheus_client import Counter, Gauge, Histogram

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Time to handle an HTTP request.",
    # route is the path template, e.g. /v1/jobs/{job_id}, never the raw path:
    # raw paths contain ids and would create a new time series per request.
    ["method", "route", "status"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTIONS = Counter(
    "predictions_total",
    "Texts classified.",
    ["label", "source"],  # source: single (POST /v1/classify) or batch (a job)
)

JOB_QUEUE_DEPTH = Gauge("job_queue_depth", "Batch jobs waiting in or being worked from the queue.")

JOBS_FINISHED = Counter("jobs_finished_total", "Batch jobs that reached a final state.", ["status"])

JOB_DURATION = Histogram(
    "job_duration_seconds",
    "Time one attempt at a batch job took.",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0),
)
