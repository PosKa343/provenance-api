"""Per-request bookkeeping: a request id, one access-log line and a latency sample."""

import re
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import FastAPI, Request, Response

from provenance_api.metrics import REQUEST_DURATION

REQUEST_ID_HEADER = "X-Request-ID"

# A caller may supply its own id so one request can be traced across services.
# Anything that does not look like an id is replaced, since it ends up in logs.
_ACCEPTABLE_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
# Polled constantly by health checks and Prometheus; logging them would bury real traffic.
_NOT_LOGGED = frozenset({"/healthz", "/readyz", "/metrics"})

access_log = structlog.get_logger("provenance_api.access")


def _request_id_for(request: Request) -> str:
    supplied = request.headers.get(REQUEST_ID_HEADER, "")
    return supplied if _ACCEPTABLE_REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex


async def _observe(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = _request_id_for(request)
    request.state.request_id = request_id
    # Bound for the life of this request, so every log line written while
    # handling it carries the id, including lines from libraries.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    started = time.perf_counter()
    status = 500  # what the client gets if the handler raises
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        duration = time.perf_counter() - started
        matched = request.scope.get("route")
        route = getattr(matched, "path", "unmatched")
        REQUEST_DURATION.labels(request.method, route, str(status)).observe(duration)
        if route not in _NOT_LOGGED:
            access_log.info(
                "request",
                method=request.method,
                route=route,
                status=status,
                duration_ms=round(duration * 1000, 1),
            )


def install_observability(app: FastAPI) -> None:
    app.middleware("http")(_observe)
