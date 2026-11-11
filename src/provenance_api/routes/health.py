import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from provenance_api.metrics import JOB_QUEUE_DEPTH
from provenance_api.schemas import HealthResponse, ReadinessResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])


@router.get("/healthz", response_model=HealthResponse, summary="Liveness: is the process up?")
async def healthz() -> HealthResponse:
    # Deliberately checks nothing else. If this depended on the database, a
    # database outage would make the orchestrator restart healthy API processes.
    return HealthResponse(status="ok")


async def _database_ok(request: Request) -> bool:
    try:
        async with request.app.state.session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Readiness: database check failed.")
        return False
    return True


async def _redis_ok(request: Request) -> bool:
    try:
        await request.app.state.queue.ping()
    except Exception:
        logger.exception("Readiness: Redis check failed.")
        return False
    return True


@router.get(
    "/readyz",
    response_model=ReadinessResponse,
    summary="Readiness: can this instance serve requests right now?",
    responses={503: {"model": ReadinessResponse}},
)
async def readyz(request: Request) -> JSONResponse:
    checks = {
        "model": hasattr(request.app.state, "inference"),
        "database": await _database_ok(request),
        "redis": await _redis_ok(request),
    }
    ready = all(checks.values())
    body = ReadinessResponse(status="ready" if ready else "not_ready", checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if ready else 503)


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    try:
        # Read at scrape time: the queue lives in Redis, not in this process.
        JOB_QUEUE_DEPTH.set(await request.app.state.queue.zcard("arq:queue"))
    except Exception:
        logger.exception("Could not read the job queue depth.")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
