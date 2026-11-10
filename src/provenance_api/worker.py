"""The background worker: takes batch jobs off the Redis queue and scores them."""

import json
import time
import uuid
from typing import Any

import structlog
from arq import Retry
from arq.connections import ArqRedis, RedisSettings
from arq.worker import Worker, func

from provenance_api.config import BackendSettings
from provenance_api.db import create_engine, create_session_factory
from provenance_api.jobs import mark_failed, run_job
from provenance_api.metrics import JOB_DURATION, JOBS_FINISHED
from provenance_api.ml.model import Classifier

job_log = structlog.get_logger(__name__)

SCORE_JOB = "score_job"


# Texts wait here, outside the queue entry, so arq never sees them and cannot
# log them. The expiry removes them even if a job is never picked up.
TEXTS_TTL_SECONDS = 24 * 60 * 60


def texts_key(job_id: uuid.UUID) -> str:
    return f"provenance:job-texts:{job_id}"


async def enqueue_scoring(redis: ArqRedis, job_id: uuid.UUID, texts: list[str]) -> None:
    await redis.set(texts_key(job_id), json.dumps(texts), ex=TEXTS_TTL_SECONDS)
    # Using the job's id as the queue id makes enqueueing the same job twice a no-op.
    await redis.enqueue_job(SCORE_JOB, str(job_id), _job_id=str(job_id))


async def score_job(ctx: dict[str, Any], job_id_text: str) -> None:
    settings: BackendSettings = ctx["settings"]
    redis: ArqRedis = ctx["redis"]
    session_factory = ctx["session_factory"]
    attempt: int = ctx["job_try"]
    job_id = uuid.UUID(job_id_text)
    # Every log line written while working on this job carries its id.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job_id=job_id_text, attempt=attempt)

    stored_texts = await redis.get(texts_key(job_id))
    if stored_texts is None:
        await mark_failed(session_factory, job_id, "The submitted texts are no longer available.")
        JOBS_FINISHED.labels("failed").inc()
        job_log.warning("job finished", status="failed", reason="texts_expired")
        return

    texts: list[str] = json.loads(stored_texts)
    started = time.perf_counter()
    status = "succeeded"
    try:
        await run_job(session_factory, ctx["classifier"], job_id, texts)
    except Exception as error:
        # Log the error type only: its message could quote a submitted text.
        if attempt < settings.job_max_tries:
            job_log.warning("job attempt failed; will retry", error=type(error).__name__)
            raise Retry(defer=settings.job_retry_backoff_seconds * 2 ** (attempt - 1)) from error
        status = "failed"
        await mark_failed(
            session_factory,
            job_id,
            f"Scoring failed after {attempt} attempts ({type(error).__name__}).",
        )
        job_log.error("job attempt failed; giving up", error=type(error).__name__)
    finally:
        JOB_DURATION.observe(time.perf_counter() - started)

    JOBS_FINISHED.labels(status).inc()
    job_log.info(
        "job finished",
        status=status,
        texts=len(texts),
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )

    # Reached on success and on final failure, never on a retry.
    await redis.delete(texts_key(job_id))


def create_worker(
    settings: BackendSettings,
    *,
    redis_pool: ArqRedis | None = None,
    burst: bool = False,
    poll_delay: float = 0.5,
) -> Worker:
    """Build a worker. burst=True makes it exit once the queue is empty (used in tests)."""

    async def startup(ctx: dict[str, Any]) -> None:
        # Loaded once per worker process, like the API does at startup.
        ctx["settings"] = settings
        ctx["classifier"] = Classifier.load()
        ctx["engine"] = create_engine(settings.database_url)
        ctx["session_factory"] = create_session_factory(ctx["engine"])

    async def shutdown(ctx: dict[str, Any]) -> None:
        await ctx["engine"].dispose()

    return Worker(
        functions=[func(score_job, name=SCORE_JOB)],
        redis_settings=RedisSettings.from_dsn(settings.redis_url),
        redis_pool=redis_pool,
        on_startup=startup,
        on_shutdown=shutdown,
        burst=burst,
        poll_delay=poll_delay,
        handle_signals=not burst,
        max_tries=settings.job_max_tries,
        # On SIGTERM, let running jobs finish for a while before cancelling them.
        job_completion_wait=settings.worker_shutdown_wait_seconds,
        # The job row in Postgres is the record of the outcome; arq need not keep one.
        keep_result=0,
    )
