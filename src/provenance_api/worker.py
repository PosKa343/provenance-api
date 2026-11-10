"""The background worker: takes batch jobs off the Redis queue and scores them."""

import json
import logging
import uuid
from typing import Any

from arq import Retry
from arq.connections import ArqRedis, RedisSettings
from arq.worker import Worker, func

from provenance_api.config import BackendSettings
from provenance_api.db import create_engine, create_session_factory
from provenance_api.jobs import mark_failed, run_job
from provenance_api.ml.model import Classifier

logger = logging.getLogger(__name__)

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

    stored_texts = await redis.get(texts_key(job_id))
    if stored_texts is None:
        await mark_failed(session_factory, job_id, "The submitted texts are no longer available.")
        return

    try:
        await run_job(session_factory, ctx["classifier"], job_id, json.loads(stored_texts))
    except Exception as error:
        # Log the error type only: its message could quote a submitted text.
        logger.error("Job %s failed on attempt %d (%s).", job_id, attempt, type(error).__name__)
        if attempt < settings.job_max_tries:
            raise Retry(defer=settings.job_retry_backoff_seconds * 2 ** (attempt - 1)) from error
        await mark_failed(
            session_factory,
            job_id,
            f"Scoring failed after {attempt} attempts ({type(error).__name__}).",
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
        # The job row in Postgres is the record of the outcome; arq need not keep one.
        keep_result=0,
    )
