"""The background worker: takes batch jobs off the Redis queue and scores them."""

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


async def enqueue_scoring(redis: ArqRedis, job_id: uuid.UUID, texts: list[str]) -> None:
    # Using the job's id as the queue id makes enqueueing the same job twice a no-op.
    await redis.enqueue_job(SCORE_JOB, str(job_id), texts, _job_id=str(job_id))


async def score_job(ctx: dict[str, Any], job_id: str, texts: list[str]) -> None:
    settings: BackendSettings = ctx["settings"]
    attempt: int = ctx["job_try"]
    try:
        await run_job(ctx["session_factory"], ctx["classifier"], uuid.UUID(job_id), texts)
    except Exception as error:
        logger.exception("Job %s failed on attempt %d.", job_id, attempt)
        if attempt >= settings.job_max_tries:
            # The details stay in the log; the user sees only the kind of error.
            await mark_failed(
                ctx["session_factory"],
                uuid.UUID(job_id),
                f"Scoring failed after {attempt} attempts ({type(error).__name__}).",
            )
            return
        raise Retry(defer=settings.job_retry_backoff_seconds * 2 ** (attempt - 1)) from error


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
        # Job arguments contain the submitted texts. Keeping no result means
        # they leave Redis as soon as the job is done.
        keep_result=0,
    )
