import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Header, Query, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from provenance_api.dependencies import ApiKeyUser, QueueDep, SessionDep, SettingsDep
from provenance_api.errors import ApiError, not_found
from provenance_api.models import Job, JobStatus, PredictionRecord, User
from provenance_api.schemas import ErrorEnvelope, JobCreate, JobOut, JobResult, JobResultsPage
from provenance_api.worker import enqueue_scoring

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/jobs",
    tags=["jobs"],
    responses={401: {"model": ErrorEnvelope}},
)


@router.post(
    "",
    response_model=JobOut,
    status_code=202,
    summary="Submit a batch of texts to classify in the background",
    responses={
        200: {"model": JobOut, "description": "This Idempotency-Key was already used; same job."},
        409: {"model": ErrorEnvelope},
        413: {"model": ErrorEnvelope},
        422: {"model": ErrorEnvelope},
        503: {"model": ErrorEnvelope},
    },
)
async def submit_job(
    body: JobCreate,
    response: Response,
    user: ApiKeyUser,
    session: SessionDep,
    queue: QueueDep,
    settings: SettingsDep,
    idempotency_key: Annotated[
        str | None,
        Header(
            max_length=255,
            description="Send the same value when retrying, to avoid creating a second job.",
        ),
    ] = None,
) -> Job:
    if len(body.texts) > settings.max_batch_texts:
        raise ApiError(
            413, "batch_too_large", f"A job can hold at most {settings.max_batch_texts} texts."
        )
    if any(len(text) > settings.max_text_chars for text in body.texts):
        raise ApiError(
            413, "text_too_long", f"Each text must be at most {settings.max_text_chars} characters."
        )

    request_sha256 = hashlib.sha256(json.dumps(body.texts).encode("utf-8")).hexdigest()
    # Read now: a rollback below expires the loaded user, and reading an expired
    # attribute would need a database call the async session cannot make implicitly.
    user_id = user.id
    job = Job(
        user_id=user_id,
        idempotency_key=idempotency_key,
        request_sha256=request_sha256,
        total=len(body.texts),
    )
    session.add(job)
    try:
        await session.commit()
    except IntegrityError:
        # The unique constraint on (user, key) fired: this key was used before.
        await session.rollback()
        existing = await _job_for_key(session, user_id, idempotency_key)
        if existing.request_sha256 != request_sha256:
            raise ApiError(
                409,
                "idempotency_key_reused",
                "This Idempotency-Key was already used with different texts.",
            ) from None
        response.status_code = 200
        return existing

    try:
        await enqueue_scoring(queue, job.id, body.texts)
    except Exception:
        # The row is committed but nothing will ever process it, so say so
        # instead of leaving a job that looks queued forever.
        logger.exception("Could not enqueue job %s.", job.id)  # a Redis error; holds no text
        job.status = JobStatus.FAILED
        job.error = "The job could not be queued."
        job.finished_at = datetime.now(UTC)
        await session.commit()
        raise ApiError(503, "queue_unavailable", "The job queue is unavailable.") from None

    await session.refresh(job)
    return job


async def _job_for_key(session: SessionDep, user_id: uuid.UUID, idempotency_key: str | None) -> Job:
    return (
        await session.execute(
            select(Job).where(Job.user_id == user_id, Job.idempotency_key == idempotency_key)
        )
    ).scalar_one()


async def _owned_job(session: SessionDep, user: User, job_id: uuid.UUID) -> Job:
    # Another user's job gets the same 404 as a missing one.
    job = (
        await session.execute(select(Job).where(Job.id == job_id, Job.user_id == user.id))
    ).scalar_one_or_none()
    if job is None:
        raise not_found("Job not found.")
    return job


@router.get(
    "/{job_id}",
    response_model=JobOut,
    summary="Get a job's status and progress",
    responses={404: {"model": ErrorEnvelope}},
)
async def get_job(job_id: uuid.UUID, user: ApiKeyUser, session: SessionDep) -> Job:
    return await _owned_job(session, user, job_id)


@router.get(
    "/{job_id}/results",
    response_model=JobResultsPage,
    summary="Get a job's results, in submission order",
    description="Results appear as texts are scored, so a running job returns a partial list.",
    responses={404: {"model": ErrorEnvelope}},
)
async def get_job_results(
    job_id: uuid.UUID,
    user: ApiKeyUser,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[
        int | None, Query(ge=0, description="Return results after this position.")
    ] = None,
) -> JobResultsPage:
    job = await _owned_job(session, user, job_id)
    query = (
        select(PredictionRecord)
        .where(PredictionRecord.job_id == job.id)
        .order_by(PredictionRecord.position)
        .limit(limit + 1)
    )
    if cursor is not None:
        query = query.where(PredictionRecord.position > cursor)

    rows = (await session.execute(query)).scalars().all()
    page = rows[:limit]
    has_more = len(rows) > limit
    return JobResultsPage(
        items=[JobResult.model_validate(row) for row in page],
        next_cursor=page[-1].position if has_more else None,
    )
