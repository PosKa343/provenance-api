"""Scoring a batch job. Safe to run more than once for the same job."""

import asyncio
import hashlib
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from provenance_api.metrics import PREDICTIONS
from provenance_api.ml.model import Classifier, Prediction
from provenance_api.models import Job, JobStatus, PredictionRecord

logger = logging.getLogger(__name__)

# Progress is saved after each chunk, so a crash loses at most this many texts of work.
CHUNK_SIZE = 10


async def run_job(
    session_factory: async_sessionmaker[AsyncSession],
    classifier: Classifier,
    job_id: uuid.UUID,
    texts: list[str],
) -> None:
    """Score every text of a job that is not scored yet, then mark it succeeded.

    The queue delivers a job at least once, so this may be called again after a
    crash or a retry. It then skips the texts that already have a result.
    """
    async with session_factory() as session:
        job = await session.get(Job, job_id)
        if job is None:
            logger.warning("Job %s does not exist; nothing to do.", job_id)
            return
        if job.status.is_finished:
            return

        job.status = JobStatus.RUNNING
        job.started_at = job.started_at or datetime.now(UTC)
        await session.commit()

        done = set(
            (
                await session.execute(
                    select(PredictionRecord.position).where(PredictionRecord.job_id == job_id)
                )
            ).scalars()
        )
        remaining = [position for position in range(len(texts)) if position not in done]

        for start in range(0, len(remaining), CHUNK_SIZE):
            positions = remaining[start : start + CHUNK_SIZE]
            # CPU-bound, so it runs in a thread and the worker's event loop stays responsive.
            predictions = await asyncio.to_thread(
                _predict_all, classifier, [texts[position] for position in positions]
            )
            await _save_chunk(session, job, positions, texts, predictions)

        job.status = JobStatus.SUCCEEDED
        job.finished_at = datetime.now(UTC)
        await session.commit()


def _predict_all(classifier: Classifier, texts: list[str]) -> list[Prediction]:
    return [classifier.predict(text) for text in texts]


async def _save_chunk(
    session: AsyncSession,
    job: Job,
    positions: list[int],
    texts: list[str],
    predictions: list[Prediction],
) -> None:
    rows = [
        {
            "id": uuid.uuid4(),
            "user_id": job.user_id,
            "job_id": job.id,
            "position": position,
            "input_sha256": hashlib.sha256(texts[position].encode("utf-8")).hexdigest(),
            "label": prediction.label.value,
            "probability_ai": prediction.probability_ai,
            "model_version": prediction.model_version,
        }
        for position, prediction in zip(positions, predictions, strict=True)
    ]
    # If another run already stored a position, keep that row and move on.
    stored = await session.execute(
        insert(PredictionRecord)
        .values(rows)
        .on_conflict_do_nothing(constraint="uq_predictions_job_id_position")
        .returning(PredictionRecord.label)
    )
    # Counted from what was actually inserted, so a re-run does not count a text twice.
    for label in stored.scalars():
        PREDICTIONS.labels(label, "batch").inc()
    # Counted from the table, not incremented, so it stays right across re-runs.
    job.processed = (
        await session.execute(
            select(func.count())
            .select_from(PredictionRecord)
            .where(PredictionRecord.job_id == job.id)
        )
    ).scalar_one()
    await session.commit()


async def mark_failed(
    session_factory: async_sessionmaker[AsyncSession], job_id: uuid.UUID, error: str
) -> None:
    async with session_factory() as session:
        job = await session.get(Job, job_id)
        if job is None or job.status.is_finished:
            return
        job.status = JobStatus.FAILED
        job.error = error
        job.finished_at = datetime.now(UTC)
        await session.commit()
