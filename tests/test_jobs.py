import hashlib
import logging
import uuid
from collections.abc import Awaitable, Callable

import httpx
import pytest
from arq.connections import ArqRedis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from provenance_api.config import Settings
from provenance_api.db import create_session_factory
from provenance_api.jobs import run_job
from provenance_api.ml.model import Classifier, Prediction
from provenance_api.worker import create_worker
from tests.conftest import Headers, MakeUser

pytestmark = pytest.mark.anyio

TEXTS = [
    "The committee met on Tuesday. It approved the budget.",
    "Rain fell all night. The river rose by morning.",
    "She opened the old door slowly. Nobody was inside.",
]

RunWorker = Callable[[], Awaitable[None]]


@pytest.fixture
def run_worker(settings: Settings, queue: ArqRedis) -> RunWorker:
    """Run a real arq worker until the queue is empty, retries included."""

    async def run() -> None:
        worker = create_worker(settings, redis_pool=queue, burst=True, poll_delay=0.01)
        try:
            await worker.main()
        finally:
            # worker.close() would also close the shared queue connection.
            await worker.ctx["engine"].dispose()

    return run


async def submit(
    client: httpx.AsyncClient, api_key: Headers, texts: list[str], key: str | None = None
) -> httpx.Response:
    headers = {**api_key, **({"Idempotency-Key": key} if key else {})}
    return await client.post("/v1/jobs", json={"texts": texts}, headers=headers)


async def count(db: AsyncEngine, table: str) -> int:
    async with db.connect() as connection:
        return int((await connection.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one())  # noqa: S608


async def test_submitted_job_is_queued_then_scored_by_the_worker(
    client: httpx.AsyncClient, db: AsyncEngine, pro_api_key: Headers, run_worker: RunWorker
) -> None:
    submitted = await submit(client, pro_api_key, TEXTS)

    assert submitted.status_code == 202
    job = submitted.json()
    assert (job["status"], job["total"], job["processed"]) == ("queued", 3, 0)
    assert job["started_at"] is None

    await run_worker()

    finished = (await client.get(f"/v1/jobs/{job['id']}", headers=pro_api_key)).json()
    assert (finished["status"], finished["processed"], finished["error"]) == ("succeeded", 3, None)
    assert finished["started_at"] is not None
    assert finished["finished_at"] is not None

    results = (await client.get(f"/v1/jobs/{job['id']}/results", headers=pro_api_key)).json()
    assert [item["position"] for item in results["items"]] == [0, 1, 2]
    assert [item["input_sha256"] for item in results["items"]] == [
        hashlib.sha256(t.encode()).hexdigest() for t in TEXTS
    ]
    assert all(item["label"] in ("human", "ai") for item in results["items"])
    assert results["next_cursor"] is None


async def test_batch_result_matches_the_single_text_endpoint(
    client: httpx.AsyncClient, db: AsyncEngine, pro_api_key: Headers, run_worker: RunWorker
) -> None:
    single = (
        await client.post("/v1/classify", json={"text": TEXTS[0]}, headers=pro_api_key)
    ).json()
    job = (await submit(client, pro_api_key, [TEXTS[0]])).json()

    await run_worker()

    item = (await client.get(f"/v1/jobs/{job['id']}/results", headers=pro_api_key)).json()["items"][
        0
    ]
    assert item["label"] == single["label"]
    assert item["probability_ai"] == pytest.approx(single["probability_ai"])


async def test_texts_are_not_kept_in_postgres_or_redis_after_the_job(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    pro_api_key: Headers,
    queue: ArqRedis,
    run_worker: RunWorker,
) -> None:
    await submit(client, pro_api_key, TEXTS)
    await run_worker()

    async with db.connect() as connection:
        rows = (await connection.execute(text("SELECT * FROM predictions"))).mappings().all()
        jobs = (await connection.execute(text("SELECT * FROM jobs"))).mappings().all()
    stored = [value for row in [*rows, *jobs] for value in row.values() if isinstance(value, str)]
    assert not any(t in value for t in TEXTS for value in stored)
    # What remains in Redis is counters only: rate-limit buckets, quotas, worker health.
    for pattern in ("provenance:job-texts:*", "arq:job:*", "arq:result:*"):
        assert await queue.keys(pattern) == []


async def test_retry_with_the_same_idempotency_key_returns_the_first_job(
    client: httpx.AsyncClient, db: AsyncEngine, pro_api_key: Headers, run_worker: RunWorker
) -> None:
    first = await submit(client, pro_api_key, TEXTS, key="order-42")
    second = await submit(client, pro_api_key, TEXTS, key="order-42")

    assert (first.status_code, second.status_code) == (202, 200)
    assert second.json()["id"] == first.json()["id"]
    assert await count(db, "jobs") == 1

    await run_worker()
    assert await count(db, "predictions") == 3

    after_finish = await submit(client, pro_api_key, TEXTS, key="order-42")
    assert after_finish.status_code == 200
    assert after_finish.json()["status"] == "succeeded"
    await run_worker()
    assert await count(db, "predictions") == 3


async def test_same_key_with_different_texts_is_a_conflict(
    client: httpx.AsyncClient, db: AsyncEngine, pro_api_key: Headers, queue: ArqRedis
) -> None:
    await submit(client, pro_api_key, TEXTS, key="order-42")

    response = await submit(client, pro_api_key, TEXTS[:1], key="order-42")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "idempotency_key_reused"
    assert await count(db, "jobs") == 1


async def test_two_users_can_use_the_same_idempotency_key(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    pro_api_key: Headers,
    make_user: MakeUser,
    queue: ArqRedis,
) -> None:
    other = await make_user("other@example.com", plan="pro")
    other_key = (await client.post("/v1/api-keys", json={"name": "k"}, headers=other)).json()["key"]

    mine = await submit(client, pro_api_key, TEXTS, key="order-42")
    theirs = await submit(client, {"X-API-Key": other_key}, TEXTS, key="order-42")

    assert (mine.status_code, theirs.status_code) == (202, 202)
    assert mine.json()["id"] != theirs.json()["id"]


async def test_without_a_key_each_submission_is_a_new_job(
    client: httpx.AsyncClient, db: AsyncEngine, pro_api_key: Headers, queue: ArqRedis
) -> None:
    first = await submit(client, pro_api_key, TEXTS)
    second = await submit(client, pro_api_key, TEXTS)

    assert first.json()["id"] != second.json()["id"]
    assert await count(db, "jobs") == 2


async def test_job_fails_after_three_attempts_when_scoring_keeps_failing(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    pro_api_key: Headers,
    queue: ArqRedis,
    run_worker: RunWorker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def explode(self: Classifier, text: str) -> Prediction:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("secret internal detail")

    job = (await submit(client, pro_api_key, TEXTS)).json()
    monkeypatch.setattr(Classifier, "predict", explode)

    await run_worker()

    failed = (await client.get(f"/v1/jobs/{job['id']}", headers=pro_api_key)).json()
    assert failed["status"] == "failed"
    assert failed["error"] == "Scoring failed after 3 attempts (RuntimeError)."
    assert failed["finished_at"] is not None
    assert attempts == 3
    assert await count(db, "predictions") == 0
    assert await queue.keys("provenance:job-texts:*") == []


async def test_job_that_fails_once_is_retried_and_resumes_without_duplicates(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    pro_api_key: Headers,
    run_worker: RunWorker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("provenance_api.jobs.CHUNK_SIZE", 2)
    real_predict = Classifier.predict
    scored: list[str] = []
    failed_once = False

    def flaky(self: Classifier, text: str) -> Prediction:
        nonlocal failed_once
        if text == TEXTS[2] and not failed_once:
            failed_once = True
            raise RuntimeError("transient")
        scored.append(text)
        return real_predict(self, text)

    job = (await submit(client, pro_api_key, TEXTS)).json()
    monkeypatch.setattr(Classifier, "predict", flaky)

    await run_worker()

    finished = (await client.get(f"/v1/jobs/{job['id']}", headers=pro_api_key)).json()
    assert (finished["status"], finished["processed"]) == ("succeeded", 3)
    assert await count(db, "predictions") == 3
    # The first two texts were saved before the failure and were not scored again.
    assert scored == TEXTS


async def test_running_a_finished_job_again_changes_nothing(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    pro_api_key: Headers,
    run_worker: RunWorker,
) -> None:
    job = (await submit(client, pro_api_key, TEXTS)).json()
    await run_worker()
    before = (await client.get(f"/v1/jobs/{job['id']}", headers=pro_api_key)).json()

    await run_job(create_session_factory(db), Classifier.load(), uuid.UUID(job["id"]), TEXTS)

    assert (await client.get(f"/v1/jobs/{job['id']}", headers=pro_api_key)).json() == before
    assert await count(db, "predictions") == 3


async def test_results_are_paginated_in_submission_order(
    client: httpx.AsyncClient, db: AsyncEngine, pro_api_key: Headers, run_worker: RunWorker
) -> None:
    job = (await submit(client, pro_api_key, TEXTS)).json()
    await run_worker()
    url = f"/v1/jobs/{job['id']}/results"

    first = (await client.get(url, params={"limit": 2}, headers=pro_api_key)).json()
    second = (
        await client.get(
            url, params={"limit": 2, "cursor": first["next_cursor"]}, headers=pro_api_key
        )
    ).json()

    assert [item["position"] for item in first["items"]] == [0, 1]
    assert first["next_cursor"] == 1
    assert [item["position"] for item in second["items"]] == [2]
    assert second["next_cursor"] is None


async def test_another_users_job_looks_like_a_missing_one(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    pro_api_key: Headers,
    make_user: MakeUser,
    queue: ArqRedis,
) -> None:
    job = (await submit(client, pro_api_key, TEXTS)).json()
    other = await make_user("other@example.com")
    other_key = {
        "X-API-Key": (await client.post("/v1/api-keys", json={"name": "k"}, headers=other)).json()[
            "key"
        ]
    }

    for path in (
        f"/v1/jobs/{job['id']}",
        f"/v1/jobs/{job['id']}/results",
        f"/v1/jobs/{uuid.uuid4()}",
    ):
        response = await client.get(path, headers=other_key)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    ("texts", "status", "code"),
    [
        ([], 422, "validation_error"),
        (["fine", "   "], 422, "validation_error"),
        (["one"] * 6, 413, "batch_too_large"),
        (["word " * 1001], 413, "text_too_long"),
    ],
)
async def test_invalid_batches_are_rejected_and_nothing_is_queued(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    pro_api_key: Headers,
    queue: ArqRedis,
    texts: list[str],
    status: int,
    code: str,
) -> None:
    response = await submit(client, pro_api_key, texts)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert await count(db, "jobs") == 0
    assert await queue.zcard("arq:queue") == 0


async def test_jobs_require_an_api_key(client: httpx.AsyncClient, queue: ArqRedis) -> None:
    assert (await client.post("/v1/jobs", json={"texts": TEXTS})).status_code == 401
    assert (await client.get(f"/v1/jobs/{uuid.uuid4()}")).status_code == 401


async def test_job_is_marked_failed_when_it_cannot_be_queued(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    pro_api_key: Headers,
    queue: ArqRedis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(*args: object, **kwargs: object) -> None:
        raise ConnectionError("redis is down")

    monkeypatch.setattr("provenance_api.routes.jobs.enqueue_scoring", unavailable)

    response = await submit(client, pro_api_key, TEXTS)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "queue_unavailable"
    async with db.connect() as connection:
        row = (await connection.execute(text("SELECT status, error FROM jobs"))).one()
    assert tuple(row) == ("failed", "The job could not be queued.")


async def test_submitted_texts_never_appear_in_logs(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    pro_api_key: Headers,
    run_worker: RunWorker,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def explode(self: Classifier, text: str) -> Prediction:
        raise RuntimeError(f"could not score: {text}")

    caplog.set_level(logging.DEBUG)
    await submit(client, pro_api_key, TEXTS)
    await run_worker()
    await submit(client, pro_api_key, TEXTS)
    monkeypatch.setattr(Classifier, "predict", explode)
    await run_worker()

    logged = "\n".join(
        record.getMessage() + str(record.exc_text or "") for record in caplog.records
    )
    assert "score_job" in logged
    assert not any(t[:20] in logged for t in TEXTS)


async def test_job_whose_texts_have_expired_is_marked_failed(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    pro_api_key: Headers,
    queue: ArqRedis,
    run_worker: RunWorker,
) -> None:
    job = (await submit(client, pro_api_key, TEXTS)).json()
    await queue.delete(f"provenance:job-texts:{job['id']}")

    await run_worker()

    failed = (await client.get(f"/v1/jobs/{job['id']}", headers=pro_api_key)).json()
    assert (failed["status"], failed["error"]) == (
        "failed",
        "The submitted texts are no longer available.",
    )
