import anyio
import httpx
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from provenance_api.config import Settings
from tests.conftest import Headers

pytestmark = pytest.mark.anyio

BODY = {"text": "The committee met on Tuesday. It approved the budget."}
BATCH = {"texts": ["First text to score.", "Second text to score.", "Third text to score."]}


async def classify(client: httpx.AsyncClient, key: Headers) -> httpx.Response:
    return await client.post("/v1/classify", json=BODY, headers=key)


async def new_key(client: httpx.AsyncClient, token: Headers) -> Headers:
    created = await client.post("/v1/api-keys", json={"name": "second"}, headers=token)
    return {"X-API-Key": created.json()["key"]}


async def test_burst_beyond_the_bucket_is_rejected_then_recovers(
    client: httpx.AsyncClient,
    api_key: Headers,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "free_burst", 3)
    monkeypatch.setattr(settings, "free_refill_per_second", 2.0)

    statuses = [(await classify(client, api_key)).status_code for _ in range(3)]
    rejected = await classify(client, api_key)

    assert statuses == [200, 200, 200]
    assert rejected.status_code == 429
    assert rejected.json()["error"]["code"] == "rate_limited"
    assert rejected.headers["Retry-After"] == "1"

    await anyio.sleep(0.6)  # one token refills in half a second
    assert (await classify(client, api_key)).status_code == 200


async def test_each_api_key_has_its_own_bucket(
    client: httpx.AsyncClient,
    member: Headers,
    api_key: Headers,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "free_burst", 1)
    monkeypatch.setattr(settings, "free_refill_per_second", 0.01)
    second_key = await new_key(client, member)

    assert (await classify(client, api_key)).status_code == 200
    assert (await classify(client, api_key)).status_code == 429
    assert (await classify(client, second_key)).status_code == 200


async def test_an_invalid_key_cannot_drain_a_bucket(
    client: httpx.AsyncClient,
    api_key: Headers,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "free_burst", 1)
    monkeypatch.setattr(settings, "free_refill_per_second", 0.01)

    for _ in range(3):
        assert (await classify(client, {"X-API-Key": "pvk_wrong"})).status_code == 401

    assert (await classify(client, api_key)).status_code == 200


async def test_daily_quota_is_enforced_and_rejected_requests_do_no_work(
    client: httpx.AsyncClient,
    db: AsyncEngine,
    api_key: Headers,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "free_daily_quota", 2)

    statuses = [(await classify(client, api_key)).status_code for _ in range(2)]
    rejected = await classify(client, api_key)

    assert statuses == [200, 200]
    assert rejected.status_code == 429
    assert rejected.json()["error"]["code"] == "quota_exceeded"
    assert 1 <= int(rejected.headers["Retry-After"]) <= 86_400
    async with db.connect() as connection:
        predictions = (await connection.execute(text("SELECT count(*) FROM predictions"))).scalar()
        usage = (await connection.execute(text("SELECT sum(units) FROM usage_events"))).scalar()
    assert (predictions, usage) == (2, 2)


async def test_quota_is_shared_by_all_of_a_users_keys(
    client: httpx.AsyncClient,
    member: Headers,
    api_key: Headers,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "free_daily_quota", 1)
    second_key = await new_key(client, member)

    assert (await classify(client, api_key)).status_code == 200
    assert (await classify(client, second_key)).status_code == 429


async def test_invalid_request_does_not_use_quota(
    client: httpx.AsyncClient,
    api_key: Headers,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "free_daily_quota", 1)

    invalid = await client.post("/v1/classify", json={"text": "   "}, headers=api_key)

    assert invalid.status_code == 422
    assert (await classify(client, api_key)).status_code == 200


async def test_pro_plan_has_its_own_higher_quota(
    client: httpx.AsyncClient,
    pro_api_key: Headers,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "free_daily_quota", 1)
    monkeypatch.setattr(settings, "pro_daily_quota", 3)

    statuses = [(await classify(client, pro_api_key)).status_code for _ in range(4)]

    assert statuses == [200, 200, 200, 429]


async def test_free_plan_cannot_submit_batch_jobs(
    client: httpx.AsyncClient, db: AsyncEngine, api_key: Headers
) -> None:
    response = await client.post("/v1/jobs", json=BATCH, headers=api_key)

    assert response.status_code == 403
    assert response.json()["error"] == {
        "code": "forbidden",
        "message": "Batch jobs need the Pro plan.",
        "request_id": response.headers["X-Request-ID"],
    }
    async with db.connect() as connection:
        assert (await connection.execute(text("SELECT count(*) FROM jobs"))).scalar() == 0


async def test_batch_job_costs_one_unit_per_text(
    client: httpx.AsyncClient,
    pro: Headers,
    pro_api_key: Headers,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "pro_daily_quota", 5)

    first = await client.post("/v1/jobs", json=BATCH, headers=pro_api_key)
    second = await client.post("/v1/jobs", json=BATCH, headers=pro_api_key)

    assert first.status_code == 202
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "quota_exceeded"
    me = (await client.get("/v1/me", headers=pro)).json()
    assert me["usage"] == {"daily_limit": 5, "used_today": 3}


async def test_resubmitting_with_the_same_idempotency_key_is_not_charged_again(
    client: httpx.AsyncClient, db: AsyncEngine, pro: Headers, pro_api_key: Headers
) -> None:
    headers = {**pro_api_key, "Idempotency-Key": "batch-1"}

    await client.post("/v1/jobs", json=BATCH, headers=headers)
    replay = await client.post("/v1/jobs", json=BATCH, headers=headers)

    assert replay.status_code == 200
    assert (await client.get("/v1/me", headers=pro)).json()["usage"]["used_today"] == 3
    async with db.connect() as connection:
        assert (await connection.execute(text("SELECT sum(units) FROM usage_events"))).scalar() == 3


async def test_usage_reports_todays_totals_by_kind(
    client: httpx.AsyncClient, pro: Headers, pro_api_key: Headers
) -> None:
    await classify(client, pro_api_key)
    await classify(client, pro_api_key)
    await client.post("/v1/jobs", json=BATCH, headers=pro_api_key)

    usage = await client.get("/v1/usage", headers=pro)

    assert usage.status_code == 200
    days = usage.json()["days"]
    assert len(days) == 1
    assert {key: days[0][key] for key in ("classify", "batch", "total")} == {
        "classify": 2,
        "batch": 3,
        "total": 5,
    }


async def test_usage_shows_only_your_own_and_needs_a_token(
    client: httpx.AsyncClient, member: Headers, pro_api_key: Headers
) -> None:
    await classify(client, pro_api_key)

    assert (await client.get("/v1/usage", headers=member)).json() == {"days": []}
    assert (await client.get("/v1/usage")).status_code == 401
    assert (await client.get("/v1/usage", headers=pro_api_key)).status_code == 401


async def test_me_reports_plan_limit_and_todays_usage(
    client: httpx.AsyncClient, member: Headers, api_key: Headers, settings: Settings
) -> None:
    await classify(client, api_key)

    me = (await client.get("/v1/me", headers=member)).json()

    assert me["plan"] == "free"
    assert me["usage"] == {"daily_limit": settings.free_daily_quota, "used_today": 1}


async def test_requests_are_allowed_when_the_limit_store_is_down(
    client: httpx.AsyncClient, api_key: Headers, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def down(*args: object, **kwargs: object) -> None:
        raise RedisConnectionError("redis is down")

    limiter = client._transport.app.state.limiter  # type: ignore[attr-defined]
    monkeypatch.setattr(limiter, "_take_from_bucket", down)
    monkeypatch.setattr(limiter, "_consume_quota", down)

    assert (await classify(client, api_key)).status_code == 200
