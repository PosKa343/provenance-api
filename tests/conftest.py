import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from arq.connections import ArqRedis, RedisSettings, create_pool
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from provenance_api.config import Settings
from provenance_api.main import create_app

REPO_ROOT = Path(__file__).parent.parent
TEST_JWT_SECRET = "test-only-secret-" + "0" * 32
TEST_WEBHOOK_SECRET = "whsec_test_only"
PASSWORD = "correct horse battery staple"

Headers = dict[str, str]
MakeUser = Callable[..., Awaitable[Headers]]


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A real PostgreSQL with the schema built by the Alembic migrations.

    Uses PROVENANCE_TEST_DATABASE_URL if set, otherwise starts a throwaway
    container, which needs a running Docker daemon.
    """
    configured = os.environ.get("PROVENANCE_TEST_DATABASE_URL")
    if configured:
        _migrate(configured)
        yield configured
        return

    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine", driver="asyncpg") as postgres:
        url = postgres.get_connection_url()
        _migrate(url)
        yield url


def _migrate(url: str) -> None:
    config = Config(REPO_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """A real Redis. Uses PROVENANCE_TEST_REDIS_URL if set, otherwise a throwaway container."""
    configured = os.environ.get("PROVENANCE_TEST_REDIS_URL")
    if configured:
        yield configured
        return

    from testcontainers.redis import RedisContainer

    with RedisContainer("redis:7-alpine") as redis:
        yield f"redis://{redis.get_container_host_ip()}:{redis.get_exposed_port(6379)}"


@pytest.fixture(scope="session")
def settings(database_url: str, redis_url: str) -> Settings:
    return Settings(
        # A developer's .env may hold real Stripe test keys; tests must never read it.
        _env_file=None,
        stripe_webhook_secret=TEST_WEBHOOK_SECRET,
        database_url=database_url,
        redis_url=redis_url,
        jwt_secret=TEST_JWT_SECRET,
        max_text_chars=5_000,
        max_batch_texts=5,
        job_retry_backoff_seconds=0,
        # In-process inference, so tests can replace Classifier.predict to force
        # failures. The worker-process mode has its own tests in test_inference.py.
        inference_workers=0,
        # High enough that only the tests about limits ever reach them.
        free_daily_quota=1_000,
        free_burst=1_000,
        pro_daily_quota=1_000,
        pro_burst=1_000,
    )


@pytest.fixture(scope="session")
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture(scope="session")
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as http,
    ):
        yield http


@pytest.fixture
async def db(database_url: str, redis_url: str) -> AsyncIterator[AsyncEngine]:
    """An engine for inspecting the database. Starts from empty tables and an empty Redis."""
    redis = await create_pool(RedisSettings.from_dsn(redis_url))
    await redis.flushdb()
    await redis.aclose()
    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        await connection.execute(
            text("TRUNCATE stripe_events, usage_events, predictions, jobs, api_keys, users CASCADE")
        )
    yield engine
    await engine.dispose()


@pytest.fixture
def make_user(client: httpx.AsyncClient, db: AsyncEngine) -> MakeUser:
    """Register a user through the API and return headers carrying their access token."""

    async def make(
        email: str = "member@example.com", role: str = "member", plan: str = "free"
    ) -> Headers:
        registered = await client.post(
            "/v1/auth/register", json={"email": email, "password": PASSWORD}
        )
        assert registered.status_code == 201, registered.text
        async with db.begin() as connection:
            await connection.execute(
                text("UPDATE users SET role = :role, plan = :plan WHERE email = :email"),
                {"role": role, "plan": plan, "email": email},
            )
        login = await client.post("/v1/auth/login", json={"email": email, "password": PASSWORD})
        return {"Authorization": f"Bearer {login.json()['access_token']}"}

    return make


@pytest.fixture
async def member(make_user: MakeUser) -> Headers:
    return await make_user()


@pytest.fixture
async def api_key(client: httpx.AsyncClient, member: Headers) -> Headers:
    created = await client.post("/v1/api-keys", json={"name": "test"}, headers=member)
    return {"X-API-Key": created.json()["key"]}


@pytest.fixture
async def queue(redis_url: str) -> AsyncIterator[ArqRedis]:
    """A connection to the job queue, emptied before each test."""
    redis = await create_pool(RedisSettings.from_dsn(redis_url))
    await redis.flushdb()
    yield redis
    await redis.aclose()


@pytest.fixture
async def pro(make_user: MakeUser) -> Headers:
    return await make_user("pro@example.com", plan="pro")


@pytest.fixture
async def pro_api_key(client: httpx.AsyncClient, pro: Headers) -> Headers:
    created = await client.post("/v1/api-keys", json={"name": "test"}, headers=pro)
    return {"X-API-Key": created.json()["key"]}
