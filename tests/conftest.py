import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from provenance_api.config import Settings
from provenance_api.main import create_app

REPO_ROOT = Path(__file__).parent.parent
TEST_JWT_SECRET = "test-only-secret-" + "0" * 32
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
async def client(database_url: str) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        Settings(database_url=database_url, max_text_chars=5_000, jwt_secret=TEST_JWT_SECRET)
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as http,
    ):
        yield http


@pytest.fixture
async def db(database_url: str) -> AsyncIterator[AsyncEngine]:
    """An engine for inspecting the database, starting from empty tables."""
    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE predictions, api_keys, users CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
def make_user(client: httpx.AsyncClient, db: AsyncEngine) -> MakeUser:
    """Register a user through the API and return headers carrying their access token."""

    async def make(email: str = "member@example.com", role: str = "member") -> Headers:
        registered = await client.post(
            "/v1/auth/register", json={"email": email, "password": PASSWORD}
        )
        assert registered.status_code == 201, registered.text
        if role != "member":
            async with db.begin() as connection:
                await connection.execute(
                    text("UPDATE users SET role = :role WHERE email = :email"),
                    {"role": role, "email": email},
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
