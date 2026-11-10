import os
from collections.abc import AsyncIterator, Iterator
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
    app = create_app(Settings(database_url=database_url, max_text_chars=5_000))
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
        await connection.execute(text("TRUNCATE predictions"))
    yield engine
    await engine.dispose()
