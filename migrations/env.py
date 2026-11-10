import asyncio

from alembic import context
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from provenance_api import models  # noqa: F401 - registers the tables on Base.metadata
from provenance_api.config import BackendSettings
from provenance_api.db import Base

# Tests pass a URL through the Alembic config; otherwise use the app's settings.
database_url = context.config.get_main_option("sqlalchemy.url") or BackendSettings().database_url


def run_offline() -> None:
    context.configure(url=database_url, target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_online() -> None:
    engine = create_async_engine(database_url)
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    asyncio.run(run_online())
