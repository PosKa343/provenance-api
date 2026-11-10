import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from provenance_api.db import create_session_factory
from provenance_api.models import Role
from provenance_api.set_role import set_role
from tests.conftest import Headers

pytestmark = pytest.mark.anyio


async def test_set_role_promotes_an_existing_user(
    client: httpx.AsyncClient, db: AsyncEngine, member: Headers
) -> None:
    changed = await set_role(create_session_factory(db), "MEMBER@example.com", Role.ADMIN)

    assert changed is True
    assert (await client.get("/v1/me", headers=member)).json()["role"] == "admin"


async def test_set_role_reports_an_unknown_email(db: AsyncEngine) -> None:
    assert await set_role(create_session_factory(db), "nobody@example.com", Role.ADMIN) is False
