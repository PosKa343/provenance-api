import hashlib

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.conftest import Headers, MakeUser

pytestmark = pytest.mark.anyio

CLASSIFY_BODY = {"text": "The committee met on Tuesday. It approved the budget."}


async def test_created_key_is_returned_once_and_only_its_hash_is_stored(
    client: httpx.AsyncClient, db: AsyncEngine, member: Headers
) -> None:
    created = await client.post("/v1/api-keys", json={"name": "laptop"}, headers=member)

    assert created.status_code == 201
    key = created.json()["key"]
    assert key.startswith("pvk_")
    assert created.json()["prefix"] == key[:12]

    async with db.connect() as connection:
        row = (await connection.execute(text("SELECT * FROM api_keys"))).mappings().one()
    assert row["key_hash"] == hashlib.sha256(key.encode()).hexdigest()
    assert key not in [value for value in row.values() if isinstance(value, str)]

    listed = await client.get("/v1/api-keys", headers=member)
    assert set(listed.json()[0]) == {
        "id",
        "name",
        "prefix",
        "created_at",
        "last_used_at",
        "revoked_at",
    }


async def test_key_authenticates_classify_and_the_prediction_belongs_to_its_owner(
    client: httpx.AsyncClient, db: AsyncEngine, member: Headers, api_key: Headers
) -> None:
    response = await client.post("/v1/classify", json=CLASSIFY_BODY, headers=api_key)

    assert response.status_code == 200
    me = await client.get("/v1/me", headers=member)
    async with db.connect() as connection:
        owner = (await connection.execute(text("SELECT user_id FROM predictions"))).scalar_one()
        last_used = (
            await connection.execute(text("SELECT last_used_at FROM api_keys"))
        ).scalar_one()
    assert str(owner) == me.json()["id"]
    assert last_used is not None


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "pvk_not-a-real-key"}])
async def test_classify_without_a_valid_key_is_unauthorized(
    client: httpx.AsyncClient, db: AsyncEngine, headers: Headers
) -> None:
    response = await client.post("/v1/classify", json=CLASSIFY_BODY, headers=headers)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    async with db.connect() as connection:
        count = (await connection.execute(text("SELECT count(*) FROM predictions"))).scalar_one()
    assert count == 0


async def test_access_token_is_not_accepted_in_place_of_an_api_key(
    client: httpx.AsyncClient, member: Headers
) -> None:
    response = await client.post("/v1/classify", json=CLASSIFY_BODY, headers=member)

    assert response.status_code == 401


async def test_revoked_key_stops_working(
    client: httpx.AsyncClient, member: Headers, api_key: Headers
) -> None:
    key_id = (await client.get("/v1/api-keys", headers=member)).json()[0]["id"]

    revoked = await client.delete(f"/v1/api-keys/{key_id}", headers=member)
    after = await client.post("/v1/classify", json=CLASSIFY_BODY, headers=api_key)

    assert revoked.status_code == 204
    assert after.status_code == 401
    listed = await client.get("/v1/api-keys", headers=member)
    assert listed.json()[0]["revoked_at"] is not None


async def test_revoking_twice_is_harmless(
    client: httpx.AsyncClient, member: Headers, api_key: Headers
) -> None:
    key_id = (await client.get("/v1/api-keys", headers=member)).json()[0]["id"]
    await client.delete(f"/v1/api-keys/{key_id}", headers=member)
    first_revoked_at = (await client.get("/v1/api-keys", headers=member)).json()[0]["revoked_at"]

    again = await client.delete(f"/v1/api-keys/{key_id}", headers=member)

    assert again.status_code == 204
    listed = await client.get("/v1/api-keys", headers=member)
    assert listed.json()[0]["revoked_at"] == first_revoked_at


async def test_user_cannot_revoke_or_see_another_users_key(
    client: httpx.AsyncClient, make_user: MakeUser, member: Headers, api_key: Headers
) -> None:
    key_id = (await client.get("/v1/api-keys", headers=member)).json()[0]["id"]
    other = await make_user("other@example.com")

    response = await client.delete(f"/v1/api-keys/{key_id}", headers=other)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert (await client.get("/v1/api-keys", headers=other)).json() == []
    still_works = await client.post("/v1/classify", json=CLASSIFY_BODY, headers=api_key)
    assert still_works.status_code == 200


async def test_viewer_can_list_keys_but_not_create_them(
    client: httpx.AsyncClient, make_user: MakeUser
) -> None:
    viewer = await make_user("viewer@example.com", role="viewer")

    created = await client.post("/v1/api-keys", json={"name": "nope"}, headers=viewer)
    listed = await client.get("/v1/api-keys", headers=viewer)

    assert created.status_code == 403
    assert created.json()["error"]["code"] == "forbidden"
    assert listed.status_code == 200


async def test_api_keys_require_a_token(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/api-keys")).status_code == 401
    assert (await client.post("/v1/api-keys", json={"name": "x"})).status_code == 401
