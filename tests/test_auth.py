import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from provenance_api.security import create_access_token
from tests.conftest import PASSWORD, TEST_JWT_SECRET, Headers, MakeUser

pytestmark = pytest.mark.anyio

EMAIL = "new@example.com"


def bearer(token: str) -> Headers:
    return {"Authorization": f"Bearer {token}"}


def assert_unauthorized(response: httpx.Response) -> None:
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert "WWW-Authenticate" in response.headers


async def test_register_creates_a_member_on_the_free_plan(
    client: httpx.AsyncClient, db: AsyncEngine
) -> None:
    response = await client.post("/v1/auth/register", json={"email": EMAIL, "password": PASSWORD})

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"id", "email", "role", "plan", "created_at"}
    assert (body["email"], body["role"], body["plan"]) == (EMAIL, "member", "free")


async def test_password_is_stored_as_an_argon2_hash(
    client: httpx.AsyncClient, db: AsyncEngine
) -> None:
    await client.post("/v1/auth/register", json={"email": EMAIL, "password": PASSWORD})

    async with db.connect() as connection:
        stored = (await connection.execute(text("SELECT password_hash FROM users"))).scalar_one()
    assert stored.startswith("$argon2id$")
    assert PASSWORD not in stored


async def test_registering_the_same_email_twice_is_a_conflict(
    client: httpx.AsyncClient, db: AsyncEngine
) -> None:
    await client.post("/v1/auth/register", json={"email": EMAIL, "password": PASSWORD})

    response = await client.post(
        "/v1/auth/register", json={"email": EMAIL.upper(), "password": PASSWORD}
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "email_taken"


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": PASSWORD},
        {"email": EMAIL, "password": "short"},
        {"email": EMAIL},
        {"email": EMAIL, "password": PASSWORD, "role": "admin"},
    ],
)
async def test_invalid_registration_is_rejected(
    client: httpx.AsyncClient, db: AsyncEngine, payload: dict[str, str]
) -> None:
    response = await client.post("/v1/auth/register", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_login_returns_a_token_that_opens_me(
    client: httpx.AsyncClient, db: AsyncEngine
) -> None:
    await client.post("/v1/auth/register", json={"email": EMAIL, "password": PASSWORD})

    login = await client.post("/v1/auth/login", json={"email": EMAIL, "password": PASSWORD})

    assert login.status_code == 200
    assert login.json()["token_type"] == "bearer"
    assert login.json()["expires_in"] == 30 * 60
    me = await client.get("/v1/me", headers=bearer(login.json()["access_token"]))
    assert me.status_code == 200
    assert me.json()["email"] == EMAIL


async def test_wrong_password_and_unknown_email_get_the_same_answer(
    client: httpx.AsyncClient, db: AsyncEngine
) -> None:
    await client.post("/v1/auth/register", json={"email": EMAIL, "password": PASSWORD})

    wrong_password = await client.post(
        "/v1/auth/login", json={"email": EMAIL, "password": "wrong password!"}
    )
    unknown_email = await client.post(
        "/v1/auth/login", json={"email": "nobody@example.com", "password": PASSWORD}
    )

    assert_unauthorized(wrong_password)
    assert_unauthorized(unknown_email)
    assert wrong_password.json()["error"]["message"] == unknown_email.json()["error"]["message"]


async def test_me_without_a_token_is_unauthorized(client: httpx.AsyncClient) -> None:
    assert_unauthorized(await client.get("/v1/me"))


async def test_me_with_a_garbage_token_is_unauthorized(client: httpx.AsyncClient) -> None:
    assert_unauthorized(await client.get("/v1/me", headers=bearer("not.a.token")))


async def test_expired_token_is_unauthorized(client: httpx.AsyncClient, db: AsyncEngine) -> None:
    registered = await client.post("/v1/auth/register", json={"email": EMAIL, "password": PASSWORD})
    expired = create_access_token(
        uuid.UUID(registered.json()["id"]),
        secret=TEST_JWT_SECRET,
        lifetime=timedelta(minutes=30),
        now=datetime.now(UTC) - timedelta(minutes=31),
    )

    assert_unauthorized(await client.get("/v1/me", headers=bearer(expired)))


async def test_token_signed_with_another_secret_is_unauthorized(
    client: httpx.AsyncClient, db: AsyncEngine
) -> None:
    registered = await client.post("/v1/auth/register", json={"email": EMAIL, "password": PASSWORD})
    forged = create_access_token(
        uuid.UUID(registered.json()["id"]),
        secret="attacker-chosen-secret-" + "1" * 32,
        lifetime=timedelta(minutes=30),
    )

    assert_unauthorized(await client.get("/v1/me", headers=bearer(forged)))


async def test_token_of_a_deleted_user_is_unauthorized(
    client: httpx.AsyncClient, db: AsyncEngine, member: Headers
) -> None:
    async with db.begin() as connection:
        await connection.execute(text("DELETE FROM users"))

    assert_unauthorized(await client.get("/v1/me", headers=member))


async def test_member_cannot_list_users(client: httpx.AsyncClient, member: Headers) -> None:
    response = await client.get("/v1/admin/users", headers=member)

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


async def test_listing_users_without_a_token_is_unauthorized(client: httpx.AsyncClient) -> None:
    assert_unauthorized(await client.get("/v1/admin/users"))


async def test_admin_can_page_through_users(client: httpx.AsyncClient, make_user: MakeUser) -> None:
    admin = await make_user("admin@example.com", role="admin")
    await make_user("b@example.com")
    await make_user("c@example.com")

    first = await client.get("/v1/admin/users", params={"limit": 2}, headers=admin)
    second = await client.get(
        "/v1/admin/users",
        params={"limit": 2, "cursor": first.json()["next_cursor"]},
        headers=admin,
    )

    assert first.status_code == 200
    assert [user["email"] for user in first.json()["items"]] == [
        "admin@example.com",
        "b@example.com",
    ]
    assert [user["email"] for user in second.json()["items"]] == ["c@example.com"]
    assert second.json()["next_cursor"] is None


async def test_role_change_applies_to_a_token_issued_earlier(
    client: httpx.AsyncClient, db: AsyncEngine, member: Headers
) -> None:
    async with db.begin() as connection:
        await connection.execute(text("UPDATE users SET role = 'admin'"))

    response = await client.get("/v1/admin/users", headers=member)

    assert response.status_code == 200


async def test_invalid_cursor_is_a_bad_request(
    client: httpx.AsyncClient, make_user: MakeUser
) -> None:
    admin = await make_user("admin@example.com", role="admin")

    response = await client.get("/v1/admin/users", params={"cursor": "garbage"}, headers=admin)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_cursor"
