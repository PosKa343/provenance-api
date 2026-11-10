import hashlib
import hmac
import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from provenance_api.billing import BillingError, gateway_from_settings
from provenance_api.config import Settings
from tests.conftest import TEST_JWT_SECRET, TEST_WEBHOOK_SECRET, Headers

pytestmark = pytest.mark.anyio

CUSTOMER = "cus_test_123"


class FakeGateway:
    """Stands in for Stripe's network API and records what was asked of it."""

    def __init__(self) -> None:
        self.customers_created: list[str] = []
        self.checkouts_created: list[str] = []
        self.fail = False

    def create_customer(self, email: str, user_id: uuid.UUID) -> str:
        if self.fail:
            raise BillingError("APIConnectionError")
        self.customers_created.append(email)
        return CUSTOMER

    def create_checkout_url(self, customer_id: str, user_id: uuid.UUID) -> str:
        self.checkouts_created.append(customer_id)
        return f"https://checkout.stripe.test/pay/{customer_id}"


@pytest.fixture
def gateway(app: FastAPI) -> Iterator[FakeGateway]:
    fake = FakeGateway()
    original, app.state.billing = app.state.billing, fake
    yield fake
    app.state.billing = original


@pytest.fixture
async def customer(client: httpx.AsyncClient, gateway: FakeGateway, member: Headers) -> Headers:
    """A free user who has started a checkout, so Stripe events can be matched to them."""
    await client.post("/v1/billing/checkout", headers=member)
    return member


def event(
    event_type: str, *, event_id: str = "evt_1", created: int = 1_000, **fields: Any
) -> dict[str, Any]:
    return {
        "id": event_id,
        "type": event_type,
        "created": created,
        "data": {"object": {"customer": CUSTOMER, **fields}},
    }


CHECKOUT_COMPLETED = event("checkout.session.completed", mode="subscription", payment_status="paid")


def sign(body: bytes, secret: str = TEST_WEBHOOK_SECRET, timestamp: int | None = None) -> str:
    """Build the Stripe-Signature header the way Stripe does."""
    timestamp = timestamp or int(time.time())
    digest = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256)
    return f"t={timestamp},v1={digest.hexdigest()}"


async def deliver(client: httpx.AsyncClient, payload: dict[str, Any]) -> httpx.Response:
    body = json.dumps(payload).encode()
    return await client.post(
        "/v1/billing/webhook", content=body, headers={"Stripe-Signature": sign(body)}
    )


async def plan_of(client: httpx.AsyncClient, user: Headers) -> str:
    return str((await client.get("/v1/me", headers=user)).json()["plan"])


async def test_checkout_creates_a_stripe_customer_once_and_returns_the_payment_page(
    client: httpx.AsyncClient, db: AsyncEngine, gateway: FakeGateway, member: Headers
) -> None:
    first = await client.post("/v1/billing/checkout", headers=member)
    second = await client.post("/v1/billing/checkout", headers=member)

    assert first.status_code == 200
    assert first.json() == {"checkout_url": f"https://checkout.stripe.test/pay/{CUSTOMER}"}
    assert second.status_code == 200
    assert gateway.customers_created == ["member@example.com"]
    assert gateway.checkouts_created == [CUSTOMER, CUSTOMER]
    async with db.connect() as connection:
        stored = (await connection.execute(text("SELECT stripe_customer_id FROM users"))).scalar()
    assert stored == CUSTOMER


async def test_checkout_needs_a_token(client: httpx.AsyncClient, gateway: FakeGateway) -> None:
    assert (await client.post("/v1/billing/checkout")).status_code == 401


async def test_checkout_is_refused_for_a_user_already_on_pro(
    client: httpx.AsyncClient, gateway: FakeGateway, pro: Headers
) -> None:
    response = await client.post("/v1/billing/checkout", headers=pro)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "already_subscribed"
    assert gateway.checkouts_created == []


async def test_checkout_reports_a_stripe_outage(
    client: httpx.AsyncClient, gateway: FakeGateway, member: Headers
) -> None:
    gateway.fail = True

    response = await client.post("/v1/billing/checkout", headers=member)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "billing_unavailable"


async def test_billing_endpoints_say_so_when_stripe_is_not_configured(
    client: httpx.AsyncClient,
    app: FastAPI,
    gateway: FakeGateway,
    member: Headers,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app.state.billing = None  # the gateway fixture restores the original afterwards
    monkeypatch.setattr(settings, "stripe_webhook_secret", None)

    checkout = await client.post("/v1/billing/checkout", headers=member)
    webhook = await client.post("/v1/billing/webhook", content=b"{}")

    assert (checkout.status_code, webhook.status_code) == (503, 503)
    assert checkout.json()["error"]["code"] == "billing_not_configured"


async def test_completed_checkout_moves_the_user_to_pro_and_unlocks_batch_jobs(
    client: httpx.AsyncClient, customer: Headers, api_key: Headers
) -> None:
    batch = {"texts": ["First text to score.", "Second text to score."]}
    assert (await client.post("/v1/jobs", json=batch, headers=api_key)).status_code == 403

    response = await deliver(client, CHECKOUT_COMPLETED)

    assert response.status_code == 200
    assert response.json() == {"status": "processed"}
    assert await plan_of(client, customer) == "pro"
    assert (await client.post("/v1/jobs", json=batch, headers=api_key)).status_code == 202


@pytest.mark.parametrize(
    "signature",
    [
        None,
        "garbage",
        "wrong-secret",
        "expired",
    ],
)
async def test_webhook_with_a_bad_signature_is_rejected_and_changes_nothing(
    client: httpx.AsyncClient, db: AsyncEngine, customer: Headers, signature: str | None
) -> None:
    body = json.dumps(CHECKOUT_COMPLETED).encode()
    header = {
        None: None,
        "garbage": "t=1,v1=deadbeef",
        "wrong-secret": sign(body, secret="whsec_attacker"),
        "expired": sign(body, timestamp=int(time.time()) - 600),
    }[signature]

    response = await client.post(
        "/v1/billing/webhook",
        content=body,
        headers={"Stripe-Signature": header} if header else {},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_webhook"
    assert await plan_of(client, customer) == "free"
    async with db.connect() as connection:
        assert (await connection.execute(text("SELECT count(*) FROM stripe_events"))).scalar() == 0


async def test_webhook_body_changed_after_signing_is_rejected(
    client: httpx.AsyncClient, customer: Headers
) -> None:
    signed_body = json.dumps(event("customer.subscription.deleted")).encode()
    sent_body = json.dumps(CHECKOUT_COMPLETED).encode()

    response = await client.post(
        "/v1/billing/webhook", content=sent_body, headers={"Stripe-Signature": sign(signed_body)}
    )

    assert response.status_code == 400
    assert await plan_of(client, customer) == "free"


async def test_redelivered_event_is_acknowledged_but_not_applied_twice(
    client: httpx.AsyncClient, db: AsyncEngine, customer: Headers
) -> None:
    await deliver(client, CHECKOUT_COMPLETED)
    # Undo its effect by hand: if the redelivery were applied again, the plan would return to pro.
    async with db.begin() as connection:
        await connection.execute(text("UPDATE users SET plan = 'free'"))

    again = await deliver(client, CHECKOUT_COMPLETED)

    assert again.status_code == 200
    assert again.json() == {"status": "already_processed"}
    assert await plan_of(client, customer) == "free"
    async with db.connect() as connection:
        assert (await connection.execute(text("SELECT count(*) FROM stripe_events"))).scalar() == 1


@pytest.mark.parametrize(
    ("status", "plan"),
    [
        ("active", "pro"),
        ("trialing", "pro"),
        ("past_due", "pro"),
        ("canceled", "free"),
        ("unpaid", "free"),
        ("incomplete_expired", "free"),
    ],
)
async def test_subscription_status_decides_the_plan(
    client: httpx.AsyncClient, customer: Headers, status: str, plan: str
) -> None:
    await deliver(client, CHECKOUT_COMPLETED)

    await deliver(
        client,
        event("customer.subscription.updated", event_id="evt_2", created=2_000, status=status),
    )

    assert await plan_of(client, customer) == plan


async def test_deleted_subscription_returns_the_user_to_free(
    client: httpx.AsyncClient, customer: Headers
) -> None:
    await deliver(client, CHECKOUT_COMPLETED)

    await deliver(client, event("customer.subscription.deleted", event_id="evt_2", created=2_000))

    assert await plan_of(client, customer) == "free"


async def test_an_older_event_arriving_late_does_not_undo_a_newer_one(
    client: httpx.AsyncClient, customer: Headers
) -> None:
    await deliver(client, event("customer.subscription.deleted", event_id="evt_2", created=2_000))

    late = await deliver(
        client,
        event("customer.subscription.updated", event_id="evt_1", created=1_000, status="active"),
    )

    assert late.status_code == 200
    assert await plan_of(client, customer) == "free"


async def test_event_for_an_unknown_customer_or_of_another_type_is_recorded_and_ignored(
    client: httpx.AsyncClient, db: AsyncEngine, customer: Headers
) -> None:
    unknown = event("customer.subscription.deleted", event_id="evt_a")
    unknown["data"]["object"]["customer"] = "cus_someone_else"
    other_type = event("invoice.paid", event_id="evt_b")

    responses = [await deliver(client, unknown), await deliver(client, other_type)]

    assert [response.status_code for response in responses] == [200, 200]
    assert await plan_of(client, customer) == "free"
    async with db.connect() as connection:
        assert (await connection.execute(text("SELECT count(*) FROM stripe_events"))).scalar() == 2


async def test_unpaid_checkout_does_not_grant_pro(
    client: httpx.AsyncClient, customer: Headers
) -> None:
    await deliver(
        client,
        event("checkout.session.completed", mode="subscription", payment_status="unpaid"),
    )

    assert await plan_of(client, customer) == "free"


def test_a_live_mode_stripe_key_is_refused_at_startup() -> None:
    with pytest.raises(ValidationError, match="test-mode key"):
        Settings(_env_file=None, jwt_secret=TEST_JWT_SECRET, stripe_secret_key="sk_live_abc123")


def test_empty_stripe_settings_mean_billing_is_off() -> None:
    settings = Settings(
        _env_file=None,
        jwt_secret=TEST_JWT_SECRET,
        stripe_secret_key="",
        stripe_webhook_secret="",
        stripe_price_id="",
    )

    assert gateway_from_settings(settings) is None
    assert settings.stripe_webhook_secret is None
