"""Stripe billing: starting a checkout, and turning webhook events into plan changes."""

import json
import uuid
from datetime import UTC, datetime
from typing import Protocol

import stripe
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from provenance_api.config import Settings
from provenance_api.models import Plan, StripeEvent, User

# A subscription Stripe is still trying to collect on (past_due) keeps Pro; one
# that has ended or never started does not.
_PRO_SUBSCRIPTION_STATUSES = frozenset({"active", "trialing", "past_due"})


class BillingError(Exception):
    """Stripe could not be reached or refused the request."""


class InvalidWebhookError(Exception):
    """The webhook's signature or body is not acceptable."""


class BillingGateway(Protocol):
    """The calls made to the payment provider. Tests replace this with a fake."""

    def create_customer(self, email: str, user_id: uuid.UUID) -> str: ...

    def create_checkout_url(self, customer_id: str, user_id: uuid.UUID) -> str: ...


class StripeGateway:
    def __init__(self, secret_key: str, price_id: str, success_url: str, cancel_url: str) -> None:
        self._client = stripe.StripeClient(secret_key, max_network_retries=2)
        self._price_id = price_id
        self._success_url = success_url
        self._cancel_url = cancel_url

    def create_customer(self, email: str, user_id: uuid.UUID) -> str:
        try:
            customer = self._client.v1.customers.create(
                params={"email": email, "metadata": {"user_id": str(user_id)}},
                # If this request is retried, Stripe returns the first customer
                # and does not create a second one.
                options={"idempotency_key": f"customer-{user_id}"},
            )
        except stripe.StripeError as error:
            raise BillingError(type(error).__name__) from error
        return customer.id

    def create_checkout_url(self, customer_id: str, user_id: uuid.UUID) -> str:
        try:
            session = self._client.v1.checkout.sessions.create(
                params={
                    "mode": "subscription",
                    "customer": customer_id,
                    "client_reference_id": str(user_id),
                    "line_items": [{"price": self._price_id, "quantity": 1}],
                    "success_url": self._success_url,
                    "cancel_url": self._cancel_url,
                }
            )
        except stripe.StripeError as error:
            raise BillingError(type(error).__name__) from error
        if session.url is None:
            raise BillingError("Checkout session has no URL")
        return session.url


def gateway_from_settings(settings: Settings) -> BillingGateway | None:
    if settings.stripe_secret_key is None or settings.stripe_price_id is None:
        return None
    return StripeGateway(
        settings.stripe_secret_key.get_secret_value(),
        settings.stripe_price_id,
        settings.billing_success_url,
        settings.billing_cancel_url,
    )


class _EventObject(BaseModel):
    customer: str | None = None
    status: str | None = None
    mode: str | None = None
    payment_status: str | None = None


class _EventData(BaseModel):
    object: _EventObject


class BillingEvent(BaseModel):
    """The few fields of a Stripe event this service reads."""

    id: str
    type: str
    created: int
    data: _EventData

    @property
    def created_at(self) -> datetime:
        return datetime.fromtimestamp(self.created, tz=UTC)

    @property
    def new_plan(self) -> Plan | None:
        """The plan this event puts its customer on, or None if it changes nothing."""
        subject = self.data.object
        if self.type == "checkout.session.completed":
            paid = subject.payment_status in ("paid", "no_payment_required")
            return Plan.PRO if subject.mode == "subscription" and paid else None
        if self.type in ("customer.subscription.created", "customer.subscription.updated"):
            return Plan.PRO if subject.status in _PRO_SUBSCRIPTION_STATUSES else Plan.FREE
        if self.type == "customer.subscription.deleted":
            return Plan.FREE
        return None


def parse_webhook(payload: bytes, signature_header: str | None, secret: str) -> BillingEvent:
    """Check that Stripe signed this exact body recently, then read it."""
    try:
        # The signature covers a timestamp too, so a captured request cannot be
        # replayed after the library's five-minute tolerance.
        stripe.WebhookSignature.verify_header(
            payload.decode("utf-8"), signature_header or "", secret, tolerance=300
        )
        return BillingEvent.model_validate(json.loads(payload))
    except (stripe.SignatureVerificationError, UnicodeDecodeError) as error:
        raise InvalidWebhookError("Invalid signature.") from error
    except (ValueError, ValidationError) as error:
        raise InvalidWebhookError("Unreadable event.") from error


async def apply_event(session: AsyncSession, event: BillingEvent) -> bool:
    """Apply an event at most once. Returns False if it had been handled before.

    Recording the event id and changing the plan share one transaction: if
    anything fails, neither is saved and Stripe's retry starts clean.
    """
    recorded = await session.execute(
        insert(StripeEvent)
        .values(id=event.id, type=event.type)
        .on_conflict_do_nothing(index_elements=[StripeEvent.id])
        .returning(StripeEvent.id)
    )
    if recorded.first() is None:
        await session.rollback()
        return False

    plan, customer_id = event.new_plan, event.data.object.customer
    if plan is not None and customer_id is not None:
        user = (
            await session.execute(
                select(User).where(User.stripe_customer_id == customer_id).with_for_update()
            )
        ).scalar_one_or_none()
        is_stale = (
            user is not None
            and user.billing_event_at is not None
            and event.created_at < user.billing_event_at
        )
        if user is not None and not is_stale:
            user.plan = plan
            user.billing_event_at = event.created_at

    await session.commit()
    return True
