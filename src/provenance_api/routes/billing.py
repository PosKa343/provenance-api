import logging

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from provenance_api.billing import BillingError, InvalidWebhookError, apply_event, parse_webhook
from provenance_api.dependencies import BillingDep, CurrentUser, SessionDep, SettingsDep
from provenance_api.errors import ApiError
from provenance_api.models import Plan
from provenance_api.schemas import AUTH_ERRORS, CheckoutOut, ErrorEnvelope, WebhookOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/billing", tags=["billing"])


def _not_configured() -> ApiError:
    return ApiError(503, "billing_not_configured", "Billing is not configured on this server.")


@router.post(
    "/checkout",
    response_model=CheckoutOut,
    summary="Start a Stripe Checkout for the Pro plan (test mode)",
    responses={
        **AUTH_ERRORS,
        409: {"model": ErrorEnvelope},
        502: {"model": ErrorEnvelope},
        503: {"model": ErrorEnvelope},
    },
)
async def create_checkout(
    user: CurrentUser, session: SessionDep, billing: BillingDep
) -> CheckoutOut:
    if billing is None:
        raise _not_configured()
    if user.plan is Plan.PRO:
        raise ApiError(409, "already_subscribed", "You are already on the Pro plan.")

    try:
        if user.stripe_customer_id is None:
            # Saved before checkout starts, so every later webhook can be matched to this user.
            # The Stripe library blocks on network calls, so they run in a worker thread.
            user.stripe_customer_id = await run_in_threadpool(
                billing.create_customer, user.email, user.id
            )
            await session.commit()
        url = await run_in_threadpool(billing.create_checkout_url, user.stripe_customer_id, user.id)
    except BillingError:
        logger.exception("Stripe request failed.")
        raise ApiError(502, "billing_unavailable", "The payment provider is unavailable.") from None
    return CheckoutOut(checkout_url=url)


@router.post(
    "/webhook",
    response_model=WebhookOut,
    summary="Receive Stripe events (called by Stripe, not by clients)",
    responses={400: {"model": ErrorEnvelope}, 503: {"model": ErrorEnvelope}},
)
async def stripe_webhook(
    request: Request, session: SessionDep, settings: SettingsDep
) -> WebhookOut:
    if settings.stripe_webhook_secret is None:
        raise _not_configured()

    # The signature is over the exact bytes received, so the raw body is read
    # here and not parsed by the framework first.
    try:
        event = parse_webhook(
            await request.body(),
            request.headers.get("Stripe-Signature"),
            settings.stripe_webhook_secret.get_secret_value(),
        )
    except InvalidWebhookError as error:
        raise ApiError(400, "invalid_webhook", str(error)) from None

    # Stripe retries anything that is not a 2xx, so a repeat is answered with success.
    applied = await apply_event(session, event)
    return WebhookOut(status="processed" if applied else "already_processed")
