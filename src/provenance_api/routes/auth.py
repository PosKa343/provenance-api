from datetime import timedelta

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from provenance_api.dependencies import CurrentUser, LimiterDep, SessionDep, SettingsDep
from provenance_api.errors import ApiError, unauthorized
from provenance_api.models import User
from provenance_api.schemas import (
    AUTH_ERRORS,
    Credentials,
    ErrorEnvelope,
    MeOut,
    QuotaOut,
    TokenResponse,
    UserOut,
)
from provenance_api.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/v1", tags=["auth"])


@router.post(
    "/auth/register",
    response_model=UserOut,
    status_code=201,
    summary="Create an account",
    responses={409: {"model": ErrorEnvelope}, 422: {"model": ErrorEnvelope}},
)
async def register(body: Credentials, session: SessionDep) -> User:
    # Argon2 is slow on purpose. It is C code that releases the interpreter lock
    # while it works, so a worker thread is enough to keep the event loop free.
    password_hash = await run_in_threadpool(hash_password, body.password)
    user = User(email=body.email, password_hash=password_hash)
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        # The unique constraint decides, so two simultaneous sign-ups cannot both win.
        await session.rollback()
        raise ApiError(409, "email_taken", "An account with this email already exists.") from None
    await session.refresh(user)
    return user


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="Exchange email and password for an access token",
    responses={401: {"model": ErrorEnvelope}, 422: {"model": ErrorEnvelope}},
)
async def login(body: Credentials, session: SessionDep, settings: SettingsDep) -> TokenResponse:
    user = (
        await session.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    password_hash = user.password_hash if user else None
    password_ok = await run_in_threadpool(verify_password, body.password, password_hash)
    if user is None or not password_ok:
        # Same answer for an unknown email and a wrong password.
        raise unauthorized("Incorrect email or password.", "Bearer")

    lifetime = timedelta(minutes=settings.access_token_minutes)
    token = create_access_token(
        user.id, secret=settings.jwt_secret.get_secret_value(), lifetime=lifetime
    )
    return TokenResponse(access_token=token, expires_in=int(lifetime.total_seconds()))


@router.get(
    "/me",
    response_model=MeOut,
    summary="The signed-in user, their plan and today's usage",
    responses=AUTH_ERRORS,
)
async def me(user: CurrentUser, settings: SettingsDep, limiter: LimiterDep) -> MeOut:
    return MeOut(
        id=user.id,
        email=user.email,
        role=user.role,
        plan=user.plan,
        created_at=user.created_at,
        usage=QuotaOut(
            daily_limit=settings.limits_for(user.plan).daily_quota,
            used_today=await limiter.used_today(user.id),
        ),
    )
