from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated

from arq.connections import ArqRedis
from fastapi import Depends, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from provenance_api.billing import BillingGateway
from provenance_api.config import Settings
from provenance_api.errors import forbidden, unauthorized
from provenance_api.limits import Limiter
from provenance_api.ml.model import Classifier
from provenance_api.models import ApiKey, Role, User
from provenance_api.security import InvalidTokenError, decode_access_token, hash_api_key

API_KEY_HEADER = "X-API-Key"

_bearer_scheme = HTTPBearer(auto_error=False, description="Access token from POST /v1/auth/login.")
_api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER, auto_error=False, description="Key from POST /v1/api-keys."
)


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_classifier(request: Request) -> Classifier:
    classifier: Classifier = request.app.state.classifier
    return classifier


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session


def get_queue(request: Request) -> ArqRedis:
    queue: ArqRedis = request.app.state.queue
    return queue


def get_limiter(request: Request) -> Limiter:
    limiter: Limiter = request.app.state.limiter
    return limiter


def get_billing(request: Request) -> BillingGateway | None:
    billing: BillingGateway | None = request.app.state.billing
    return billing


SettingsDep = Annotated[Settings, Depends(get_settings)]
ClassifierDep = Annotated[Classifier, Depends(get_classifier)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
QueueDep = Annotated[ArqRedis, Depends(get_queue)]
LimiterDep = Annotated[Limiter, Depends(get_limiter)]
BillingDep = Annotated[BillingGateway | None, Depends(get_billing)]


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    session: SessionDep,
    settings: SettingsDep,
) -> User:
    """The user named by a valid access token. Used by dashboard endpoints."""
    if credentials is None:
        raise unauthorized("Missing bearer token.", "Bearer")
    try:
        user_id = decode_access_token(
            credentials.credentials, secret=settings.jwt_secret.get_secret_value()
        )
    except InvalidTokenError:
        raise unauthorized("Invalid or expired token.", "Bearer") from None

    # The role is read from the database on every request, not from the token,
    # so a role change or a deleted account takes effect immediately.
    user = await session.get(User, user_id)
    if user is None:
        raise unauthorized("Invalid or expired token.", "Bearer")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(*allowed: Role) -> Callable[[User], Awaitable[User]]:
    """Build a dependency that only lets the given roles through."""

    async def check(user: CurrentUser) -> User:
        if user.role not in allowed:
            raise forbidden("Your role does not allow this action.")
        return user

    return check


async def get_api_key_user(
    key: Annotated[str | None, Depends(_api_key_scheme)],
    session: SessionDep,
    limiter: LimiterDep,
    settings: SettingsDep,
) -> User:
    """The owner of a valid, unrevoked API key, after a rate-limit check for that key."""
    if key is None:
        raise unauthorized(f"Missing {API_KEY_HEADER} header.", "ApiKey")

    row = (
        await session.execute(
            select(ApiKey, User)
            .join(User, ApiKey.user_id == User.id)
            .where(ApiKey.key_hash == hash_api_key(key), ApiKey.revoked_at.is_(None))
        )
    ).one_or_none()
    if row is None:
        raise unauthorized("Invalid or revoked API key.", "ApiKey")

    api_key, user = row
    # Checked after authentication, so an invalid key cannot drain a real key's bucket.
    await limiter.check_rate(api_key.id, settings.limits_for(user.plan))
    api_key.last_used_at = datetime.now(UTC)
    await session.commit()
    return user


ApiKeyUser = Annotated[User, Depends(get_api_key_user)]
