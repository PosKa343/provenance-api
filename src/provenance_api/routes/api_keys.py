import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select

from provenance_api.dependencies import CurrentUser, SessionDep, require_role
from provenance_api.errors import not_found
from provenance_api.models import ApiKey, Role, User
from provenance_api.schemas import (
    AUTH_ERRORS,
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyOut,
    ErrorEnvelope,
)
from provenance_api.security import generate_api_key

router = APIRouter(prefix="/v1/api-keys", tags=["api keys"], responses=AUTH_ERRORS)

# Viewers are read-only: they can see their keys but not create or revoke them.
KeyManager = Annotated[User, Depends(require_role(Role.ADMIN, Role.MEMBER))]


@router.post("", response_model=ApiKeyCreated, status_code=201, summary="Create an API key")
async def create_api_key(
    body: ApiKeyCreate, user: KeyManager, session: SessionDep
) -> ApiKeyCreated:
    new_key = generate_api_key()
    api_key = ApiKey(
        user_id=user.id, name=body.name, prefix=new_key.prefix, key_hash=new_key.key_hash
    )
    session.add(api_key)
    await session.commit()
    await session.refresh(api_key)
    # The only time the full key leaves the server. It cannot be recovered later.
    return ApiKeyCreated(
        id=api_key.id,
        name=api_key.name,
        prefix=api_key.prefix,
        created_at=api_key.created_at,
        last_used_at=None,
        revoked_at=None,
        key=new_key.plaintext,
    )


@router.get("", response_model=list[ApiKeyOut], summary="List your API keys")
async def list_api_keys(user: CurrentUser, session: SessionDep) -> Sequence[ApiKey]:
    result = await session.execute(
        select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc())
    )
    return result.scalars().all()


@router.delete(
    "/{key_id}",
    status_code=204,
    summary="Revoke an API key",
    responses={404: {"model": ErrorEnvelope}},
)
async def revoke_api_key(key_id: uuid.UUID, user: KeyManager, session: SessionDep) -> Response:
    # Filtering by owner means another user's key looks exactly like a missing
    # one, so the response does not reveal which key ids exist.
    api_key = (
        await session.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id))
    ).scalar_one_or_none()
    if api_key is None:
        raise not_found("API key not found.")

    if api_key.revoked_at is None:
        api_key.revoked_at = datetime.now(UTC)
        await session.commit()
    return Response(status_code=204)
