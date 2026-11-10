from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, tuple_

from provenance_api.dependencies import SessionDep, require_role
from provenance_api.models import Role, User
from provenance_api.pagination import Cursor
from provenance_api.schemas import AUTH_ERRORS, ErrorEnvelope, UserOut, UserPage

router = APIRouter(prefix="/v1/admin", tags=["admin"], responses=AUTH_ERRORS)

Admin = Annotated[User, Depends(require_role(Role.ADMIN))]


@router.get(
    "/users",
    response_model=UserPage,
    summary="List all users (admin only)",
    responses={400: {"model": ErrorEnvelope}},
)
async def list_users(
    _admin: Admin,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: str | None = None,
) -> UserPage:
    query = select(User).order_by(User.created_at, User.id).limit(limit + 1)
    if cursor is not None:
        after = Cursor.decode(cursor)
        query = query.where(tuple_(User.created_at, User.id) > (after.created_at, after.id))

    # One extra row is fetched to learn whether another page exists.
    users = (await session.execute(query)).scalars().all()
    page = users[:limit]
    has_more = len(users) > limit
    return UserPage(
        items=[UserOut.model_validate(user) for user in page],
        next_cursor=Cursor(page[-1].created_at, page[-1].id).encode() if has_more else None,
    )
