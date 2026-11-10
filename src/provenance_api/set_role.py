"""Change a user's role. There is no API for this; see `provenance-api set-role`."""

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from provenance_api.models import Role, User


async def set_role(
    session_factory: async_sessionmaker[AsyncSession], email: str, role: Role
) -> bool:
    """Return False if no user has that email."""
    async with session_factory() as session:
        result = await session.execute(
            update(User).where(User.email == email.lower()).values(role=role).returning(User.id)
        )
        await session.commit()
        return result.first() is not None
