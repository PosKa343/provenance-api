"""Change a user's role. There is no API for this: the first admin is made here.

Usage: python -m provenance_api.set_role <email> <admin|member|viewer>
"""

import asyncio
import sys

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from provenance_api.config import DatabaseSettings
from provenance_api.db import create_engine, create_session_factory
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


async def _main(email: str, role: Role) -> int:
    engine = create_engine(DatabaseSettings().database_url)
    try:
        found = await set_role(create_session_factory(engine), email, role)
    finally:
        await engine.dispose()
    print(f"{email} is now {role.value}." if found else f"No user with email {email}.")
    return 0 if found else 1


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in [role.value for role in Role]:
        sys.exit(__doc__)
    sys.exit(asyncio.run(_main(sys.argv[1], Role(sys.argv[2]))))
