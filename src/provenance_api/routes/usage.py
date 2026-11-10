from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from provenance_api.dependencies import CurrentUser, SessionDep
from provenance_api.models import UsageEvent, UsageKind
from provenance_api.schemas import AUTH_ERRORS, UsageDay, UsageOut

router = APIRouter(prefix="/v1", tags=["usage"], responses=AUTH_ERRORS)


@router.get("/usage", response_model=UsageOut, summary="Your metered usage by day")
async def get_usage(
    user: CurrentUser,
    session: SessionDep,
    days: Annotated[int, Query(ge=1, le=90, description="How many days back to include.")] = 30,
) -> UsageOut:
    since = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
        days=days - 1
    )
    day = func.date(func.timezone("UTC", UsageEvent.occurred_at))
    rows = await session.execute(
        select(day, UsageEvent.kind, func.sum(UsageEvent.units))
        .where(UsageEvent.user_id == user.id, UsageEvent.occurred_at >= since)
        .group_by(day, UsageEvent.kind)
    )

    by_day: dict[date, dict[UsageKind, int]] = {}
    for event_day, kind, units in rows:
        by_day.setdefault(event_day, {})[kind] = int(units)

    return UsageOut(
        days=[
            UsageDay(
                date=event_day,
                classify=kinds.get(UsageKind.CLASSIFY, 0),
                batch=kinds.get(UsageKind.BATCH, 0),
                total=sum(kinds.values()),
            )
            for event_day, kinds in sorted(by_day.items(), reverse=True)
        ]
    )
