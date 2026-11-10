"""Rate limiting and daily quotas, enforced in Redis so every API replica shares them."""

import logging
import math
import uuid
from datetime import UTC, datetime, timedelta

from redis.asyncio import Redis
from redis.exceptions import RedisError

from provenance_api.config import PlanLimits
from provenance_api.errors import ApiError

logger = logging.getLogger(__name__)

# Each script runs atomically inside Redis, so two requests arriving together
# cannot both spend the last token or the last unit of quota.

# Token bucket: holds up to `capacity` tokens and gains `refill` tokens per
# second. A request needs one token. Uses the Redis server's clock, so API
# replicas with slightly different clocks still agree.
_TAKE_FROM_BUCKET = """
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local time = redis.call('TIME')
local now = tonumber(time[1]) + tonumber(time[2]) / 1000000

local state = redis.call('HMGET', KEYS[1], 'tokens', 'updated')
local tokens = tonumber(state[1]) or capacity
local updated = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + (now - updated) * refill)

local retry_after = 0
if tokens >= 1 then
  tokens = tokens - 1
else
  retry_after = (1 - tokens) / refill
end

redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'updated', tostring(now))
-- An idle bucket refills in capacity / refill seconds; after that it can be forgotten.
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / refill) + 1)
return tostring(retry_after)
"""

# Daily quota: add `units` to today's counter unless that would pass the limit.
_CONSUME_QUOTA = """
local units = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local used = tonumber(redis.call('GET', KEYS[1]) or '0')
if used + units > limit then
  return 0
end
redis.call('INCRBY', KEYS[1], units)
redis.call('EXPIRE', KEYS[1], 172800)
return 1
"""


def _quota_key(user_id: uuid.UUID, now: datetime) -> str:
    return f"provenance:quota:{user_id}:{now:%Y%m%d}"


def _seconds_until_utc_midnight(now: datetime) -> int:
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, math.ceil((tomorrow - now).total_seconds()))


def _too_many_requests(code: str, message: str, retry_after_seconds: int) -> ApiError:
    return ApiError(429, code, message, headers={"Retry-After": str(retry_after_seconds)})


class Limiter:
    """If Redis is unreachable every check passes: an outage must not take the API down."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._take_from_bucket = redis.register_script(_TAKE_FROM_BUCKET)
        self._consume_quota = redis.register_script(_CONSUME_QUOTA)

    async def check_rate(self, api_key_id: uuid.UUID, limits: PlanLimits) -> None:
        """Spend one token from this API key's bucket, or raise a 429."""
        try:
            retry_after = float(
                await self._take_from_bucket(
                    keys=[f"provenance:bucket:{api_key_id}"],
                    args=[limits.burst, limits.refill_per_second],
                )
            )
        except (RedisError, OSError):
            logger.exception("Rate limiter unavailable; allowing the request.")
            return
        if retry_after > 0:
            raise _too_many_requests(
                "rate_limited", "Too many requests. Slow down.", math.ceil(retry_after)
            )

    async def consume_quota(self, user_id: uuid.UUID, units: int, limits: PlanLimits) -> None:
        """Count `units` against today's quota, or raise a 429 if they do not fit."""
        now = datetime.now(UTC)
        try:
            allowed = await self._consume_quota(
                keys=[_quota_key(user_id, now)], args=[units, limits.daily_quota]
            )
        except (RedisError, OSError):
            logger.exception("Quota store unavailable; allowing the request.")
            return
        if not allowed:
            raise _too_many_requests(
                "quota_exceeded",
                f"Daily quota of {limits.daily_quota} reached. It resets at 00:00 UTC.",
                _seconds_until_utc_midnight(now),
            )

    async def refund_quota(self, user_id: uuid.UUID, units: int) -> None:
        try:
            await self._redis.decrby(_quota_key(user_id, datetime.now(UTC)), units)
        except (RedisError, OSError):
            logger.exception("Quota store unavailable; refund skipped.")

    async def used_today(self, user_id: uuid.UUID) -> int:
        try:
            return int(await self._redis.get(_quota_key(user_id, datetime.now(UTC))) or 0)
        except (RedisError, OSError):
            logger.exception("Quota store unavailable; reporting zero usage.")
            return 0
