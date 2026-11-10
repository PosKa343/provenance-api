from dataclasses import dataclass

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class PlanLimits:
    daily_quota: int
    burst: int
    refill_per_second: float
    batch_jobs: bool


class BackendSettings(BaseSettings):
    """What the worker, migrations and admin commands need. Read from PROVENANCE_* variables."""

    model_config = SettingsConfigDict(env_prefix="PROVENANCE_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://provenance:provenance@localhost:5432/provenance"
    redis_url: str = "redis://localhost:6379"

    # A batch job is attempted this many times before it is marked failed.
    job_max_tries: int = 3
    # Wait before retry n is this many seconds times 2^(n-1).
    job_retry_backoff_seconds: float = 5.0


class Settings(BackendSettings):
    """Full configuration for the API server."""

    max_text_chars: int = 20_000
    max_batch_texts: int = 100

    # No default: the service must not start with a guessable signing key.
    jwt_secret: SecretStr = Field(min_length=32)
    access_token_minutes: int = 30

    # Billing is optional: without these three the billing endpoints answer 503.
    stripe_secret_key: SecretStr | None = None
    stripe_webhook_secret: SecretStr | None = None
    stripe_price_id: str | None = None
    billing_success_url: str = "http://localhost:8000/docs"
    billing_cancel_url: str = "http://localhost:8000/docs"

    @field_validator("stripe_secret_key", "stripe_webhook_secret", "stripe_price_id", mode="before")
    @classmethod
    def empty_means_not_configured(cls, value: object) -> object:
        # compose passes unset variables through as empty strings.
        return None if value == "" else value

    @field_validator("stripe_secret_key")
    @classmethod
    def only_test_mode_keys(cls, value: SecretStr | None) -> SecretStr | None:
        # This project must never take real money, so a live key is a startup error.
        if value is not None and not value.get_secret_value().startswith("sk_test_"):
            raise ValueError("stripe_secret_key must be a test-mode key (sk_test_...)")
        return value

    # Daily quota counts single classifications plus every text in a batch job.
    # Burst and refill describe the per-key token bucket.
    free_daily_quota: int = 100
    free_burst: int = 10
    free_refill_per_second: float = 1.0
    pro_daily_quota: int = 10_000
    pro_burst: int = 50
    pro_refill_per_second: float = 10.0

    def limits_for(self, plan: str) -> PlanLimits:
        if plan == "pro":
            return PlanLimits(
                self.pro_daily_quota, self.pro_burst, self.pro_refill_per_second, batch_jobs=True
            )
        return PlanLimits(
            self.free_daily_quota, self.free_burst, self.free_refill_per_second, batch_jobs=False
        )
