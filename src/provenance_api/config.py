from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
