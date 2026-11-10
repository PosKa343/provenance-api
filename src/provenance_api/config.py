from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """All that migrations and admin commands need. Read from PROVENANCE_* variables."""

    model_config = SettingsConfigDict(env_prefix="PROVENANCE_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://provenance:provenance@localhost:5432/provenance"


class Settings(DatabaseSettings):
    """Full configuration for the API server."""

    max_text_chars: int = 20_000

    # No default: the service must not start with a guessable signing key.
    jwt_secret: SecretStr = Field(min_length=32)
    access_token_minutes: int = 30
