from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from PROVENANCE_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="PROVENANCE_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://provenance:provenance@localhost:5432/provenance"
    max_text_chars: int = 20_000
