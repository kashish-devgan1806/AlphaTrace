"""Central settings object. Reads from environment variables / a local .env
file so nothing secret (DB password, SEC contact email) is hardcoded."""
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SEC EDGAR
    sec_user_agent: str = "AlphaTrace research-bot unset@example.com"

    # Postgres
    database_url: str = "postgresql://alphatrace:alphatrace@localhost:5432/alphatrace"

    # FastAPI
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000


settings = Settings()
