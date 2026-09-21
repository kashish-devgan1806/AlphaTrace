"""Central settings object. Reads from environment variables / a local .env
file so nothing secret (DB password, SEC contact email) is hardcoded."""
from pathlib import Path

# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the project root, not the current directory: a bare ".env" is
# resolved against wherever the process was started, so running a script from
# another directory silently skipped the file and fell back to the defaults
# below (placeholder SEC user-agent included).
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Values that mean "nobody filled this in". SEC's fair-access policy needs a
# real contact in the User-Agent, so scripts warn when they see one of these.
PLACEHOLDER_USER_AGENTS = ("unset@example.com", "your-email@example.com")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # SEC EDGAR
    sec_user_agent: str = "AlphaTrace research-bot unset@example.com"

    # Postgres
    # Same credentials docker-compose.yml falls back to when .env sets none.
    database_url: str = "postgresql://alphatrace:change-me@localhost:5432/alphatrace"

    # FastAPI
    app_env: str = "development"
    # Loopback by default so a dev server isn't exposed to the LAN; a container
    # sets APP_HOST=0.0.0.0 explicitly.
    app_host: str = "127.0.0.1"
    app_port: int = 8000


settings = Settings()
