"""Offline tests for app/config.py defaults and app/db.py's connect call."""
from __future__ import annotations

import logging
from pathlib import Path

import app.db as db_module
import scripts.edgar_pull as edgar_pull
from app.config import ENV_FILE, Settings, settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_env_file_is_anchored_to_the_project_root_not_the_cwd():
    assert ENV_FILE == PROJECT_ROOT / ".env"
    assert ENV_FILE.is_absolute()
    assert Settings.model_config["env_file"] == ENV_FILE


def test_defaults_match_compose_and_are_not_exposed_to_the_lan():
    defaults = Settings(_env_file=None)

    # Same fallback credentials docker-compose.yml uses when .env sets none.
    assert ":change-me@" in defaults.database_url
    assert defaults.app_host == "127.0.0.1"


def test_get_connection_passes_a_connect_timeout(monkeypatch):
    seen = {}

    def fake_connect(url, **kwargs):
        seen["url"] = url
        seen["kwargs"] = kwargs
        return "conn"

    registered = []
    monkeypatch.setattr(db_module.psycopg, "connect", fake_connect)
    monkeypatch.setattr(db_module, "register_vector", lambda conn: registered.append(conn))

    assert db_module.get_connection() == "conn"
    assert seen["kwargs"] == {"connect_timeout": db_module.CONNECT_TIMEOUT_SECONDS}
    assert registered == ["conn"]


def test_placeholder_user_agent_warns_once(monkeypatch, caplog):
    monkeypatch.setattr(settings, "sec_user_agent", "AlphaTrace research-bot unset@example.com")
    monkeypatch.setattr(edgar_pull, "_warned_placeholder_user_agent", False)

    with caplog.at_level(logging.WARNING, logger="scripts.edgar_pull"):
        edgar_pull._headers()
        edgar_pull._headers()

    warnings = [r for r in caplog.records if "placeholder" in r.getMessage()]
    assert len(warnings) == 1


def test_real_user_agent_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(settings, "sec_user_agent", "AlphaTrace someone@real-domain.org")
    monkeypatch.setattr(edgar_pull, "_warned_placeholder_user_agent", False)

    with caplog.at_level(logging.WARNING, logger="scripts.edgar_pull"):
        edgar_pull._headers()

    assert not [r for r in caplog.records if "placeholder" in r.getMessage()]
