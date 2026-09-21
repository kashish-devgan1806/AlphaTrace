"""
Apply every db/init/*.sql migration, in filename order, to the configured
database.

docker-compose only runs db/init/ automatically against an *empty* data
volume, so on an existing volume new migrations have to be applied by hand.
Every file here is idempotent (CREATE ... IF NOT EXISTS / ADD COLUMN IF NOT
EXISTS), so running this script again is always safe.

Usage:
    python scripts/migrate.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.db import CONNECT_TIMEOUT_SECONDS  # noqa: E402

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "init"


def migration_files(directory: Path = MIGRATIONS_DIR) -> list[Path]:
    return sorted(directory.glob("*.sql"))


def apply_migrations(conn: psycopg.Connection, files: list[Path]) -> list[str]:
    """Run each file as one multi-statement script; return the names applied.
    Stops at the first failure (later files may depend on earlier ones)."""
    applied: list[str] = []
    for path in files:
        conn.execute(path.read_text(encoding="utf-8"))
        applied.append(path.name)
    return applied


def main(argv: Optional[list[str]] = None) -> int:
    files = migration_files()
    if not files:
        print(f"ERROR: no .sql files found in {MIGRATIONS_DIR}", file=sys.stderr)
        return 1
    # Plain connect, not app.db.get_connection(): that registers the vector
    # adapter, which fails on a database where 01_enable_pgvector.sql hasn't
    # created the extension yet — exactly the state this script fixes.
    try:
        with psycopg.connect(settings.database_url, connect_timeout=CONNECT_TIMEOUT_SECONDS, autocommit=True) as conn:
            for name in apply_migrations(conn, files):
                print(f"applied {name}")
    except psycopg.Error as exc:
        print(f"ERROR: migration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
