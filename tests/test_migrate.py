"""Offline tests for scripts/migrate.py — a fake connection records the SQL."""
from __future__ import annotations

import pytest

from scripts.migrate import apply_migrations, migration_files


class FakeConn:
    def __init__(self, fail_on: str | None = None) -> None:
        self.executed: list[str] = []
        self._fail_on = fail_on

    def execute(self, sql: str) -> None:
        if self._fail_on and self._fail_on in sql:
            raise RuntimeError("boom")
        self.executed.append(sql)


def test_migration_files_are_returned_in_filename_order(tmp_path):
    for name in ("03_c.sql", "01_a.sql", "02_b.sql", "notes.txt"):
        (tmp_path / name).write_text("select 1;", encoding="utf-8")

    assert [p.name for p in migration_files(tmp_path)] == ["01_a.sql", "02_b.sql", "03_c.sql"]


def test_apply_migrations_runs_each_file_in_order_and_reports_names(tmp_path):
    (tmp_path / "01_a.sql").write_text("-- a\nselect 1;", encoding="utf-8")
    (tmp_path / "02_b.sql").write_text("-- b\nselect 2;", encoding="utf-8")
    conn = FakeConn()

    applied = apply_migrations(conn, migration_files(tmp_path))

    assert applied == ["01_a.sql", "02_b.sql"]
    assert conn.executed == ["-- a\nselect 1;", "-- b\nselect 2;"]


def test_apply_migrations_stops_at_the_first_failure(tmp_path):
    (tmp_path / "01_a.sql").write_text("FAIL", encoding="utf-8")
    (tmp_path / "02_b.sql").write_text("select 2;", encoding="utf-8")
    conn = FakeConn(fail_on="FAIL")

    with pytest.raises(RuntimeError):
        apply_migrations(conn, migration_files(tmp_path))
    assert conn.executed == []  # 02 never ran


def test_real_migration_directory_has_the_expected_files():
    names = [p.name for p in migration_files()]

    assert names[:3] == [
        "01_enable_pgvector.sql",
        "02_create_chunks_table.sql",
        "03_add_chunks_content_hash.sql",
    ]
