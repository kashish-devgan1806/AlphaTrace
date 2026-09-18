"""Offline tests for scripts/build_corpus.py's multi-ticker loop and result
aggregation. process_ticker() itself is already covered by
tests/test_chunk_filing.py, so this file monkeypatches it directly and
focuses purely on orchestration: does the loop keep going after one
ticker's failure, does the summary/exit code reflect every ticker, and is
the DB connection shared across the whole run rather than reopened per
ticker."""
from __future__ import annotations

import httpx

import scripts.build_corpus as build_corpus
from scripts.chunk_filing import ProcessResult


def _fake_load_ticker_map(client):
    return {"AAPL": 1, "MSFT": 2, "NVDA": 3}


def test_build_corpus_mixed_outcomes_continues_past_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(build_corpus, "load_ticker_map", _fake_load_ticker_map)

    canned = {
        "AAPL": ProcessResult("AAPL", "ok", chunk_count=5, section_count=2),
        "MSFT": ProcessResult("MSFT", "error", stage="fetch_submissions", error="boom"),
        "NVDA": ProcessResult("NVDA", "ok", chunk_count=8, section_count=3),
    }
    calls = []

    def fake_process_ticker(client, ticker_map, ticker, form, insert, conn=None):
        calls.append(ticker)
        return canned[ticker]

    monkeypatch.setattr(build_corpus, "process_ticker", fake_process_ticker)

    exit_code = build_corpus.main(["AAPL", "MSFT", "NVDA"])

    assert calls == ["AAPL", "MSFT", "NVDA"]  # all 3 attempted, loop didn't stop at MSFT's failure
    assert exit_code == 1

    out = capsys.readouterr().out
    assert "AAPL" in out and "MSFT" in out and "NVDA" in out


def test_build_corpus_all_succeed_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(build_corpus, "load_ticker_map", _fake_load_ticker_map)

    def fake_process_ticker(client, ticker_map, ticker, form, insert, conn=None):
        return ProcessResult(ticker, "ok", chunk_count=3, section_count=1)

    monkeypatch.setattr(build_corpus, "process_ticker", fake_process_ticker)

    exit_code = build_corpus.main(["AAPL", "MSFT"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "ok" in out


def test_build_corpus_ticker_map_failure_short_circuits(monkeypatch):
    def failing_load_ticker_map(client):
        raise httpx.HTTPError("network down")

    monkeypatch.setattr(build_corpus, "load_ticker_map", failing_load_ticker_map)

    calls = []

    def fake_process_ticker(client, ticker_map, ticker, form, insert, conn=None):
        calls.append(ticker)
        return ProcessResult(ticker, "ok")

    monkeypatch.setattr(build_corpus, "process_ticker", fake_process_ticker)

    exit_code = build_corpus.main(["AAPL", "MSFT"])

    assert exit_code == 1
    assert calls == []  # never reached the per-ticker loop


def test_build_corpus_insert_shares_one_connection_across_tickers(monkeypatch):
    monkeypatch.setattr(build_corpus, "load_ticker_map", _fake_load_ticker_map)

    class FakeConn:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    fake_conn = FakeConn()
    monkeypatch.setattr(build_corpus, "get_connection", lambda: fake_conn)

    seen_conns = []

    def fake_process_ticker(client, ticker_map, ticker, form, insert, conn=None):
        seen_conns.append(conn)
        return ProcessResult(ticker, "ok", chunk_count=1, section_count=1, inserted=1)

    monkeypatch.setattr(build_corpus, "process_ticker", fake_process_ticker)

    exit_code = build_corpus.main(["AAPL", "MSFT", "NVDA", "--insert"])

    assert exit_code == 0
    assert seen_conns == [fake_conn, fake_conn, fake_conn]  # same connection every time
    assert fake_conn.closed == 1  # closed once after the loop, not per ticker
