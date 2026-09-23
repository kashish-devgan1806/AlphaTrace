# AlphaTrace

A multi-agent, self-verifying equity-research copilot.

A LangGraph-orchestrated crew of 7 agents ingests SEC filings (10-K/10-Q/8-K), earnings-call transcripts, and slide decks, then answers analyst questions with inline citations. Before any answer ships, a Critic agent checks every claim against its source chunk and cross-checks every number against the SEC's own machine-readable XBRL data — flagging what it can't verify instead of guessing.

## Status

🚧 Early build — following a 12-week / 72-session build schedule. See commit history for day-by-day progress.

- [x] Session 1 — repo skeleton, Postgres+pgvector via Docker Compose, first EDGAR submissions pull
- [x] Session 2 — XBRL `companyfacts` pull, GAAP tag extraction (Revenues, GrossProfit, NetIncomeLoss)
- [x] Session 3 — `chunks` table migration (pgvector HNSW index, cosine distance)
- [x] Session 4 — `embed_text()` (BAAI/bge-small-en-v1.5) + batch insert into `chunks`
- [x] Session 5 — section-aware chunker: split 10-K/10-Q text into Part/Item sections, pack into `ChunkRecord`s sized for the embedding model
- [x] Session 6 — wired pull → chunk → embed → insert into one multi-ticker pipeline (`scripts/build_corpus.py`), live-verified against AAPL, MSFT, NVDA
- [x] Session 6 follow-up — `chunks` gained a `content_hash` unique index (`db/init/03_add_chunks_content_hash.sql`); re-running the pipeline against an already-ingested filing now skips duplicates instead of re-inserting them

- [x] Session 7 — `search(conn, query, k, ticker=None)` (`app/search.py`): cosine top-k over pgvector with an optional ticker filter; `EXPLAIN ANALYZE` review (planner seq-scans at ~700 rows, HNSW path verified equal on top-10); first precision@k / recall@k sketch
- [x] Session 7 cleanup — chunker fixes (MSFT page-header mislabelling, hidden inline-XBRL text, 512-token overflow, page-header/whitespace noise), `--replace` re-ingest, retry/backoff on SEC calls, distinct exit codes, config/compose hardening (`127.0.0.1` binding), `scripts/migrate.py`

- [x] Session 8 — LangGraph `AgentState` schema (`app/state.py`, with a field-by-field write/read ownership doc) and a trivial `ingest -> index` `StateGraph` (`app/graph.py`) wrapping the existing EDGAR pull and section-aware chunker into two LangGraph nodes, proving the wiring end to end with no LLM call yet
- [x] Session 9 — packaged Days 1-5 (EDGAR client, XBRL client, DB layer, chunker, graph scaffold) behind one importable surface, `app/toolkit.py`; re-ran the pull-chunk-embed-insert pipeline plus the new graph scaffold live against AAPL, MSFT, NVDA — **Phase 0 (Setup & Shared Infrastructure) complete**

## Architecture (evolving)

- **Ingestion:** EDGAR filings + XBRL facts, earnings-call transcripts, slide decks
- **Storage:** Postgres + pgvector
- **Orchestration:** LangGraph (cyclic verify-and-revise loop)
- **Agents:** Research Analyst, Sentiment/Tone, Quant/Forecast, Critic (citation + XBRL cross-check), and others
- **Eval:** RAGAS faithfulness, citation accuracy, XBRL-grounded hallucination checks
- **LLMOps:** Langfuse tracing, CI-gated eval regression gate, cost/latency-aware model tiering

## Getting Started

```bash
cp .env.example .env        # then fill in SEC_USER_AGENT and Postgres creds
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d        # Postgres + pgvector; db/init/*.sql only auto-applies on a fresh volume
python scripts/migrate.py   # applies every db/init/*.sql in order (idempotent) — run after pulling new migrations
uvicorn app.main:app --reload --port 8000   # http://localhost:8000/health

python scripts/edgar_pull.py AAPL           # pulls live filing metadata from EDGAR
python scripts/edgar_pull.py AAPL --facts   # + extracts Revenues/GrossProfit/NetIncomeLoss from XBRL companyfacts
python scripts/chunk_filing.py AAPL         # fetches latest 10-K, chunks it, prints a per-section summary
python scripts/chunk_filing.py AAPL --insert  # + embeds and writes the chunks into Postgres
python scripts/build_corpus.py AAPL MSFT NVDA            # same pipeline, 3 tickers in one run
python scripts/build_corpus.py AAPL MSFT NVDA --insert   # + embeds and writes all 3 into Postgres
python scripts/build_corpus.py AAPL MSFT NVDA --insert --replace   # re-ingest: delete each filing's old rows first (use after the chunker changes)
python scripts/build_corpus.py AAPL --refresh-ticker-cache         # force a fresh SEC ticker→CIK download
python -c "from app.db import get_connection; from app.search import search; print(search(get_connection(), 'NVIDIA export controls', k=3, ticker='NVDA'))"
python -c "from app.toolkit import build_graph; print(build_graph().invoke({'ticker': 'AAPL'})['chunk_count'])"  # ingest -> index graph scaffold
pytest -q                                   # offline tests, no network required
```

## License

TBD
