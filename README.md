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
pip install -r requirements-dev.txt

docker compose up -d        # Postgres + pgvector
uvicorn app.main:app --reload --port 8000   # http://localhost:8000/health

python scripts/edgar_pull.py AAPL           # pulls live filing metadata from EDGAR
python scripts/edgar_pull.py AAPL --facts   # + extracts Revenues/GrossProfit/NetIncomeLoss from XBRL companyfacts
python scripts/chunk_filing.py AAPL         # fetches latest 10-K, chunks it, prints a per-section summary
python scripts/chunk_filing.py AAPL --insert  # + embeds and writes the chunks into Postgres
pytest -q                                   # offline tests, no network required
```

## License

TBD
