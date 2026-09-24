# AlphaTrace

A multi-agent, self-verifying equity-research copilot.

A LangGraph-orchestrated crew of 7 agents ingests SEC filings (10-K/10-Q/8-K), earnings-call transcripts, and slide decks, then answers analyst questions with inline citations. Before any answer ships, a Critic agent checks every claim against its source chunk and cross-checks every number against the SEC's own machine-readable XBRL data — flagging what it can't verify instead of guessing.

## Status

🚧 Early build — following a 56-day, agent-paired build schedule (Phase 0 Setup → Phase 1: 7 agents, 2 days each → Phase 2 Integration/Eval/Observability → Phase 3 Productionize & Ship). See commit history for day-by-day progress.

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

- [x] Agent 1 of 7 (Ingestion, `a1d1`) — grew `ingest_node` into the real spec: pulls the latest 10-K *and* 10-Q *and* 8-K, the XBRL facts, and (when the 8-K has one) its Exhibit 99.x slide deck into one `document_bundle`; a missing form/exhibit degrades gracefully instead of failing. Lives in the new `app/agents/` package; live-verified against AAPL, MSFT, NVDA — **Phase 1 started**

- [x] Agent 2 of 7 (Indexing, `a2d1`) — grew `index_node` to chunk every filing form present in the bundle (not just the primary one), extract financial-statement tables with row/column structure preserved (`app/tables.py`, serialized to Markdown, a distinct `chunk_type` from prose), and rasterize a PDF slide-deck exhibit into page images with a CLIP visual embedding (`app/visual.py`, `sentence-transformers/clip-ViT-B-32`, its own `page_embeddings` table — an HTML exhibit has no natural page boundary, so it's chunked as text instead). Lives in `app/agents/indexing.py`; live-verified against AAPL, MSFT, NVDA

- [x] Agent 3 of 7 (Research Analyst, `a3d1`) — the project's first LLM call: `analyst_node` retrieves a k=20 candidate pool from pgvector (`app/search.py`, now with an optional `chunk_type` filter), reranks it with a cross-encoder (`app/rerank.py`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 5 kept), and calls an LLM (`app/llm.py`, Groq/`openai/gpt-oss-120b`) with a grounded-answer prompt that returns a cited draft answer — every citation resolved against the retrieved evidence, a hallucinated or unbacked citation dropped and flagged rather than trusted. Lives in `app/agents/analyst.py`; live-verified against AAPL, MSFT, NVDA (reranking improved P@5 0.63→0.70 and fixed the known NVDA-foundry hard case from rank 7 to rank 1; 3 real answers' citations hand-checked against source text)

## Architecture (evolving)

- **Ingestion:** EDGAR filings + XBRL facts, earnings-call transcripts, slide decks
- **Storage:** Postgres + pgvector
- **Orchestration:** LangGraph (cyclic verify-and-revise loop)
- **Agents:** Research Analyst, Sentiment/Tone, Quant/Forecast, Critic (citation + XBRL cross-check), and others
- **Eval:** RAGAS faithfulness, citation accuracy, XBRL-grounded hallucination checks
- **LLMOps:** Langfuse tracing, CI-gated eval regression gate, cost/latency-aware model tiering

## Getting Started

```bash
cp .env.example .env        # then fill in SEC_USER_AGENT, Postgres creds, and GROQ_API_KEY (console.groq.com)
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
python -c "
from app.toolkit import build_graph
b = build_graph().invoke({'ticker': 'AAPL'})['document_bundle']
print({f: b['filings'][f] is not None for f in b['filings']}, 'xbrl:', b['xbrl_facts'] is not None, 'slide_deck:', b['slide_deck'] is not None)
"  # Agent 1: 10-K/10-Q/8-K + XBRL facts + 8-K slide-deck exhibit, all in one bundle
python -c "
from app.toolkit import build_graph
r = build_graph().invoke({'ticker': 'AAPL'})
print('chunks:', r['chunk_count'], 'visual pages:', r['visual_chunk_count'])
"  # Agent 2: text + table chunks across every form, plus slide-deck page images when the exhibit is a PDF
python -c "
from app.toolkit import build_graph
r = build_graph().invoke({'ticker': 'AAPL', 'question': 'How does Apple describe competition for its products?'})
print(r['draft_answer']); [print(' ', c) for c in r['citations']]
"  # Agent 3: retrieve -> rerank -> grounded LLM answer with citations (needs GROQ_API_KEY, and AAPL already inserted via build_corpus.py --insert)
pytest -q                                   # offline tests, no network required
```

## License

TBD
