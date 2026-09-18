-- Session 3: the chunks table itself. Session 1's 01_enable_pgvector.sql
-- only turned the extension on; this is the first thing that actually uses
-- it.
--
-- Column choices, justified:
--   id        BIGSERIAL PRIMARY KEY — an opaque surrogate key. Nothing about
--             a chunk (doc_id + section + position) is a natural key worth
--             building joins/foreign keys against later, so a plain
--             auto-incrementing int keeps things simple.
--   doc_id    TEXT NOT NULL — identifies the source filing (e.g. an EDGAR
--             accession number). TEXT, not a foreign key: there's no
--             `documents` table yet (out of scope for Month 1's ingestion
--             work so far), so this is deliberately a loose reference for
--             now, not a front-loaded schema for a table that doesn't exist.
--   section   TEXT NOT NULL — which part of the filing this chunk came from
--             (e.g. "Item 1A Risk Factors", "MD&A"). Every chunk should be
--             attributable to a section once Session 5's section-aware
--             chunker exists, so this is NOT NULL now rather than loosened
--             later.
--   text      TEXT NOT NULL — the chunk's raw text. No length cap: Postgres
--             TEXT has no practical limit, and enforcing a max length here
--             would just duplicate whatever limit the chunker (Session 5)
--             or the embedding model (Session 4) already has to enforce for
--             their own reasons.
--   embedding vector(384) NOT NULL — 384 = BAAI/bge-small-en-v1.5's output
--             dimension (see app/embeddings.py). Fixed-width by pgvector's
--             own design: every row in one `vector` column must share a
--             dimension, so this number is pinned to whichever model
--             Session 4 picked, not left generic. NOT NULL because a chunk
--             row only exists to be searched — an un-embedded chunk isn't
--             useful to store here yet.
--   metadata  JSONB NOT NULL DEFAULT '{}'::jsonb — everything else that
--             doesn't earn its own column yet (ticker, form type, fiscal
--             year, chunk index within its section, ...). JSONB over plain
--             TEXT/JSON because it's binary-parsed and indexable (e.g. a
--             future GIN index) without needing to know today what every
--             future filter will be.
CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    section   TEXT NOT NULL,
    text      TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    metadata  JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Every real query pattern so far ("this filing's chunks", "this filing's
-- MD&A chunks") starts from doc_id, so it gets a plain btree index. No
-- separate index on `section` alone yet — YAGNI until a query actually
-- needs to scan across documents by section, which isn't a thing Month 1
-- does.
CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);

-- HNSW over IVFFlat, and why:
--   IVFFlat needs to be trained on a representative sample of the data
--   *before* it's useful — its lists are built from a k-means pass over
--   whatever's in the table at CREATE INDEX time, so an index built on an
--   empty (or nearly empty) chunks table gives poor recall until it's
--   rebuilt later on real data. HNSW has no training step: it builds
--   incrementally as rows are inserted, which matches how this table is
--   actually going to be filled (one filing, then another, indefinitely) —
--   not one big bulk load with a stable final size known up front.
--
--   m and ef_construction — what they actually trade off:
--     m: max number of graph edges kept per node, per layer. Bigger m means
--        each node has more neighbors to search through, so recall goes up
--        and search is more likely to find the true nearest neighbors — at
--        the cost of a larger index (more edges to store) and slower
--        inserts (more edges to maintain per new node). Smaller m means a
--        sparser graph: faster/smaller, but easier for the search to get
--        stuck in a local neighborhood and miss a true nearest neighbor.
--     ef_construction: the size of the candidate list explored *while
--        building* the graph — how many candidate neighbors get considered
--        per node before picking the best m of them. Bigger ef_construction
--        means a more thoroughly-searched (higher-quality) graph, at the
--        cost of slower index builds. It only affects build time, not
--        query time (that's ef_search, a per-query GUC, not an index
--        parameter).
--   pgvector's defaults (m=16, ef_construction=64) are used here rather
--   than tuned values — right for Month 1's data volume (a handful of
--   filings' worth of chunks), and premature to tune against a corpus size
--   the project hasn't reached yet.
--
--   Cosine over L2: BAAI/bge-small-en-v1.5's model card recommends cosine
--   similarity for retrieval (it's trained/evaluated that way), and
--   `embed_text()` normalizes every output vector to unit length — for
--   normalized vectors cosine distance and L2 distance rank results
--   identically, but `vector_cosine_ops` is the operator class that
--   matches how this model is documented to be used, so that's what the
--   index is built with rather than relying on the L2/cosine equivalence
--   as an implementation detail.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
