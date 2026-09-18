-- Session 3: the chunks table indexing/retrieval will read and write.
--
-- Column choices:
--   id          BIGSERIAL   surrogate key; nothing about a chunk is naturally
--                            unique on its own, and callers never need to
--                            invent one themselves.
--   doc_id      TEXT        identifies the source filing (e.g. an SEC
--                            accession number). TEXT, not a FK, because
--                            there's no documents table yet in Month 1 —
--                            adding a real FK is a later session's job once
--                            one exists, not something to front-load here.
--   section     TEXT        the filing section a chunk came from (e.g.
--                            "Item 1A Risk Factors"), set by the
--                            section-aware chunker (Session 5).
--   text        TEXT        the chunk's raw content. Unbounded length on
--                            purpose — the chunker, not this schema, is
--                            responsible for keeping chunks a sane size.
--   embedding   VECTOR(384) fixed to 384 dims to match Session 4's chosen
--                            model, BAAI/bge-small-en-v1.5 (see
--                            app/embeddings.py:EMBEDDING_DIM). Changing the
--                            model means changing both places and
--                            re-embedding against a fresh volume — pgvector
--                            enforces the dimension at insert time, so a
--                            mismatch fails loudly rather than corrupting
--                            data.
--   metadata    JSONB       everything else worth keeping per chunk (page
--                            number, fiscal period, filing form type, ...)
--                            without a schema migration every time a new
--                            field is needed.
CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL,
    section   TEXT NOT NULL,
    text      TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    metadata  JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id);

-- HNSW over IVFFlat: IVFFlat's `lists` parameter has to be sized against the
-- row count up front and re-tuned (effectively rebuilt) as the corpus grows,
-- which doesn't fit a table that keeps gaining chunks every time a new
-- filing is ingested. HNSW builds and refines its graph incrementally as
-- rows are inserted, needs no row-count guess, and gives better recall at a
-- given query speed — at the cost of a slower, more memory-hungry build,
-- which is a fine trade for a table this size (thousands, not billions, of
-- rows in this project's scope) where query latency is what's demoable.
--
-- vector_cosine_ops because embeddings from Session 4's model are
-- normalized (unit-length) at embed time — cosine distance on normalized
-- vectors is the metric that model is actually tuned to produce meaningful
-- rankings under.
--
-- m (max connections per graph node, default 16) and ef_construction
-- (candidate list size while building, default 64) are pgvector's own
-- defaults, made explicit here rather than left implicit: higher m improves
-- recall but grows the index and slows inserts; higher ef_construction
-- improves graph quality (and therefore recall) at the cost of slower
-- builds, with diminishing returns past a few hundred. Defaults are the
-- right starting point at this corpus size — worth revisiting only if a
-- later eval run shows retrieval recall is actually the bottleneck.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
