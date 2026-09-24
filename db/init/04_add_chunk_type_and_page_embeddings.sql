-- Two additions for the Indexing Agent: a way to tell a text chunk from a
-- table chunk within the existing `chunks` table, and a separate table for
-- slide-deck page visual embeddings (a different fixed width than text
-- embeddings, so it can't share the `chunks.embedding vector(384)` column).

-- chunk_type distinguishes a table chunk (Markdown-serialized financial
-- statement, app/tables.py) from an ordinary prose chunk (app/chunker.py)
-- within the same table/index/search path -- both still go through
-- batch_insert_chunks() and chunks_embedding_hnsw_idx unchanged. Defaulted
-- to 'text' so every row inserted before this migration existed reads as
-- what it always was.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS chunk_type TEXT NOT NULL DEFAULT 'text';

ALTER TABLE chunks
    DROP CONSTRAINT IF EXISTS chunks_chunk_type_check;
ALTER TABLE chunks
    ADD CONSTRAINT chunks_chunk_type_check CHECK (chunk_type IN ('text', 'table'));

CREATE INDEX IF NOT EXISTS chunks_chunk_type_idx ON chunks (chunk_type);

-- page_embeddings holds one row per rasterized slide-deck page (a PDF
-- Exhibit 99.x, app/visual.py). embedding vector(512) is
-- sentence-transformers/clip-ViT-B-32's output width (app/embeddings.py) --
-- a different model and dimension than the text embedder, hence its own
-- table rather than a nullable second embedding column on `chunks`.
-- doc_id here is the slide deck's own identifier (its exhibit name/URL,
-- not a filing accession number -- a deck isn't "a filing" the way a
-- 10-K/10-Q/8-K primary document is).
CREATE TABLE IF NOT EXISTS page_embeddings (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    page_number INT NOT NULL,
    embedding   vector(512) NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS page_embeddings_doc_id_idx ON page_embeddings (doc_id);

-- One row per (deck, page) -- re-rasterizing and re-inserting the same deck
-- is a safe no-op per page, mirroring chunks_content_hash_idx's role for
-- the chunks table.
CREATE UNIQUE INDEX IF NOT EXISTS page_embeddings_doc_page_idx
    ON page_embeddings (doc_id, page_number);

CREATE INDEX IF NOT EXISTS page_embeddings_embedding_hnsw_idx
    ON page_embeddings USING hnsw (embedding vector_cosine_ops);
