-- Session 6 follow-up: chunks had no natural uniqueness key (id is an
-- opaque surrogate), so re-running the ingestion pipeline against a filing
-- it had already processed silently created duplicate rows — exactly what
-- happened running scripts/build_corpus.py twice against AAPL.
--
-- A generated, hashed column instead of a plain unique index on
-- (doc_id, section, text) directly: those are TEXT columns with no length
-- cap (by design, db/init/02_create_chunks_table.sql), and a chunk's text
-- can run to ~1600 chars — a composite btree index over the raw columns
-- risks exceeding Postgres' ~2704-byte-per-index-entry limit for a long
-- chunk. Hashing first keeps the index entry a fixed 32 bytes regardless of
-- how long any individual chunk's text is.
--
-- chr(31) (ASCII Unit Separator) as the field delimiter rather than a
-- printable character like '|': Postgres TEXT can legally contain almost
-- anything a filing might, including '|', so a printable delimiter could
-- theoretically let two different (doc_id, section, text) triples hash
-- identically if the delimiter itself appears inside an earlier field
-- (e.g. a doc_id containing "|"). chr(31) is a control character no SEC
-- filing text or accession number will ever contain, so the concatenation
-- is unambiguous in practice.
--
-- GENERATED ... STORED (not computed at insert time in application code):
-- means every row gets the same hash regardless of which code path
-- inserted it (app.chunks.batch_insert_chunks today, anything else later),
-- and it's usable directly in a WHERE/ON CONFLICT clause like any other
-- column.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_hash TEXT
        GENERATED ALWAYS AS (md5(doc_id || chr(31) || section || chr(31) || text)) STORED;

-- The uniqueness guard itself. app.chunks.batch_insert_chunks's INSERT now
-- targets this column with ON CONFLICT ... DO NOTHING, so reprocessing an
-- already-ingested filing is a safe no-op per chunk instead of a duplicate
-- row.
CREATE UNIQUE INDEX IF NOT EXISTS chunks_content_hash_idx ON chunks (content_hash);
