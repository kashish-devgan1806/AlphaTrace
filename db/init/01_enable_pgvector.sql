-- Runs once against an empty data volume. The pgvector/pgvector image ships
-- the extension's binaries; this just turns it on for our database so
-- Session 3's `vector` column type is available.
CREATE EXTENSION IF NOT EXISTS vector;
