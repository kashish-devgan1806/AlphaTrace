"""Embedding utilities, built around BAAI/bge-small-en-v1.5.

Why this model (Session 4's study step):
- It's a retrieval-tuned sentence-transformer (trained specifically for
  semantic search, not general sentence similarity), small enough to run on
  CPU with no GPU dependency for local dev, and its model card explicitly
  recommends cosine similarity — matching the HNSW index Session 3 built
  with `vector_cosine_ops`.
- EMBEDDING_DIM below (384) is *not* independent of the model choice — it's
  bge-small-en-v1.5's fixed output width. pgvector's `vector(384)` column
  in db/init/02_create_chunks_table.sql has to match this number exactly;
  a mismatched dimension isn't a slow query, it's an insert-time error.
  If the model ever changes, this constant and that column both have to
  change together.
- The model card also recommends prefixing *queries* (not indexed passages)
  with an instruction string to get its best retrieval performance —
  QUERY_INSTRUCTION below exists for that asymmetry. embed_text()/
  embed_texts() are for the chunks going into the index; embed_query() is
  for a question at search time (Session 7's search(), not used yet).
"""
from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _warn_on_truncation(model: SentenceTransformer, texts: list[str]) -> None:
    """Flag inputs the model will silently truncate rather than let it
    happen unnoticed.

    sentence-transformers tokenizes with truncation=True by default: a text
    longer than the model's max_seq_length (512 tokens for
    bge-small-en-v1.5) doesn't error or wrap — everything past token 512 is
    dropped before encoding, with no exception and no default log line. For
    a chunk-sized piece of text (Session 5's section-aware chunker keeps
    chunks well under 512 tokens) that's fine — the input was never going
    to exceed it. It stops being fine the moment something upstream calls
    this on a whole raw section or document instead of an actual chunk, so
    this warns instead of trusting every caller to already know the limit.
    """
    max_len = model.max_seq_length
    tokenizer = model.tokenizer
    for text in texts:
        token_count = len(tokenizer.encode(text, add_special_tokens=True))
        if token_count > max_len:
            logger.warning(
                "embed_text: input has %d tokens, exceeds model max_seq_length=%d "
                "- it will be silently truncated to the first %d tokens.",
                token_count,
                max_len,
                max_len,
            )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts (passages), no query instruction."""
    if not texts:
        return []
    model = _get_model()
    _warn_on_truncation(model, texts)
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.tolist()


def embed_text(text: str) -> list[float]:
    """Embed a single chunk text. Thin wrapper over embed_texts() so
    call sites with one chunk don't have to wrap/unwrap a list."""
    return embed_texts([text])[0]


def embed_query(text: str) -> list[float]:
    """Embed a search query — prefixed per the model card's instruction so
    query and passage embeddings live in the same retrieval-optimized
    space. Not used yet (search() is a later session); added now because
    it's a fixed property of this model, not a speculative feature."""
    return embed_texts([QUERY_INSTRUCTION + text])[0]
