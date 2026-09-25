"""Embedding utilities, built around BAAI/bge-small-en-v1.5.

Why this model:
- It's a retrieval-tuned sentence-transformer (trained specifically for
  semantic search, not general sentence similarity), small enough to run on
  CPU with no GPU dependency for local dev, and its model card explicitly
  recommends cosine similarity — matching the HNSW index built with
  `vector_cosine_ops`.
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
  for a question at search time (app/search.py's search()).
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from PIL.Image import Image

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# CLIP over a genuine ColPali (multi-vector, late-interaction): CLIP is a
# single dense vector per image, so it fits the existing pgvector/cosine
# shape (page_embeddings.embedding vector(512), same vector_cosine_ops
# search app/search.py already does for text) with just a new table --
# no multi-vector storage or MaxSim scoring to build. ColPali's own
# advantage (patch-level late interaction) is real but out of scope for
# Month 1; this is a working analog, not the same architecture.
VISUAL_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
VISUAL_EMBEDDING_DIM = 512

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_visual_model: SentenceTransformer | None = None
_model_lock = threading.Lock()
_visual_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_visual_model() -> SentenceTransformer:
    global _visual_model
    if _visual_model is None:
        with _visual_model_lock:
            if _visual_model is None:
                _visual_model = SentenceTransformer(VISUAL_MODEL_NAME)
    return _visual_model


def _warn_on_truncation(model: SentenceTransformer, texts: list[str]) -> None:
    """Flag inputs the model will silently truncate rather than let it
    happen unnoticed.

    sentence-transformers tokenizes with truncation=True by default: a text
    longer than the model's max_seq_length (512 tokens for
    bge-small-en-v1.5) doesn't error or wrap — everything past token 512 is
    dropped before encoding, with no exception and no default log line. For
    a chunk-sized piece of text (the section-aware chunker keeps chunks
    well under 512 tokens) that's fine — the input was never going
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


def count_tokens(text: str) -> int:
    """Token count under the embedding model's own tokenizer, special tokens
    included — the number that max_seq_length (512) is compared against."""
    tokenizer = _get_model().tokenizer
    return len(tokenizer.encode(text, add_special_tokens=True, verbose=False))


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
    space. Called by app/search.py's search(); the prefix is a fixed property
    of this model, not a tunable."""
    return embed_texts([QUERY_INSTRUCTION + text])[0]


def embed_images(images: list["Image"]) -> list[list[float]]:
    """Embed a batch of rasterized slide-deck page images with CLIP --
    the visual-index counterpart to embed_texts(). Not normalized to a
    fixed instruction prefix the way text queries are: CLIP's image and
    text towers share one embedding space natively, so a plain text query
    embedded with this same model (not embed_query(), which is bge-only)
    is what a future visual search() would compare against."""
    if not images:
        return []
    model = _get_visual_model()
    vectors = model.encode(images, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.tolist()


def embed_image(image: "Image") -> list[float]:
    """Embed a single page image. Thin wrapper over embed_images(), same
    reasoning as embed_text() over embed_texts()."""
    return embed_images([image])[0]
