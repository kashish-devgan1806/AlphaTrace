"""app/toolkit.py is a pure re-export surface — this test only guards that
every name in __all__ actually resolves and is identical to the object in
its home module, so a rename/removal in the underlying module surfaces
here instead of silently breaking whatever agent imported it from
app.toolkit."""
from __future__ import annotations

from app import chunker, chunks, db, embeddings, graph, state, toolkit
from scripts import chunk_filing, edgar_pull

_HOME_MODULE = {
    "load_ticker_map": edgar_pull,
    "fetch_submissions": edgar_pull,
    "fetch_primary_document": edgar_pull,
    "find_latest_filing": chunk_filing,
    "fetch_filing_index": edgar_pull,
    "find_exhibit_99": edgar_pull,
    "fetch_document_bytes": edgar_pull,
    "fetch_companyfacts": edgar_pull,
    "extract_gaap_facts": edgar_pull,
    "get_connection": db,
    "embed_text": embeddings,
    "embed_texts": embeddings,
    "embed_query": embeddings,
    "ChunkRecord": chunks,
    "batch_insert_chunks": chunks,
    "chunk_filing": chunker,
    "split_into_sections": chunker,
    "strip_html_to_text": chunker,
    "AgentState": state,
    "build_graph": graph,
    "ingest_node": graph,
    "index_node": graph,
}


def test_all_matches_home_module_mapping():
    assert set(toolkit.__all__) == set(_HOME_MODULE)


def test_every_export_is_identical_to_its_home_module_object():
    for name, home_module in _HOME_MODULE.items():
        assert getattr(toolkit, name) is getattr(home_module, name), (
            f"app.toolkit.{name} has drifted from {home_module.__name__}.{name}"
        )
