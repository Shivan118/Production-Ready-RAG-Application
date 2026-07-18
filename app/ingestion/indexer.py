"""Vector store indexing — ChromaDB persistent collection.

`get_vectorstore()` is the single access point to Chroma used by both
ingestion and retrieval, so they always share the same collection.
"""

from functools import lru_cache
from pathlib import Path

import logfire
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from app.config import get_settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import load_file


@lru_cache(maxsize=8)
def _vectorstore_for(api_key: str) -> Chroma:
    from app.ingestion.embedder import _embeddings_for

    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=_embeddings_for(api_key),
        persist_directory=str(settings.chroma_persist_dir),
        client_settings=ChromaSettings(anonymized_telemetry=False),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_vectorstore() -> Chroma:
    from app.runtime_keys import effective_openai_key

    return _vectorstore_for(effective_openai_key())


def index_file(path: Path) -> int:
    """Load → chunk → embed → upsert one file. Returns number of chunks."""
    with logfire.span("index_file", filename=path.name) as span:
        documents = load_file(path)
        chunks = chunk_documents(documents)
        if not chunks:
            span.set_attribute("chunks_indexed", 0)
            return 0

        vectorstore = get_vectorstore()
        # chunk_id as the Chroma ID makes re-ingesting the same file idempotent
        ids = [c.metadata["chunk_id"] for c in chunks]
        vectorstore.add_documents(chunks, ids=ids)

        # keep the in-memory BM25 index in sync with the collection
        from app.retrieval.hybrid import invalidate_bm25

        invalidate_bm25()

        _extract_graph(chunks)
        span.set_attribute("chunks_indexed", len(chunks))
        return len(chunks)


def _extract_graph(chunks) -> None:
    """Populate Neo4j from chunks (skipped when disabled or Neo4j is down)."""
    from app.graph.client import is_enabled
    from app.graph.extractor import extract_graph
    from app.graph.store import store_extraction

    settings = get_settings()
    if not settings.graph_extraction_on_ingest or not is_enabled():
        return

    with logfire.span("graph.extract_chunks", num_chunks=len(chunks)):
        for chunk in chunks:
            extraction = extract_graph(chunk.page_content)
            store_extraction(
                extraction,
                chunk_id=chunk.metadata["chunk_id"],
                source=chunk.metadata.get("source", "unknown"),
            )


def document_count() -> int:
    return get_vectorstore()._collection.count()


def list_sources() -> dict[str, int]:
    """Indexed filenames with their chunk counts."""
    data = get_vectorstore()._collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        source = (meta or {}).get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def delete_source(source: str) -> int:
    """Delete all chunks of one file from Chroma. Returns chunks removed."""
    with logfire.span("delete_source", source=source) as span:
        collection = get_vectorstore()._collection
        data = collection.get(where={"source": source})
        ids = data["ids"]
        if ids:
            collection.delete(ids=ids)
            from app.retrieval.hybrid import invalidate_bm25

            invalidate_bm25()
        span.set_attribute("chunks_deleted", len(ids))
        return len(ids)
