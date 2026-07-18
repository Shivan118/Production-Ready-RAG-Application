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


@lru_cache
def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=_embeddings(),
        persist_directory=str(settings.chroma_persist_dir),
        client_settings=ChromaSettings(anonymized_telemetry=False),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _embeddings():
    from app.ingestion.embedder import get_embeddings

    return get_embeddings()


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
        span.set_attribute("chunks_indexed", len(chunks))
        return len(chunks)


def document_count() -> int:
    return get_vectorstore()._collection.count()
