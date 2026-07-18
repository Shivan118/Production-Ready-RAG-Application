"""Dense (semantic) retrieval over ChromaDB.

Phase 1 baseline strategy. Later phases add hybrid, multi-query,
self-query, compression, reranking and graph retrieval behind the
same `retrieve()` signature.
"""

import logfire
from langchain_core.documents import Document

from app.config import get_settings
from app.ingestion.indexer import get_vectorstore
from app.models.schemas import SourceChunk


def retrieve(question: str, top_k: int | None = None) -> list[SourceChunk]:
    """Semantic similarity search; returns chunks with relevance scores."""
    settings = get_settings()
    k = top_k or settings.top_k

    with logfire.span("retrieve.dense", question=question, top_k=k) as span:
        vectorstore = get_vectorstore()
        # raw cosine distance in [0, 2]; clamp relevance into [0, 1]
        results: list[tuple[Document, float]] = vectorstore.similarity_search_with_score(
            question, k=k
        )
        chunks = [
            SourceChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                chunk_id=doc.metadata.get("chunk_id", ""),
                score=round(max(0.0, 1.0 - distance), 4),
            )
            for doc, distance in results
        ]
        span.set_attribute("chunks_returned", len(chunks))
        span.set_attribute("top_score", chunks[0].score if chunks else None)
        return chunks
