"""Hybrid search: dense (Chroma) + sparse keyword (BM25), fused with RRF.

The BM25 index is built in memory from the full Chroma collection and
rebuilt lazily after each ingest (see `invalidate()`), so both retrievers
always search the same corpus.
"""

import re
import threading

import logfire
from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.ingestion.indexer import get_vectorstore
from app.models.schemas import SourceChunk
from app.retrieval import dense
from app.retrieval.fusion import rrf_fuse

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _BM25Index:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[SourceChunk] = []
        self._stale = True

    def invalidate(self) -> None:
        with self._lock:
            self._stale = True

    def _build(self) -> None:
        data = get_vectorstore()._collection.get(include=["documents", "metadatas"])
        chunks = [
            SourceChunk(
                content=doc,
                source=(meta or {}).get("source", "unknown"),
                chunk_id=(meta or {}).get("chunk_id", ""),
                score=0.0,
            )
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
        corpus = [_tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._chunks = chunks
        self._stale = False
        logfire.info("bm25_index_built", num_chunks=len(chunks))

    def search(self, question: str, k: int) -> list[SourceChunk]:
        with self._lock:
            if self._stale:
                self._build()
            if self._bm25 is None:
                return []
            scores = self._bm25.get_scores(_tokenize(question))
            ranked = sorted(
                zip(self._chunks, scores), key=lambda x: x[1], reverse=True
            )[:k]
            return [
                c.model_copy(update={"score": round(float(s), 4)})
                for c, s in ranked
                if s > 0
            ]


_index = _BM25Index()


def invalidate_bm25() -> None:
    _index.invalidate()


def sparse_search(question: str, k: int) -> list[SourceChunk]:
    with logfire.span("retrieve.sparse_bm25", question=question, top_k=k) as span:
        results = _index.search(question, k)
        span.set_attribute("chunks_returned", len(results))
        return results


def retrieve(
    question: str,
    top_k: int | None = None,
    fetch_k: int | None = None,
    dense_query: str | None = None,
) -> list[SourceChunk]:
    """Dense + BM25, RRF-fused.

    `dense_query` lets HyDE substitute a hypothetical document for the
    embedding side while BM25 still matches the user's literal keywords.
    """
    settings = get_settings()
    k = top_k or settings.top_k
    fk = fetch_k or settings.fetch_k

    with logfire.span("retrieve.hybrid", question=question, top_k=k) as span:
        dense_results = dense.retrieve(dense_query or question, top_k=fk)
        sparse_results = sparse_search(question, k=fk)
        fused = rrf_fuse([dense_results, sparse_results], top_n=k)
        span.set_attribute("dense_candidates", len(dense_results))
        span.set_attribute("sparse_candidates", len(sparse_results))
        span.set_attribute("chunks_returned", len(fused))
        return fused
