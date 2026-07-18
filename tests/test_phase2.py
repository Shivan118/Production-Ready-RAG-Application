"""Phase 2 unit tests — RRF fusion, BM25 tokenization, schema strategy,
reranker fallback. All offline (no OpenAI/Cohere calls)."""

import os

import pytest
from pydantic import ValidationError

os.environ.setdefault("OPENAI_API_KEY", "sk-test-0000000000")

from app.models.schemas import (  # noqa: E402
    QueryRequest,
    RetrievalStrategy,
    SourceChunk,
)
from app.retrieval.fusion import rrf_fuse  # noqa: E402
from app.retrieval.hybrid import _tokenize  # noqa: E402


def _chunk(cid: str, score: float = 0.0) -> SourceChunk:
    return SourceChunk(content=f"content {cid}", source="s.md", chunk_id=cid, score=score)


class TestRRF:
    def test_item_in_both_lists_ranks_first(self):
        a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
        fused = rrf_fuse([[a, b], [c, a]])
        assert fused[0].chunk_id == "a"
        assert len(fused) == 3

    def test_top_n_limits(self):
        chunks = [_chunk(str(i)) for i in range(10)]
        fused = rrf_fuse([chunks], top_n=3)
        assert len(fused) == 3

    def test_preserves_order_within_single_list(self):
        a, b = _chunk("a"), _chunk("b")
        fused = rrf_fuse([[a, b]])
        assert [c.chunk_id for c in fused] == ["a", "b"]

    def test_empty_input(self):
        assert rrf_fuse([]) == []
        assert rrf_fuse([[], []]) == []

    def test_scores_are_rrf_not_original(self):
        a = _chunk("a", score=0.99)
        fused = rrf_fuse([[a]])
        assert fused[0].score == pytest.approx(1 / 61, abs=1e-4)


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert _tokenize("Hybrid-Search with BM25!") == [
            "hybrid", "search", "with", "bm25",
        ]

    def test_empty(self):
        assert _tokenize("") == []


class TestStrategySchema:
    def test_default_strategy_is_advanced(self):
        req = QueryRequest(question="what is RAG?")
        assert req.strategy == RetrievalStrategy.ADVANCED
        assert req.use_rerank is True
        assert req.use_compression is False

    def test_accepts_all_strategies(self):
        for s in RetrievalStrategy:
            req = QueryRequest(question="what is RAG?", strategy=s.value)
            assert req.strategy == s

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="what is RAG?", strategy="quantum")


class TestRerankerFallback:
    def test_no_key_passthrough(self, monkeypatch):
        from app.retrieval import reranker

        monkeypatch.setattr(reranker, "_client", lambda: None)
        chunks = [_chunk(str(i)) for i in range(10)]
        out = reranker.rerank("q", chunks, top_n=4)
        assert out == chunks[:4]

    def test_empty_chunks(self):
        from app.retrieval import reranker

        assert reranker.rerank("q", [], top_n=5) == []
