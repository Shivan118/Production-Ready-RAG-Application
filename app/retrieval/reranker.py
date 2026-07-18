"""Cohere Rerank — cross-encoder scoring of candidate chunks against the query.

Applied after retrieval to pick the truly relevant few from a wide candidate
set. Degrades gracefully: without COHERE_API_KEY the candidates pass through
unreranked (with a warning logged) so the API never breaks."""

from functools import lru_cache

import cohere
import logfire

from app.config import get_settings
from app.models.schemas import SourceChunk


@lru_cache
def _client() -> cohere.ClientV2 | None:
    settings = get_settings()
    if not settings.cohere_api_key:
        return None
    return cohere.ClientV2(api_key=settings.cohere_api_key)


def is_enabled() -> bool:
    return _client() is not None


def rerank(
    question: str, chunks: list[SourceChunk], top_n: int
) -> list[SourceChunk]:
    if not chunks:
        return []

    client = _client()
    if client is None:
        logfire.warn("rerank_skipped_no_cohere_key", candidates=len(chunks))
        return chunks[:top_n]

    settings = get_settings()
    with logfire.span(
        "rerank.cohere",
        question=question,
        candidates=len(chunks),
        top_n=top_n,
        model=settings.cohere_rerank_model,
    ) as span:
        response = client.rerank(
            model=settings.cohere_rerank_model,
            query=question,
            documents=[c.content for c in chunks],
            top_n=min(top_n, len(chunks)),
        )
        reranked = [
            chunks[r.index].model_copy(
                update={"score": round(r.relevance_score, 4)}
            )
            for r in response.results
        ]
        span.set_attribute("top_score", reranked[0].score if reranked else None)
        return reranked
