"""Reciprocal Rank Fusion — merge ranked result lists from multiple retrievers.

RRF score for a chunk = sum over lists of 1 / (k + rank + 1).
Robust to incomparable score scales (BM25 vs cosine), which is exactly
the hybrid-search problem.
"""

from app.models.schemas import SourceChunk

RRF_K = 60


def rrf_fuse(
    result_lists: list[list[SourceChunk]],
    top_n: int | None = None,
    k: int = RRF_K,
) -> list[SourceChunk]:
    scores: dict[str, float] = {}
    first_seen: dict[str, SourceChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            key = chunk.chunk_id or chunk.content[:100]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            first_seen.setdefault(key, chunk)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]

    return [
        first_seen[key].model_copy(update={"score": round(score, 5)})
        for key, score in ranked
    ]
