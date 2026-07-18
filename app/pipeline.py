"""RAG pipeline orchestrator.

Phase 1 flow: retrieve (dense) → generate.
Later phases slot in: guardrails → query transforms → hybrid retrieval →
graph augmentation → rerank → compression → generate → output guardrails.
"""

import time

import logfire

from app.config import get_settings
from app.generation.generator import generate_answer
from app.models.schemas import QueryRequest, QueryResponse
from app.retrieval import dense


def run_query(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    start = time.perf_counter()

    with logfire.span("rag_pipeline", question=request.question) as span:
        chunks = dense.retrieve(request.question, top_k=request.top_k)
        answer = generate_answer(request.question, chunks)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("num_sources", len(chunks))

        return QueryResponse(
            answer=answer,
            sources=chunks,
            model=settings.openai_chat_model,
            retrieval_strategy="dense",
            latency_ms=latency_ms,
        )
