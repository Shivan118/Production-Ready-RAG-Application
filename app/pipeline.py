"""RAG pipeline orchestrator.

Flow: retrieve (strategy) → [Cohere rerank] → [contextual compression] → generate.
Strategies that fan out (multi_query, hyde, advanced) return a wide candidate
pool so the reranker has real choices; without reranking they are cut to top_k.
"""

import time

import logfire

from app.config import get_settings
from app.generation.generator import generate_answer
from app.guardrails import rails
from app.models.schemas import (
    GuardrailReport,
    QueryRequest,
    QueryResponse,
    RetrievalStrategy,
    SourceChunk,
)
from app.graph import retriever as graph_retriever
from app.retrieval import (
    compression,
    dense,
    hybrid,
    multi_query,
    query_expansion,
    reranker,
    self_query,
)
from app.retrieval.fusion import rrf_fuse


def _retrieve_advanced(question: str, fetch_k: int) -> list[SourceChunk]:
    """Multi-query variants + HyDE passage + graph results, RRF-fused."""
    settings = get_settings()
    queries = [question] + multi_query.generate_variants(question)
    result_lists = [
        hybrid.retrieve(q, top_k=fetch_k, fetch_k=fetch_k) for q in queries
    ]
    passage = query_expansion.hypothetical_document(question)
    result_lists.append(
        hybrid.retrieve(question, top_k=fetch_k, dense_query=passage)
    )
    graph_results = graph_retriever.retrieve(question, top_k=fetch_k)
    if graph_results:
        result_lists.append(graph_results)
    return rrf_fuse(result_lists, top_n=settings.fetch_k)


def _retrieve(request: QueryRequest, k: int) -> list[SourceChunk]:
    """Dispatch to the requested strategy.

    Returns a candidate pool of size fetch_k when reranking will follow,
    otherwise exactly k.
    """
    settings = get_settings()
    pool = settings.fetch_k if request.use_rerank else k

    match request.strategy:
        case RetrievalStrategy.DENSE:
            return dense.retrieve(request.question, top_k=pool)
        case RetrievalStrategy.HYBRID:
            return hybrid.retrieve(request.question, top_k=pool)
        case RetrievalStrategy.MULTI_QUERY:
            return multi_query.retrieve(request.question, top_k=pool)
        case RetrievalStrategy.HYDE:
            return query_expansion.retrieve(request.question, top_k=pool)
        case RetrievalStrategy.SELF_QUERY:
            return self_query.retrieve(request.question, top_k=pool)
        case RetrievalStrategy.GRAPH:
            return graph_retriever.retrieve(request.question, top_k=pool)
        case RetrievalStrategy.ADVANCED:
            return _retrieve_advanced(request.question, settings.fetch_k)


def _blocked_response(
    report: GuardrailReport, strategy_label: str, start: float
) -> QueryResponse:
    return QueryResponse(
        answer=report.blocked_reason or "Your request was blocked by guardrails.",
        sources=[],
        model=get_settings().openai_chat_model,
        retrieval_strategy=strategy_label,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        blocked=True,
        guardrails=report,
    )


def run_query(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    start = time.perf_counter()
    k = request.top_k or settings.top_k

    strategy_label = request.strategy.value
    guardrails_on = settings.guardrails_enabled and request.use_guardrails

    with logfire.span(
        "rag_pipeline", question=request.question, strategy=strategy_label
    ) as span:
        # --- input guardrails (may block before any retrieval/generation) ---
        report = GuardrailReport()
        if guardrails_on:
            report = rails.check_input(request.question)
            if not report.passed:
                span.set_attribute("guardrail_blocked", True)
                return _blocked_response(report, strategy_label, start)

        chunks = _retrieve(request, k)

        if request.use_rerank:
            chunks = reranker.rerank(request.question, chunks, top_n=k)
            if reranker.is_enabled():
                strategy_label += "+rerank"
        chunks = chunks[:k]

        if request.use_compression:
            chunks = compression.compress(request.question, chunks)
            strategy_label += "+compression"

        answer = generate_answer(request.question, chunks)

        # --- output guardrails (redact PII, optional grounding check) ---
        grounded: bool | None = None
        if guardrails_on:
            answer, grounded = rails.apply_output(answer, chunks, report)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("num_sources", len(chunks))
        span.set_attribute("final_strategy", strategy_label)
        span.set_attribute("grounded", grounded)

        return QueryResponse(
            answer=answer,
            sources=chunks,
            model=settings.openai_chat_model,
            retrieval_strategy=strategy_label,
            latency_ms=latency_ms,
            blocked=False,
            guardrails=report if guardrails_on else None,
            grounded=grounded,
        )
