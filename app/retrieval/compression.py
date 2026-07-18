"""Contextual Compression — an LLM extractor strips irrelevant sentences from
each retrieved chunk, so the generator sees only what matters for the query.
Chunks with no relevant content are dropped entirely."""

from functools import lru_cache

import logfire
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

from app.generation.generator import _llm
from app.models.schemas import SourceChunk


@lru_cache(maxsize=8)
def _compressor_for(api_key: str) -> LLMChainExtractor:
    from app.generation.generator import _llm_for

    return LLMChainExtractor.from_llm(_llm_for(api_key))


def _compressor() -> LLMChainExtractor:
    from app.runtime_keys import effective_openai_key

    return _compressor_for(effective_openai_key())


def compress(question: str, chunks: list[SourceChunk]) -> list[SourceChunk]:
    if not chunks:
        return []

    with logfire.span(
        "compress.llm_extract", question=question, input_chunks=len(chunks)
    ) as span:
        docs = [
            Document(
                page_content=c.content,
                metadata={"chunk_id": c.chunk_id, "source": c.source, "score": c.score},
            )
            for c in chunks
        ]
        compressed = _compressor().compress_documents(docs, query=question)
        result = [
            SourceChunk(
                content=d.page_content,
                source=d.metadata.get("source", "unknown"),
                chunk_id=d.metadata.get("chunk_id", ""),
                score=d.metadata.get("score", 0.0),
            )
            for d in compressed
        ]
        span.set_attribute("output_chunks", len(result))
        span.set_attribute(
            "chars_before", sum(len(c.content) for c in chunks)
        )
        span.set_attribute(
            "chars_after", sum(len(c.content) for c in result)
        )
        return result
