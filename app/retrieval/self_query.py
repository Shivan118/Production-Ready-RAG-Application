"""Self-Query Retrieval — the LLM turns natural-language constraints into
metadata filters ("in attention.pdf after page 5" → filter source/page),
then runs the cleaned semantic query with that filter."""

from functools import lru_cache

import logfire
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from app.config import get_settings
from app.generation.generator import _llm
from app.ingestion.indexer import get_vectorstore
from app.models.schemas import SourceChunk

_METADATA_FIELDS = [
    AttributeInfo(
        name="source",
        description="The filename of the document, e.g. 'attention.pdf' or 'notes.md'",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="Zero-based page number within the document (PDFs only)",
        type="integer",
    ),
]

_CONTENT_DESCRIPTION = "Text chunks from the user's uploaded documents"


@lru_cache
def _retriever() -> SelfQueryRetriever:
    settings = get_settings()
    return SelfQueryRetriever.from_llm(
        llm=_llm(),
        vectorstore=get_vectorstore(),
        document_contents=_CONTENT_DESCRIPTION,
        metadata_field_info=_METADATA_FIELDS,
        search_kwargs={"k": settings.top_k},
        verbose=False,
    )


def retrieve(question: str, top_k: int | None = None) -> list[SourceChunk]:
    with logfire.span("retrieve.self_query", question=question) as span:
        retriever = _retriever()
        if top_k:
            retriever.search_kwargs["k"] = top_k
        docs = retriever.invoke(question)
        # SelfQueryRetriever returns docs without scores; encode rank instead
        chunks = [
            SourceChunk(
                content=d.page_content,
                source=d.metadata.get("source", "unknown"),
                chunk_id=d.metadata.get("chunk_id", ""),
                score=round(1.0 / (i + 1), 4),
            )
            for i, d in enumerate(docs)
        ]
        span.set_attribute("chunks_returned", len(chunks))
        return chunks
