"""Query Expansion via HyDE (Hypothetical Document Embeddings).

The LLM writes a hypothetical answer passage; that passage is embedded and
used for dense search (documents match documents better than questions
match documents), while BM25 keeps matching the user's literal keywords.
Results are fused with a plain hybrid pass on the original question."""

import logfire
from langchain_core.prompts import ChatPromptTemplate

from app.config import get_settings
from app.generation.generator import _llm
from app.models.schemas import SourceChunk
from app.retrieval import hybrid
from app.retrieval.fusion import rrf_fuse

_HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a short factual passage (3-5 sentences) that would answer "
            "the user's question, as if it came from a technical document. "
            "Write only the passage — no preamble.",
        ),
        ("user", "{question}"),
    ]
)


def hypothetical_document(question: str) -> str:
    with logfire.span("hyde.generate_document", question=question) as span:
        passage = (_HYDE_PROMPT | _llm()).invoke({"question": question}).content
        span.set_attribute("passage", passage)
        return passage


def retrieve(question: str, top_k: int | None = None) -> list[SourceChunk]:
    settings = get_settings()
    k = top_k or settings.top_k

    with logfire.span("retrieve.hyde", question=question) as span:
        passage = hypothetical_document(question)
        hyde_results = hybrid.retrieve(
            question, top_k=settings.fetch_k, dense_query=passage
        )
        plain_results = hybrid.retrieve(question, top_k=settings.fetch_k)
        fused = rrf_fuse([hyde_results, plain_results], top_n=k)
        span.set_attribute("chunks_returned", len(fused))
        return fused
