"""Multi-Query Retrieval — rephrase the question N ways, retrieve for each,
fuse with RRF. Counters vocabulary mismatch between the user and the corpus."""

import logfire
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import get_settings
from app.generation.generator import _llm
from app.models.schemas import SourceChunk
from app.retrieval import hybrid
from app.retrieval.fusion import rrf_fuse


class QueryVariants(BaseModel):
    variants: list[str] = Field(..., description="Rephrased versions of the question")


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate {n} different rephrasings of the user's question for "
            "document retrieval. Vary the vocabulary and angle; keep the "
            "original meaning. Return only the rephrasings.",
        ),
        ("user", "{question}"),
    ]
)


def generate_variants(question: str, n: int | None = None) -> list[str]:
    settings = get_settings()
    n = n or settings.multi_query_count
    with logfire.span("multi_query.generate_variants", question=question, n=n) as span:
        chain = _PROMPT | _llm().with_structured_output(QueryVariants)
        result: QueryVariants = chain.invoke({"question": question, "n": n})
        variants = [v.strip() for v in result.variants if v.strip()][:n]
        span.set_attribute("variants", variants)
        return variants


def retrieve(question: str, top_k: int | None = None) -> list[SourceChunk]:
    settings = get_settings()
    k = top_k or settings.top_k

    with logfire.span("retrieve.multi_query", question=question) as span:
        queries = [question] + generate_variants(question)
        result_lists = [
            hybrid.retrieve(q, top_k=settings.fetch_k, fetch_k=settings.fetch_k)
            for q in queries
        ]
        fused = rrf_fuse(result_lists, top_n=k)
        span.set_attribute("num_queries", len(queries))
        span.set_attribute("chunks_returned", len(fused))
        return fused
