"""Grounded answer generation with source citations."""

from functools import lru_cache

import logfire
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.models.schemas import SourceChunk

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY the provided context.

Rules:
- Answer strictly from the context below. Do not use outside knowledge.
- Cite sources inline as [1], [2] matching the numbered context blocks.
- If the context does not contain the answer, say exactly: \
"I could not find this information in the indexed documents."
- Be concise and factual."""

USER_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


@lru_cache(maxsize=8)
def _llm_for(api_key: str) -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=api_key,
        temperature=0.0,
    )


def _llm() -> ChatOpenAI:
    from app.runtime_keys import effective_openai_key

    return _llm_for(effective_openai_key())


def _format_context(chunks: list[SourceChunk]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {c.source})\n{c.content}" for i, c in enumerate(chunks)
    )


def generate_answer(question: str, chunks: list[SourceChunk]) -> str:
    with logfire.span("generate_answer", num_chunks=len(chunks)) as span:
        if not chunks:
            return "I could not find this information in the indexed documents."

        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
        )
        chain = prompt | _llm()
        response = chain.invoke(
            {"context": _format_context(chunks), "question": question}
        )
        span.set_attribute("answer_length", len(response.content))
        return response.content
