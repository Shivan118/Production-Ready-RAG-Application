"""Output grounding / faithfulness check.

An LLM judge decides whether the generated answer is supported by the
retrieved context. Optional (adds a call), so it's off by default; when on
it flags — not blocks — so the user still sees the answer plus a warning.
"""

import logfire
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.generation.generator import _llm
from app.models.schemas import SourceChunk

# refusal sentinel from the generator — always grounded, skip the check
_REFUSAL = "I could not find this information in the indexed documents."


class GroundingVerdict(BaseModel):
    grounded: bool = Field(..., description="Is every claim supported by the context?")
    reason: str = Field(..., description="Brief justification")


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict fact-checker. Decide whether the ANSWER is fully "
            "supported by the CONTEXT. Mark grounded=false if the answer states "
            "anything not present in the context.",
        ),
        ("user", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
    ]
)


def check(answer: str, chunks: list[SourceChunk]) -> tuple[bool, str]:
    """Return (grounded, reason). Fails open (grounded=True) on any error."""
    if not chunks or answer.strip() == _REFUSAL:
        return True, "no claims to verify"

    with logfire.span("guardrail.grounding") as span:
        context = "\n\n".join(c.content for c in chunks)
        try:
            chain = _PROMPT | _llm().with_structured_output(GroundingVerdict)
            verdict: GroundingVerdict = chain.invoke(
                {"context": context, "answer": answer}
            )
        except Exception as e:  # noqa: BLE001
            logfire.warn("grounding_check_failed", error=str(e))
            return True, "check unavailable"

        span.set_attribute("grounded", verdict.grounded)
        span.set_attribute("reason", verdict.reason)
        return verdict.grounded, verdict.reason
