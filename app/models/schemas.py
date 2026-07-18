"""Pydantic request/response schemas for the API layer."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RetrievalStrategy(str, Enum):
    DENSE = "dense"                # plain semantic top-k
    HYBRID = "hybrid"              # dense + BM25, RRF-fused
    MULTI_QUERY = "multi_query"    # N rephrasings -> hybrid -> RRF
    HYDE = "hyde"                  # hypothetical document embedding
    SELF_QUERY = "self_query"      # LLM-extracted metadata filters
    GRAPH = "graph"                # Neo4j entity traversal + linked chunks
    ADVANCED = "advanced"          # multi-query + HyDE + graph -> RRF


# ---------- Ingestion ----------

class IngestedFile(BaseModel):
    filename: str
    chunks: int = Field(..., ge=0)
    status: str = "indexed"


class IngestResponse(BaseModel):
    files: list[IngestedFile]
    total_chunks: int
    collection: str
    latency_ms: float


# ---------- Guardrails ----------

class GuardrailCheck(BaseModel):
    name: str                          # e.g. "prompt_injection", "output_pii"
    passed: bool                       # True = clear, False = triggered
    detail: str | None = None          # what triggered it


class GuardrailReport(BaseModel):
    passed: bool = True                # False if any input rail blocked the request
    blocked_reason: str | None = None  # user-facing reason when blocked
    checks: list[GuardrailCheck] = []
    pii_detected: list[str] = []       # entity types found (input or output)


# ---------- Query ----------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20, description="Override default top_k")
    strategy: RetrievalStrategy = Field(
        RetrievalStrategy.ADVANCED, description="Retrieval strategy to use"
    )
    use_rerank: bool = Field(
        True, description="Apply Cohere rerank to candidates (skipped if no key)"
    )
    use_compression: bool = Field(
        False, description="LLM-extract only relevant sentences from final chunks"
    )
    use_guardrails: bool = Field(
        True, description="Run input/output guardrails (per-server config decides which)"
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class SourceChunk(BaseModel):
    content: str
    source: str
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model: str
    retrieval_strategy: str
    latency_ms: float
    blocked: bool = False              # input guardrail refused the request
    guardrails: GuardrailReport | None = None
    grounded: bool | None = None       # output grounding verdict (if checked)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------- Documents ----------

class DocumentInfo(BaseModel):
    source: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class DeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    graph_chunks_deleted: int = 0
    file_removed: bool


# ---------- Evaluation ----------

class EvalMetric(str, Enum):
    FAITHFULNESS = "faithfulness"            # is the answer grounded in retrieved context?
    ANSWER_RELEVANCY = "answer_relevancy"    # does the answer address the question?
    CONTEXT_PRECISION = "context_precision"  # are the retrieved chunks relevant/ranked well?
    CONTEXT_RECALL = "context_recall"        # did retrieval find everything the answer needs?


class GoldenItem(BaseModel):
    question: str
    ground_truth: str
    category: str | None = None


class GoldenDatasetResponse(BaseModel):
    description: str
    size: int
    items: list[GoldenItem]


class EvalRequest(BaseModel):
    strategies: list[RetrievalStrategy] = Field(
        default_factory=lambda: [RetrievalStrategy.HYBRID, RetrievalStrategy.ADVANCED]
    )
    metrics: list[EvalMetric] = Field(
        default_factory=lambda: list(EvalMetric)
    )
    num_questions: int | None = Field(
        None, ge=1, le=50, description="Cap on golden questions (None = all)"
    )
    top_k: int = Field(5, ge=1, le=20)
    use_rerank: bool = True


class PerQuestionResult(BaseModel):
    question: str
    answer: str
    ground_truth: str
    num_contexts: int
    scores: dict[str, float | None]         # metric name -> score (None if not computable)


class StrategyEvalResult(BaseModel):
    strategy: str
    num_questions: int
    aggregate: dict[str, float | None]      # metric name -> mean score
    per_question: list[PerQuestionResult]
    avg_latency_ms: float


class EvalResponse(BaseModel):
    metrics: list[str]
    dataset_size: int
    num_questions: int
    results: list[StrategyEvalResult]
    total_latency_ms: float


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str
    environment: str
    collection: str
    documents_indexed: int
    details: dict[str, Any] = {}
