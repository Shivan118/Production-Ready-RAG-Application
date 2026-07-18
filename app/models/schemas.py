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


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str
    environment: str
    collection: str
    documents_indexed: int
    details: dict[str, Any] = {}
