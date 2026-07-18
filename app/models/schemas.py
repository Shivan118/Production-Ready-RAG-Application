"""Pydantic request/response schemas for the API layer."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


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
    retrieval_strategy: str = "dense"
    latency_ms: float
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: str
    environment: str
    collection: str
    documents_indexed: int
    details: dict[str, Any] = {}
