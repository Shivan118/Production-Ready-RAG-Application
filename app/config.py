"""Application configuration via Pydantic Settings.

Every value is validated at startup — the app fails fast on a bad config
instead of failing mid-request.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(..., min_length=10, description="OpenAI API key")
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # --- ChromaDB ---
    chroma_persist_dir: Path = Path("./chroma_db")
    chroma_collection: str = Field("rag_documents", min_length=1)

    # --- Chunking ---
    chunk_size: int = Field(1000, ge=100, le=8000)
    chunk_overlap: int = Field(150, ge=0)

    # --- Retrieval ---
    top_k: int = Field(5, ge=1, le=50)
    fetch_k: int = Field(20, ge=5, le=100, description="Candidate pool before rerank/fusion")
    multi_query_count: int = Field(3, ge=1, le=5)

    # --- Cohere Rerank ---
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"

    # --- Neo4j (Graph RAG) ---
    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    graph_extraction_on_ingest: bool = Field(
        True, description="Extract entities/relations into Neo4j during /ingest"
    )
    graph_max_hops: int = Field(2, ge=1, le=3)

    # --- Guardrails ---
    guardrails_enabled: bool = True
    guardrail_injection: bool = True          # heuristic prompt-injection detection
    guardrail_moderation: bool = True         # OpenAI moderation on the query
    guardrail_input_pii: bool = True          # detect PII in the question
    guardrail_output_pii: bool = True         # redact PII from the answer
    guardrail_grounding: bool = False         # LLM faithfulness check (adds latency+cost)
    block_on_input_pii: bool = False          # block vs. just flag PII in the query
    pii_score_threshold: float = Field(0.5, ge=0.0, le=1.0)
    pii_language: str = "en"
    spacy_model: str = "en_core_web_sm"
    # True-identity PII only. Deliberately excludes ORGANIZATION / DATE_TIME /
    # URL / LOCATION / NRP — spaCy tags "OpenAI", "Transformer", dates as those,
    # and redacting them would gut legitimate RAG answers.
    pii_entities: list[str] = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "CRYPTO",
        "IBAN_CODE",
        "IP_ADDRESS",
        "US_SSN",
        "US_PASSPORT",
        "US_DRIVER_LICENSE",
        "US_BANK_NUMBER",
        "US_ITIN",
        "MEDICAL_LICENSE",
    ]

    # --- Observability ---
    logfire_token: str = ""

    # --- App ---
    environment: Literal["development", "staging", "production"] = "development"

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_smaller_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be smaller than chunk_size ({chunk_size})"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
