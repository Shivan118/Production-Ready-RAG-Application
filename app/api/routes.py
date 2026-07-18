"""API routes: /ingest, /query, /health."""

import shutil
import time
from pathlib import Path

import logfire
import openai
from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile

from app.config import get_settings
from app.ingestion.indexer import document_count, index_file
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileType
from app.models.schemas import (
    HealthResponse,
    IngestedFile,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.pipeline import run_query
from app.runtime_keys import cohere_key_override, openai_key_override


async def apply_key_overrides(
    x_openai_api_key: str | None = Header(None),
    x_cohere_api_key: str | None = Header(None),
) -> None:
    """Let clients (e.g. the Streamlit UI) supply their own API keys per request."""
    if x_openai_api_key and x_openai_api_key.strip():
        openai_key_override.set(x_openai_api_key.strip())
    if x_cohere_api_key and x_cohere_api_key.strip():
        cohere_key_override.set(x_cohere_api_key.strip())


router = APIRouter(dependencies=[Depends(apply_key_overrides)])

UPLOAD_DIR = Path("data/uploads")


def _rerank_enabled() -> bool:
    from app.retrieval.reranker import is_enabled

    return is_enabled()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile]) -> IngestResponse:
    """Upload and index documents (.pdf, .txt, .md, .docx)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    settings = get_settings()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    results: list[IngestedFile] = []

    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=400, detail="File has no filename")
        ext = Path(upload.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"'{upload.filename}': unsupported type '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            )

        dest = UPLOAD_DIR / Path(upload.filename).name
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

        try:
            chunks = index_file(dest)
        except UnsupportedFileType as e:
            raise HTTPException(status_code=415, detail=str(e))
        except Exception as e:
            logfire.error("ingest_failed", filename=upload.filename, error=str(e))
            raise HTTPException(
                status_code=500, detail=f"Failed to index '{upload.filename}': {e}"
            )
        results.append(IngestedFile(filename=upload.filename, chunks=chunks))

    return IngestResponse(
        files=results,
        total_chunks=sum(r.chunks for r in results),
        collection=settings.chroma_collection,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Ask a question over the indexed documents."""
    if document_count() == 0:
        raise HTTPException(
            status_code=409,
            detail="No documents indexed yet. Upload files via /ingest first.",
        )
    try:
        return run_query(request)
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        collection=settings.chroma_collection,
        documents_indexed=document_count(),
        details={
            "chat_model": settings.openai_chat_model,
            "embedding_model": settings.openai_embedding_model,
            "rerank_enabled": _rerank_enabled(),
            "rerank_model": settings.cohere_rerank_model,
        },
    )
