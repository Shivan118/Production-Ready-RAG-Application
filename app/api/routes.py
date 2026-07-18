"""API routes: /ingest, /query, /health."""

import shutil
import time
from pathlib import Path

import logfire
import openai
from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile

from app.config import get_settings
from app.ingestion.indexer import (
    delete_source,
    document_count,
    index_file,
    list_sources,
)
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileType
from app.models.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentsResponse,
    EvalRequest,
    EvalResponse,
    GoldenDatasetResponse,
    HealthResponse,
    IngestedFile,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievalStrategy,
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


def _raise_for_openai(e: openai.OpenAIError) -> None:
    """Map OpenAI SDK errors to clean HTTP responses (no raw 500 tracebacks)."""
    if isinstance(e, openai.AuthenticationError):
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
    if isinstance(e, openai.APIConnectionError):
        raise HTTPException(
            status_code=503,
            detail="Could not reach OpenAI — check your internet/DNS connection "
            "and retry.",
        )
    if isinstance(e, openai.RateLimitError):
        raise HTTPException(
            status_code=429,
            detail="OpenAI rate limit or quota exceeded — please retry shortly.",
        )
    raise HTTPException(status_code=502, detail=f"Upstream LLM error: {type(e).__name__}")


def _rerank_enabled() -> bool:
    from app.retrieval.reranker import is_enabled

    return is_enabled()


def _graph_enabled() -> bool:
    from app.graph.client import is_enabled

    return is_enabled()


def _graph_stats() -> dict:
    from app.graph.store import graph_stats

    return graph_stats()


def _guardrails_status() -> dict:
    from app.guardrails.rails import status

    return status()


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
    if request.strategy == RetrievalStrategy.GRAPH and not _graph_enabled():
        raise HTTPException(
            status_code=503,
            detail="Graph strategy requires Neo4j. Set NEO4J_URI in .env "
            "(docker compose up -d, or a free Neo4j Aura instance).",
        )
    try:
        return run_query(request)
    except openai.OpenAIError as e:
        _raise_for_openai(e)


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents() -> DocumentsResponse:
    """List indexed files with their chunk counts."""
    sources = list_sources()
    return DocumentsResponse(
        documents=[DocumentInfo(source=s, chunks=n) for s, n in sources.items()],
        total_chunks=sum(sources.values()),
    )


@router.delete("/documents/{filename}", response_model=DeleteResponse)
async def delete_document(filename: str) -> DeleteResponse:
    """Delete one uploaded file everywhere: Chroma chunks, BM25 index,
    Neo4j graph data, and the stored upload. Explicit call only."""
    safe_name = Path(filename).name  # no path traversal
    chunks_deleted = delete_source(safe_name)
    if chunks_deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed document named '{safe_name}'. "
            f"See GET /api/v1/documents for the list.",
        )

    from app.graph.store import delete_source_graph

    graph_deleted = delete_source_graph(safe_name)

    file_removed = False
    upload_path = UPLOAD_DIR / safe_name
    if upload_path.is_file():
        upload_path.unlink()
        file_removed = True

    logfire.info(
        "document_deleted",
        filename=safe_name,
        chunks_deleted=chunks_deleted,
        graph_chunks_deleted=graph_deleted,
        file_removed=file_removed,
    )
    return DeleteResponse(
        filename=safe_name,
        chunks_deleted=chunks_deleted,
        graph_chunks_deleted=graph_deleted,
        file_removed=file_removed,
    )


@router.get("/evals/dataset", response_model=GoldenDatasetResponse)
async def eval_dataset() -> GoldenDatasetResponse:
    """The golden Q&A set used for evaluation."""
    from app.evaluation import dataset

    return GoldenDatasetResponse(
        description=dataset.description(),
        size=dataset.size(),
        items=dataset.golden_items(),
    )


# NOTE: sync `def` on purpose — FastAPI runs it in a threadpool, so RAGAS can
# spin up its own asyncio loop without clashing with the server's event loop.
@router.post("/evals/run", response_model=EvalResponse)
def run_evals(request: EvalRequest) -> EvalResponse:
    """Evaluate the selected strategies against the golden dataset with RAGAS.

    Slow and LLM-heavy: ~(strategies x questions x (1 answer + metrics)) calls.
    """
    if not request.strategies:
        raise HTTPException(status_code=400, detail="Select at least one strategy")
    if document_count() == 0:
        raise HTTPException(
            status_code=409, detail="No documents indexed — ingest first."
        )
    from app.evaluation.runner import run_evaluation

    try:
        return run_evaluation(request)
    except openai.OpenAIError as e:
        _raise_for_openai(e)


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
            "graph": _graph_stats(),
            "guardrails": _guardrails_status(),
        },
    )
