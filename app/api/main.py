"""FastAPI application entrypoint.

Run:  uvicorn app.api.main:app --reload
Docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.observability.logfire_setup import instrument_fastapi, setup_logfire

setup_logfire()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast on invalid config and warm up the Chroma collection
    get_settings()
    from app.ingestion.indexer import get_vectorstore

    get_vectorstore()
    yield


app = FastAPI(
    title="RAG End-to-End",
    description="Production-ready RAG API — Phase 1: dense retrieval baseline",
    version="0.1.0",
    lifespan=lifespan,
)
instrument_fastapi(app)
app.include_router(router, prefix="/api/v1")
