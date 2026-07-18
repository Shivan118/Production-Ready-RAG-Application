"""Document chunking with recursive character splitting."""

import hashlib

import logfire
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks with stable chunk IDs."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    with logfire.span(
        "chunk_documents",
        input_docs=len(documents),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ) as span:
        chunks = splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:12]
            chunk.metadata["chunk_id"] = f"{source}::{i}::{content_hash}"
        span.set_attribute("output_chunks", len(chunks))
        return chunks
