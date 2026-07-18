"""File loaders — turn uploaded files into LangChain Documents.

Supported: .pdf, .txt, .md, .docx
"""

from pathlib import Path

import logfire
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


class UnsupportedFileType(ValueError):
    pass


def load_file(path: Path) -> list[Document]:
    """Load a single file into Documents (one per page for PDFs)."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    with logfire.span("load_file", filename=path.name, extension=ext):
        if ext == ".pdf":
            docs = PyPDFLoader(str(path)).load()
        elif ext == ".docx":
            docs = Docx2txtLoader(str(path)).load()
        else:  # .txt / .md
            docs = TextLoader(str(path), encoding="utf-8").load()

        # Normalize metadata: keep the original filename as the source
        for doc in docs:
            doc.metadata["source"] = path.name
        return docs
