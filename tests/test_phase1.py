"""Phase 1 unit tests — config validation, schemas, chunking, loaders.

These run without an OpenAI key or network access.
"""

import os

import pytest
from pydantic import ValidationError

os.environ.setdefault("OPENAI_API_KEY", "sk-test-0000000000")

from app.config import Settings  # noqa: E402
from app.ingestion.chunking import chunk_documents  # noqa: E402
from app.ingestion.loaders import UnsupportedFileType, load_file  # noqa: E402
from app.models.schemas import QueryRequest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


class TestConfig:
    def test_valid_settings(self):
        s = Settings(openai_api_key="sk-test-0000000000", _env_file=None)
        assert s.chunk_size == 1000
        assert s.top_k == 5

    def test_rejects_overlap_larger_than_chunk(self):
        with pytest.raises(ValidationError):
            Settings(
                openai_api_key="sk-test-0000000000",
                chunk_size=200,
                chunk_overlap=300,
                _env_file=None,
            )

    def test_rejects_short_api_key(self):
        with pytest.raises(ValidationError):
            Settings(openai_api_key="x", _env_file=None)


class TestSchemas:
    def test_query_request_strips_whitespace(self):
        req = QueryRequest(question="  what is RAG?  ")
        assert req.question == "what is RAG?"

    def test_query_request_rejects_blank(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="   ")

    def test_query_request_rejects_bad_top_k(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="what is RAG?", top_k=0)


class TestChunking:
    def test_chunks_have_stable_ids(self):
        doc = Document(
            page_content="RAG combines retrieval with generation. " * 100,
            metadata={"source": "test.md"},
        )
        chunks = chunk_documents([doc])
        assert len(chunks) > 1
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "chunk IDs must be unique"
        assert all(i.startswith("test.md::") for i in ids)

    def test_empty_input(self):
        assert chunk_documents([]) == []


class TestLoaders:
    def test_loads_markdown(self, tmp_path):
        f = tmp_path / "note.md"
        f.write_text("# Title\n\nSome content about RAG.", encoding="utf-8")
        docs = load_file(f)
        assert len(docs) == 1
        assert docs[0].metadata["source"] == "note.md"
        assert "RAG" in docs[0].page_content

    def test_rejects_unsupported_extension(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")
        with pytest.raises(UnsupportedFileType):
            load_file(f)
