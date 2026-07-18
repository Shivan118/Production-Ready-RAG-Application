# RAG End-to-End — Production-Ready RAG Application

Advanced RAG with OpenAI + ChromaDB: hybrid search, multi-query retrieval,
query expansion, self-query, contextual compression, Cohere reranking,
Graph RAG (Neo4j), guardrails (Presidio PII), RAGAS evals, and full
observability with Pydantic Logfire.

## Phase Status

| Phase | Scope | Status |
|---|---|---|
| 1. Foundation | Config, ingestion → ChromaDB, dense retrieval, generation, FastAPI, Logfire | ✅ |
| 2. Advanced retrieval | Hybrid+RRF, multi-query, query expansion, self-query, compression, Cohere rerank | ⏳ |
| 3. Graph RAG | Neo4j entity/relation extraction + graph-augmented retrieval | ⏳ |
| 4. Guardrails | Prompt-injection + topic rails, Presidio PII detection/redaction | ⏳ |
| 5. Evals | RAGAS golden dataset, per-strategy comparison | ⏳ |
| 6. Ship | Custom metrics, Streamlit UI, Docker | ⏳ |

## Setup

```bash
python3.11 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # then fill in OPENAI_API_KEY (LOGFIRE_TOKEN optional)
```

## Run

```bash
uvicorn app.api.main:app --reload
```

Open http://localhost:8000/docs

| Endpoint | Description |
|---|---|
| `POST /api/v1/ingest` | Upload `.pdf` / `.txt` / `.md` / `.docx` files |
| `POST /api/v1/query` | `{"question": "...", "top_k": 5}` → grounded answer + cited sources |
| `GET /api/v1/health` | Status + indexed document count |

## Tests

```bash
pytest tests/ -v
```

## Architecture (Phase 1)

```
upload → load → chunk (recursive, stable IDs) → embed (OpenAI) → ChromaDB
query  → dense retrieval (top-k similarity) → grounded generation with citations
```

Every stage emits a Logfire span (latency, token usage, retrieval scores).
Set `LOGFIRE_TOKEN` to stream traces to the dashboard; without it, spans
print to the console.
