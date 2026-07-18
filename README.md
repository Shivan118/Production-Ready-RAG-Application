# RAG End-to-End — Production-Ready RAG Application

Advanced RAG with OpenAI + ChromaDB: hybrid search, multi-query retrieval,
query expansion, self-query, contextual compression, Cohere reranking,
Graph RAG (Neo4j), guardrails (Presidio PII), RAGAS evals, and full
observability with Pydantic Logfire.

## Phase Status

| Phase | Scope | Status |
|---|---|---|
| 1. Foundation | Config, ingestion → ChromaDB, dense retrieval, generation, FastAPI, Logfire | ✅ |
| 2. Advanced retrieval | Hybrid+RRF, multi-query, query expansion, self-query, compression, Cohere rerank | ✅ |
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
| `POST /api/v1/query` | `{"question": "...", "strategy": "advanced", "use_rerank": true, "use_compression": false}` → grounded answer + cited sources |
| `GET /api/v1/health` | Status + indexed document count |

## Tests

```bash
pytest tests/ -v
```

## Architecture

```
upload → load → chunk (recursive, stable IDs) → embed (OpenAI) → ChromaDB (cosine)
                                                              ↘ BM25 index (in-memory, auto-synced)

query → strategy retrieval → [Cohere rerank] → [LLM compression] → grounded generation
```

### Retrieval strategies (`strategy` field on /query)

| Strategy | What it does |
|---|---|
| `dense` | Plain semantic top-k over ChromaDB |
| `hybrid` | Dense + BM25 keyword search, fused with Reciprocal Rank Fusion |
| `multi_query` | LLM rephrases the question 3 ways → hybrid each → RRF |
| `hyde` | LLM writes a hypothetical answer passage, embeds that for dense search |
| `self_query` | LLM extracts metadata filters ("in attention.pdf") → filtered search |
| `advanced` (default) | multi-query + HyDE → hybrid → RRF over everything |

`use_rerank: true` (default) sends the candidate pool through Cohere Rerank
(`rerank-v3.5`) — silently skipped if `COHERE_API_KEY` is not set.
`use_compression: true` LLM-extracts only the relevant sentences from the
final chunks before generation.

Every stage emits a Logfire span (latency, token usage, retrieval scores).
Set `LOGFIRE_TOKEN` to stream traces to the dashboard; without it, spans
print to the console.
