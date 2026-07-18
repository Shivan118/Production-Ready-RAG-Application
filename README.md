# Production-Ready RAG Application

An end-to-end Retrieval-Augmented Generation system built with **OpenAI + ChromaDB + FastAPI + Streamlit**, featuring six retrieval strategies, Cohere reranking, per-request API keys, and full observability with **Pydantic Logfire**.

## Phase Status

| Phase | Scope | Status |
|---|---|---|
| 1. Foundation | Config, ingestion → ChromaDB, dense retrieval, generation, FastAPI, Logfire | ✅ |
| 2. Advanced retrieval | Hybrid+RRF, multi-query, HyDE, self-query, compression, Cohere rerank | ✅ |
| 2.5 Frontend | Streamlit chatbot + session dashboard + document manager, sidebar API keys | ✅ |
| 3. Graph RAG | Neo4j entity/relation extraction + graph-augmented retrieval | ⏳ |
| 4. Guardrails | Prompt-injection + topic rails, Presidio PII detection/redaction | ⏳ |
| 5. Evals | RAGAS golden dataset, per-strategy comparison | ⏳ |
| 6. Ship | Custom metrics, Docker, CI | ⏳ |

## Architecture

```
                        ┌─ INGESTION ─────────────────────────────────────┐
 upload (pdf/txt/md/docx) → load → chunk → embed (OpenAI) → ChromaDB (cosine)
                                                          ↘ BM25 index (in-memory, auto-synced)

                        ┌─ QUERY ─────────────────────────────────────────┐
 question → strategy retrieval → RRF fusion → Cohere rerank → compression → grounded answer + citations
                        └─ every stage emits a Logfire span ──────────────┘
```

## What We Implemented — and Why

### Ingestion (`app/ingestion/`)
- **Recursive character chunking** (1000 chars, 150 overlap) — keeps paragraphs/sentences intact so chunks stay semantically coherent; overlap prevents losing context at boundaries.
- **Stable chunk IDs** (`source::index::content-hash`) used as Chroma IDs — **why:** re-uploading the same file *upserts* instead of duplicating, making ingestion idempotent.
- **ChromaDB with cosine space** — **why:** Chroma defaults to L2 distance, which produced negative/meaningless relevance scores with OpenAI embeddings; cosine gives interpretable 0–1 scores.

### Retrieval strategies (`app/retrieval/`) — `strategy` field on `/query`

| Strategy | What | Why |
|---|---|---|
| `dense` | Semantic top-k over ChromaDB | Baseline; captures meaning but misses exact terms |
| `hybrid` | Dense + BM25 keyword search, fused with **Reciprocal Rank Fusion** | BM25 catches exact tokens (error codes, names, acronyms) that embeddings blur; RRF merges rankings without comparing incompatible score scales |
| `multi_query` | LLM rephrases the question 3 ways → hybrid each → RRF | Users and documents use different vocabulary; multiple phrasings raise recall |
| `hyde` | LLM writes a hypothetical answer passage, embeds *that* for dense search | Documents match documents better than questions match documents |
| `self_query` | LLM extracts metadata filters ("in attention.pdf" → `source='attention.pdf'`) | Natural-language constraints become real filters instead of hoping similarity finds the right file |
| `advanced` (default) | multi-query + HyDE → hybrid → RRF over all lists | Maximum recall pass feeding the reranker |

### Post-retrieval (`use_rerank`, `use_compression` flags)
- **Cohere Rerank (`rerank-v3.5`)** — a cross-encoder reads query+chunk *together*, far more accurate than vector similarity. **Why the wide-then-narrow design:** retrieval casts a wide net (fetch_k=20 candidates), the reranker picks the truly relevant top-k. Degrades gracefully — no Cohere key → candidates pass through with a logged warning, the API never breaks.
- **Contextual compression** — an LLM extractor strips irrelevant sentences from final chunks. **Why:** less noise in the prompt = fewer hallucinations + fewer tokens.

### Generation (`app/generation/`)
- Answers **only** from retrieved context, with inline `[1]` `[2]` citations mapped to sources; refuses explicitly when context lacks the answer — **why:** grounding + verifiability are the whole point of RAG.

### Per-request API keys (`app/runtime_keys.py`)
- Clients can send `X-OpenAI-Api-Key` / `X-Cohere-Api-Key` headers; resolved via request-scoped contextvars with per-key client caches, falling back to `.env`. **Why:** lets the Streamlit UI (or any tenant) bring their own keys without restarting the server — a real multi-tenant SaaS pattern. Invalid keys return a clean `401`.

### Observability (Pydantic Logfire, `app/observability/`)
- FastAPI + OpenAI auto-instrumented; every pipeline stage wrapped in a custom span with attributes (chunks returned, top scores, latency, variants generated). **Why:** when an answer is bad you can see *which stage* failed — retrieval? rerank? generation? — plus token counts per call for cost tracking. Works without a token (console) and streams to the dashboard with one.

### Validation everywhere (Pydantic)
- `Settings` fail fast at startup (bad chunk config, missing key = app won't boot); every request/response is a typed schema (strategy enum, bounded top_k, non-blank question → `422` not `500`). **Why:** production APIs fail loudly at the edge, not deep in a pipeline.

### Frontend (`frontend/app.py` — Streamlit)
- **💬 Chat** — history, per-answer strategy/latency/sources with scores.
- **📊 Dashboard** — session metrics: query count, avg/P95 latency, avg top score, latency chart, strategy usage, history table; deep traces link to Logfire.
- **📁 Documents** — drag-and-drop ingestion with per-file chunk counts.
- **Sidebar** — API keys (password-masked, sent per-request), strategy/top-k/rerank/compression controls, live backend health.

## Setup

```bash
python3.11 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env         # fill in OPENAI_API_KEY (COHERE_API_KEY + LOGFIRE_TOKEN optional)
```

## Run

```bash
# Terminal 1 — API
uvicorn app.api.main:app --reload
# Terminal 2 — UI
streamlit run frontend/app.py
```

API docs: http://localhost:8000/docs · UI: http://localhost:8501

## API

| Endpoint | Description |
|---|---|
| `POST /api/v1/ingest` | Upload `.pdf` / `.txt` / `.md` / `.docx` (multipart) |
| `POST /api/v1/query` | `{"question", "strategy", "top_k", "use_rerank", "use_compression"}` |
| `GET /api/v1/health` | Status, chunks indexed, rerank enabled |

Optional headers on any endpoint: `X-OpenAI-Api-Key`, `X-Cohere-Api-Key`.

## Tests

```bash
pytest tests/ -v      # 22 tests: config, schemas, chunking, RRF math, rerank fallback
```

## Project Structure

```
app/
├── config.py              # Pydantic Settings — validated at startup
├── runtime_keys.py        # per-request API key overrides (contextvars)
├── models/schemas.py      # request/response schemas + RetrievalStrategy enum
├── ingestion/             # loaders → chunking → embeddings → Chroma indexer
├── retrieval/             # dense, hybrid+BM25, fusion(RRF), multi_query,
│                          # query_expansion(HyDE), self_query, reranker, compression
├── generation/            # grounded generation with citations
├── pipeline.py            # orchestrator: retrieve → rerank → compress → generate
├── observability/         # Logfire setup + instrumentation
└── api/                   # FastAPI app + routes
frontend/app.py            # Streamlit chat + dashboard + documents
tests/                     # offline unit tests (no API keys needed)
```
