# Production-Ready RAG Application

An end-to-end Retrieval-Augmented Generation system built with **OpenAI + ChromaDB + FastAPI + Streamlit**, featuring six retrieval strategies, Cohere reranking, per-request API keys, and full observability with **Pydantic Logfire**.

## Phase Status

| Phase | Scope | Status |
|---|---|---|
| 1. Foundation | Config, ingestion → ChromaDB, dense retrieval, generation, FastAPI, Logfire | ✅ |
| 2. Advanced retrieval | Hybrid+RRF, multi-query, HyDE, self-query, compression, Cohere rerank | ✅ |
| 2.5 Frontend | Streamlit chatbot + session dashboard + document manager, sidebar API keys | ✅ |
| 3. Graph RAG | Neo4j entity/relation extraction + graph-augmented retrieval | ✅ |
| 4. Guardrails | Prompt-injection + moderation rails, Presidio PII detection/redaction, grounding check | ✅ |
| 5. Evals | RAGAS golden dataset, per-strategy comparison, in-UI reports | ✅ |
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
| `graph` | Query entities → Neo4j traversal → relation facts + linked chunks | Answers multi-hop questions ("who proposed X and where do they work?") that similarity search can't connect |
| `advanced` (default) | multi-query + HyDE + graph → hybrid → RRF over all lists | Maximum recall pass feeding the reranker |

### Graph RAG (`app/graph/`) — Phase 3
- **Ingest-time extraction**: an LLM (structured output) pulls entities (`Person`, `Technology`, `Concept`, …) and relations (`PROPOSED`, `USES`, …) from each chunk into Neo4j: `(:Entity)-[:REL]->(:Entity)` plus `(:Entity)-[:MENTIONED_IN]->(:Chunk)` — **why:** vector search finds *similar text*; a graph finds *connected facts* across documents.
- **Query time**: entities spotted in the question → up to 2-hop traversal → triples returned as a "knowledge graph facts" context block + the real chunks those entities appear in (hydrated from Chroma).
- **MERGE everywhere** → re-ingestion never duplicates graph data. Relationship types stored as properties → no Cypher injection.
- **Graceful**: no `NEO4J_URI` → graph strategy returns a clear `503`, `advanced` silently skips graph, everything else unaffected.

**Start Neo4j** (pick one):
```bash
docker compose up -d      # local, needs Docker Desktop; browser at http://localhost:7474
# or create a free instance at https://neo4j.com/product/auradb/ (no install)
```
Then set `NEO4J_URI` / `NEO4J_PASSWORD` in `.env`, restart the API, and re-ingest your documents to populate the graph.

### Post-retrieval (`use_rerank`, `use_compression` flags)
- **Cohere Rerank (`rerank-v3.5`)** — a cross-encoder reads query+chunk *together*, far more accurate than vector similarity. **Why the wide-then-narrow design:** retrieval casts a wide net (fetch_k=20 candidates), the reranker picks the truly relevant top-k. Degrades gracefully — no Cohere key → candidates pass through with a logged warning, the API never breaks.
- **Contextual compression** — an LLM extractor strips irrelevant sentences from final chunks. **Why:** less noise in the prompt = fewer hallucinations + fewer tokens.

### Generation (`app/generation/`)
- Answers **only** from retrieved context, with inline `[1]` `[2]` citations mapped to sources; refuses explicitly when context lacks the answer — **why:** grounding + verifiability are the whole point of RAG.

### Evaluation (`app/evaluation/` + `evals/`) — Phase 5
- **Golden dataset** (`evals/golden_dataset.json`): 12 Q&A pairs with reference answers, grounded in the indexed sample docs. **Why:** you can't improve what you can't measure — a fixed reference set makes retrieval changes comparable.
- **RAGAS metrics** (all 0–1, higher is better): **faithfulness** (is the answer grounded in the retrieved context?), **answer relevancy** (does it address the question?), **context precision** (are retrieved chunks relevant/well-ranked?), **context recall** (did retrieval find everything the ground-truth answer needs?). Metrics are read back by each RAGAS metric's own `.name`, so the wrapper survives RAGAS's cross-version renames.
- **Per-strategy comparison**: the runner executes the *real* pipeline for every (strategy × question) with guardrails off, collects answer + contexts, and scores them — so you see, e.g., whether `advanced` actually beats `hybrid` on faithfulness for your data.
- **In the UI**: a dedicated **📈 Evals tab** — pick strategies + metrics + question count, run, and get an aggregate comparison table, a grouped per-metric bar chart, and per-question score breakdowns, all rendered in Streamlit. No notebook required.
- Runs server-side via `POST /evals/run` (sync endpoint → threadpool, so RAGAS's asyncio loop doesn't clash with the server's).

### Guardrails (`app/guardrails/`) — Phase 4
A defense-in-depth wrapper around the pipeline. **Input rails run before any retrieval/generation** (so a bad request costs nothing); **output rails transform/flag the answer**. Every rail is individually toggleable and degrades gracefully.

| Rail | Stage | Action | Why |
|---|---|---|---|
| **Prompt injection** (`injection.py`) | input | **block** | Regex heuristics catch "ignore previous instructions / reveal system prompt / act as / jailbreak" — instant, zero-cost, before tokens are spent |
| **Moderation** (`moderation.py`) | input | **block** | OpenAI's free moderation endpoint flags hate/violence/self-harm/sexual content |
| **Input PII** (`pii.py`) | input | flag (or block) | Presidio detects personal data in the question; flagged by default, `BLOCK_ON_INPUT_PII=true` to reject |
| **Output PII** (`pii.py`) | output | **redact** | Presidio replaces PII in the answer with `<PERSON>`, `<EMAIL_ADDRESS>`, … before it leaves the server |
| **Grounding** (`grounding.py`) | output | flag | Optional LLM fact-checker verifies the answer is supported by context (off by default — adds a call) |

- **PII entity allowlist**: redaction is scoped to true-identity types (PERSON, EMAIL, PHONE, SSN, credit card, …) and **deliberately excludes** ORGANIZATION/DATE/URL — spaCy tags "OpenAI"/"Transformer"/dates as those, and redacting them would gut legitimate answers. Configurable via `pii_entities`.
- **Graceful degradation**: if Presidio/spaCy aren't installed, PII rails report `unavailable` and the pipeline runs unaffected. Moderation/grounding fail *open* (logged, treated as pass) so a guardrail outage never takes the API down.
- Blocked requests return `200` with `blocked: true` and a `guardrails` report (every check + reason) — the UI shows a 🛡️ notice; the API stays uniform. Per-request `use_guardrails: false` opts out.

**Setup:** `pip install presidio-analyzer presidio-anonymizer spacy` then `python -m spacy download en_core_web_sm` (both in `requirements.txt`).

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
| `POST /api/v1/query` | `{"question", "strategy", "top_k", "use_rerank", "use_compression", "use_guardrails"}` → answer, sources, `blocked`, `guardrails` report, `grounded` |
| `GET /api/v1/documents` | List indexed files with chunk counts |
| `DELETE /api/v1/documents/{filename}` | Remove one file everywhere: Chroma chunks, BM25, Neo4j graph data, stored upload |
| `GET /api/v1/evals/dataset` | The golden Q&A dataset |
| `POST /api/v1/evals/run` | Run RAGAS eval for selected strategies → per-strategy metric scores |
| `GET /api/v1/health` | Status, chunks indexed, rerank/graph/guardrails status |

Optional headers on any endpoint: `X-OpenAI-Api-Key`, `X-Cohere-Api-Key`.

## Tests

```bash
pytest tests/ -v      # 45 tests: config, schemas, chunking, RRF, graph, guardrails
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
├── graph/                 # Neo4j client, entity/relation extractor, store, retriever
├── guardrails/            # injection, moderation, pii (Presidio), grounding, rails
├── evaluation/            # golden dataset loader, RAGAS wrapper, per-strategy runner
├── pipeline.py            # orchestrator: guardrails → retrieve → rerank → compress → generate → guardrails
├── observability/         # Logfire setup + instrumentation
└── api/                   # FastAPI app + routes
evals/golden_dataset.json  # golden Q&A pairs for evaluation
frontend/app.py            # Streamlit chat + dashboard + evals + documents
tests/                     # offline unit tests (no API keys needed)
```
