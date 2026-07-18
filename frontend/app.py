"""RAG End-to-End — Streamlit chatbot + observability dashboard.

Run:  streamlit run frontend/app.py
Requires the FastAPI backend:  uvicorn app.api.main:app --reload
"""

from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="RAG Chat", page_icon="🔎", layout="wide")

ACCENT = "#2a78d6"  # categorical slot 1 (validated palette)
LOGFIRE_URL = "https://logfire-us.pydantic.dev/shivan3446/starter-project"
STRATEGIES = ["advanced", "hybrid", "dense", "multi_query", "hyde", "self_query", "graph"]

USER_AVATAR = "🧑"
BOT_AVATAR = "🤖"
SUGGESTIONS = [
    "What is multi-head attention?",
    "Summarize the key ideas in my documents",
    "How does hybrid search work?",
    "Who are the authors and where do they work?",
]

# Friendly, ChatGPT-style refusals keyed by the guardrail that fired.
FRIENDLY_BLOCK = {
    "prompt_injection": "I can only help with questions about your uploaded "
    "documents. Mind rephrasing what you'd like to know? 🙂",
    "moderation": "I'm not able to help with that request. Let's keep things "
    "focused on your documents.",
    "input_pii": "For your privacy, please remove personal details (emails, "
    "phone numbers, etc.) from your message and ask again.",
}

# --- ChatGPT-like look: centered column, tinted bubbles, avatars, pinned input ---
st.markdown(
    """
    <style>
    /* leave room for the top toolbar (so tabs aren't clipped) and the
       bottom-pinned chat input */
    .block-container { max-width: 820px; padding-top: 3rem; padding-bottom: 7rem; }

    /* make the tab bar clearly visible */
    div[data-baseweb="tab-list"] { gap: 1.2rem; }
    button[data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }

    /* chat bubbles */
    [data-testid="stChatMessage"] {
        padding: 0.55rem 0.85rem;
        margin-bottom: 0.4rem;
        border-radius: 16px;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: rgba(42, 120, 214, 0.10);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: rgba(128, 128, 128, 0.08);
    }

    /* pin the chat input to the bottom, centered over the main content area
       (main area sits to the right of the ~336px sidebar) */
    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 1.2rem;
        left: calc(50% + 168px);
        transform: translateX(-50%);
        width: min(760px, 80vw);
        z-index: 100;
        border-radius: 24px;
        box-shadow: 0 2px 18px rgba(0, 0, 0, 0.18);
    }
    @media (max-width: 900px) {
        [data-testid="stChatInput"] { left: 50%; width: 92vw; }
    }

    .stButton button { border-radius: 12px; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []  # one record per query, feeds the dashboard


# ---------- backend client ----------

def api_headers() -> dict:
    headers = {}
    if st.session_state.get("openai_key", "").strip():
        headers["X-OpenAI-Api-Key"] = st.session_state.openai_key.strip()
    if st.session_state.get("cohere_key", "").strip():
        headers["X-Cohere-Api-Key"] = st.session_state.cohere_key.strip()
    return headers


def backend() -> str:
    return st.session_state.get("backend_url", "http://localhost:8000/api/v1").rstrip("/")


def get_health() -> dict | None:
    try:
        r = requests.get(f"{backend()}/health", headers=api_headers(), timeout=5)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def post_query(question: str) -> tuple[dict | None, str | None]:
    payload = {
        "question": question,
        "strategy": st.session_state.strategy,
        "top_k": st.session_state.top_k,
        "use_rerank": st.session_state.use_rerank,
        "use_compression": st.session_state.use_compression,
        "use_guardrails": st.session_state.use_guardrails,
    }
    try:
        r = requests.post(
            f"{backend()}/query", json=payload, headers=api_headers(), timeout=180
        )
    except requests.RequestException as e:
        return None, f"Backend unreachable: {e}"
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"HTTP {r.status_code}: {detail}"
    return r.json(), None


def get_eval_dataset() -> dict | None:
    try:
        r = requests.get(f"{backend()}/evals/dataset", headers=api_headers(), timeout=10)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def post_eval_run(payload: dict) -> tuple[dict | None, str | None]:
    try:
        r = requests.post(
            f"{backend()}/evals/run", json=payload, headers=api_headers(), timeout=1800
        )
    except requests.RequestException as e:
        return None, f"Backend unreachable: {e}"
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"HTTP {r.status_code}: {detail}"
    return r.json(), None


def post_ingest(files) -> tuple[dict | None, str | None]:
    payload = [("files", (f.name, f.getvalue())) for f in files]
    try:
        r = requests.post(
            f"{backend()}/ingest", files=payload, headers=api_headers(), timeout=600
        )
    except requests.RequestException as e:
        return None, f"Backend unreachable: {e}"
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        return None, f"HTTP {r.status_code}: {detail}"
    return r.json(), None


# ---------- rendering ----------

def _blocked_message(data: dict) -> str:
    """Map the failed guardrail to a friendly, ChatGPT-style refusal."""
    for check in (data.get("guardrails") or {}).get("checks", []):
        if not check.get("passed", True):
            return FRIENDLY_BLOCK.get(
                check["name"],
                "I can't help with that one — try rephrasing your question "
                "about your documents.",
            )
    return "I can't help with that request."


def render_answer(data: dict) -> None:
    """Render one assistant turn: blocked notice, answer, guardrail badges, sources."""
    if data.get("blocked"):
        st.markdown(_blocked_message(data))
        st.caption("🛡️ Filtered by guardrails")
        return

    st.markdown(data["answer"])

    top_score = data["sources"][0]["score"] if data["sources"] else 0.0
    st.caption(
        f"`{data['retrieval_strategy']}` · {data['latency_ms']:.0f} ms · "
        f"{len(data['sources'])} sources · top score {top_score}"
    )

    g = data.get("guardrails") or {}
    pii_found = g.get("pii_detected") or []
    if pii_found:
        st.caption(f"🔒 PII redacted: {', '.join(pii_found)}")
    grounded = data.get("grounded")
    if grounded is True:
        st.caption("✅ Answer grounded in sources")
    elif grounded is False:
        st.caption("⚠️ Answer may not be fully supported by sources")

    if data["sources"]:
        with st.expander("📄 Sources"):
            for i, s in enumerate(data["sources"], 1):
                st.markdown(
                    f"**[{i}] {s['source']}** — score `{s['score']}`\n\n"
                    f"> {s['content'][:400]}{'…' if len(s['content']) > 400 else ''}"
                )


# ---------- sidebar ----------

with st.sidebar:
    st.title("🔎 RAG End-to-End")

    st.subheader("🔑 API Keys")
    st.text_input(
        "OpenAI API key",
        type="password",
        key="openai_key",
        help="Leave empty to use the key from the server's .env",
    )
    st.text_input(
        "Cohere API key (rerank)",
        type="password",
        key="cohere_key",
        help="Leave empty to use the key from the server's .env",
    )

    st.subheader("⚙️ Retrieval")
    st.selectbox("Strategy", STRATEGIES, key="strategy")
    st.slider("Top K sources", 1, 10, 5, key="top_k")
    st.toggle("Cohere rerank", value=True, key="use_rerank")
    st.toggle("Contextual compression", value=False, key="use_compression")
    st.toggle("Guardrails", value=True, key="use_guardrails")

    st.subheader("🖥️ Backend")
    st.text_input("API URL", value="http://localhost:8000/api/v1", key="backend_url")

    health = get_health()
    if health:
        st.success(
            f"Connected · {health['documents_indexed']} chunks indexed · "
            f"env: {health['environment']}"
        )
        rerank_on = health["details"].get("rerank_enabled", False)
        if st.session_state.use_rerank and not rerank_on and not st.session_state.cohere_key:
            st.warning("Rerank is ON but no Cohere key is configured — it will be skipped.")
        graph = health["details"].get("graph", {})
        if graph.get("enabled"):
            st.info(
                f"🕸️ Graph: {graph['entities']} entities · "
                f"{graph['relations']} relations"
            )
        elif st.session_state.strategy in ("graph", "advanced"):
            st.caption("🕸️ Neo4j not connected — graph retrieval inactive.")

        guard = health["details"].get("guardrails", {})
        if guard.get("enabled") and st.session_state.use_guardrails:
            active = [
                n for n in ("injection", "moderation", "input_pii", "output_pii", "grounding")
                if guard.get(n)
            ]
            st.info(f"🛡️ Guardrails: {', '.join(active)}")
            if guard.get("pii_backend") != "presidio":
                st.caption("⚠️ Presidio unavailable — PII rails inactive.")
    else:
        st.error("Backend offline — start it:\n`uvicorn app.api.main:app --reload`")

    st.link_button("🔥 Open Logfire traces", LOGFIRE_URL, use_container_width=True)


# ---------- tabs ----------

tab_chat, tab_dash, tab_evals, tab_docs = st.tabs(
    ["💬 Chat", "📊 Dashboard", "📈 Evals", "📁 Documents"]
)


with tab_chat:
    # conversation history
    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and msg.get("data"):
                render_answer(msg["data"])
            else:
                st.markdown(msg["content"])

    # welcome screen with suggested prompts (only before the first message)
    if not st.session_state.messages and not st.session_state.get("pending_q"):
        st.markdown(
            "<div style='text-align:center; margin-top:6vh; opacity:0.9'>"
            "<div style='font-size:3.2rem'>🔎</div>"
            "<h2 style='margin:0.3rem 0'>Ask your documents</h2>"
            "<p style='color:gray; margin-top:0'>Grounded answers with citations, "
            "powered by your RAG pipeline.</p></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, suggestion in enumerate(SUGGESTIONS):
            if cols[i % 2].button(suggestion, use_container_width=True, key=f"sug_{i}"):
                st.session_state.pending_q = suggestion
                st.rerun()

    # accept a typed message or a clicked suggestion
    question = st.chat_input("Message your documents…")
    if not question and st.session_state.get("pending_q"):
        question = st.session_state.pop("pending_q")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            with st.spinner(f"Retrieving with `{st.session_state.strategy}`…"):
                data, error = post_query(question)

            if error:
                st.error(error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {error}"}
                )
            else:
                render_answer(data)
                st.session_state.messages.append(
                    {"role": "assistant", "content": data["answer"], "data": data}
                )
                top_score = data["sources"][0]["score"] if data["sources"] else 0.0
                st.session_state.history.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "question": question,
                        "strategy": data["retrieval_strategy"],
                        "latency_ms": data["latency_ms"],
                        "sources": len(data["sources"]),
                        "top_score": top_score,
                        "blocked": data.get("blocked", False),
                        "model": data["model"],
                    }
                )


with tab_dash:
    history = st.session_state.history
    if not history:
        st.info("No queries yet — ask something in the **Chat** tab and metrics will appear here.")
    else:
        df = pd.DataFrame(history)
        df.index = range(1, len(df) + 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queries", len(df))
        c2.metric("Avg latency", f"{df['latency_ms'].mean():,.0f} ms")
        c3.metric("P95 latency", f"{df['latency_ms'].quantile(0.95):,.0f} ms")
        c4.metric("Avg top score", f"{df['top_score'].mean():.3f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Latency per query (ms)**")
            st.line_chart(df["latency_ms"], color=ACCENT, height=260)
        with col_b:
            st.markdown("**Queries by strategy**")
            counts = df["strategy"].value_counts()
            st.bar_chart(counts, color=ACCENT, height=260, horizontal=True)

        st.markdown("**Query history**")
        st.dataframe(
            df[["time", "question", "strategy", "latency_ms", "sources", "top_score"]],
            use_container_width=True,
        )

        st.markdown(
            f"Session metrics only — full traces (per-stage spans, prompts, "
            f"token counts, costs) live in [Logfire]({LOGFIRE_URL})."
        )


EVAL_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _fmt(v) -> float | None:
    return round(v, 3) if isinstance(v, (int, float)) else None


with tab_evals:
    st.subheader("📈 RAG Evaluation — RAGAS")
    st.caption(
        "Run the selected strategies against the golden Q&A dataset and score them "
        "with RAGAS. All scores are 0–1, higher is better. Faithfulness = answer "
        "grounded in context; answer relevancy = answer addresses the question; "
        "context precision/recall = retrieval quality vs. the ground truth."
    )

    ds = get_eval_dataset()
    if not ds:
        st.error("Could not load the golden dataset — is the backend running?")
    else:
        st.markdown(f"**Golden dataset:** {ds['size']} questions · {ds['description']}")
        with st.expander("📋 View golden dataset"):
            st.dataframe(pd.DataFrame(ds["items"]), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        eval_strategies = c1.multiselect(
            "Strategies to compare", STRATEGIES, default=["hybrid", "advanced"]
        )
        eval_metrics = c2.multiselect(
            "Metrics", EVAL_METRICS, default=EVAL_METRICS
        )
        c3, c4, c5 = st.columns(3)
        num_q = c3.slider("Questions", 1, ds["size"], min(3, ds["size"]))
        eval_top_k = c4.slider("Top K", 1, 10, 5, key="eval_top_k")
        eval_rerank = c5.toggle("Rerank", value=True, key="eval_rerank")

        est = len(eval_strategies) * num_q * (1 + len(eval_metrics))
        st.caption(
            f"⏳ ~{est} LLM calls ({len(eval_strategies)} strategies × {num_q} "
            f"questions × (1 answer + {len(eval_metrics)} metrics)) — can take minutes."
        )

        if st.button(
            "▶️ Run evaluation",
            type="primary",
            disabled=not (eval_strategies and eval_metrics),
        ):
            with st.spinner("Running evaluation… (LLM-heavy, please wait)"):
                result, error = post_eval_run(
                    {
                        "strategies": eval_strategies,
                        "metrics": eval_metrics,
                        "num_questions": num_q,
                        "top_k": eval_top_k,
                        "use_rerank": eval_rerank,
                    }
                )
            if error:
                st.error(error)
            else:
                st.session_state.eval_results = result

        result = st.session_state.get("eval_results")
        if result:
            metrics = result["metrics"]
            st.divider()
            st.markdown(
                f"### Results — {result['num_questions']} questions · "
                f"{result['total_latency_ms'] / 1000:.1f}s total"
            )

            # aggregate comparison table (strategies × metrics)
            rows = []
            for r in result["results"]:
                row = {"strategy": r["strategy"]}
                row.update({m: _fmt(r["aggregate"].get(m)) for m in metrics})
                row["avg_latency_ms"] = r["avg_latency_ms"]
                rows.append(row)
            agg_df = pd.DataFrame(rows).set_index("strategy")

            st.markdown("**Aggregate scores** (mean per metric — best per column is the winner)")
            st.dataframe(agg_df, use_container_width=True)

            # grouped bar chart: x = metric, series = strategy
            st.markdown("**Per-metric comparison**")
            chart_df = agg_df[metrics].T  # metrics on x-axis, strategies as series
            st.bar_chart(chart_df, height=320, stack=False)

            # per-question breakdown
            st.markdown("**Per-question breakdown**")
            for r in result["results"]:
                with st.expander(f"🔍 {r['strategy']} — {r['num_questions']} questions"):
                    pq_rows = []
                    for pq in r["per_question"]:
                        prow = {"question": pq["question"], "ctx": pq["num_contexts"]}
                        prow.update({m: _fmt(pq["scores"].get(m)) for m in metrics})
                        pq_rows.append(prow)
                    st.dataframe(
                        pd.DataFrame(pq_rows), use_container_width=True, hide_index=True
                    )


def get_documents() -> list[dict]:
    try:
        r = requests.get(f"{backend()}/documents", headers=api_headers(), timeout=10)
        return r.json()["documents"] if r.ok else []
    except requests.RequestException:
        return []


def delete_document(source: str) -> tuple[dict | None, str | None]:
    try:
        r = requests.delete(
            f"{backend()}/documents/{source}", headers=api_headers(), timeout=60
        )
    except requests.RequestException as e:
        return None, str(e)
    if not r.ok:
        try:
            return None, r.json().get("detail", r.text)
        except ValueError:
            return None, r.text
    return r.json(), None


with tab_docs:
    health = get_health()
    if health:
        st.metric("Chunks indexed", health["documents_indexed"])

    docs = get_documents()
    if docs:
        st.markdown("**Indexed documents**")
        for d in docs:
            col_name, col_chunks, col_del = st.columns([5, 2, 2])
            col_name.markdown(f"📄 `{d['source']}`")
            col_chunks.caption(f"{d['chunks']} chunks")
            if col_del.button("🗑️ Delete", key=f"del_{d['source']}"):
                result, error = delete_document(d["source"])
                if error:
                    st.error(error)
                else:
                    st.toast(
                        f"Deleted {result['filename']} — "
                        f"{result['chunks_deleted']} chunks removed"
                    )
                    st.rerun()
        st.divider()

    uploads = st.file_uploader(
        "Upload documents (.pdf, .txt, .md, .docx)",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    if uploads and st.button("📥 Ingest", type="primary"):
        with st.spinner(f"Indexing {len(uploads)} file(s)…"):
            data, error = post_ingest(uploads)
        if error:
            st.error(error)
        else:
            st.success(
                f"Indexed {data['total_chunks']} chunks in {data['latency_ms']:.0f} ms"
            )
            st.table(
                pd.DataFrame(
                    [{"file": f["filename"], "chunks": f["chunks"]} for f in data["files"]]
                )
            )
