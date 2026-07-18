"""RAG End-to-End — Streamlit chatbot + observability dashboard.

Run:  streamlit run frontend/app.py
Requires the FastAPI backend:  uvicorn app.api.main:app --reload
"""

from datetime import datetime

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="RAG End-to-End", page_icon="🔎", layout="wide")

ACCENT = "#2a78d6"  # categorical slot 1 (validated palette)
LOGFIRE_URL = "https://logfire-us.pydantic.dev/shivan3446/starter-project"
STRATEGIES = ["advanced", "hybrid", "dense", "multi_query", "hyde", "self_query"]

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
    else:
        st.error("Backend offline — start it:\n`uvicorn app.api.main:app --reload`")

    st.link_button("🔥 Open Logfire traces", LOGFIRE_URL, use_container_width=True)


# ---------- tabs ----------

tab_chat, tab_dash, tab_docs = st.tabs(["💬 Chat", "📊 Dashboard", "📁 Documents"])


with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                m = msg["meta"]
                st.caption(
                    f"`{m['strategy']}` · {m['latency_ms']:.0f} ms · "
                    f"{m['num_sources']} sources · top score {m['top_score']}"
                )
                with st.expander("📄 Sources"):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**[{i}] {s['source']}** — score `{s['score']}`\n\n"
                            f"> {s['content'][:400]}{'…' if len(s['content']) > 400 else ''}"
                        )

    if question := st.chat_input("Ask a question about your documents…"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(f"Retrieving with `{st.session_state.strategy}`…"):
                data, error = post_query(question)

            if error:
                st.error(error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {error}"}
                )
            else:
                top_score = data["sources"][0]["score"] if data["sources"] else 0.0
                meta = {
                    "strategy": data["retrieval_strategy"],
                    "latency_ms": data["latency_ms"],
                    "num_sources": len(data["sources"]),
                    "top_score": top_score,
                }
                st.markdown(data["answer"])
                st.caption(
                    f"`{meta['strategy']}` · {meta['latency_ms']:.0f} ms · "
                    f"{meta['num_sources']} sources · top score {top_score}"
                )
                with st.expander("📄 Sources"):
                    for i, s in enumerate(data["sources"], 1):
                        st.markdown(
                            f"**[{i}] {s['source']}** — score `{s['score']}`\n\n"
                            f"> {s['content'][:400]}{'…' if len(s['content']) > 400 else ''}"
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "meta": meta,
                        "sources": data["sources"],
                    }
                )
                st.session_state.history.append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "question": question,
                        "strategy": data["retrieval_strategy"],
                        "latency_ms": data["latency_ms"],
                        "sources": len(data["sources"]),
                        "top_score": top_score,
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


with tab_docs:
    health = get_health()
    if health:
        st.metric("Chunks indexed", health["documents_indexed"])

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
