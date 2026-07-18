"""Graph RAG retrieval.

Flow: extract entities from the question → match Entity nodes → traverse
up to graph_max_hops of relations → return
  1. a synthetic "knowledge graph facts" chunk (the triples as text), and
  2. the real document chunks those entities are MENTIONED_IN
     (hydrated from Chroma so the generator sees full text).
"""

import logfire

from app.config import get_settings
from app.graph.client import get_driver
from app.graph.extractor import extract_graph
from app.ingestion.indexer import get_vectorstore
from app.models.schemas import SourceChunk

GRAPH_SOURCE_LABEL = "knowledge-graph"


def _query_entities(question: str) -> list[str]:
    """Cheap entity spotting: reuse the extractor on the question itself."""
    extraction = extract_graph(question)
    return [e.name for e in extraction.entities]


def _traverse(names: list[str]) -> tuple[list[str], list[str]]:
    """Return (facts, chunk_ids) for entities matching any of the names."""
    driver = get_driver()
    if driver is None or not names:
        return [], []

    settings = get_settings()
    with driver.session() as session:
        records = session.run(
            f"""
            UNWIND $names AS name
            MATCH (e:Entity)
            WHERE e.name_lower CONTAINS toLower(name)
            WITH DISTINCT e
            OPTIONAL MATCH path = (e)-[:REL*1..{settings.graph_max_hops}]-(other:Entity)
            WITH e, path
            LIMIT 200
            UNWIND (CASE WHEN path IS NULL THEN [NULL] ELSE relationships(path) END) AS r
            WITH e, r
            WHERE r IS NOT NULL
            WITH DISTINCT startNode(r) AS s, r, endNode(r) AS t
            RETURN s.name AS source, r.type AS relation, t.name AS target
            LIMIT 50
            """,
            names=names,
        )
        facts = [
            f"{rec['source']} —[{rec['relation']}]→ {rec['target']}"
            for rec in records
        ]

        chunk_records = session.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.name_lower CONTAINS toLower(name)
            RETURN DISTINCT c.chunk_id AS chunk_id
            LIMIT 20
            """,
            names=names,
        )
        chunk_ids = [rec["chunk_id"] for rec in chunk_records]
    return facts, chunk_ids


def _hydrate_chunks(chunk_ids: list[str], limit: int) -> list[SourceChunk]:
    if not chunk_ids:
        return []
    data = get_vectorstore()._collection.get(
        ids=chunk_ids[:limit], include=["documents", "metadatas"]
    )
    return [
        SourceChunk(
            content=doc,
            source=(meta or {}).get("source", "unknown"),
            chunk_id=(meta or {}).get("chunk_id", ""),
            score=0.5,  # graph-linked: rank via rerank/RRF downstream
        )
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]


def retrieve(question: str, top_k: int | None = None) -> list[SourceChunk]:
    settings = get_settings()
    k = top_k or settings.top_k

    with logfire.span("retrieve.graph", question=question) as span:
        if get_driver() is None:
            logfire.warn("graph_retrieval_skipped_neo4j_unavailable")
            span.set_attribute("chunks_returned", 0)
            return []

        names = _query_entities(question)
        span.set_attribute("query_entities", names)
        facts, chunk_ids = _traverse(names)
        span.set_attribute("facts_found", len(facts))
        span.set_attribute("linked_chunks", len(chunk_ids))

        chunks: list[SourceChunk] = []
        if facts:
            chunks.append(
                SourceChunk(
                    content="Knowledge graph facts:\n" + "\n".join(facts),
                    source=GRAPH_SOURCE_LABEL,
                    chunk_id="graph::facts",
                    score=1.0,
                )
            )
        chunks.extend(_hydrate_chunks(chunk_ids, limit=k))
        span.set_attribute("chunks_returned", len(chunks))
        return chunks[: k + 1]  # facts chunk + up to k document chunks
