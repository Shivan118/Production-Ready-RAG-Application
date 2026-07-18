"""Persist extracted entities/relations in Neo4j.

Model:  (:Entity {name, type}) -[:REL {type}]-> (:Entity)
        (:Entity) -[:MENTIONED_IN]-> (:Chunk {chunk_id, source})

MERGE everywhere → ingesting the same file twice never duplicates graph data.
Relationship types are stored as a property (not dynamic Cypher types) so
queries stay injection-safe.
"""

import logfire

from app.graph.client import get_driver
from app.graph.extractor import GraphExtraction

_CONSTRAINTS = [
    "CREATE CONSTRAINT entity_name IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
]

_ensured = False


def _ensure_constraints() -> None:
    global _ensured
    if _ensured:
        return
    driver = get_driver()
    if driver is None:
        return
    with driver.session() as session:
        for stmt in _CONSTRAINTS:
            session.run(stmt)
    _ensured = True


def store_extraction(
    extraction: GraphExtraction, chunk_id: str, source: str
) -> None:
    driver = get_driver()
    if driver is None or (not extraction.entities and not extraction.relations):
        return

    _ensure_constraints()
    with logfire.span(
        "graph.store",
        chunk_id=chunk_id,
        entities=len(extraction.entities),
        relations=len(extraction.relations),
    ):
        with driver.session() as session:
            session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.source = $source
                WITH c
                UNWIND $entities AS ent
                MERGE (e:Entity {name_lower: toLower(ent.name)})
                SET e.name = ent.name, e.type = ent.type
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                chunk_id=chunk_id,
                source=source,
                entities=[e.model_dump() for e in extraction.entities],
            )
            if extraction.relations:
                session.run(
                    """
                    UNWIND $relations AS rel
                    MATCH (s:Entity {name_lower: toLower(rel.source)})
                    MATCH (t:Entity {name_lower: toLower(rel.target)})
                    MERGE (s)-[r:REL {type: rel.relation}]->(t)
                    """,
                    relations=[r.model_dump() for r in extraction.relations],
                )


def delete_source_graph(source: str) -> int:
    """Remove a file's Chunk nodes and any entities left orphaned by it."""
    driver = get_driver()
    if driver is None:
        return 0
    with logfire.span("graph.delete_source", source=source) as span:
        with driver.session() as session:
            record = session.run(
                """
                MATCH (c:Chunk {source: $source})
                DETACH DELETE c
                RETURN count(c) AS deleted
                """,
                source=source,
            ).single()
            session.run(
                """
                MATCH (e:Entity)
                WHERE NOT (e)-[:MENTIONED_IN]->(:Chunk)
                DETACH DELETE e
                """
            )
            deleted = record["deleted"] if record else 0
            span.set_attribute("chunks_deleted", deleted)
            return deleted


def graph_stats() -> dict:
    driver = get_driver()
    if driver is None:
        return {"enabled": False, "entities": 0, "relations": 0, "chunks": 0}
    with driver.session() as session:
        record = session.run(
            """
            MATCH (e:Entity)
            OPTIONAL MATCH (:Entity)-[r:REL]->(:Entity)
            OPTIONAL MATCH (c:Chunk)
            RETURN count(DISTINCT e) AS entities,
                   count(DISTINCT r) AS relations,
                   count(DISTINCT c) AS chunks
            """
        ).single()
        return {
            "enabled": True,
            "entities": record["entities"],
            "relations": record["relations"],
            "chunks": record["chunks"],
        }
