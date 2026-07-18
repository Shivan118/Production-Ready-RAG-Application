"""Phase 3 unit tests — extraction schemas, normalization, graceful degradation
without Neo4j. All offline."""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-0000000000")

from app.graph import client as graph_client  # noqa: E402
from app.graph.extractor import (  # noqa: E402
    Entity,
    GraphExtraction,
    Relation,
)
from app.graph.store import graph_stats  # noqa: E402
from app.models.schemas import QueryRequest, RetrievalStrategy  # noqa: E402


class TestExtractionSchemas:
    def test_entity_name_normalized(self):
        e = Entity(name="  The   Transformer  ", type="Technology")
        assert e.name == "The Transformer"

    def test_entity_unknown_type_falls_back(self):
        assert Entity(name="X", type="Alien").type == "Other"

    def test_relation_type_cypher_safe(self):
        r = Relation(source="A", relation="was proposed by!", target="B")
        assert r.relation == "WAS_PROPOSED_BY"

    def test_relation_empty_after_clean_falls_back(self):
        r = Relation(source="A", relation="!!!", target="B")
        assert r.relation == "RELATED_TO"

    def test_extraction_defaults_empty(self):
        x = GraphExtraction()
        assert x.entities == [] and x.relations == []


class TestGracefulDegradation:
    """`graph_disabled` (conftest) forces Neo4j off regardless of .env."""

    def test_driver_none_without_uri(self, graph_disabled):
        assert graph_client.get_driver() is None
        assert graph_client.is_enabled() is False

    def test_stats_disabled(self, graph_disabled):
        stats = graph_stats()
        assert stats["enabled"] is False
        assert stats["entities"] == 0

    def test_graph_retrieve_returns_empty(self, graph_disabled):
        from app.graph.retriever import retrieve

        assert retrieve("who proposed the transformer?") == []


class TestStrategyEnum:
    def test_graph_strategy_accepted(self):
        req = QueryRequest(question="what is X?", strategy="graph")
        assert req.strategy == RetrievalStrategy.GRAPH
