"""LLM entity/relation extraction — turns chunk text into graph triples.

Uses structured output so the result is always schema-valid; entity names
are normalized so "The Transformer" and "transformer" merge into one node.
"""

import logfire
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

from app.generation.generator import _llm

ENTITY_TYPES = ["Person", "Organization", "Technology", "Concept", "Location", "Other"]


class Entity(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    type: str = Field("Other", description=f"One of {ENTITY_TYPES}")

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return " ".join(v.strip().split())

    @field_validator("type")
    @classmethod
    def valid_type(cls, v: str) -> str:
        return v if v in ENTITY_TYPES else "Other"


class Relation(BaseModel):
    source: str = Field(..., min_length=1)
    relation: str = Field(..., min_length=1, max_length=60)
    target: str = Field(..., min_length=1)

    @field_validator("relation")
    @classmethod
    def normalize_relation(cls, v: str) -> str:
        # Cypher-safe relationship type: UPPER_SNAKE_CASE
        cleaned = "".join(c if c.isalnum() else "_" for c in v.strip().upper())
        return cleaned.strip("_")[:60] or "RELATED_TO"


class GraphExtraction(BaseModel):
    entities: list[Entity] = []
    relations: list[Relation] = []


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract the key entities and factual relations from the text.\n"
            f"- Entity types: {', '.join(ENTITY_TYPES)}\n"
            "- Max 10 entities and 10 relations; only ones clearly stated in the text.\n"
            "- relation should be a short verb phrase, e.g. 'PROPOSED', 'USES', "
            "'WORKS_AT', 'PART_OF'.\n"
            "- source/target of relations must be entity names from your entity list.",
        ),
        ("user", "{text}"),
    ]
)


def extract_graph(text: str) -> GraphExtraction:
    with logfire.span("graph.extract", text_length=len(text)) as span:
        chain = _PROMPT | _llm().with_structured_output(GraphExtraction)
        try:
            result: GraphExtraction = chain.invoke({"text": text})
        except Exception as e:
            logfire.warn("graph_extraction_failed", error=str(e))
            return GraphExtraction()

        # keep only relations whose endpoints are known entities
        names = {e.name.lower() for e in result.entities}
        result.relations = [
            r
            for r in result.relations
            if r.source.lower() in names and r.target.lower() in names
        ]
        span.set_attribute("entities", len(result.entities))
        span.set_attribute("relations", len(result.relations))
        return result
