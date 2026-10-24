"""Entity-Relationship schema definitions for Knowledge Graph and Agent reasoning.

Built with Pydantic v2 models for strict type-safety, input normalization,
confidence scoring, and seamless JSON/Cypher serialization.
"""

from enum import Enum
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntityType(str, Enum):
    """Categorical classification of Knowledge Graph entities."""

    ORGANIZATION = "ORGANIZATION"
    PERSON = "PERSON"
    TECHNOLOGY = "TECHNOLOGY"
    FINANCIAL_METRIC = "FINANCIAL_METRIC"
    PRODUCT = "PRODUCT"
    LOCATION = "LOCATION"
    REGULATORY_BODY = "REGULATORY_BODY"
    CONCEPT = "CONCEPT"


class RelationType(str, Enum):
    """Semantic relation types defining directed graph edges."""

    ACQUIRED = "ACQUIRED"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    DEVELOPS = "DEVELOPS"
    COMPETES_WITH = "COMPETES_WITH"
    INVESTS_IN = "INVESTS_IN"
    PARTNERS_WITH = "PARTNERS_WITH"
    REGULATES = "REGULATES"
    REPORTED_EARNINGS = "REPORTED_EARNINGS"
    AFFECTS = "AFFECTS"
    LEADS = "LEADS"
    LOCATED_IN = "LOCATED_IN"


def slugify_id(text: str) -> str:
    """Normalize text into a canonical lowercase entity ID slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")


class Entity(BaseModel):
    """Represents a node in the Knowledge Graph with properties and aliases."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    entity_id: str = Field(..., description="Unique slug identifier (e.g., 'nvidia_corp')")
    name: str = Field(..., description="Canonical display name (e.g., 'NVIDIA Corporation')")
    entity_type: EntityType = Field(..., description="Category enum of the entity")
    description: Optional[str] = Field(default=None, description="Natural language description")
    aliases: List[str] = Field(default_factory=list, description="Synonyms/acronyms for text scanning")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary domain attributes")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        slug = slugify_id(v)
        if not slug:
            raise ValueError("entity_id must contain at least one alphanumeric character")
        return slug

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name cannot be empty")
        return v

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, v: List[str]) -> List[str]:
        cleaned: Set[str] = set()
        for alias in v:
            a = alias.strip().lower()
            if a:
                cleaned.add(a)
        return sorted(list(cleaned))

    def all_surface_forms(self) -> List[str]:
        """Return canonical name plus all aliases for Prefix Trie dictionary scanning."""
        forms = {self.name.lower()}
        forms.update(self.aliases)
        return sorted(list(forms))


class Relation(BaseModel):
    """Represents a directed or undirected edge connecting two entities."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    relation_id: str = Field(..., description="Unique edge identifier")
    relation_type: Union[RelationType, str] = Field(..., description="Type of semantic relation")
    weight: float = Field(default=1.0, gt=0.0, description="Edge weight for traversal algorithms")
    directed: bool = Field(default=True, description="Whether relation is directed (subject -> object)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with relation")

    @field_validator("relation_id")
    @classmethod
    def validate_relation_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("relation_id cannot be empty")
        return v

    @property
    def relation_name(self) -> str:
        """String representation of the relation type."""
        if isinstance(self.relation_type, Enum):
            return self.relation_type.value
        return str(self.relation_type)


class FactTriple(BaseModel):
    """Atomic factual statement (Subject, Predicate, Object) with provenance tracking."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    triple_id: str = Field(..., description="Unique identifier for the fact triple")
    subject: Entity = Field(..., description="Source entity node")
    relation: Relation = Field(..., description="Connecting predicate/edge")
    object: Entity = Field(..., description="Target entity node")
    evidence_text: Optional[str] = Field(default=None, description="Original source passage proving the fact")
    source_doc_id: Optional[str] = Field(default=None, description="Provenance document identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Belief certainty score")
    created_at: float = Field(default_factory=time.time, description="Unix timestamp of creation")

    @property
    def subject_id(self) -> str:
        return self.subject.entity_id

    @property
    def object_id(self) -> str:
        return self.object.entity_id

    def to_tuple(self) -> Tuple[str, str, str]:
        """Return canonical 3-tuple (subject_name, relation_name, object_name)."""
        return (self.subject.name, self.relation.relation_name, self.object.name)

    def to_cypher(self) -> str:
        """Format as a Neo4j Cypher-compatible CREATE pattern."""
        dir_arrow = "->" if self.relation.directed else "-"
        return (
            f"(:{self.subject.entity_type.value} {{name: '{self.subject.name}', id: '{self.subject.entity_id}'}})"
            f"-[:{self.relation.relation_name} {{weight: {self.relation.weight}}}]"
            f"{dir_arrow}"
            f"(:{self.object.entity_type.value} {{name: '{self.object.name}', id: '{self.object.entity_id}'}})"
        )

    def to_text(self) -> str:
        """Render factual triple as a concise human-readable sentence."""
        return f"{self.subject.name} [{self.relation.relation_name}] {self.object.name} (confidence: {self.confidence:.2f})"


class GraphQuery(BaseModel):
    """Parameter specification for associative entity graph retrieval."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    source_entity_id: str = Field(..., description="Root entity slug to expand from")
    target_entity_id: Optional[str] = Field(default=None, description="Optional target for pathfinding")
    max_hops: int = Field(default=2, ge=1, le=10, description="Maximum traversal depth")
    relation_types: Optional[List[Union[RelationType, str]]] = Field(
        default=None, description="Allowlist of relation types to follow"
    )
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum edge confidence")


class SubgraphResult(BaseModel):
    """Structured knowledge payload returned by graph traversal for Agent synthesis."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    root_entity: Entity
    entities: List[Entity] = Field(default_factory=list)
    triples: List[FactTriple] = Field(default_factory=list)
    paths: List[List[str]] = Field(default_factory=list, description="Sequences of entity IDs traversed")

    def summary(self) -> str:
        """Generate structured text summary of the retrieved subgraph."""
        lines = [f"Subgraph rooted at '{self.root_entity.name}' ({len(self.entities)} entities, {len(self.triples)} facts):"]
        for triple in self.triples:
            lines.append(f"  • {triple.to_text()}")
        return "\n".join(lines)
