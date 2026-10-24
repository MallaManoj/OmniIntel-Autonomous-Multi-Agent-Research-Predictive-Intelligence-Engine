"""Pydantic v2 API Request/Response Schemas and Strict Validation Contracts.

Enforces zero-trust boundary validation with:
  - ConfigDict(extra="forbid", str_strip_whitespace=True, validate_default=True)
  - Custom field validators for string sanitization and control-character filtering
  - Cross-field model validators for semantic consistency (e.g. source != target)
  - Comprehensive OpenAPI documentation with json_schema_extra examples
"""

import re
import time
from typing import Any, Dict, List, Literal, Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# -----------------------------------------------------------------------------
# Base Configuration
# -----------------------------------------------------------------------------

class OmniIntelBaseModel(BaseModel):
    """Base model enforcing strict Pydantic v2 enterprise validation standards.

    Attributes:
        extra="forbid": Rejects undeclared or typo fields to prevent parameter injection.
        str_strip_whitespace=True: Automatically trims leading/trailing whitespace.
        validate_default=True: Validates default values against constraints.
        populate_by_name=True: Allows aliased field population.
        from_attributes=True: Enables direct ORM/dataclass translation.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        populate_by_name=True,
        from_attributes=True,
    )


# -----------------------------------------------------------------------------
# Standardized Error Schemas
# -----------------------------------------------------------------------------

class ValidationErrorDetail(OmniIntelBaseModel):
    """Individual field validation failure item."""

    loc: List[Any] = Field(..., description="Location of the invalid attribute", example=["body", "query"])
    msg: str = Field(..., description="Human-readable validation failure reason")
    type: str = Field(..., description="Pydantic validation error type tag", example="string_too_short")


class APIErrorResponse(OmniIntelBaseModel):
    """Standardized error envelope returned on all non-2xx API responses."""

    error: bool = Field(default=True, description="Always true for error responses")
    status_code: int = Field(..., description="HTTP status code integer", example=422)
    message: str = Field(..., description="Summary explanation of the error")
    details: Optional[List[ValidationErrorDetail]] = Field(
        default=None, description="Detailed field-level failures if applicable"
    )
    correlation_id: str = Field(..., description="Trace correlation ID for distributed log auditing")


# -----------------------------------------------------------------------------
# Health & Telemetry Schemas
# -----------------------------------------------------------------------------

class HealthStatusResponse(OmniIntelBaseModel):
    """Liveness probe response model."""

    status: str = Field(default="healthy", description="Current service health status")
    service: str = Field(default="OmniIntel Autonomous Multi-Agent Engine")
    version: str = Field(default="1.0.0", description="SemVer release version")
    uptime_seconds: float = Field(..., ge=0.0, description="Service elapsed uptime in seconds")
    timestamp: float = Field(..., description="Epoch timestamp of check")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "service": "OmniIntel Autonomous Multi-Agent Engine",
                "version": "1.0.0",
                "uptime_seconds": 124.58,
                "timestamp": 1729780000.0,
            }
        }
    )


class ReadinessResponse(OmniIntelBaseModel):
    """Readiness probe response model verifying underlying subsystems."""

    status: str = Field(..., description="'ready' if subsystems are operational")
    is_ready: bool = Field(..., description="Boolean readiness state")
    knowledge_graph_entities: int = Field(default=0, ge=0)
    knowledge_graph_triples: int = Field(default=0, ge=0)
    vector_store_documents: int = Field(default=0, ge=0)
    reranker_active: bool = Field(default=False)
    orchestrator_active: bool = Field(default=False)
    timestamp: float = Field(...)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ready",
                "is_ready": True,
                "knowledge_graph_entities": 15,
                "knowledge_graph_triples": 22,
                "vector_store_documents": 8,
                "reranker_active": True,
                "orchestrator_active": True,
                "timestamp": 1729780000.0,
            }
        }
    )


class TelemetryResponse(OmniIntelBaseModel):
    """Real-time system telemetry and performance metrics."""

    total_events_dispatched: int = Field(default=0, ge=0)
    event_types_recorded: Dict[str, int] = Field(default_factory=dict)
    total_rag_queries: int = Field(default=0, ge=0)
    rag_latency_avg_ms: float = Field(default=0.0, ge=0.0)
    cross_encoder_cache_hits: int = Field(default=0, ge=0)
    cross_encoder_cache_misses: int = Field(default=0, ge=0)
    cache_hit_rate_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    uptime_seconds: float = Field(default=0.0, ge=0.0)


# -----------------------------------------------------------------------------
# Mission & Multi-Agent Planning Schemas
# -----------------------------------------------------------------------------

class MissionRequest(OmniIntelBaseModel):
    """Strategic intelligence mission submission payload."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="High-level research prompt or strategic intelligence directive",
    )
    max_replans: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum dynamic self-correction cycles permitted upon Critic rejection",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional domain parameters, ticker overrides, or seed constraints",
    )

    @field_validator("query")
    @classmethod
    def validate_query_content(cls, val: str) -> str:
        """Ensure query is not empty whitespace or malicious control characters."""
        stripped = val.strip()
        if len(stripped) < 3:
            raise ValueError("Query must contain at least 3 non-whitespace characters")
        if re.search(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", stripped):
            raise ValueError("Query contains prohibited control characters")
        return stripped

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Compare NVIDIA and TSMC in sovereign AI infrastructure and evaluate gross margins.",
                "max_replans": 2,
                "context": {"priority": "HIGH", "report_depth": "COMPREHENSIVE"},
            }
        }
    )


class MissionResponse(OmniIntelBaseModel):
    """Consolidated outcome of an end-to-end multi-agent research mission."""

    query: str
    success: bool
    plan_id: str
    total_latency_ms: float = Field(..., ge=0.0)
    total_stages_executed: int = Field(..., ge=0)
    replan_count: int = Field(..., ge=0)
    verdict: Optional[str] = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    target_entities: List[str] = Field(default_factory=list)
    dossier_markdown: str = Field(default="")
    errors: List[str] = Field(default_factory=list)
    created_at: float


class PlanOnlyRequest(OmniIntelBaseModel):
    """Request model for DAG planning decomposition without execution."""

    query: str = Field(..., min_length=3, max_length=2000)
    context: Optional[Dict[str, Any]] = None

    @field_validator("query")
    @classmethod
    def validate_query_content(cls, val: str) -> str:
        stripped = val.strip()
        if len(stripped) < 3:
            raise ValueError("Query must contain at least 3 non-whitespace characters")
        return stripped


class TaskDetailSchema(OmniIntelBaseModel):
    """Individual decomposed task within an execution plan."""

    name: str
    agent_role: str
    description: str
    dependencies: List[str] = Field(default_factory=list)
    expected_output: str


class PlanResponse(OmniIntelBaseModel):
    """Decomposed Kahn's algorithm DAG task schedule."""

    plan_id: str
    query: str
    complexity: str
    total_tasks: int = Field(..., ge=0)
    total_stages: int = Field(..., ge=0)
    topological_order: List[str]
    batches: List[List[str]]
    tasks: Dict[str, Dict[str, Any]]


# -----------------------------------------------------------------------------
# Knowledge Graph Schemas
# -----------------------------------------------------------------------------

class GraphStatsResponse(OmniIntelBaseModel):
    """High-level topology statistics of the Knowledge Graph."""

    num_entities: int = Field(..., ge=0)
    num_triples: int = Field(..., ge=0)
    density: float = Field(..., ge=0.0)
    entity_types: Dict[str, int]
    relation_types: Dict[str, int]


class EntityDetailResponse(OmniIntelBaseModel):
    """Detailed metadata and local adjacency list for an entity."""

    entity_id: str
    name: str
    entity_type: str
    aliases: List[str] = Field(default_factory=list)
    in_degree: int = Field(..., ge=0)
    out_degree: int = Field(..., ge=0)
    outgoing_relations: List[Dict[str, Any]] = Field(default_factory=list)
    incoming_relations: List[Dict[str, Any]] = Field(default_factory=list)


class PathQueryRequest(OmniIntelBaseModel):
    """Multi-hop path exploration query payload."""

    source_entity: str = Field(..., min_length=1, max_length=100, description="Source entity ID or canonical name")
    target_entity: str = Field(..., min_length=1, max_length=100, description="Target entity ID or canonical name")

    @model_validator(mode="after")
    def validate_distinct_endpoints(self) -> "PathQueryRequest":
        """Disallow trivial identical source and target self-loops."""
        if self.source_entity.strip().lower() == self.target_entity.strip().lower():
            raise ValueError("Source entity and target entity cannot be identical")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "source_entity": "nvidia",
                "target_entity": "asml",
            }
        }
    )


class PathQueryResponse(OmniIntelBaseModel):
    """Multi-hop path exploration response between source and target entities."""

    source: str
    target: str
    found: bool
    hop_count: int = Field(default=0, ge=0)
    aggregate_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    path_explanation: str = ""


class SubgraphRequest(OmniIntelBaseModel):
    """Egocentric subgraph extraction request."""

    root_entity_id: str = Field(..., min_length=1, description="Root entity anchor ID")
    max_hops: int = Field(default=2, ge=1, le=4, description="Maximum BFS expansion depth")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum edge confidence filter")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "root_entity_id": "nvidia",
                "max_hops": 2,
                "min_confidence": 0.75,
            }
        }
    )


class SubgraphResponse(OmniIntelBaseModel):
    """Egocentric subgraph topological output."""

    root_entity_id: str
    total_entities: int = Field(..., ge=0)
    total_triples: int = Field(..., ge=0)
    entities: List[Dict[str, Any]]
    triples: List[Dict[str, Any]]
    hop_depth: int = Field(..., ge=1)


# -----------------------------------------------------------------------------
# Hybrid Search & Reranking Schemas
# -----------------------------------------------------------------------------

class HybridSearchRequest(OmniIntelBaseModel):
    """Dual-channel RAG retrieval search request."""

    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Search text query to match across sparse, dense, and graph channels",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Final number of documents to return")
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Interpolation balance: 1.0 = Pure Vector/Dense, 0.0 = Pure Sparse/BM25",
    )
    fusion_mode: Literal["rrf", "linear", "auto"] = Field(
        default="rrf",
        description="Fusion algorithm: Reciprocal Rank Fusion ('rrf') or Linear Weighted Sum ('linear')",
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to execute Stage-2 Deep Neural Cross-Encoder Reranking",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, val: str) -> str:
        s = val.strip()
        if len(s) < 2:
            raise ValueError("Search query must contain at least 2 non-whitespace characters")
        return s

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "CoWoS advanced packaging capacity TSMC NVIDIA",
                "top_k": 3,
                "alpha": 0.5,
                "fusion_mode": "rrf",
                "use_reranker": True,
            }
        }
    )


class SearchResultItem(OmniIntelBaseModel):
    """Individual retrieved and scored document passage."""

    doc_id: str
    text: str
    score: float
    stage: str = Field(default="hybrid", description="Retrieval stage: 'hybrid' or 'cross_encoder'")
    source_channel: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HybridSearchResponse(OmniIntelBaseModel):
    """Consolidated retrieval result with provenance breakdown."""

    query: str
    total_hits: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0.0)
    cross_encoder_reranked: bool
    results: List[SearchResultItem]


class RerankCandidateItem(OmniIntelBaseModel):
    """Candidate passage submitted for neural cross-encoder scoring."""

    doc_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=5000)


class RerankRequest(OmniIntelBaseModel):
    """Dedicated Cross-Encoder batch scoring request."""

    query: str = Field(..., min_length=2, max_length=1000)
    candidates: List[RerankCandidateItem] = Field(..., min_length=1, max_length=100)
    return_top_k: Optional[int] = Field(default=None, ge=1, le=100)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Who is the premier foundry for NVIDIA chips?",
                "candidates": [
                    {"doc_id": "c1", "text": "TSMC manufactures Hopper and Blackwell GPUs."},
                    {"doc_id": "c2", "text": "ASML manufactures photolithography steppers."},
                ],
                "return_top_k": 2,
            }
        }
    )


class RerankResultItem(OmniIntelBaseModel):
    """Neural Cross-Encoder ranked passage."""

    doc_id: str
    text: str
    score: float
    rank: int = Field(..., ge=1)


class RerankResponse(OmniIntelBaseModel):
    """Response containing neural relevance scores."""

    query: str
    total_candidates: int = Field(..., ge=1)
    latency_ms: float = Field(..., ge=0.0)
    ranked_results: List[RerankResultItem]


# -----------------------------------------------------------------------------
# Entity Scanning & Recognition Schemas
# -----------------------------------------------------------------------------

class EntityScanRequest(OmniIntelBaseModel):
    """Prefix Trie entity scanner request."""

    text: str = Field(..., min_length=1, max_length=20000, description="Raw text string to scan for entities")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "NVIDIA depends heavily on TSMC and ASML for advanced AI silicon production."
            }
        }
    )


class EntityMatchItem(OmniIntelBaseModel):
    """Individual entity recognized in text via Prefix Trie."""

    matched_text: str
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    payload_value: Optional[str] = None


class EntityScanResponse(OmniIntelBaseModel):
    """Prefix Trie entity extraction response."""

    total_matches: int = Field(..., ge=0)
    scan_latency_ms: float = Field(..., ge=0.0)
    matches: List[EntityMatchItem]
