"""Pydantic v2 schemas for OmniIntel REST endpoints and WebSocket payloads."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Health & Telemetry Schemas
# -----------------------------------------------------------------------------

class HealthStatusResponse(BaseModel):
    """Liveness probe response model."""

    status: str = Field(default="healthy", description="Current service health status")
    service: str = Field(default="OmniIntel Autonomous Multi-Agent Engine")
    version: str = Field(default="1.0.0", description="SemVer release version")
    uptime_seconds: float = Field(..., description="Service elapsed uptime in seconds")
    timestamp: float = Field(..., description="Epoch timestamp of check")


class ReadinessResponse(BaseModel):
    """Readiness probe response model verifying underlying subsystems."""

    status: str = Field(..., description="'ready' if subsystems are operational")
    is_ready: bool = Field(..., description="Boolean readiness state")
    knowledge_graph_entities: int = Field(default=0)
    knowledge_graph_triples: int = Field(default=0)
    vector_store_documents: int = Field(default=0)
    reranker_active: bool = Field(default=False)
    orchestrator_active: bool = Field(default=False)
    timestamp: float = Field(...)


class TelemetryResponse(BaseModel):
    """Real-time system telemetry and performance metrics."""

    total_events_dispatched: int = Field(default=0)
    event_types_recorded: Dict[str, int] = Field(default_factory=dict)
    total_rag_queries: int = Field(default=0)
    rag_latency_avg_ms: float = Field(default=0.0)
    cross_encoder_cache_hits: int = Field(default=0)
    cross_encoder_cache_misses: int = Field(default=0)
    cache_hit_rate_pct: float = Field(default=0.0)
    uptime_seconds: float = Field(default=0.0)


# -----------------------------------------------------------------------------
# Mission & Planning Schemas
# -----------------------------------------------------------------------------

class MissionRequest(BaseModel):
    """Strategic intelligence mission submission payload."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="High-level research prompt or strategic intelligence directive",
        example="Compare NVIDIA and TSMC in sovereign AI infrastructure and evaluate gross margins.",
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


class MissionResponse(BaseModel):
    """Consolidated outcome of an end-to-end multi-agent research mission."""

    query: str
    success: bool
    plan_id: str
    total_latency_ms: float
    total_stages_executed: int
    replan_count: int
    verdict: Optional[str] = None
    confidence_score: float = 0.0
    target_entities: List[str] = Field(default_factory=list)
    dossier_markdown: str = Field(default="")
    errors: List[str] = Field(default_factory=list)
    created_at: float


class PlanOnlyRequest(BaseModel):
    """Request model for DAG planning decomposition without execution."""

    query: str = Field(..., min_length=3, max_length=2000)
    context: Optional[Dict[str, Any]] = None


class PlanResponse(BaseModel):
    """Decomposed Kahn's algorithm DAG task schedule."""

    plan_id: str
    query: str
    complexity: str
    total_tasks: int
    total_stages: int
    topological_order: List[str]
    batches: List[List[str]]
    tasks: Dict[str, Dict[str, Any]]


# -----------------------------------------------------------------------------
# Knowledge Graph Schemas
# -----------------------------------------------------------------------------

class GraphStatsResponse(BaseModel):
    """High-level topology statistics of the Knowledge Graph."""

    num_entities: int
    num_triples: int
    density: float
    entity_types: Dict[str, int]
    relation_types: Dict[str, int]


class EntityDetailResponse(BaseModel):
    """Detailed metadata and local adjacency list for an entity."""

    entity_id: str
    name: str
    entity_type: str
    aliases: List[str] = Field(default_factory=list)
    in_degree: int
    out_degree: int
    outgoing_relations: List[Dict[str, Any]] = Field(default_factory=list)
    incoming_relations: List[Dict[str, Any]] = Field(default_factory=list)


class PathQueryResponse(BaseModel):
    """Multi-hop path exploration response between source and target entities."""

    source: str
    target: str
    found: bool
    hop_count: int = 0
    aggregate_confidence: float = 0.0
    path_explanation: str = ""
