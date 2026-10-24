"""Health check, readiness, and runtime telemetry endpoints."""

from collections import Counter
import time
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import HealthStatusResponse, ReadinessResponse, TelemetryResponse
from src.api.state import AppState, get_app_state

router = APIRouter(prefix="/health", tags=["Health & Monitoring"])


@router.get("", response_model=HealthStatusResponse, summary="Service Liveness Probe")
async def get_health(state: AppState = Depends(get_app_state)) -> HealthStatusResponse:
    """Return process liveness, uptime, and service identification."""
    return HealthStatusResponse(
        status="healthy",
        service="OmniIntel Autonomous Multi-Agent Engine",
        version=state.version,
        uptime_seconds=round(state.uptime_seconds, 2),
        timestamp=time.time(),
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Subsystem Readiness Probe")
async def get_readiness(state: AppState = Depends(get_app_state)) -> ReadinessResponse:
    """Verify that Knowledge Graph, Hybrid RAG, and Master Orchestrator are initialized."""
    if not state.is_ready or state.orchestrator is None or state.knowledge_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Subsystems not yet initialized or undergoing background warmup.",
        )

    return ReadinessResponse(
        status="ready",
        is_ready=True,
        knowledge_graph_entities=state.knowledge_graph.num_entities,
        knowledge_graph_triples=state.knowledge_graph.num_triples,
        vector_store_documents=state.pipeline.hybrid_engine.vector_store.count if state.pipeline else 0,
        reranker_active=(state.pipeline.reranker is not None) if state.pipeline else False,
        orchestrator_active=True,
        timestamp=time.time(),
    )


@router.get("/telemetry", response_model=TelemetryResponse, summary="Live Performance & Telemetry")
async def get_telemetry(state: AppState = Depends(get_app_state)) -> TelemetryResponse:
    """Aggregate live event statistics, RAG latencies, and neural cache metrics."""
    # 1. EventBus stats
    history = state.event_bus.history
    event_counts = dict(Counter(e.event_type for e in history))

    # 2. RAG Telemetry
    total_rag_queries = 0
    rag_latency_avg_ms = 0.0
    if state.pipeline:
        t_rag = state.pipeline.telemetry
        total_rag_queries = t_rag.get("total_queries", 0)
        rag_latency_avg_ms = t_rag.get("avg_latency_ms", 0.0)

    # 3. Neural Cross-Encoder Reranker Cache Telemetry
    cache_hits = 0
    cache_misses = 0
    cache_hit_rate = 0.0
    if state.pipeline and state.pipeline.reranker:
        t_rerank = state.pipeline.reranker.telemetry
        cache_hits = t_rerank.get("cache_hits", 0)
        cache_misses = t_rerank.get("cache_misses", 0)
        total_cache_queries = cache_hits + cache_misses
        if total_cache_queries > 0:
            cache_hit_rate = (cache_hits / total_cache_queries) * 100.0

    return TelemetryResponse(
        total_events_dispatched=len(history),
        event_types_recorded=event_counts,
        total_rag_queries=total_rag_queries,
        rag_latency_avg_ms=round(rag_latency_avg_ms, 2),
        cross_encoder_cache_hits=cache_hits,
        cross_encoder_cache_misses=cache_misses,
        cache_hit_rate_pct=round(cache_hit_rate, 2),
        uptime_seconds=round(state.uptime_seconds, 2),
    )
