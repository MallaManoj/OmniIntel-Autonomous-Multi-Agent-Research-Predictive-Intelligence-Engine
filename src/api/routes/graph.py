"""Knowledge Graph inspection and multi-hop reasoning API routes."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.schemas import (
    EntityDetailResponse,
    GraphStatsResponse,
    PathQueryResponse,
)
from src.api.state import AppState, get_app_state
from src.knowledge_graph.traversal import GraphTraversalEngine

router = APIRouter(prefix="/api/v1/graph", tags=["Knowledge Graph"])


@router.get("/stats", response_model=GraphStatsResponse, summary="Knowledge Graph Global Stats")
async def get_graph_stats(state: AppState = Depends(get_app_state)) -> GraphStatsResponse:
    """Retrieve global topology metrics, node/edge counts, and type distributions."""
    if not state.is_ready or state.knowledge_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge graph is initializing.",
        )

    kg = state.knowledge_graph
    triples = kg.get_all_triples()

    entity_types: dict[str, int] = {}
    for e in kg.get_all_entities():
        t = e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type)
        entity_types[t] = entity_types.get(t, 0) + 1

    relation_types: dict[str, int] = {}
    for trip in triples:
        rt = trip.relation.relation_type
        r_str = rt.value if hasattr(rt, "value") else str(rt)
        relation_types[r_str] = relation_types.get(r_str, 0) + 1

    return GraphStatsResponse(
        num_entities=kg.num_entities,
        num_triples=kg.num_triples,
        density=round(kg.density, 4),
        entity_types=entity_types,
        relation_types=relation_types,
    )


@router.get("/entities", summary="List Entities")
async def list_entities(
    entity_type: Optional[str] = Query(None, description="Filter by EntityType (e.g. ORGANIZATION, PRODUCT)"),
    state: AppState = Depends(get_app_state),
) -> List[dict]:
    """Retrieve registered entities with optional type filtering."""
    if not state.is_ready or state.knowledge_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge graph is initializing.",
        )

    kg = state.knowledge_graph
    entities = kg.get_all_entities()

    if entity_type:
        filtered = [
            e for e in entities
            if (e.entity_type.value if hasattr(e.entity_type, "value") else str(e.entity_type)).upper() == entity_type.upper()
        ]
    else:
        filtered = entities

    return [e.model_dump() for e in filtered]


@router.get("/entity/{entity_id}", response_model=EntityDetailResponse, summary="Entity Details & Adjacency")
async def get_entity_details(
    entity_id: str,
    state: AppState = Depends(get_app_state),
) -> EntityDetailResponse:
    """Retrieve focal entity details along with incoming and outgoing relations."""
    if not state.is_ready or state.knowledge_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge graph is initializing.",
        )

    kg = state.knowledge_graph
    entity = kg.resolve_entity(entity_id) or kg.get_entity(entity_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Entity '{entity_id}' not found in knowledge graph.",
        )

    outgoing = [
        {
            "target": neighbor.name,
            "target_id": neighbor.entity_id,
            "relation": edge.relation.relation_type if not hasattr(edge.relation.relation_type, "value") else edge.relation.relation_type.value,
            "confidence": edge.confidence,
        }
        for neighbor, edge in kg.get_neighbors(entity.entity_id, direction="out")
    ]

    incoming = [
        {
            "source": neighbor.name,
            "source_id": neighbor.entity_id,
            "relation": edge.relation.relation_type if not hasattr(edge.relation.relation_type, "value") else edge.relation.relation_type.value,
            "confidence": edge.confidence,
        }
        for neighbor, edge in kg.get_neighbors(entity.entity_id, direction="in")
    ]

    return EntityDetailResponse(
        entity_id=entity.entity_id,
        name=entity.name,
        entity_type=entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type),
        aliases=entity.aliases,
        in_degree=kg.in_degree(entity.entity_id),
        out_degree=kg.out_degree(entity.entity_id),
        outgoing_relations=outgoing,
        incoming_relations=incoming,
    )


@router.get("/path", response_model=PathQueryResponse, summary="Dijkstra Multi-Hop Path Reasoning")
async def find_reasoning_path(
    source: str = Query(..., description="Source entity name or alias"),
    target: str = Query(..., description="Target entity name or alias"),
    state: AppState = Depends(get_app_state),
) -> PathQueryResponse:
    """Find the most reliable path between two entities via Dijkstra's algorithm."""
    if not state.is_ready or state.knowledge_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge graph is initializing.",
        )

    kg = state.knowledge_graph
    src_ent = kg.resolve_entity(source) or kg.get_entity(source)
    tgt_ent = kg.resolve_entity(target) or kg.get_entity(target)

    if not src_ent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source entity '{source}' not recognized.",
        )
    if not tgt_ent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target entity '{target}' not recognized.",
        )

    traversal = GraphTraversalEngine(kg)
    path_res = traversal.find_most_reliable_path(src_ent.entity_id, tgt_ent.entity_id)

    return PathQueryResponse(
        source=src_ent.name,
        target=tgt_ent.name,
        found=path_res.found,
        hop_count=path_res.hop_count,
        aggregate_confidence=round(path_res.confidence_score, 4),
        path_explanation=path_res.to_explanation() if path_res.found else f"No path found between {src_ent.name} and {tgt_ent.name}.",
    )
