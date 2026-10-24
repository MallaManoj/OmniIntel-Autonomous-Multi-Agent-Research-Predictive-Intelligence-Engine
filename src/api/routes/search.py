"""Search, Neural Reranking, and Entity Recognition API endpoints."""

import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.schemas import (
    EntityMatchItem,
    EntityScanRequest,
    EntityScanResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    RerankRequest,
    RerankResponse,
    RerankResultItem,
    SearchResultItem,
)
from src.api.state import AppState, get_app_state
from src.knowledge_graph.hybrid_search import HybridSearchConfig
from src.knowledge_graph.pipeline import RetrievalPipelineConfig

router = APIRouter(prefix="/api/v1/search", tags=["Hybrid RAG & Neural Search"])


@router.post("/hybrid", response_model=HybridSearchResponse, summary="Dual-Channel Hybrid Retrieval with Cross-Encoder")
async def execute_hybrid_search(
    request: HybridSearchRequest,
    state: AppState = Depends(get_app_state),
) -> HybridSearchResponse:
    """Execute dual-channel retrieval (BM25 sparse + dense vector embeddings) with Cross-Encoder reranking."""
    if not state.is_ready or state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval pipeline is initializing.",
        )

    start_t = time.perf_counter()

    # Configure pipeline parameters
    dense_w = request.alpha
    sparse_w = 1.0 - request.alpha
    graph_w = 0.2
    total_w = dense_w + sparse_w + graph_w

    h_config = HybridSearchConfig(
        dense_weight=round(dense_w / total_w, 2),
        sparse_weight=round(sparse_w / total_w, 2),
        graph_weight=round(graph_w / total_w, 2),
        fusion_method=request.fusion_mode if request.fusion_mode != "auto" else "rrf",
        top_k=request.top_k * 2,  # Expand Stage-1 recall pool for reranker
    )
    p_config = RetrievalPipelineConfig(
        hybrid_config=h_config,
        candidate_pool_size=request.top_k * 2,
        final_top_k=request.top_k,
        enable_reranker=request.use_reranker,
    )

    output = state.pipeline.retrieve(request.query, config=p_config)
    total_ms = (time.perf_counter() - start_t) * 1000.0

    items = [
        SearchResultItem(
            doc_id=item.doc_id,
            text=item.text,
            score=round(float(item.final_score), 4),
            stage="cross_encoder" if item.stage2_score is not None else "hybrid",
            source_channel="hybrid_rrf",
            metadata=item.metadata,
        )
        for item in output.items
    ]

    return HybridSearchResponse(
        query=request.query,
        total_hits=len(items),
        latency_ms=round(total_ms, 2),
        cross_encoder_reranked=output.total_reranked > 0,
        results=items,
    )


@router.post("/rerank", response_model=RerankResponse, summary="Cross-Encoder Neural Batch Reranking")
async def execute_neural_rerank(
    request: RerankRequest,
    state: AppState = Depends(get_app_state),
) -> RerankResponse:
    """Score candidate passages using Deep Transformer Cross-Encoder with LRU Cache."""
    if not state.is_ready or state.pipeline is None or state.pipeline.reranker is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neural Cross-Encoder reranker is not active.",
        )

    start_t = time.perf_counter()
    reranker = state.pipeline.reranker
    texts = [cand.text for cand in request.candidates]

    rerank_results = reranker.rerank(
        query=request.query,
        documents=texts,
        top_k=request.return_top_k or len(texts),
    )
    total_ms = (time.perf_counter() - start_t) * 1000.0

    ranked_items: List[RerankResultItem] = []
    for rank_idx, res in enumerate(rerank_results, start=1):
        orig_cand = request.candidates[res.original_index]
        ranked_items.append(
            RerankResultItem(
                doc_id=orig_cand.doc_id,
                text=res.document,
                score=round(float(res.score), 4),
                rank=rank_idx,
            )
        )

    return RerankResponse(
        query=request.query,
        total_candidates=len(request.candidates),
        latency_ms=round(total_ms, 2),
        ranked_results=ranked_items,
    )


@router.post("/scan-entities", response_model=EntityScanResponse, summary="Prefix Trie Fast Entity Scanner")
async def scan_entities_in_text(
    request: EntityScanRequest,
    state: AppState = Depends(get_app_state),
) -> EntityScanResponse:
    """Scan arbitrary input text for recognized financial and corporate entities in O(N) using Prefix Trie."""
    if not state.is_ready or state.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Entity indexing trie is initializing.",
        )

    start_t = time.perf_counter()
    trie = state.pipeline.hybrid_engine.trie

    raw_matches = trie.extract_entities(request.text)
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    matches = [
        EntityMatchItem(
            matched_text=m[0],
            start_char=m[1],
            end_char=m[2],
            payload_value=str(m[3]) if m[3] is not None else None,
        )
        for m in raw_matches
    ]

    return EntityScanResponse(
        total_matches=len(matches),
        scan_latency_ms=round(latency_ms, 3),
        matches=matches,
    )
