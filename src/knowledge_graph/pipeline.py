"""Integrated End-to-End Hybrid Retrieval and Cross-Encoder Reranking Pipeline.

Combines Stage 1 high-recall multi-channel Hybrid Search (Dense vectors + Sparse TF-IDF +
Knowledge Graph expansion) with Stage 2 high-precision neural Transformer Cross-Encoder
reranking under torch.inference_mode() and O(1) LRU caching.
Produces grounded citation contexts with graph reasoning explanations for Agent synthesis.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from src.knowledge_graph.chunker import DocumentChunk, TextChunker
from src.knowledge_graph.embeddings import DenseEmbeddingModel, EmbeddingEngine
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.hybrid_search import (
    HybridSearchConfig,
    HybridSearchEngine,
    HybridSearchResult,
)
from src.knowledge_graph.schema import Entity, FactTriple
from src.knowledge_graph.traversal import PathResult
from src.knowledge_graph.vector_store import VectorStore
from src.models.dataset import SimpleTokenizer
from src.models.device import get_device
from src.models.inference import CrossEncoderInferenceEngine, RerankResult


@dataclass
class RetrievalPipelineConfig:
    """Configuration governing candidate pool sizing, neural reranking, and citation formatting."""

    candidate_pool_size: int = 20
    final_top_k: int = 5
    enable_reranker: bool = True
    min_relevance_threshold: float = 0.0
    hybrid_config: Optional[HybridSearchConfig] = None


@dataclass
class PipelineContextItem:
    """Individual retrieved and reranked passage with full attribution and graph provenance."""

    doc_id: str
    text: str
    final_score: float
    stage1_rank: int
    stage1_score: float
    stage2_rank: Optional[int] = None
    stage2_score: Optional[float] = None
    matched_entities: List[str] = field(default_factory=list)
    graph_paths: List[PathResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalPipelineOutput:
    """Complete retrieval payload ready for Agent synthesis and LLM prompting."""

    query: str
    items: List[PipelineContextItem]
    grounded_context_str: str
    total_candidates_retrieved: int
    total_reranked: int
    latency_stage1_ms: float
    latency_stage2_ms: float
    total_latency_ms: float

    @property
    def has_results(self) -> bool:
        return len(self.items) > 0


class HybridRAGPipeline:
    """Two-Stage Retrieve-and-Rerank Pipeline with Knowledge Graph Attribution.

    Architecture:
      Stage 1: Multi-Channel Hybrid Search (Dense + Sparse + Graph) -> Candidate Pool of size N.
      Stage 2: Cross-Encoder Transformer Reranker (Micro-batched + LRU Cache) -> Final Top-K.
      Stage 3: Grounded Context Assembly with multi-hop graph reasoning chains.
    """

    def __init__(
        self,
        hybrid_engine: HybridSearchEngine,
        reranker: Optional[CrossEncoderInferenceEngine] = None,
        default_config: Optional[RetrievalPipelineConfig] = None,
    ) -> None:
        self.hybrid_engine = hybrid_engine
        self.reranker = reranker
        self.default_config = default_config or RetrievalPipelineConfig()

        # Telemetry
        self.total_pipeline_queries: int = 0
        self.total_pipeline_time_ms: float = 0.0

    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Convenience delegation to index documents across dense, sparse, and graph stores."""
        return self.hybrid_engine.index_document(doc_id=doc_id, text=text, metadata=metadata)

    def query(
        self,
        query_text: str,
        config: Optional[RetrievalPipelineConfig] = None,
    ) -> RetrievalPipelineOutput:
        """Execute end-to-end retrieve-and-rerank pipeline.

        Args:
            query_text: Natural language user question or research prompt.
            config: Optional override configuration.

        Returns:
            RetrievalPipelineOutput containing reranked passages and formatted prompt context.
        """
        start_pipeline = time.perf_counter()
        cfg = config or self.default_config
        clean_q = query_text.strip()

        if not clean_q:
            return RetrievalPipelineOutput(
                query=query_text,
                items=[],
                grounded_context_str="",
                total_candidates_retrieved=0,
                total_reranked=0,
                latency_stage1_ms=0.0,
                latency_stage2_ms=0.0,
                total_latency_ms=0.0,
            )

        # -------------------------------------------------------------
        # STAGE 1: High-Recall Multi-Channel Hybrid Search
        # -------------------------------------------------------------
        start_stage1 = time.perf_counter()
        h_cfg = cfg.hybrid_config or HybridSearchConfig(top_k=cfg.candidate_pool_size)
        h_cfg.top_k = cfg.candidate_pool_size

        stage1_hits: List[HybridSearchResult] = self.hybrid_engine.search(
            clean_q,
            top_k=cfg.candidate_pool_size,
            config=h_cfg,
        )
        latency_stage1_ms = (time.perf_counter() - start_stage1) * 1000.0

        if not stage1_hits:
            return RetrievalPipelineOutput(
                query=query_text,
                items=[],
                grounded_context_str="No relevant documents found.",
                total_candidates_retrieved=0,
                total_reranked=0,
                latency_stage1_ms=round(latency_stage1_ms, 3),
                latency_stage2_ms=0.0,
                total_latency_ms=round((time.perf_counter() - start_pipeline) * 1000.0, 3),
            )

        # -------------------------------------------------------------
        # STAGE 2: Deep Cross-Encoder Neural Reranking
        # -------------------------------------------------------------
        start_stage2 = time.perf_counter()
        final_items: List[PipelineContextItem] = []
        total_reranked = 0

        if cfg.enable_reranker and self.reranker is not None:
            candidate_texts = [hit.text for hit in stage1_hits]
            rerank_results: List[RerankResult] = self.reranker.rerank(
                query=clean_q,
                documents=candidate_texts,
                top_k=cfg.final_top_k,
            )
            total_reranked = len(candidate_texts)

            for rank_idx, r_res in enumerate(rerank_results):
                orig_hit = stage1_hits[r_res.original_index]
                if r_res.score < cfg.min_relevance_threshold:
                    continue

                final_items.append(
                    PipelineContextItem(
                        doc_id=orig_hit.doc_id,
                        text=orig_hit.text,
                        final_score=round(r_res.score, 4),
                        stage1_rank=r_res.original_index + 1,
                        stage1_score=orig_hit.score,
                        stage2_rank=rank_idx + 1,
                        stage2_score=round(r_res.score, 4),
                        matched_entities=orig_hit.matched_entities,
                        graph_paths=orig_hit.graph_paths,
                        metadata=orig_hit.metadata,
                    )
                )
        else:
            # Fallback directly to Stage 1 Top-K
            for rank_idx, orig_hit in enumerate(stage1_hits[: cfg.final_top_k]):
                if orig_hit.score < cfg.min_relevance_threshold:
                    continue

                final_items.append(
                    PipelineContextItem(
                        doc_id=orig_hit.doc_id,
                        text=orig_hit.text,
                        final_score=round(orig_hit.score, 4),
                        stage1_rank=rank_idx + 1,
                        stage1_score=orig_hit.score,
                        stage2_rank=None,
                        stage2_score=None,
                        matched_entities=orig_hit.matched_entities,
                        graph_paths=orig_hit.graph_paths,
                        metadata=orig_hit.metadata,
                    )
                )

        latency_stage2_ms = (time.perf_counter() - start_stage2) * 1000.0

        # -------------------------------------------------------------
        # STAGE 3: Context Assembly for Agent Synthesis
        # -------------------------------------------------------------
        grounded_context_str = self._format_grounded_context(clean_q, final_items)

        total_latency_ms = (time.perf_counter() - start_pipeline) * 1000.0
        self.total_pipeline_queries += 1
        self.total_pipeline_time_ms += total_latency_ms

        return RetrievalPipelineOutput(
            query=query_text,
            items=final_items,
            grounded_context_str=grounded_context_str,
            total_candidates_retrieved=len(stage1_hits),
            total_reranked=total_reranked,
            latency_stage1_ms=round(latency_stage1_ms, 3),
            latency_stage2_ms=round(latency_stage2_ms, 3),
            total_latency_ms=round(total_latency_ms, 3),
        )

    def _format_grounded_context(
        self,
        query: str,
        items: List[PipelineContextItem],
    ) -> str:
        """Format retrieved passages and graph reasoning chains into an LLM-ready prompt context block."""
        if not items:
            return "No relevant context found."

        context_lines: List[str] = [
            f"=== GROUNDED KNOWLEDGE BASE CONTEXT (Query: '{query}') ===",
        ]

        # Collect unique graph paths discovered across items
        seen_paths: Set[str] = set()
        graph_explanations: List[str] = []

        for item in items:
            for path in item.graph_paths:
                explanation = path.to_explanation()
                if explanation not in seen_paths:
                    seen_paths.add(explanation)
                    graph_explanations.append(explanation)

        if graph_explanations:
            context_lines.append("\n[STRUCTURED KNOWLEDGE GRAPH REASONING]")
            for exp in graph_explanations:
                context_lines.append(f"• {exp}")

        context_lines.append("\n[RETRIEVED & RERANKED EVIDENCE PASSAGES]")
        for i, item in enumerate(items, 1):
            score_desc = (
                f"Relevance: {item.final_score:.3f} (Neural Rerank #{item.stage2_rank}, Hybrid #{item.stage1_rank})"
                if item.stage2_rank is not None
                else f"Relevance: {item.final_score:.4f} (Hybrid #{item.stage1_rank})"
            )
            context_lines.append(f"[{i}] Document ID: {item.doc_id} | {score_desc}")
            if item.matched_entities:
                context_lines.append(f"    Entities: {', '.join(item.matched_entities)}")
            context_lines.append(f"    Content: {item.text.strip()}\n")

        return "\n".join(context_lines)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        embedding_dim: int = 64,
        max_seq_len: int = 64,
    ) -> "HybridRAGPipeline":
        """Instantiate complete pipeline loaded from a trained CrossEncoder checkpoint."""
        # 1. Base stores
        vector_store = VectorStore(dimension=embedding_dim, metric="cosine")
        embedding_engine = EmbeddingEngine(embedding_dim=embedding_dim, max_seq_len=max_seq_len)
        kg = KnowledgeGraph()
        hybrid_engine = HybridSearchEngine(
            vector_store=vector_store,
            embedding_engine=embedding_engine,
            knowledge_graph=kg,
        )

        # 2. Neural Cross-Encoder
        reranker = CrossEncoderInferenceEngine.from_checkpoint(
            checkpoint_path=checkpoint_path,
        )

        return cls(hybrid_engine=hybrid_engine, reranker=reranker)

    @property
    def telemetry(self) -> Dict[str, Any]:
        """Telemetry covering pipeline throughput and component statistics."""
        avg_lat = (
            self.total_pipeline_time_ms / self.total_pipeline_queries
            if self.total_pipeline_queries > 0
            else 0.0
        )
        stats = {
            "total_queries": self.total_pipeline_queries,
            "average_pipeline_latency_ms": round(avg_lat, 3),
            "hybrid_engine": self.hybrid_engine.telemetry,
        }
        if self.reranker:
            stats["reranker"] = self.reranker.telemetry
        return stats
