"""Information Retrieval & Hybrid Search Evaluation Suite.

Computes industry-standard IR ranking metrics:
  - Hit Rate@K (Success@K)
  - Recall@K
  - Precision@K
  - Mean Reciprocal Rank (MRR@K)
  - Normalized Discounted Cumulative Gain (NDCG@K)
  - Mean Average Precision (MAP@K)

Includes automated modality ablation benchmarking (Dense vs Sparse vs Graph vs Hybrid).
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Set, Union

from src.knowledge_graph.hybrid_search import (
    HybridSearchConfig,
    HybridSearchEngine,
    HybridSearchResult,
)
from src.knowledge_graph.pipeline import HybridRAGPipeline, RetrievalPipelineConfig


@dataclass
class RetrievalTestCase:
    """Benchmark test instance defining a query and ground truth relevant documents."""

    query: str
    relevant_doc_ids: Set[str]
    category: Optional[str] = None
    graded_relevance: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        if isinstance(self.relevant_doc_ids, (list, tuple)):
            self.relevant_doc_ids = set(self.relevant_doc_ids)


@dataclass
class RetrievalMetrics:
    """Aggregated evaluation metrics over a test collection."""

    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    total_queries: int = 0
    average_latency_ms: float = 0.0

    def summary(self) -> str:
        """Render a clean Markdown summary table of metrics."""
        lines = [
            "### Information Retrieval Benchmark Summary",
            f"**Total Queries Evaluated:** {self.total_queries} | **Average Latency:** {self.average_latency_ms:.2f} ms",
            f"**MRR:** {self.mrr:.4f} | **MAP:** {self.map_score:.4f}",
            "",
            "| Metric | " + " | ".join(f"@{k}" for k in sorted(self.hit_rate_at_k.keys())) + " |",
            "| :--- | " + " | ".join(":---:" for _ in self.hit_rate_at_k) + " |",
        ]

        # Hit Rate row
        hit_row = "| Hit Rate | " + " | ".join(
            f"{self.hit_rate_at_k[k]:.4f}" for k in sorted(self.hit_rate_at_k.keys())
        ) + " |"
        lines.append(hit_row)

        # Recall row
        rec_row = "| Recall | " + " | ".join(
            f"{self.recall_at_k[k]:.4f}" for k in sorted(self.recall_at_k.keys())
        ) + " |"
        lines.append(rec_row)

        # Precision row
        prec_row = "| Precision | " + " | ".join(
            f"{self.precision_at_k[k]:.4f}" for k in sorted(self.precision_at_k.keys())
        ) + " |"
        lines.append(prec_row)

        # NDCG row
        ndcg_row = "| NDCG | " + " | ".join(
            f"{self.ndcg_at_k[k]:.4f}" for k in sorted(self.ndcg_at_k.keys())
        ) + " |"
        lines.append(ndcg_row)

        return "\n".join(lines)


class RetrievalEvaluator:
    """Evaluates search engines and RAG pipelines against labeled ground truth callsets."""

    @staticmethod
    def compute_hit_rate(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Hit Rate@K: 1.0 if at least one relevant document is in top-K, else 0.0."""
        top_k = retrieved_ids[:k]
        return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0

    @staticmethod
    def compute_recall(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Recall@K: fraction of relevant documents retrieved in top-K."""
        if not relevant_ids:
            return 1.0
        top_k = set(retrieved_ids[:k])
        hits = len(top_k.intersection(relevant_ids))
        return hits / len(relevant_ids)

    @staticmethod
    def compute_precision(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        """Precision@K: fraction of top-K documents that are relevant."""
        if k <= 0:
            return 0.0
        top_k = set(retrieved_ids[:k])
        hits = len(top_k.intersection(relevant_ids))
        return hits / k

    @staticmethod
    def compute_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Reciprocal Rank: 1 / rank of the first relevant document."""
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def compute_ndcg(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
        graded: Optional[Dict[str, int]] = None,
    ) -> float:
        """NDCG@K: Normalized Discounted Cumulative Gain."""
        if k <= 0 or not relevant_ids:
            return 1.0

        top_k = retrieved_ids[:k]

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(top_k, 1):
            rel = graded.get(doc_id, 1) if (graded and doc_id in graded) else (1 if doc_id in relevant_ids else 0)
            if rel > 0:
                dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

        # Calculate IDCG (Ideal DCG)
        ideal_rels = []
        if graded:
            ideal_rels = sorted(graded.values(), reverse=True)
        else:
            ideal_rels = [1] * len(relevant_ids)

        idcg = 0.0
        for i, rel in enumerate(ideal_rels[:k], 1):
            idcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def compute_average_precision(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
        """Average Precision@K (AP@K)."""
        if not relevant_ids:
            return 1.0

        num_hits = 0
        sum_precisions = 0.0
        top_k = retrieved_ids[:k]

        for rank, doc_id in enumerate(top_k, 1):
            if doc_id in relevant_ids:
                num_hits += 1
                sum_precisions += num_hits / rank

        denom = min(k, len(relevant_ids))
        return sum_precisions / denom if denom > 0 else 0.0

    def evaluate_search_engine(
        self,
        engine: HybridSearchEngine,
        test_cases: List[RetrievalTestCase],
        k_values: Optional[List[int]] = None,
        config: Optional[HybridSearchConfig] = None,
    ) -> RetrievalMetrics:
        """Evaluate a HybridSearchEngine across a test collection."""
        if not test_cases:
            return RetrievalMetrics()

        ks = sorted(k_values or [1, 3, 5])
        max_k = max(ks)

        hit_rates: Dict[int, List[float]] = {k: [] for k in ks}
        recalls: Dict[int, List[float]] = {k: [] for k in ks}
        precisions: Dict[int, List[float]] = {k: [] for k in ks}
        ndcgs: Dict[int, List[float]] = {k: [] for k in ks}
        rrs: List[float] = []
        aps: List[float] = []
        latencies: List[float] = []

        cfg = config or HybridSearchConfig(top_k=max_k)
        cfg.top_k = max_k

        for tc in test_cases:
            t0 = time.perf_counter()
            results = engine.search(tc.query, top_k=max_k, config=cfg)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            retrieved_ids = [r.doc_id for r in results]

            for k in ks:
                hit_rates[k].append(self.compute_hit_rate(retrieved_ids, tc.relevant_doc_ids, k))
                recalls[k].append(self.compute_recall(retrieved_ids, tc.relevant_doc_ids, k))
                precisions[k].append(self.compute_precision(retrieved_ids, tc.relevant_doc_ids, k))
                ndcgs[k].append(self.compute_ndcg(retrieved_ids, tc.relevant_doc_ids, k, tc.graded_relevance))

            rrs.append(self.compute_reciprocal_rank(retrieved_ids, tc.relevant_doc_ids))
            aps.append(self.compute_average_precision(retrieved_ids, tc.relevant_doc_ids, k=max_k))

        n = len(test_cases)
        return RetrievalMetrics(
            hit_rate_at_k={k: round(sum(vals) / n, 4) for k, vals in hit_rates.items()},
            recall_at_k={k: round(sum(vals) / n, 4) for k, vals in recalls.items()},
            precision_at_k={k: round(sum(vals) / n, 4) for k, vals in precisions.items()},
            ndcg_at_k={k: round(sum(vals) / n, 4) for k, vals in ndcgs.items()},
            mrr=round(sum(rrs) / n, 4),
            map_score=round(sum(aps) / n, 4),
            total_queries=n,
            average_latency_ms=round(sum(latencies) / n, 3),
        )

    def evaluate_pipeline(
        self,
        pipeline: HybridRAGPipeline,
        test_cases: List[RetrievalTestCase],
        k_values: Optional[List[int]] = None,
        config: Optional[RetrievalPipelineConfig] = None,
    ) -> RetrievalMetrics:
        """Evaluate full retrieve-and-rerank HybridRAGPipeline."""
        if not test_cases:
            return RetrievalMetrics()

        ks = sorted(k_values or [1, 3, 5])
        max_k = max(ks)

        hit_rates: Dict[int, List[float]] = {k: [] for k in ks}
        recalls: Dict[int, List[float]] = {k: [] for k in ks}
        precisions: Dict[int, List[float]] = {k: [] for k in ks}
        ndcgs: Dict[int, List[float]] = {k: [] for k in ks}
        rrs: List[float] = []
        aps: List[float] = []
        latencies: List[float] = []

        cfg = config or RetrievalPipelineConfig(final_top_k=max_k)
        cfg.final_top_k = max_k

        for tc in test_cases:
            t0 = time.perf_counter()
            out = pipeline.query(tc.query, config=cfg)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            retrieved_ids = [item.doc_id for item in out.items]

            for k in ks:
                hit_rates[k].append(self.compute_hit_rate(retrieved_ids, tc.relevant_doc_ids, k))
                recalls[k].append(self.compute_recall(retrieved_ids, tc.relevant_doc_ids, k))
                precisions[k].append(self.compute_precision(retrieved_ids, tc.relevant_doc_ids, k))
                ndcgs[k].append(self.compute_ndcg(retrieved_ids, tc.relevant_doc_ids, k, tc.graded_relevance))

            rrs.append(self.compute_reciprocal_rank(retrieved_ids, tc.relevant_doc_ids))
            aps.append(self.compute_average_precision(retrieved_ids, tc.relevant_doc_ids, k=max_k))

        n = len(test_cases)
        return RetrievalMetrics(
            hit_rate_at_k={k: round(sum(vals) / n, 4) for k, vals in hit_rates.items()},
            recall_at_k={k: round(sum(vals) / n, 4) for k, vals in recalls.items()},
            precision_at_k={k: round(sum(vals) / n, 4) for k, vals in precisions.items()},
            ndcg_at_k={k: round(sum(vals) / n, 4) for k, vals in ndcgs.items()},
            mrr=round(sum(rrs) / n, 4),
            map_score=round(sum(aps) / n, 4),
            total_queries=n,
            average_latency_ms=round(sum(latencies) / n, 3),
        )

    def compare_modalities(
        self,
        engine: HybridSearchEngine,
        test_cases: List[RetrievalTestCase],
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, RetrievalMetrics]:
        """Ablation study evaluating Dense-Only vs Sparse-Only vs Graph-Only vs Hybrid RRF."""
        ks = k_values or [1, 3, 5]
        max_k = max(ks)

        configs = {
            "Dense-Only": HybridSearchConfig(dense_weight=1.0, sparse_weight=0.0, graph_weight=0.0, top_k=max_k),
            "Sparse-Only": HybridSearchConfig(dense_weight=0.0, sparse_weight=1.0, graph_weight=0.0, top_k=max_k),
            "Graph-Only": HybridSearchConfig(dense_weight=0.0, sparse_weight=0.0, graph_weight=1.0, top_k=max_k),
            "Hybrid-RRF": HybridSearchConfig(dense_weight=0.5, sparse_weight=0.3, graph_weight=0.2, fusion_method="rrf", top_k=max_k),
        }

        results: Dict[str, RetrievalMetrics] = {}
        for modality_name, cfg in configs.items():
            metrics = self.evaluate_search_engine(engine, test_cases, k_values=ks, config=cfg)
            results[modality_name] = metrics

        return results
