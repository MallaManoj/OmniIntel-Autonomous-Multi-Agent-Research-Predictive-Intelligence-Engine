"""Unit tests for Information Retrieval evaluation metrics and modality ablation.

Validates Hit Rate@K, Recall@K, Precision@K, MRR, NDCG@K, MAP,
end-to-end pipeline evaluation, and modality comparison.
Compatible with standard library `unittest` and `pytest`.
"""

import unittest

from src.knowledge_graph.embeddings import EmbeddingEngine
from src.knowledge_graph.eval import (
    RetrievalEvaluator,
    RetrievalMetrics,
    RetrievalTestCase,
)
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.hybrid_search import HybridSearchEngine
from src.knowledge_graph.pipeline import HybridRAGPipeline
from src.knowledge_graph.schema import (
    Entity,
    EntityType,
    FactTriple,
    Relation,
    RelationType,
)
from src.knowledge_graph.vector_store import VectorStore


class TestRetrievalEvaluator(unittest.TestCase):
    """Test suite covering IR ranking metric calculations and benchmark evaluators."""

    def setUp(self) -> None:
        self.evaluator = RetrievalEvaluator()

    def test_compute_hit_rate(self) -> None:
        """Hit Rate@K must return 1.0 if any relevant doc is in top-K, else 0.0."""
        retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
        relevant = {"doc_c"}

        self.assertEqual(self.evaluator.compute_hit_rate(retrieved, relevant, k=1), 0.0)
        self.assertEqual(self.evaluator.compute_hit_rate(retrieved, relevant, k=2), 0.0)
        self.assertEqual(self.evaluator.compute_hit_rate(retrieved, relevant, k=3), 1.0)
        self.assertEqual(self.evaluator.compute_hit_rate(retrieved, relevant, k=5), 1.0)

    def test_compute_recall(self) -> None:
        """Recall@K must calculate the fraction of all relevant documents retrieved."""
        retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
        relevant = {"doc_a", "doc_c", "doc_z"}  # 3 total relevant docs

        self.assertAlmostEqual(self.evaluator.compute_recall(retrieved, relevant, k=1), 1.0 / 3.0)
        self.assertAlmostEqual(self.evaluator.compute_recall(retrieved, relevant, k=2), 1.0 / 3.0)
        self.assertAlmostEqual(self.evaluator.compute_recall(retrieved, relevant, k=3), 2.0 / 3.0)
        self.assertAlmostEqual(self.evaluator.compute_recall(retrieved, relevant, k=4), 2.0 / 3.0)

    def test_compute_precision(self) -> None:
        """Precision@K must calculate the fraction of retrieved top-K that are relevant."""
        retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
        relevant = {"doc_a", "doc_c"}

        self.assertAlmostEqual(self.evaluator.compute_precision(retrieved, relevant, k=1), 1.0)  # 1/1
        self.assertAlmostEqual(self.evaluator.compute_precision(retrieved, relevant, k=2), 0.5)  # 1/2
        self.assertAlmostEqual(self.evaluator.compute_precision(retrieved, relevant, k=4), 0.5)  # 2/4

    def test_compute_reciprocal_rank(self) -> None:
        """MRR must evaluate 1 / rank of the first relevant document."""
        self.assertEqual(self.evaluator.compute_reciprocal_rank(["d1", "d2"], {"d1"}), 1.0)
        self.assertEqual(self.evaluator.compute_reciprocal_rank(["d0", "d1"], {"d1"}), 0.5)
        self.assertEqual(self.evaluator.compute_reciprocal_rank(["d0", "d1", "d2", "d3"], {"d3"}), 0.25)
        self.assertEqual(self.evaluator.compute_reciprocal_rank(["d0", "d1"], {"dx"}), 0.0)

    def test_compute_ndcg_perfect_and_imperfect(self) -> None:
        """NDCG must be 1.0 for ideal rankings and strictly lower for suboptimal orderings."""
        perfect_order = ["doc_rel_high", "doc_rel_med", "doc_irrelevant"]
        inverted_order = ["doc_irrelevant", "doc_rel_med", "doc_rel_high"]
        relevant = {"doc_rel_high", "doc_rel_med"}
        graded = {"doc_rel_high": 2, "doc_rel_med": 1, "doc_irrelevant": 0}

        ndcg_perfect = self.evaluator.compute_ndcg(perfect_order, relevant, k=3, graded=graded)
        ndcg_inverted = self.evaluator.compute_ndcg(inverted_order, relevant, k=3, graded=graded)

        self.assertAlmostEqual(ndcg_perfect, 1.0, places=4)
        self.assertLess(ndcg_inverted, ndcg_perfect)
        self.assertGreater(ndcg_inverted, 0.0)

    def test_metrics_summary_table(self) -> None:
        """RetrievalMetrics.summary must format a valid Markdown table."""
        metrics = RetrievalMetrics(
            hit_rate_at_k={1: 0.60, 3: 0.90, 5: 1.00},
            recall_at_k={1: 0.40, 3: 0.80, 5: 1.00},
            precision_at_k={1: 0.60, 3: 0.50, 5: 0.40},
            ndcg_at_k={1: 0.60, 3: 0.85, 5: 0.92},
            mrr=0.75,
            map_score=0.72,
            total_queries=10,
            average_latency_ms=1.45,
        )

        md = metrics.summary()
        self.assertIn("### Information Retrieval Benchmark Summary", md)
        self.assertIn("| Metric | @1 | @3 | @5 |", md)
        self.assertIn("| Hit Rate | 0.6000 | 0.9000 | 1.0000 |", md)
        self.assertIn("MRR:** 0.7500", md)


class TestRetrievalBenchmarkSuite(unittest.TestCase):
    """End-to-end benchmark testing on an indexed test corpus."""

    def setUp(self) -> None:
        self.vector_store = VectorStore(dimension=32)
        self.embedding_engine = EmbeddingEngine(embedding_dim=32, max_seq_len=32)
        self.kg = KnowledgeGraph()

        # Build Knowledge Graph
        e_nvidia = Entity(entity_id="nvidia", name="NVIDIA", entity_type=EntityType.ORGANIZATION)
        e_tsmc = Entity(entity_id="tsmc", name="TSMC", entity_type=EntityType.ORGANIZATION)
        self.kg.add_triple(
            FactTriple(
                triple_id="t1",
                subject=e_nvidia,
                relation=Relation(relation_id="r1", relation_type=RelationType.PARTNERS_WITH),
                object=e_tsmc,
            )
        )

        self.engine = HybridSearchEngine(
            vector_store=self.vector_store,
            embedding_engine=self.embedding_engine,
            knowledge_graph=self.kg,
        )

        # Index documents
        self.engine.index_document("doc1", "NVIDIA GPUs and high performance compute clusters.")
        self.engine.index_document("doc2", "TSMC semiconductor fab and advanced wafer packaging.")
        self.engine.index_document("doc3", "Central bank interest rates and macro liquidity.")

        self.evaluator = RetrievalEvaluator()

        self.test_cases = [
            RetrievalTestCase(query="NVIDIA GPU clusters", relevant_doc_ids={"doc1"}),
            RetrievalTestCase(query="TSMC semiconductor wafer", relevant_doc_ids={"doc2"}),
        ]

    def test_evaluate_search_engine(self) -> None:
        """evaluate_search_engine must compute metrics across test cases."""
        metrics = self.evaluator.evaluate_search_engine(
            engine=self.engine,
            test_cases=self.test_cases,
            k_values=[1, 3],
        )

        self.assertEqual(metrics.total_queries, 2)
        self.assertEqual(metrics.hit_rate_at_k[1], 1.0)
        self.assertEqual(metrics.recall_at_k[1], 1.0)
        self.assertEqual(metrics.mrr, 1.0)
        self.assertGreater(metrics.average_latency_ms, 0.0)

    def test_compare_modalities_ablation(self) -> None:
        """compare_modalities must return comparative metrics for all 4 channels."""
        ablation = self.evaluator.compare_modalities(
            engine=self.engine,
            test_cases=self.test_cases,
            k_values=[1, 3],
        )

        self.assertIn("Dense-Only", ablation)
        self.assertIn("Sparse-Only", ablation)
        self.assertIn("Graph-Only", ablation)
        self.assertIn("Hybrid-RRF", ablation)

        # Hybrid RRF should achieve top hit rate
        self.assertEqual(ablation["Hybrid-RRF"].hit_rate_at_k[1], 1.0)


if __name__ == "__main__":
    unittest.main()
