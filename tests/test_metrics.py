"""Unit tests for Information Retrieval and Reranking evaluation metrics.

Compatible with both standard library `unittest` and `pytest`.
"""

import unittest
from src.models.metrics import (
    RankingEvaluator,
    compute_dcg_at_k,
    compute_hit_rate_at_k,
    compute_ndcg_at_k,
    compute_reciprocal_rank,
)


class TestRankingMetrics(unittest.TestCase):
    """Test suite verifying MRR, NDCG@K, HitRate, and RankingEvaluator."""

    def test_reciprocal_rank_positions(self) -> None:
        """Verify Reciprocal Rank computation across various candidate positions."""
        # Relevant item at rank 1 -> RR = 1.0
        scores1 = [0.9, 0.4, 0.1]
        labels1 = [1, 0, 0]
        self.assertEqual(compute_reciprocal_rank(scores1, labels1), 1.0)

        # Relevant item at rank 2 -> RR = 0.5
        # Scores descending order of indices: index 1 (0.8, rank 1), index 0 (0.2, rank 2), index 2 (0.1, rank 3)
        scores2 = [0.2, 0.8, 0.1]
        labels2 = [1, 0, 0]        # Relevant is at index 0 (rank 2)
        self.assertEqual(compute_reciprocal_rank(scores2, labels2), 0.5)

        # Relevant item at rank 3 -> RR = 1/3
        scores3 = [0.9, 0.8, 0.7]
        labels3 = [0, 0, 1]
        self.assertAlmostEqual(compute_reciprocal_rank(scores3, labels3), 1.0 / 3.0)

        # No relevant items -> RR = 0.0
        scores4 = [0.9, 0.8]
        labels4 = [0, 0]
        self.assertEqual(compute_reciprocal_rank(scores4, labels4), 0.0)

    def test_ndcg_perfect_and_imperfect(self) -> None:
        """NDCG must be 1.0 for a perfect ranking and strictly lower for inverted rankings."""
        # Perfect ordering: higher predicted score maps to higher graded relevance
        scores_perf = [0.95, 0.80, 0.50, 0.10]
        labels_perf = [3, 2, 1, 0]
        ndcg_perf = compute_ndcg_at_k(scores_perf, labels_perf, k=4)
        self.assertAlmostEqual(ndcg_perf, 1.0, places=4)

        # Inverted ordering: model predicted lowest score for most relevant item
        scores_inv = [0.10, 0.50, 0.80, 0.95]
        ndcg_inv = compute_ndcg_at_k(scores_inv, labels_perf, k=4)
        self.assertLess(ndcg_inv, 1.0)
        self.assertGreater(ndcg_inv, 0.0)

        # All zero relevance -> NDCG = 0.0
        self.assertEqual(compute_ndcg_at_k([0.9, 0.8], [0, 0], k=2), 0.0)

    def test_hit_rate_at_k(self) -> None:
        """Hit Rate@K should return 1.0 if any relevant doc is in top-K, else 0.0."""
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        labels = [0, 0, 1, 0, 0]  # Relevant item is at rank 3

        self.assertEqual(compute_hit_rate_at_k(scores, labels, k=1), 0.0)
        self.assertEqual(compute_hit_rate_at_k(scores, labels, k=2), 0.0)
        self.assertEqual(compute_hit_rate_at_k(scores, labels, k=3), 1.0)
        self.assertEqual(compute_hit_rate_at_k(scores, labels, k=5), 1.0)

    def test_ranking_evaluator_summary(self) -> None:
        """RankingEvaluator must aggregate mean metrics across multiple queries."""
        evaluator = RankingEvaluator(k_thresholds=(1, 3, 5))

        # Query 1: Perfect retrieval (MRR = 1.0)
        evaluator.evaluate_query(
            predicted_scores=[0.9, 0.4, 0.1],
            ground_truth_relevance=[1, 0, 0],
        )

        # Query 2: Sub-optimal retrieval (relevant at rank 2 -> MRR = 0.5)
        evaluator.evaluate_query(
            predicted_scores=[0.9, 0.4, 0.1],
            ground_truth_relevance=[0, 1, 0],
        )

        summary = evaluator.compute_summary()
        self.assertEqual(summary["total_queries"], 2.0)
        # Mean MRR = (1.0 + 0.5) / 2 = 0.75
        self.assertAlmostEqual(summary["mean_mrr"], 0.75, places=4)
        self.assertIn("mean_ndcg@3", summary)
        self.assertIn("mean_hit_rate@1", summary)

    def test_length_mismatch_error(self) -> None:
        """Metric functions must raise ValueError on mismatched array lengths."""
        with self.assertRaises(ValueError):
            compute_reciprocal_rank([0.5], [1, 0])
        with self.assertRaises(ValueError):
            compute_ndcg_at_k([0.5, 0.2], [1])
        with self.assertRaises(ValueError):
            compute_hit_rate_at_k([0.5], [1, 0])


if __name__ == "__main__":
    unittest.main()
