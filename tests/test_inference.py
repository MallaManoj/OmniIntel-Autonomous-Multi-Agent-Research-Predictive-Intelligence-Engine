"""Unit tests for CrossEncoderInferenceEngine, micro-batching, and LRU cache integration.

Compatible with both standard library `unittest` and `pytest`.
"""

import os
import shutil
import tempfile
import unittest

from src.models.checkpoint import ModelCheckpointManager
from src.models.dataset import SimpleTokenizer
from src.models.device import get_device
from src.models.inference import CrossEncoderInferenceEngine, RerankResult
from src.models.reranker import CrossEncoderConfig, CrossEncoderReranker

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch required for InferenceEngine tests")
class TestInferenceEngine(unittest.TestCase):
    """Test suite verifying inference micro-batching, LRU caching, and top-K reranking."""

    def setUp(self) -> None:
        self.device = get_device()
        self.temp_dir = tempfile.mkdtemp()
        self.tokenizer = SimpleTokenizer(vocab=["nvidia", "revenue", "gpu", "quarter", "growth", "gaming"])
        self.config = CrossEncoderConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=32,
            embedding_dim=32,
            num_heads=2,
            num_layers=1,
            dim_feedforward=64,
        )
        self.model = CrossEncoderReranker(self.config).to(self.device)
        self.engine = CrossEncoderInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            cache_capacity=100,
            max_seq_len=32,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_predict_score_and_caching(self) -> None:
        """predict_score must compute score, cache it, and return cache hit on repeat."""
        q = "nvidia revenue"
        doc = "gpu growth in quarter"

        score1 = self.engine.predict_score(q, doc)
        self.assertTrue(0.0 <= score1 <= 1.0)
        self.assertEqual(self.engine.cache.size, 1)
        self.assertEqual(self.engine.cache.hits, 0)
        self.assertEqual(self.engine.cache.misses, 1)

        # Second call must hit LRU Cache
        score2 = self.engine.predict_score(q, doc)
        self.assertEqual(score1, score2)
        self.assertEqual(self.engine.cache.hits, 1)

    def test_rerank_micro_batching_and_order(self) -> None:
        """rerank must return results sorted by score descending with original indices."""
        query = "nvidia gpu"
        docs = [
            "nvidia quarterly revenue growth",
            "gaming graphics cards",
            "completely unrelated document",
            "another text passage",
        ]
        results = self.engine.rerank(query, docs, top_k=3, batch_size=2)

        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], RerankResult)

        # Verify descending order
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # Verify original index integrity
        for r in results:
            self.assertEqual(docs[r.original_index], r.document)

    def test_empty_documents_handling(self) -> None:
        """rerank with empty list should safely return empty list."""
        results = self.engine.rerank("query", [])
        self.assertEqual(results, [])

    def test_telemetry_metrics(self) -> None:
        """telemetry property must return latency and cache statistics."""
        self.engine.rerank("query", ["doc 1", "doc 2"])
        stats = self.engine.telemetry
        self.assertEqual(stats["total_queries"], 1)
        self.assertEqual(stats["total_documents_scored"], 2)
        self.assertIn("cache_stats", stats)
        self.assertGreaterEqual(stats["average_latency_ms"], 0.0)

    def test_from_checkpoint_factory(self) -> None:
        """Engine should instantiate seamlessly from a saved checkpoint file."""
        ckpt_path = os.path.join(self.temp_dir, "model.pt")
        ModelCheckpointManager.save_checkpoint(
            model=self.model,
            filepath=ckpt_path,
            tokenizer=self.tokenizer,
        )

        loaded_engine = CrossEncoderInferenceEngine.from_checkpoint(
            checkpoint_path=ckpt_path,
            device=self.device,
        )
        score = loaded_engine.predict_score("nvidia", "revenue")
        self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()
