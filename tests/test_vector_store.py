"""Unit tests for VectorStore with Cosine Similarity and Min-Heap Top-K Retrieval.

Validates vector ingestion, matrix synchronization, metadata filtering,
Min-Heap selection ordering, disk persistence round-trip, and telemetry.
Compatible with standard library `unittest` and `pytest`.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.knowledge_graph.vector_store import (
    VectorRecord,
    VectorSearchResult,
    VectorStore,
)


class TestVectorStore(unittest.TestCase):
    """Test suite covering vector storage, mathematical retrieval, and persistence."""

    def setUp(self) -> None:
        self.dimension = 4
        self.store = VectorStore(dimension=self.dimension, metric="cosine")

    def test_invalid_initialization(self) -> None:
        """Store must reject non-positive dimension or invalid metric."""
        with self.assertRaises(ValueError):
            VectorStore(dimension=0)
        with self.assertRaises(ValueError):
            VectorStore(dimension=-5)
        with self.assertRaises(ValueError):
            VectorStore(dimension=4, metric="invalid_metric")

    def test_add_and_get(self) -> None:
        """Single item addition and O(1) dictionary retrieval."""
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        meta = {"title": "Doc 1", "category": "tech"}
        self.store.add("doc1", vec, metadata=meta)

        self.assertEqual(len(self.store), 1)
        self.assertEqual(self.store.count, 1)

        record = self.store.get("doc1")
        self.assertIsNotNone(record)
        self.assertEqual(record.id, "doc1")
        self.assertEqual(record.metadata["title"], "Doc 1")
        np.testing.assert_allclose(record.vector, vec, atol=1e-5)

    def test_add_dimension_mismatch(self) -> None:
        """Vector with incompatible dimension must raise ValueError."""
        with self.assertRaises(ValueError):
            self.store.add("doc1", [1.0, 0.0])  # dimension 2 instead of 4

    def test_add_batch(self) -> None:
        """add_batch must bulk-insert vectors and metadatas correctly."""
        ids = ["v1", "v2", "v3"]
        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        metas = [{"tag": "a"}, {"tag": "b"}, {"tag": "c"}]

        self.store.add_batch(ids, vectors, metas)
        self.assertEqual(self.store.count, 3)

        for i, doc_id in enumerate(ids):
            rec = self.store.get(doc_id)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.metadata["tag"], metas[i]["tag"])

    def test_delete_and_clear(self) -> None:
        """Record deletion and index clearing."""
        self.store.add("d1", [1.0, 0.0, 0.0, 0.0])
        self.store.add("d2", [0.0, 1.0, 0.0, 0.0])

        self.assertTrue(self.store.delete("d1"))
        self.assertEqual(self.store.count, 1)
        self.assertIsNone(self.store.get("d1"))

        # Deleting nonexistent key returns False
        self.assertFalse(self.store.delete("d1"))

        self.store.clear()
        self.assertEqual(self.store.count, 0)
        self.assertEqual(self.store.search([1.0, 0.0, 0.0, 0.0]), [])

    def test_search_exact_match(self) -> None:
        """An identical vector query must yield similarity score ~1.0 as top hit."""
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        self.store.add("doc1", v1)
        self.store.add("doc2", v2)
        self.store.add("doc3", v3)

        results = self.store.search(v1, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "doc1")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)
        # Orthogonal vectors have dot product = 0.0
        self.assertAlmostEqual(results[1].score, 0.0, places=4)

    def test_min_heap_top_k_ordering(self) -> None:
        """Min-Heap must correctly rank candidates in descending order of similarity."""
        # Query direction: [1.0, 0.0, 0.0, 0.0]
        # Angles with query:
        # v_close: high cosine similarity (~0.95)
        # v_mid: medium cosine similarity (~0.71)
        # v_far: low cosine similarity (~0.20)
        # v_ortho: orthogonal similarity (0.0)
        v_close = [0.95, 0.31, 0.0, 0.0]
        v_mid = [0.71, 0.71, 0.0, 0.0]
        v_far = [0.20, 0.98, 0.0, 0.0]
        v_ortho = [0.0, 1.0, 0.0, 0.0]

        self.store.add("close", v_close)
        self.store.add("mid", v_mid)
        self.store.add("far", v_far)
        self.store.add("ortho", v_ortho)

        query = [1.0, 0.0, 0.0, 0.0]
        results = self.store.search(query, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual([r.id for r in results], ["close", "mid", "far"])
        # Verify monotonically descending scores
        self.assertGreater(results[0].score, results[1].score)
        self.assertGreater(results[1].score, results[2].score)

    def test_score_threshold_filter(self) -> None:
        """Candidates below score_threshold must be excluded from results."""
        self.store.add("high", [1.0, 0.0, 0.0, 0.0])
        self.store.add("low", [0.1, 0.99, 0.0, 0.0])

        query = [1.0, 0.0, 0.0, 0.0]
        results = self.store.search(query, top_k=5, score_threshold=0.5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "high")

    def test_metadata_predicate_filtering(self) -> None:
        """filter_fn must restrict search to matching metadata attributes."""
        self.store.add("t1", [1.0, 0.0, 0.0, 0.0], metadata={"category": "finance"})
        self.store.add("t2", [0.9, 0.1, 0.0, 0.0], metadata={"category": "sports"})
        self.store.add("t3", [0.8, 0.2, 0.0, 0.0], metadata={"category": "finance"})

        query = [1.0, 0.0, 0.0, 0.0]
        # Only search within finance documents
        results = self.store.search(
            query,
            top_k=5,
            filter_fn=lambda meta: meta.get("category") == "finance",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([r.id for r in results], ["t1", "t3"])
        for r in results:
            self.assertEqual(r.metadata["category"], "finance")

    def test_persistence_save_and_load(self) -> None:
        """VectorStore must accurately serialize and deserialize to disk."""
        self.store.add("recA", [0.6, 0.8, 0.0, 0.0], metadata={"source": "sec_10k"})
        self.store.add("recB", [0.0, 0.0, 1.0, 0.0], metadata={"source": "earnings_call"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test_index.npz"
            self.store.save(path)
            self.assertTrue(path.exists())

            loaded = VectorStore.load(path)
            self.assertEqual(loaded.count, 2)
            self.assertEqual(loaded.dimension, self.dimension)
            self.assertEqual(loaded.metric, "cosine")

            rec_a = loaded.get("recA")
            self.assertIsNotNone(rec_a)
            self.assertEqual(rec_a.metadata["source"], "sec_10k")

            # Search on loaded store produces identical results
            res_orig = self.store.search([0.6, 0.8, 0.0, 0.0], top_k=1)
            res_loaded = loaded.search([0.6, 0.8, 0.0, 0.0], top_k=1)

            self.assertEqual(res_orig[0].id, res_loaded[0].id)
            self.assertAlmostEqual(res_orig[0].score, res_loaded[0].score, places=5)

    def test_telemetry(self) -> None:
        """Search count and average latency must be recorded accurately."""
        self.store.add("d1", [1.0, 0.0, 0.0, 0.0])
        self.store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
        self.store.search([0.0, 1.0, 0.0, 0.0], top_k=1)

        stats = self.store.telemetry
        self.assertEqual(stats["total_searches"], 2)
        self.assertEqual(stats["total_records"], 1)
        self.assertEqual(stats["dimension"], 4)
        self.assertGreaterEqual(stats["average_search_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
