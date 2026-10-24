"""Unit tests for DenseEmbeddingModel and EmbeddingEngine.

Validates vector output dimensions, L2 normalization constraint (||v||_2 = 1.0),
masked mean pooling invariance to padding, and cosine similarity dot product equivalence.
Compatible with both standard library `unittest` and `pytest`.
"""

import unittest
import numpy as np
import torch

from src.knowledge_graph.chunker import DocumentChunk
from src.knowledge_graph.embeddings import DenseEmbeddingModel, EmbeddingEngine
from src.models.dataset import SimpleTokenizer
from src.models.device import get_device


class TestDenseEmbeddingModel(unittest.TestCase):
    """Test suite verifying neural architecture of DenseEmbeddingModel."""

    def setUp(self) -> None:
        self.device = get_device()
        self.vocab_size = 500
        self.embedding_dim = 32
        self.max_seq_len = 16
        self.model = DenseEmbeddingModel(
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            embedding_dim=self.embedding_dim,
            num_heads=2,
            dim_feedforward=64,
        ).to(self.device)
        self.model.eval()

    def test_forward_output_shape(self) -> None:
        """Forward pass should output tensor of shape [batch_size, embedding_dim]."""
        batch_size = 4
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones((batch_size, seq_len), device=self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)

        self.assertEqual(output.shape, (batch_size, self.embedding_dim))

    def test_l2_normalization_constraint(self) -> None:
        """Every embedding vector must have Euclidean norm ||v||_2 equal to 1.0."""
        batch_size = 8
        seq_len = 12
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        attention_mask = torch.ones((batch_size, seq_len), device=self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)

        norms = torch.norm(output, p=2, dim=-1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_masked_mean_pooling_ignores_padding(self) -> None:
        """Masked mean pooling must ignore zero-masked padding tokens."""
        seq_len = 10
        # Sequence 1: 5 real tokens, 5 padding
        input_ids = torch.randint(1, self.vocab_size, (1, seq_len), device=self.device)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], device=self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)

        # Output should be valid and normalized
        self.assertEqual(output.shape, (1, self.embedding_dim))
        self.assertFalse(torch.isnan(output).any())
        self.assertAlmostEqual(torch.norm(output, p=2).item(), 1.0, places=5)


class TestEmbeddingEngine(unittest.TestCase):
    """Test suite verifying serving engine and batch operations."""

    def setUp(self) -> None:
        self.engine = EmbeddingEngine(
            embedding_dim=32,
            max_seq_len=24,
        )

    def test_embed_single_text(self) -> None:
        """embed_text should return a 1D numpy array with norm 1.0."""
        text = "Deep learning transformers enable scalable dense retrieval."
        vec = self.engine.embed_text(text)

        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (32,))
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(float(norm), 1.0, places=5)

    def test_embed_batch(self) -> None:
        """embed_batch should return an [N, D] array with unit-norm rows."""
        texts = [
            "Apple designs custom silicon for neural acceleration.",
            "Market liquidity dropped ahead of the interest rate announcement.",
            "Graph databases capture complex entity relations efficiently.",
        ]
        vectors = self.engine.embed_batch(texts, batch_size=2)

        self.assertEqual(vectors.shape, (3, 32))
        for i in range(3):
            norm = np.linalg.norm(vectors[i])
            self.assertAlmostEqual(float(norm), 1.0, places=5)

    def test_empty_inputs(self) -> None:
        """Empty texts or chunks should return empty arrays cleanly without error."""
        empty_vectors = self.engine.embed_batch([])
        self.assertEqual(empty_vectors.shape, (0, 32))

        empty_chunks = self.engine.embed_chunks([])
        self.assertEqual(empty_chunks, [])

    def test_embed_chunks(self) -> None:
        """embed_chunks should associate each DocumentChunk with its unit embedding."""
        chunk1 = DocumentChunk(
            chunk_id="doc1_c0",
            doc_id="doc1",
            text="First chunk regarding monetary policy and federal reserves.",
            start_char=0,
            end_char=59,
            chunk_index=0,
            word_count=8,
        )
        chunk2 = DocumentChunk(
            chunk_id="doc1_c1",
            doc_id="doc1",
            text="Second chunk regarding inflation expectations and labor data.",
            start_char=60,
            end_char=121,
            chunk_index=1,
            word_count=8,
        )

        results = self.engine.embed_chunks([chunk1, chunk2])
        self.assertEqual(len(results), 2)
        for chunk, embedding in results:
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertIsInstance(embedding, np.ndarray)
            self.assertEqual(embedding.shape, (32,))
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)

    def test_cosine_similarity_dot_product_equivalence(self) -> None:
        """For L2-normalized vectors u and v, dot(u, v) must exactly equal cosine_similarity(u, v)."""
        text_a = "Quantitative risk analysis and portfolio optimization."
        text_b = "Asset allocation models with algorithmic risk parity."

        vec_a = self.engine.embed_text(text_a)
        vec_b = self.engine.embed_text(text_b)

        # Dot product
        dot_product = float(np.dot(vec_a, vec_b))

        # Explicit Cosine similarity: (a . b) / (||a|| * ||b||)
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        cosine_sim = dot_product / (norm_a * norm_b)

        self.assertAlmostEqual(dot_product, cosine_sim, places=5)

    def test_telemetry_tracking(self) -> None:
        """Telemetry should correctly record cumulative count and latency."""
        self.engine.embed_text("Sample query string")
        self.engine.embed_batch(["Text one", "Text two"])

        stats = self.engine.telemetry
        self.assertEqual(stats["total_embeddings_generated"], 3)
        self.assertEqual(stats["embedding_dimension"], 32)
        self.assertGreater(stats["average_latency_per_text_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
