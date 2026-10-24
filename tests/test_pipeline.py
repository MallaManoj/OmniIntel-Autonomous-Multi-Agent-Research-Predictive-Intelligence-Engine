"""Unit tests for HybridRAGPipeline end-to-end retrieve-and-rerank system.

Validates Stage 1 coarse retrieval, Stage 2 Cross-Encoder deep reranking,
grounded citation context assembly, fallback behaviors, and checkpoint loading.
Compatible with standard library `unittest` and `pytest`.
"""

from pathlib import Path
import unittest

from src.knowledge_graph.embeddings import EmbeddingEngine
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.hybrid_search import HybridSearchConfig, HybridSearchEngine
from src.knowledge_graph.pipeline import (
    HybridRAGPipeline,
    PipelineContextItem,
    RetrievalPipelineConfig,
    RetrievalPipelineOutput,
)
from src.knowledge_graph.schema import (
    Entity,
    EntityType,
    FactTriple,
    Relation,
    RelationType,
)
from src.knowledge_graph.vector_store import VectorStore
from src.models.dataset import SimpleTokenizer
from src.models.inference import CrossEncoderInferenceEngine
from src.models.reranker import CrossEncoderConfig, CrossEncoderReranker


class TestHybridRAGPipeline(unittest.TestCase):
    """Test suite covering the integrated two-stage Hybrid RAG pipeline."""

    def setUp(self) -> None:
        self.embedding_dim = 32
        self.vector_store = VectorStore(dimension=self.embedding_dim, metric="cosine")
        self.embedding_engine = EmbeddingEngine(
            embedding_dim=self.embedding_dim,
            max_seq_len=32,
        )
        self.kg = KnowledgeGraph()

        # Build Knowledge Graph
        self.e_jensen = Entity(entity_id="jensen", name="Jensen Huang", entity_type=EntityType.PERSON, aliases=["Jensen"])
        self.e_nvidia = Entity(entity_id="nvidia", name="NVIDIA", entity_type=EntityType.ORGANIZATION, aliases=["NVDA"])
        self.e_tsmc = Entity(entity_id="tsmc", name="TSMC", entity_type=EntityType.ORGANIZATION)

        self.kg.add_triple(
            FactTriple(
                triple_id="t1",
                subject=self.e_jensen,
                relation=Relation(relation_id="r1", relation_type=RelationType.LEADS),
                object=self.e_nvidia,
                confidence=1.0,
            )
        )
        self.kg.add_triple(
            FactTriple(
                triple_id="t2",
                subject=self.e_nvidia,
                relation=Relation(relation_id="r2", relation_type=RelationType.PARTNERS_WITH),
                object=self.e_tsmc,
                confidence=0.95,
            )
        )

        self.hybrid_engine = HybridSearchEngine(
            vector_store=self.vector_store,
            embedding_engine=self.embedding_engine,
            knowledge_graph=self.kg,
        )

        # Build or load neural reranker
        checkpoint_path = Path("checkpoints/reranker_best.pt")
        if checkpoint_path.exists():
            self.reranker_engine = CrossEncoderInferenceEngine.from_checkpoint(str(checkpoint_path))
        else:
            self.reranker_model = CrossEncoderReranker(
                CrossEncoderConfig(
                    vocab_size=1000,
                    max_seq_len=64,
                    embedding_dim=32,
                    num_heads=2,
                    num_layers=1,
                    dim_feedforward=64,
                )
            )
            self.reranker_engine = CrossEncoderInferenceEngine(
                model=self.reranker_model,
                tokenizer=SimpleTokenizer(),
                max_seq_len=64,
            )

        self.pipeline = HybridRAGPipeline(
            hybrid_engine=self.hybrid_engine,
            reranker=self.reranker_engine,
        )

        # Index corpus
        self.pipeline.index_document(
            doc_id="doc_ai_chips",
            text="NVIDIA and TSMC partner to manufacture cutting edge accelerator silicon for cloud deep learning.",
            metadata={"domain": "hardware"},
        )
        self.pipeline.index_document(
            doc_id="doc_inflation",
            text="Consumer price indices rose moderately as monetary policy makers tracked interest rates.",
            metadata={"domain": "economics"},
        )

    def test_end_to_end_query(self) -> None:
        """Pipeline must retrieve Stage 1 hits, rerank in Stage 2, and assemble context."""
        out = self.pipeline.query("NVIDIA accelerator silicon partnership")
        self.assertIsInstance(out, RetrievalPipelineOutput)
        self.assertTrue(out.has_results)
        self.assertGreater(len(out.items), 0)

        top_hit = out.items[0]
        self.assertEqual(top_hit.doc_id, "doc_ai_chips")
        self.assertEqual(top_hit.stage1_rank, 1)
        self.assertEqual(top_hit.stage2_rank, 1)
        self.assertGreater(top_hit.final_score, 0.0)

        # Verify grounded context formatting
        self.assertIn("GROUNDED KNOWLEDGE BASE CONTEXT", out.grounded_context_str)
        self.assertIn("Document ID: doc_ai_chips", out.grounded_context_str)
        self.assertGreater(out.total_latency_ms, 0.0)

    def test_reranker_disabled_fallback(self) -> None:
        """When enable_reranker=False, pipeline should fall back to Stage 1 rankings."""
        cfg = RetrievalPipelineConfig(enable_reranker=False, final_top_k=2)
        out = self.pipeline.query("NVIDIA accelerator", config=cfg)

        self.assertTrue(out.has_results)
        for item in out.items:
            self.assertIsNone(item.stage2_rank)
            self.assertIsNone(item.stage2_score)

    def test_empty_query(self) -> None:
        """Empty query should gracefully return empty output without errors."""
        out = self.pipeline.query("")
        self.assertFalse(out.has_results)
        self.assertEqual(out.items, [])
        self.assertEqual(out.total_latency_ms, 0.0)

    def test_context_includes_graph_reasoning(self) -> None:
        """When query matches graph entities, grounded context must present graph reasoning."""
        out = self.pipeline.query("Jensen Huang partner TSMC")
        self.assertTrue(out.has_results)
        self.assertIn("STRUCTURED KNOWLEDGE GRAPH REASONING", out.grounded_context_str)
        self.assertIn("Jensen Huang", out.grounded_context_str)
        self.assertIn("NVIDIA", out.grounded_context_str)

    def test_from_checkpoint_factory(self) -> None:
        """Pipeline must instantiate cleanly from a saved model checkpoint file."""
        checkpoint_path = Path("checkpoints/reranker_best.pt")
        if checkpoint_path.exists():
            pipeline_ckpt = HybridRAGPipeline.from_checkpoint(checkpoint_path)
            self.assertIsNotNone(pipeline_ckpt.reranker)
            self.assertGreater(pipeline_ckpt.reranker.model.config.vocab_size, 0)

    def test_telemetry(self) -> None:
        """Pipeline telemetry must track query count and latencies."""
        self.pipeline.query("AI chips")
        self.pipeline.query("Monetary inflation")

        stats = self.pipeline.telemetry
        self.assertEqual(stats["total_queries"], 2)
        self.assertGreater(stats["average_pipeline_latency_ms"], 0.0)
        self.assertIn("hybrid_engine", stats)
        self.assertIn("reranker", stats)


if __name__ == "__main__":
    unittest.main()
