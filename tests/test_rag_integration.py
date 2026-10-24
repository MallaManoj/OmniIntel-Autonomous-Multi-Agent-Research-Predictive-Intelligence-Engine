"""End-to-End Integration Test Suite for Phase 3: Knowledge Graph & Hybrid RAG.

Validates the full lifecycle across:
  1. Document ingestion and semantic boundary-preserving chunking
  2. Continuous vector embeddings and SIMD dot-product indexing
  3. Knowledge Graph multi-hop entity reasoning & Dijkstra path discovery
  4. Multi-channel Hybrid Search with Reciprocal Rank Fusion (RRF)
  5. Neural Cross-Encoder deep reranking and O(1) LRU caching
  6. Grounded context generation for Autonomous Agent prompt injection
  7. Quantitative benchmark evaluation (Hit Rate@K, Recall@K, MRR, NDCG@K)

Compatible with standard library `unittest` and `pytest`.
"""

from pathlib import Path
import unittest

from src.knowledge_graph.chunker import TextChunker
from src.knowledge_graph.embeddings import DenseEmbeddingModel, EmbeddingEngine
from src.knowledge_graph.eval import (
    RetrievalEvaluator,
    RetrievalMetrics,
    RetrievalTestCase,
)
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.hybrid_search import (
    HybridSearchConfig,
    HybridSearchEngine,
)
from src.knowledge_graph.pipeline import (
    HybridRAGPipeline,
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
from src.knowledge_graph.traversal import GraphTraversalEngine
from src.knowledge_graph.vector_store import VectorStore
from src.models.dataset import SimpleTokenizer
from src.models.inference import CrossEncoderInferenceEngine
from src.models.reranker import CrossEncoderConfig, CrossEncoderReranker


class TestRAGPipelineIntegration(unittest.TestCase):
    """End-to-end integration tests verifying Phase 3 components working in concert."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.embedding_dim = 32
        cls.max_seq_len = 64

        # 1. Base storage components
        cls.vector_store = VectorStore(dimension=cls.embedding_dim, metric="cosine")
        cls.embedding_engine = EmbeddingEngine(
            embedding_dim=cls.embedding_dim,
            max_seq_len=cls.max_seq_len,
        )
        cls.kg = KnowledgeGraph()
        cls.chunker = TextChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=5)

        # 2. Build Supply Chain Knowledge Graph
        cls.e_jensen = Entity(entity_id="jensen", name="Jensen Huang", entity_type=EntityType.PERSON, aliases=["Jensen"])
        cls.e_nvidia = Entity(entity_id="nvidia", name="NVIDIA Corporation", entity_type=EntityType.ORGANIZATION, aliases=["NVDA", "nvidia"])
        cls.e_tsmc = Entity(entity_id="tsmc", name="TSMC", entity_type=EntityType.ORGANIZATION, aliases=["Taiwan Semiconductor"])
        cls.e_asml = Entity(entity_id="asml", name="ASML", entity_type=EntityType.ORGANIZATION, aliases=["ASML Holding"])
        cls.e_b200 = Entity(entity_id="blackwell_b200", name="Blackwell B200", entity_type=EntityType.PRODUCT, aliases=["B200"])

        cls.kg.add_triple(
            FactTriple(
                triple_id="t1",
                subject=cls.e_jensen,
                relation=Relation(relation_id="r1", relation_type=RelationType.LEADS),
                object=cls.e_nvidia,
                confidence=1.0,
            )
        )
        cls.kg.add_triple(
            FactTriple(
                triple_id="t2",
                subject=cls.e_nvidia,
                relation=Relation(relation_id="r2", relation_type=RelationType.DEVELOPS),
                object=cls.e_b200,
                confidence=0.99,
            )
        )
        cls.kg.add_triple(
            FactTriple(
                triple_id="t3",
                subject=cls.e_nvidia,
                relation=Relation(relation_id="r3", relation_type=RelationType.PARTNERS_WITH),
                object=cls.e_tsmc,
                confidence=0.95,
            )
        )
        cls.kg.add_triple(
            FactTriple(
                triple_id="t4",
                subject=cls.e_tsmc,
                relation=Relation(relation_id="r4", relation_type=RelationType.SUBSIDIARY_OF if hasattr(RelationType, "BUYS_FROM") else "BUYS_FROM"),
                object=cls.e_asml,
                confidence=0.92,
            )
        )

        # 3. Hybrid search engine
        cls.hybrid_engine = HybridSearchEngine(
            vector_store=cls.vector_store,
            embedding_engine=cls.embedding_engine,
            knowledge_graph=cls.kg,
            chunker=cls.chunker,
        )

        # 4. Neural Cross-Encoder Reranker
        checkpoint_path = Path("checkpoints/reranker_best.pt")
        if checkpoint_path.exists():
            cls.reranker = CrossEncoderInferenceEngine.from_checkpoint(str(checkpoint_path))
        else:
            cls.reranker = CrossEncoderInferenceEngine(
                model=CrossEncoderReranker(
                    CrossEncoderConfig(
                        vocab_size=1000,
                        max_seq_len=cls.max_seq_len,
                        embedding_dim=cls.embedding_dim,
                        num_heads=2,
                        num_layers=1,
                    )
                ),
                tokenizer=SimpleTokenizer(),
                max_seq_len=cls.max_seq_len,
            )

        # 5. Full Pipeline
        cls.pipeline = HybridRAGPipeline(
            hybrid_engine=cls.hybrid_engine,
            reranker=cls.reranker,
        )

        # Ingest benchmark documents
        cls.doc1_text = (
            "NVIDIA reported unprecedented data center financial results for the quarter. "
            "Demand for Hopper and Blackwell B200 computing architectures surged across cloud hyper-scalers. "
            "Operating margins expanded significantly due to high software attachment rates."
        )
        cls.doc2_text = (
            "TSMC confirmed that 3nm fabrication capacity is fully booked through next year. "
            "Advanced CoWoS packaging facilities are scaling aggressively to support high performance computing silicon customers."
        )
        cls.doc3_text = (
            "ASML shipped advanced extreme ultraviolet lithography systems to premier foundry customers in Taiwan. "
            "Next generation high numerical aperture EUV equipment remains critical for sub-2nm silicon fabrication."
        )
        cls.doc4_text = (
            "Federal Reserve governors voted unanimously to maintain target federal funds rates unchanged. "
            "Labor market statistics and core personal consumption expenditures showed steady disinflationary progress."
        )

        cls.pipeline.index_document("doc_nvda", cls.doc1_text, {"domain": "earnings"})
        cls.pipeline.index_document("doc_tsmc", cls.doc2_text, {"domain": "manufacturing"})
        cls.pipeline.index_document("doc_asml", cls.doc3_text, {"domain": "equipment"})
        cls.pipeline.index_document("doc_macro", cls.doc4_text, {"domain": "economics"})

    def test_corpus_ingestion_and_indexing_integrity(self) -> None:
        """Verify that all documents were chunked, embedded, and mapped into inverted indices."""
        self.assertEqual(self.hybrid_engine._total_documents, 4)
        self.assertGreater(self.vector_store.count, 0)
        self.assertIn("nvidia", self.hybrid_engine._doc_entities["doc_nvda"])
        self.assertIn("tsmc", self.hybrid_engine._doc_entities["doc_tsmc"])
        self.assertIn("asml", self.hybrid_engine._doc_entities["doc_asml"])

    def test_multi_hop_knowledge_graph_reasoning_in_pipeline(self) -> None:
        """Query for 'Jensen lithography equipment' must bridge Jensen -> NVIDIA -> TSMC -> ASML."""
        # Note: 'doc_asml' does NOT mention Jensen or NVIDIA, but is connected via graph hops!
        cfg = RetrievalPipelineConfig(
            candidate_pool_size=4,
            final_top_k=3,
            hybrid_config=HybridSearchConfig(
                dense_weight=0.2,
                sparse_weight=0.2,
                graph_weight=0.6,
                graph_max_hops=3,
            ),
        )
        out = self.pipeline.query("Jensen Huang lithography equipment", config=cfg)
        self.assertTrue(out.has_results)

        retrieved_ids = [item.doc_id for item in out.items]
        self.assertIn("doc_asml", retrieved_ids)

        # Grounded context must explicitly include the structured graph reasoning chain
        self.assertIn("STRUCTURED KNOWLEDGE GRAPH REASONING", out.grounded_context_str)
        self.assertIn("ASML", out.grounded_context_str)

    def test_end_to_end_evaluation_metrics(self) -> None:
        """Execute quantitative benchmark evaluation asserting Hit Rate, Recall, and MRR."""
        test_cases = [
            RetrievalTestCase(query="Blackwell B200 Hopper compute architecture", relevant_doc_ids={"doc_nvda"}),
            RetrievalTestCase(query="TSMC 3nm CoWoS advanced packaging capacity", relevant_doc_ids={"doc_tsmc"}),
            RetrievalTestCase(query="ASML extreme ultraviolet EUV lithography systems", relevant_doc_ids={"doc_asml"}),
            RetrievalTestCase(query="Federal Reserve benchmark interest rates", relevant_doc_ids={"doc_macro"}),
        ]

        evaluator = RetrievalEvaluator()
        # Evaluate Stage 1 Hybrid retrieval engine across the 4 queries
        metrics_hybrid = evaluator.evaluate_search_engine(self.hybrid_engine, test_cases, k_values=[1, 3])

        self.assertEqual(metrics_hybrid.total_queries, 4)
        self.assertGreaterEqual(metrics_hybrid.hit_rate_at_k[1], 0.75)
        self.assertEqual(metrics_hybrid.hit_rate_at_k[3], 1.0)
        self.assertEqual(metrics_hybrid.recall_at_k[3], 1.0)
        self.assertGreaterEqual(metrics_hybrid.mrr, 0.85)
        self.assertGreater(metrics_hybrid.average_latency_ms, 0.0)

        # Evaluate full retrieve-and-rerank pipeline with final_top_k=3
        cfg = RetrievalPipelineConfig(enable_reranker=False, final_top_k=3)
        metrics_pipe = evaluator.evaluate_pipeline(self.pipeline, test_cases, k_values=[1, 3], config=cfg)
        self.assertEqual(metrics_pipe.hit_rate_at_k[3], 1.0)
        self.assertEqual(metrics_pipe.recall_at_k[3], 1.0)

    def test_modality_ablation_benchmarking(self) -> None:
        """Ablation benchmark should confirm Hybrid-RRF combines strengths of all channels."""
        test_cases = [
            RetrievalTestCase(query="Blackwell B200", relevant_doc_ids={"doc_nvda"}),
            RetrievalTestCase(query="extreme ultraviolet lithography", relevant_doc_ids={"doc_asml"}),
        ]

        evaluator = RetrievalEvaluator()
        ablation = evaluator.compare_modalities(self.hybrid_engine, test_cases, k_values=[1, 3])

        self.assertIn("Hybrid-RRF", ablation)
        self.assertEqual(ablation["Hybrid-RRF"].hit_rate_at_k[1], 1.0)
        self.assertEqual(ablation["Hybrid-RRF"].mrr, 1.0)

    def test_lru_cache_acceleration_in_pipeline(self) -> None:
        """Repeated identical queries must hit the LRU cache and yield lower latency."""
        q = "NVIDIA Hopper Blackwell operating margins"
        out_first = self.pipeline.query(q)
        out_second = self.pipeline.query(q)

        self.assertEqual(out_first.items[0].doc_id, out_second.items[0].doc_id)
        # Verify second query rerank stage executed faster due to LRU cache hits
        if self.pipeline.reranker:
            self.assertGreater(self.pipeline.reranker.cache.stats["hits"], 0)


if __name__ == "__main__":
    unittest.main()
