"""Unit tests for HybridSearchEngine multi-channel retrieval and rank fusion.

Validates multi-modal indexing, dense semantic retrieval, sparse lexical matching,
Knowledge Graph K-hop path expansion, Reciprocal Rank Fusion (RRF), and telemetry.
Compatible with standard library `unittest` and `pytest`.
"""

import unittest

from src.knowledge_graph.embeddings import DenseEmbeddingModel, EmbeddingEngine
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.hybrid_search import (
    HybridSearchConfig,
    HybridSearchEngine,
    HybridSearchResult,
)
from src.knowledge_graph.schema import (
    Entity,
    EntityType,
    FactTriple,
    Relation,
    RelationType,
)
from src.knowledge_graph.vector_store import VectorStore


class TestHybridSearchEngine(unittest.TestCase):
    """Test suite covering Hybrid Search across dense, sparse, and graph modalities."""

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
        self.e_tsmc = Entity(entity_id="tsmc", name="TSMC", entity_type=EntityType.ORGANIZATION, aliases=["Taiwan Semiconductor"])
        self.e_blackwell = Entity(entity_id="blackwell", name="Blackwell B200", entity_type=EntityType.PRODUCT, aliases=["B200"])

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
                relation=Relation(relation_id="r2", relation_type=RelationType.DEVELOPS),
                object=self.e_blackwell,
                confidence=0.98,
            )
        )
        self.kg.add_triple(
            FactTriple(
                triple_id="t3",
                subject=self.e_nvidia,
                relation=Relation(relation_id="r3", relation_type=RelationType.PARTNERS_WITH),
                object=self.e_tsmc,
                confidence=0.95,
            )
        )

        self.engine = HybridSearchEngine(
            vector_store=self.vector_store,
            embedding_engine=self.embedding_engine,
            knowledge_graph=self.kg,
        )

        # Ingest benchmark documents
        self.engine.index_document(
            doc_id="doc_nvda_earnings",
            text="NVIDIA reported record quarterly revenue driven by immense demand for AI compute infrastructure and Hopper GPUs.",
            metadata={"category": "earnings"},
        )
        self.engine.index_document(
            doc_id="doc_tsmc_fab",
            text="TSMC advanced packaging and 3nm wafer foundry nodes are running at full manufacturing capacity for high performance compute partners.",
            metadata={"category": "semiconductors"},
        )
        self.engine.index_document(
            doc_id="doc_fed_policy",
            text="Federal Reserve officials maintained benchmark interest rates steady to monitor labor market liquidity and core inflation metrics.",
            metadata={"category": "macroeconomics"},
        )

    def test_indexing_pipeline(self) -> None:
        """Indexing must populate inverted index, vector store, and entity bindings."""
        self.assertEqual(self.engine._total_documents, 3)
        self.assertGreater(self.vector_store.count, 0)
        self.assertIn("nvidia", self.engine._inverted_index)
        self.assertIn("tsmc", self.engine._inverted_index)

        # Check entity recognition in documents
        self.assertIn("nvidia", self.engine._doc_entities["doc_nvda_earnings"])
        self.assertIn("tsmc", self.engine._doc_entities["doc_tsmc_fab"])

    def test_sparse_exact_keyword_matching(self) -> None:
        """Rare exact keywords like 'packaging' or 'inflation' must trigger strong sparse signals."""
        results = self.engine.search("advanced packaging foundry wafer", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].doc_id, "doc_tsmc_fab")
        self.assertIsNotNone(results[0].sparse_score)
        self.assertGreater(results[0].sparse_score, 0.0)

    def test_graph_associative_expansion(self) -> None:
        """Query mentioning 'Jensen' should boost 'doc_tsmc_fab' via graph path Jensen -> NVIDIA -> TSMC."""
        # Query: 'What is Jensen Huang manufacturing capacity?'
        # Note: 'doc_tsmc_fab' does NOT mention Jensen directly! It mentions TSMC.
        # But Knowledge Graph has Jensen -> NVIDIA -> TSMC!
        cfg = HybridSearchConfig(
            dense_weight=0.2,
            sparse_weight=0.2,
            graph_weight=0.6,  # Boost graph
            top_k=3,
        )
        results = self.engine.search("Jensen Huang foundry capacity", config=cfg)
        self.assertGreater(len(results), 0)

        hit_ids = [r.doc_id for r in results]
        self.assertIn("doc_tsmc_fab", hit_ids)

        # Confirm graph score was credited to TSMC doc
        tsmc_hit = next(r for r in results if r.doc_id == "doc_tsmc_fab")
        self.assertIsNotNone(tsmc_hit.graph_score)
        self.assertGreater(tsmc_hit.graph_score, 0.0)

    def test_rrf_and_linear_fusion_modes(self) -> None:
        """Both RRF and Linear fusion strategies must execute cleanly and return ordered hits."""
        cfg_rrf = HybridSearchConfig(fusion_method="rrf", top_k=2)
        results_rrf = self.engine.search("Federal reserve benchmark interest rates", config=cfg_rrf)
        self.assertGreater(len(results_rrf), 0)
        self.assertEqual(results_rrf[0].doc_id, "doc_fed_policy")

        cfg_linear = HybridSearchConfig(fusion_method="linear", top_k=2)
        results_linear = self.engine.search("Federal reserve benchmark interest rates", config=cfg_linear)
        self.assertGreater(len(results_linear), 0)
        self.assertEqual(results_linear[0].doc_id, "doc_fed_policy")

    def test_empty_query_returns_empty(self) -> None:
        """Empty query must return an empty list."""
        self.assertEqual(self.engine.search(""), [])
        self.assertEqual(self.engine.search("   "), [])

    def test_telemetry(self) -> None:
        """Search telemetry must record cumulative query metrics."""
        self.engine.search("NVIDIA AI infrastructure")
        self.engine.search("TSMC manufacturing")

        stats = self.engine.telemetry
        self.assertEqual(stats["total_documents"], 3)
        self.assertEqual(stats["total_queries"], 2)
        self.assertGreater(stats["average_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
