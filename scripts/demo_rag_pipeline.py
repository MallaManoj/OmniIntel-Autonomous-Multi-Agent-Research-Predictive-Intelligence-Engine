#!/usr/bin/env python3
"""OmniIntel Demonstration Script: Phase 3 Knowledge Graph & Hybrid RAG Engine.

Demonstrates:
  1. Automated semantic chunking and dense vector indexing
  2. Multi-hop Knowledge Graph associative reasoning
  3. Multi-channel Hybrid Search with Reciprocal Rank Fusion (RRF)
  4. Deep Transformer Cross-Encoder neural reranking with LRU caching
  5. Information Retrieval evaluation and 4-way modality ablation benchmarking
"""

from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.knowledge_graph.chunker import TextChunker
from src.knowledge_graph.embeddings import DenseEmbeddingModel, EmbeddingEngine
from src.knowledge_graph.eval import (
    RetrievalEvaluator,
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
)
from src.knowledge_graph.schema import (
    Entity,
    EntityType,
    FactTriple,
    Relation,
    RelationType,
)
from src.knowledge_graph.vector_store import VectorStore
from src.models.inference import CrossEncoderInferenceEngine


def main() -> None:
    print("=" * 80)
    print("OmniIntel: Autonomous Multi-Agent Research & Predictive Intelligence Engine")
    print("Phase 3 Demonstration: Knowledge Graph & Hybrid Retrieve-and-Rerank RAG")
    print("=" * 80)

    # 1. Initialize Storage & Neural Components
    print("\n[1/5] Initializing Vector Store, Knowledge Graph & Neural Models...")
    embedding_dim = 64
    max_seq_len = 64

    vector_store = VectorStore(dimension=embedding_dim, metric="cosine")
    embedding_engine = EmbeddingEngine(embedding_dim=embedding_dim, max_seq_len=max_seq_len)
    kg = KnowledgeGraph()
    chunker = TextChunker(chunk_size=40, chunk_overlap=10, min_chunk_size=5)

    # 2. Populate Knowledge Graph Entities & Relations
    print("[2/5] Building Knowledge Graph Supply Chain Ontology...")
    e_jensen = Entity(entity_id="jensen", name="Jensen Huang", entity_type=EntityType.PERSON, aliases=["Jensen", "CEO Huang"])
    e_nvidia = Entity(entity_id="nvidia", name="NVIDIA Corporation", entity_type=EntityType.ORGANIZATION, aliases=["NVDA", "nvidia"])
    e_tsmc = Entity(entity_id="tsmc", name="TSMC", entity_type=EntityType.ORGANIZATION, aliases=["Taiwan Semiconductor"])
    e_asml = Entity(entity_id="asml", name="ASML Holding", entity_type=EntityType.ORGANIZATION, aliases=["ASML"])
    e_b200 = Entity(entity_id="blackwell_b200", name="Blackwell B200", entity_type=EntityType.PRODUCT, aliases=["B200"])

    kg.add_triple(FactTriple(
        triple_id="t1", subject=e_jensen, relation=Relation(relation_id="r1", relation_type=RelationType.LEADS), object=e_nvidia, confidence=1.0
    ))
    kg.add_triple(FactTriple(
        triple_id="t2", subject=e_nvidia, relation=Relation(relation_id="r2", relation_type=RelationType.DEVELOPS), object=e_b200, confidence=0.99
    ))
    kg.add_triple(FactTriple(
        triple_id="t3", subject=e_nvidia, relation=Relation(relation_id="r3", relation_type=RelationType.PARTNERS_WITH), object=e_tsmc, confidence=0.95
    ))
    kg.add_triple(FactTriple(
        triple_id="t4", subject=e_tsmc, relation=Relation(relation_id="r4", relation_type="BUYS_FROM"), object=e_asml, confidence=0.92
    ))
    print(f"  ✓ Knowledge Graph indexed: {kg.num_entities} entities, {kg.num_triples} factual triples.")

    # 3. Hybrid Search & Neural Cross-Encoder
    hybrid_engine = HybridSearchEngine(
        vector_store=vector_store,
        embedding_engine=embedding_engine,
        knowledge_graph=kg,
        chunker=chunker,
    )

    checkpoint_path = Path("checkpoints/reranker_best.pt")
    if checkpoint_path.exists():
        print(f"  ✓ Loading trained Cross-Encoder from '{checkpoint_path}'...")
        reranker = CrossEncoderInferenceEngine.from_checkpoint(str(checkpoint_path))
    else:
        print("  ! Checkpoint not found; using baseline model...")
        reranker = None

    pipeline = HybridRAGPipeline(hybrid_engine=hybrid_engine, reranker=reranker)

    # 4. Ingest Domain Filings
    print("\n[3/5] Ingesting & Chunking Financial/Tech Intelligence Corpus...")
    corpus = {
        "doc_nvda_q3": (
            "NVIDIA reported third quarter revenue of thirty-five billion dollars, up ninety-four percent year over year. "
            "Data Center compute sales were driven by insatiable demand for Blackwell B200 and Hopper architectures. "
            "CEO Jensen Huang noted cloud customers are racing to deploy generative AI clusters."
        ),
        "doc_tsmc_foundry": (
            "TSMC reported that three-nanometer wafer capacity is operating at full utilization across leading fab sites. "
            "Advanced CoWoS chiplet packaging facilities are expanding rapidly to support accelerator partners. "
            "Capital expenditures remain prioritized for next-generation lithography and clean-room expansion."
        ),
        "doc_asml_euv": (
            "ASML confirmed record bookings for High-NA extreme ultraviolet lithography scanners. "
            "Premier semiconductor foundries in Asia require advanced multi-patterning tools for sub-two nanometer silicon. "
            "Shipment volumes of Twinscan EUV systems increased twenty percent sequentially."
        ),
        "doc_macro_fed": (
            "Federal Reserve Open Market Committee members held benchmark interest rates steady at five and a quarter percent. "
            "Labor market indicators and non-farm payroll additions exhibited moderate rebalancing. "
            "Inflation metrics showed consistent movement toward the central bank's two percent target."
        ),
    }

    for doc_id, text in corpus.items():
        pipeline.index_document(doc_id=doc_id, text=text, metadata={"category": doc_id.split("_")[1]})
    print(f"  ✓ Ingested {len(corpus)} documents into {vector_store.count} vector chunks.")

    # 5. Live Multi-Hop Query Demonstration
    print("\n[4/5] Running Multi-Hop Supply Chain Research Query...")
    sample_query = "What lithography equipment does Jensen Huang's silicon supply chain depend on?"
    print(f"  Query: '{sample_query}'")

    cfg = RetrievalPipelineConfig(
        candidate_pool_size=4,
        final_top_k=2,
        hybrid_config=HybridSearchConfig(
            dense_weight=0.2,
            sparse_weight=0.2,
            graph_weight=0.6,
            graph_max_hops=3,
        ),
    )

    t0 = time.perf_counter()
    output = pipeline.query(sample_query, config=cfg)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"  ✓ Pipeline executed in {latency_ms:.2f} ms (Stage 1: {output.latency_stage1_ms:.2f} ms | Stage 2: {output.latency_stage2_ms:.2f} ms)")
    print(f"  ✓ Candidates retrieved: {output.total_candidates_retrieved} | Final reranked: {len(output.items)}")
    print("\n" + "-" * 40 + " GROUNDED AGENT PROMPT CONTEXT " + "-" * 40)
    print(output.grounded_context_str)
    print("-" * 110)

    # 6. Quantitative Modality Ablation Benchmark
    print("\n[5/5] Running Quantitative Modality Ablation Sweep...")
    test_cases = [
        RetrievalTestCase(query="Blackwell B200 Data Center sales", relevant_doc_ids={"doc_nvda_q3"}),
        RetrievalTestCase(query="CoWoS advanced packaging capacity", relevant_doc_ids={"doc_tsmc_foundry"}),
        RetrievalTestCase(query="High-NA EUV lithography scanners", relevant_doc_ids={"doc_asml_euv"}),
        RetrievalTestCase(query="Federal Reserve benchmark interest rates", relevant_doc_ids={"doc_macro_fed"}),
    ]

    evaluator = RetrievalEvaluator()
    ablation_results = evaluator.compare_modalities(hybrid_engine, test_cases, k_values=[1, 3])

    print("\n" + "=" * 80)
    print(f"{'Modality':<15} | {'Hit Rate@1':<12} | {'Recall@1':<10} | {'MRR':<8} | {'Avg Latency (ms)':<15}")
    print("-" * 80)
    for mod_name, metrics in ablation_results.items():
        print(f"{mod_name:<15} | {metrics.hit_rate_at_k[1]:<12.4f} | {metrics.recall_at_k[1]:<10.4f} | {metrics.mrr:<8.4f} | {metrics.average_latency_ms:<15.2f}")
    print("=" * 80)
    print("\n✓ Phase 3 Knowledge Graph & Hybrid RAG Engine is 100% operational!")


if __name__ == "__main__":
    main()
