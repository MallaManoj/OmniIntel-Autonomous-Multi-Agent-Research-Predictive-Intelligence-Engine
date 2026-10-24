"""Application state and dependency management for OmniIntel FastAPI service."""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, Optional

from src.agents.orchestrator import MasterOrchestrator
from src.core.events import EventBus, global_event_bus
from src.knowledge_graph.chunker import TextChunker
from src.knowledge_graph.embeddings import DenseEmbeddingModel, EmbeddingEngine
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.hybrid_search import HybridSearchEngine
from src.knowledge_graph.pipeline import HybridRAGPipeline
from src.knowledge_graph.schema import Entity, EntityType, FactTriple, Relation
from src.knowledge_graph.vector_store import VectorStore
from src.models.dataset import SimpleTokenizer
from src.models.inference import CrossEncoderInferenceEngine
from src.models.reranker import CrossEncoderConfig, CrossEncoderReranker


@dataclass
class AppState:
    """Encapsulates shared backend runtime components."""

    event_bus: EventBus = field(default_factory=lambda: global_event_bus)
    knowledge_graph: Optional[KnowledgeGraph] = None
    pipeline: Optional[HybridRAGPipeline] = None
    orchestrator: Optional[MasterOrchestrator] = None
    market_feed: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    is_ready: bool = False
    version: str = "1.0.0"

    @property
    def uptime_seconds(self) -> float:
        """Calculate running service uptime in seconds."""
        return time.time() - self.start_time


# Singleton global application state
app_state = AppState()


def get_app_state() -> AppState:
    """Dependency provider returning singleton application state."""
    return app_state


def initialize_knowledge_base() -> AppState:
    """Bootstrap and index default semiconductor and tech ecosystem knowledge base."""
    # 1. Knowledge Graph Fixture
    kg = KnowledgeGraph()
    e_nvda = Entity(
        entity_id="nvidia",
        name="NVIDIA Corporation",
        entity_type=EntityType.ORGANIZATION,
        aliases=["NVDA", "nvidia", "Nvidia"],
    )
    e_tsmc = Entity(
        entity_id="tsmc",
        name="TSMC",
        entity_type=EntityType.ORGANIZATION,
        aliases=["Taiwan Semiconductor", "tsmc", "TSM"],
    )
    e_asml = Entity(
        entity_id="asml",
        name="ASML Holding",
        entity_type=EntityType.ORGANIZATION,
        aliases=["ASML", "asml"],
    )
    e_amd = Entity(
        entity_id="amd",
        name="Advanced Micro Devices",
        entity_type=EntityType.ORGANIZATION,
        aliases=["AMD", "amd"],
    )
    e_blackwell = Entity(
        entity_id="blackwell",
        name="Blackwell B200",
        entity_type=EntityType.PRODUCT,
        aliases=["Blackwell", "B200"],
    )
    e_cowos = Entity(
        entity_id="cowos",
        name="CoWoS Packaging",
        entity_type=EntityType.PRODUCT,
        aliases=["CoWoS"],
    )

    kg.add_triple(FactTriple(triple_id="t1", subject=e_asml, relation=Relation(relation_id="r1", relation_type="supplier_to"), object=e_tsmc, confidence=0.98))
    kg.add_triple(FactTriple(triple_id="t2", subject=e_tsmc, relation=Relation(relation_id="r2", relation_type="supplier_to"), object=e_nvda, confidence=0.99))
    kg.add_triple(FactTriple(triple_id="t3", subject=e_tsmc, relation=Relation(relation_id="r3", relation_type="supplier_to"), object=e_amd, confidence=0.94))
    kg.add_triple(FactTriple(triple_id="t4", subject=e_nvda, relation=Relation(relation_id="r4", relation_type="develops"), object=e_blackwell, confidence=1.0))
    kg.add_triple(FactTriple(triple_id="t5", subject=e_tsmc, relation=Relation(relation_id="r5", relation_type="manufactures"), object=e_cowos, confidence=0.97))

    # 2. Hybrid RAG Pipeline
    embedding_dim = 32
    max_seq_len = 64
    vs = VectorStore(dimension=embedding_dim, metric="cosine")
    ee = EmbeddingEngine(embedding_dim=embedding_dim, max_seq_len=max_seq_len)
    ch = TextChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=5)
    hse = HybridSearchEngine(vector_store=vs, embedding_engine=ee, knowledge_graph=kg, chunker=ch)

    checkpoint_path = Path("checkpoints/reranker_best.pt")
    if checkpoint_path.exists():
        reranker = CrossEncoderInferenceEngine.from_checkpoint(str(checkpoint_path))
    else:
        reranker = CrossEncoderInferenceEngine(
            model=CrossEncoderReranker(
                CrossEncoderConfig(vocab_size=1000, max_seq_len=max_seq_len, embedding_dim=embedding_dim, num_heads=2, num_layers=1)
            ),
            tokenizer=SimpleTokenizer(),
            max_seq_len=max_seq_len,
        )

    pipeline = HybridRAGPipeline(hybrid_engine=hse, reranker=reranker)

    # Ingest baseline filings
    pipeline.index_document(
        doc_id="sec_nvda_10q",
        text=(
            "NVIDIA Corporation reported record revenue driven by hyper-scale demand for AI accelerated compute. "
            "Data center consolidated gross margin reached 75.10%, supported by strong software attachment rates. "
            "Blackwell B200 accelerators scaled into volume manufacturing on TSMC 4NP foundry nodes."
        ),
        metadata={"source": "NVIDIA Q3 SEC Form 10-Q", "entity": "NVDA"},
    )
    pipeline.index_document(
        doc_id="sec_tsmc_20f",
        text=(
            "TSMC reported maximum utilization across 3nm and 4nm semiconductor manufacturing capacity. "
            "Annual capital expenditure prioritized doubling CoWoS advanced packaging capacity to alleviate customer supply constraints. "
            "High performance computing customers including NVIDIA and AMD dominate advanced packaging allocations."
        ),
        metadata={"source": "TSMC Annual Form 20-F", "entity": "TSM"},
    )
    pipeline.index_document(
        doc_id="sec_amd_10q",
        text=(
            "Advanced Micro Devices accelerated data center deployments of Instinct MI300X AI solutions. "
            "Consolidated gross margin expanded to 51.50% as enterprise cloud deployments scaled."
        ),
        metadata={"source": "AMD Quarterly Report", "entity": "AMD"},
    )
    pipeline.index_document(
        doc_id="industry_asml_euv",
        text=(
            "ASML Holding shipped next-generation extreme ultraviolet lithography systems to premier foundries in Taiwan. "
            "High numerical aperture EUV equipment remains critical for fabricating sub-2nm silicon wafers."
        ),
        metadata={"source": "Semiconductor Industry Analysis", "entity": "ASML"},
    )

    # 3. Market Feed
    market_feed = {
        "nvda": {
            "ticker": "NVDA",
            "name": "NVIDIA Corporation",
            "market_cap_billions": 3120.0,
            "revenue_ttm_billions": 96.3,
            "gross_margin": 0.751,
            "operating_margin": 0.618,
            "pe_ratio": 46.2,
            "debt_to_equity": 0.18,
            "free_cash_flow_billions": 39.2,
            "primary_suppliers": ["TSMC", "ASML"],
        },
        "tsm": {
            "ticker": "TSM",
            "name": "Taiwan Semiconductor",
            "market_cap_billions": 890.0,
            "revenue_ttm_billions": 76.5,
            "gross_margin": 0.543,
            "operating_margin": 0.431,
            "pe_ratio": 24.8,
            "debt_to_equity": 0.28,
            "free_cash_flow_billions": 28.5,
            "primary_suppliers": ["ASML"],
        },
        "amd": {
            "ticker": "AMD",
            "name": "Advanced Micro Devices",
            "market_cap_billions": 255.0,
            "revenue_ttm_billions": 23.2,
            "gross_margin": 0.515,
            "operating_margin": 0.234,
            "pe_ratio": 115.0,
            "debt_to_equity": 0.05,
            "free_cash_flow_billions": 2.1,
            "primary_suppliers": ["TSMC"],
        },
    }

    # 4. Master Orchestrator
    orchestrator = MasterOrchestrator(
        event_bus=app_state.event_bus,
        pipeline=pipeline,
        graph=kg,
        market_feed=market_feed,
    )

    app_state.knowledge_graph = kg
    app_state.pipeline = pipeline
    app_state.orchestrator = orchestrator
    app_state.market_feed = market_feed
    app_state.is_ready = True

    return app_state
