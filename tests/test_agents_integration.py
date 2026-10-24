"""End-to-End Multi-Agent Integration Test Suite for Phase 4: Autonomous Agentic AI System.

Validates the full multi-agent cognitive architecture working in concert with real
deep learning and knowledge graph subsystems:
  1. High-level directive planning and Kahn's algorithm DAG decomposition
  2. Concurrent agent execution (ResearcherAgent and QuantAgent) via asyncio.gather()
  3. Real Hybrid RAG retrieval (BM25 + Dense + Cross-Encoder reranking)
  4. Real Knowledge Graph traversal and Dijkstra most reliable path discovery
  5. Deterministic AST-sandboxed financial modeling with verified formulas
  6. Adversarial Critic cross-examination and citation grounding audits
  7. Closed-loop dynamic self-correction and graph surgery on Critic rejection
  8. Executive synthesis into C-suite Markdown briefings with strategic recommendations
  9. Asynchronous EventBus telemetry streaming and monotonic audit logging
 10. Resilient error boundaries and loop guards (max_replans)

Compatible with standard library `unittest` and `pytest`.
"""

import asyncio
from pathlib import Path
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from src.agents.base import AgentMemory, AgentState, ToolRegistry
from src.agents.critic import (
    CriticAgent,
    CriticIssue,
    VerificationAudit,
    VerificationVerdict,
)
from src.agents.orchestrator import MasterOrchestrator, OrchestratorResult
from src.agents.planner import ExecutionPlan, PlanComplexity, PlannerAgent
from src.agents.quant import QuantAgent, QuantReport
from src.agents.researcher import CitationItem, ResearchDossier, ResearcherAgent
from src.agents.synthesizer import ExecutiveDossier, SynthesizerAgent
from src.core.dsa.dag import TaskState
from src.core.events import Event, EventBus, EventType
from src.knowledge_graph.chunker import TextChunker
from src.knowledge_graph.embeddings import DenseEmbeddingModel, EmbeddingEngine
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.hybrid_search import HybridSearchConfig, HybridSearchEngine
from src.knowledge_graph.pipeline import (
    HybridRAGPipeline,
    PipelineContextItem,
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


class TestAgentsIntegration(unittest.TestCase):
    """Production-grade End-to-End Multi-Agent Integration Test Suite."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize real multi-component stack: KG, VectorStore, Reranker, and RAG."""
        torch.manual_seed(42)
        np.random.seed(42)
        cls.embedding_dim = 32
        cls.max_seq_len = 64

        # 1. Knowledge Graph Fixture: Global Semiconductor & Cloud Ecosystem
        cls.kg = KnowledgeGraph()
        cls.e_nvda = Entity(
            entity_id="nvidia",
            name="NVIDIA Corporation",
            entity_type=EntityType.ORGANIZATION,
            aliases=["NVDA", "nvidia", "Nvidia"],
        )
        cls.e_tsmc = Entity(
            entity_id="tsmc",
            name="TSMC",
            entity_type=EntityType.ORGANIZATION,
            aliases=["Taiwan Semiconductor", "tsmc", "TSM"],
        )
        cls.e_asml = Entity(
            entity_id="asml",
            name="ASML Holding",
            entity_type=EntityType.ORGANIZATION,
            aliases=["ASML", "asml"],
        )
        cls.e_amd = Entity(
            entity_id="amd",
            name="Advanced Micro Devices",
            entity_type=EntityType.ORGANIZATION,
            aliases=["AMD", "amd"],
        )
        cls.e_blackwell = Entity(
            entity_id="blackwell",
            name="Blackwell B200",
            entity_type=EntityType.PRODUCT,
            aliases=["Blackwell", "B200"],
        )

        cls.kg.add_triple(
            FactTriple(
                triple_id="t1",
                subject=cls.e_asml,
                relation=Relation(relation_id="r1", relation_type="supplier_to"),
                object=cls.e_tsmc,
                confidence=0.98,
            )
        )
        cls.kg.add_triple(
            FactTriple(
                triple_id="t2",
                subject=cls.e_tsmc,
                relation=Relation(relation_id="r2", relation_type="supplier_to"),
                object=cls.e_nvda,
                confidence=0.99,
            )
        )
        cls.kg.add_triple(
            FactTriple(
                triple_id="t3",
                subject=cls.e_tsmc,
                relation=Relation(relation_id="r3", relation_type="supplier_to"),
                object=cls.e_amd,
                confidence=0.94,
            )
        )
        cls.kg.add_triple(
            FactTriple(
                triple_id="t4",
                subject=cls.e_nvda,
                relation=Relation(relation_id="r4", relation_type="develops"),
                object=cls.e_blackwell,
                confidence=1.0,
            )
        )

        # 2. Vector Store, Embedding Engine, and Chunker
        cls.vector_store = VectorStore(dimension=cls.embedding_dim, metric="cosine")
        cls.embedding_engine = EmbeddingEngine(
            embedding_dim=cls.embedding_dim,
            max_seq_len=cls.max_seq_len,
        )
        cls.chunker = TextChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=5)

        cls.hybrid_engine = HybridSearchEngine(
            vector_store=cls.vector_store,
            embedding_engine=cls.embedding_engine,
            knowledge_graph=cls.kg,
            chunker=cls.chunker,
        )

        # 3. Neural Cross-Encoder Reranker
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

        # 4. Full Real Hybrid RAG Pipeline
        cls.pipeline = HybridRAGPipeline(
            hybrid_engine=cls.hybrid_engine,
            reranker=cls.reranker,
        )

        # Ingest multi-source intelligence reports
        cls.pipeline.index_document(
            doc_id="sec_nvda_10q",
            text=(
                "NVIDIA Corporation delivered outstanding financial performance driven by hyper-scale demand "
                "for accelerated computing and generative artificial intelligence architectures. "
                "Data center revenue reached record levels, supporting high consolidated gross margin of 75.1%. "
                "NVIDIA Blackwell B200 GPUs entered full-scale volume production utilizing advanced TSMC 4NP foundry nodes."
            ),
            metadata={"source": "NVIDIA Q3 SEC Form 10-Q", "entity": "NVDA"},
        )
        cls.pipeline.index_document(
            doc_id="sec_tsmc_20f",
            text=(
                "TSMC confirmed full utilization across 3nm and 4nm advanced semiconductor manufacturing lines. "
                "Capital expenditures reached $30.0 billion to accelerate CoWoS advanced packaging capacity expansion. "
                "High performance computing customers including NVIDIA and AMD dominate advanced node allocation."
            ),
            metadata={"source": "TSMC Annual Form 20-F", "entity": "TSM"},
        )
        cls.pipeline.index_document(
            doc_id="industry_asml_euv",
            text=(
                "ASML Holding shipped next-generation extreme ultraviolet lithography systems to premier foundries in Taiwan. "
                "High numerical aperture EUV equipment is essential for fabricating sub-2nm silicon wafers."
            ),
            metadata={"source": "Semiconductor Industry Analysis", "entity": "ASML"},
        )
        cls.pipeline.index_document(
            doc_id="sec_amd_10q",
            text=(
                "Advanced Micro Devices reported rapid datacenter adoption of Instinct MI300X AI accelerators. "
                "AMD consolidated gross margin expanded to 51.5% as enterprise cloud deployments scaled."
            ),
            metadata={"source": "AMD Quarterly Report", "entity": "AMD"},
        )

        # 5. Verified Market Data Feed (Fraction-format margins matching tools.py)
        cls.market_feed = {
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
                "gross_margin": 0.485,
                "operating_margin": 0.114,
                "pe_ratio": 115.0,
                "debt_to_equity": 0.05,
                "free_cash_flow_billions": 2.1,
                "primary_suppliers": ["TSMC"],
            },
        }

    # -------------------------------------------------------------------------
    # Test 1: Real End-to-End Single-Entity Research Mission
    # -------------------------------------------------------------------------

    def test_end_to_end_real_pipeline_single_entity(self) -> None:
        """MasterOrchestrator must execute real RAG + KG + Quant + Critic + Synthesizer to completion."""
        event_bus = EventBus()
        captured_events: list[str] = []
        event_bus.subscribe("*", lambda e: captured_events.append(e.event_type))

        orchestrator = MasterOrchestrator(
            event_bus=event_bus,
            pipeline=self.pipeline,
            graph=self.kg,
            market_feed=self.market_feed,
        )

        query = (
            "Analyze NVIDIA Blackwell computing architecture, data center gross margins, "
            "and TSMC supplier dependency for sovereign AI infrastructure."
        )

        result: OrchestratorResult = orchestrator.run(query)

        # 1. Mission Completion & Performance Invariants
        self.assertTrue(result.success, f"Mission failed with errors: {result.errors}")
        self.assertGreaterEqual(result.total_stages_executed, 3)
        self.assertGreater(result.total_latency_ms, 0.0)
        self.assertEqual(len(result.errors), 0)

        # 2. Plan Invariants
        self.assertIsNotNone(result.plan)
        self.assertGreaterEqual(result.plan.total_tasks, 3)

        # 3. Research Dossier Invariants (Grounding & Citations)
        dossier: ExecutiveDossier = result.dossier
        self.assertIsNotNone(dossier)
        self.assertIn("NVIDIA", dossier.target_entities)
        self.assertGreaterEqual(len(dossier.citations), 1)
        self.assertGreaterEqual(len(dossier.qualitative_findings), 1)

        # 4. Supply Chain Graph Provenance (Dijkstra Traversal)
        self.assertGreaterEqual(len(dossier.supply_chain_links), 1)
        path_str = " ".join(dossier.supply_chain_links)
        self.assertTrue("TSMC" in path_str or "asml" in path_str.lower() or "supplier" in path_str.lower())

        # 5. Financial Modeling Invariants
        self.assertIn("75.10%", dossier.financial_table_markdown)
        self.assertIn("NVDA", dossier.financial_table_markdown)

        # 6. Critic Verification Audit Invariants
        self.assertIsNotNone(result.audit)
        self.assertEqual(result.audit.verdict, VerificationVerdict.APPROVED)
        self.assertGreaterEqual(result.audit.confidence_score, 0.70)

        # 7. Strategic Synthesis & Markdown Output
        self.assertGreaterEqual(dossier.overall_confidence, 0.70)
        self.assertEqual(dossier.audit_verdict, "APPROVED")
        self.assertGreaterEqual(len(dossier.recommendations), 1)

        markdown = result.to_markdown()
        self.assertIn("# Strategic Intelligence Dossier", markdown)
        self.assertIn("Executive Summary", markdown)
        self.assertIn("Key Qualitative & Competitive Moats", markdown)
        self.assertIn("Quantitative Financial Modeling", markdown)
        self.assertIn("Strategic Recommendations", markdown)

        # 8. Event Telemetry Invariants
        self.assertIn("SYSTEM_STARTED", captured_events)
        self.assertIn("TASK_SCHEDULED", captured_events)
        self.assertIn("TASK_STARTED", captured_events)
        self.assertIn("TOOL_CALLED", captured_events)
        self.assertIn("TOOL_RETURNED", captured_events)
        self.assertIn("TASK_COMPLETED", captured_events)
        self.assertIn("REPORT_GENERATED", captured_events)

    # -------------------------------------------------------------------------
    # Test 2: Multi-Entity Comparative Mission Execution
    # -------------------------------------------------------------------------

    def test_end_to_end_multi_entity_comparative_mission(self) -> None:
        """MasterOrchestrator must handle multi-entity queries, parallel batching, and side-by-side matrices."""
        orchestrator = MasterOrchestrator(
            pipeline=self.pipeline,
            graph=self.kg,
            market_feed=self.market_feed,
        )

        query = (
            "Compare NVIDIA and AMD in datacenter AI market share, gross margins, "
            "valuation multiples, and TSMC foundry manufacturing capacity."
        )

        result: OrchestratorResult = orchestrator.run(query)

        self.assertTrue(result.success)
        dossier = result.dossier
        self.assertIsNotNone(dossier)

        # Verify multiple entities recognized
        analyzed_entities = set(dossier.target_entities)
        self.assertTrue("NVIDIA" in analyzed_entities or "NVDA" in analyzed_entities)
        self.assertIn("AMD", analyzed_entities)

        # Check side-by-side comparative markdown table presence
        markdown = dossier.to_markdown()
        self.assertIn("| Metric | Category | NVDA | AMD |", markdown)

    # -------------------------------------------------------------------------
    # Test 3: ReAct Cognitive Lifecycle & Memory Invariants
    # -------------------------------------------------------------------------

    def test_react_agent_lifecycle_and_state_transitions(self) -> None:
        """Agents must transition strictly through ReAct states and maintain clean memory bounds."""
        orchestrator = MasterOrchestrator(
            pipeline=self.pipeline,
            graph=self.kg,
            market_feed=self.market_feed,
        )

        query = (
            "Analyze NVIDIA GPU supply chain, TSMC foundry reliance, and compute gross margins "
            "for strategic intelligence briefing."
        )
        result = orchestrator.run(query)

        self.assertTrue(result.success)

        # Verify that agents completed their tasks successfully
        self.assertEqual(orchestrator.researcher.state, AgentState.COMPLETED)
        self.assertEqual(orchestrator.quant.state, AgentState.COMPLETED)
        self.assertEqual(orchestrator.critic.state, AgentState.COMPLETED)
        self.assertEqual(orchestrator.synthesizer.state, AgentState.COMPLETED)

        # Verify memory structures: message history recorded and max bounds respected
        for agent in orchestrator.agent_pool.values():
            self.assertGreaterEqual(agent.memory.max_messages, 1)
            self.assertIsInstance(agent.memory.get_messages(), list)

    # -------------------------------------------------------------------------
    # Test 4: Closed-Loop Self-Correction with Dynamic Graph Surgery
    # -------------------------------------------------------------------------

    def test_closed_loop_self_correction_with_adversarial_injection(self) -> None:
        """When Critic rejects an ungrounded or conflicting claim, Orchestrator must dynamically replan and heal."""
        event_bus = EventBus()
        rejected_events: list[dict] = []
        event_bus.subscribe(
            EventType.CRITIC_REJECTED,
            lambda e: rejected_events.append(e.payload),
        )

        # Custom mock Critic that forces a rejection on the 1st cycle and approves on the 2nd cycle
        mock_critic = MagicMock(spec=CriticAgent)
        mock_critic.agent_id = "critic"
        mock_critic.name = "Critic / Verifier Agent"
        mock_critic.tools = MagicMock()

        audit_rejected = VerificationAudit(
            task="Audit preliminary research",
            verdict=VerificationVerdict.REJECTED,
            confidence_score=0.42,
            issues=[
                CriticIssue(
                    severity="CRITICAL",
                    category="NUMERICAL_MISMATCH",
                    description="Qualitative section claims 45.0% gross margin conflicting with 75.1% quant metric.",
                    location="Research Dossier",
                    remediation_advice="Re-extract verified gross margin from SEC 10-Q filing.",
                )
            ],
            remediation_feedback="Remediate conflicting gross margin metric to match verified 75.1% 10-Q filing.",
        )

        audit_approved = VerificationAudit(
            task="Audit remediated research",
            verdict=VerificationVerdict.APPROVED,
            confidence_score=0.95,
            issues=[],
            remediation_feedback=None,
        )

        mock_critic.audit.side_effect = [audit_rejected, audit_approved]

        orchestrator = MasterOrchestrator(
            event_bus=event_bus,
            pipeline=self.pipeline,
            graph=self.kg,
            market_feed=self.market_feed,
            critic=mock_critic,
        )

        query = "Analyze NVIDIA gross margin and TSMC supply chain"
        result = orchestrator.run(query, max_replans=2)

        # Invariants of successful self-correction
        self.assertTrue(result.success)
        self.assertEqual(result.replan_count, 1)
        self.assertEqual(len(result.remediation_history), 1)
        self.assertEqual(len(rejected_events), 1)

        remediation_record = result.remediation_history[0]
        self.assertEqual(remediation_record["replan_index"], 1)
        self.assertIn("Remediate", remediation_record["feedback"])

        # Final audit must be the approved audit
        self.assertIsNotNone(result.audit)
        self.assertEqual(result.audit.verdict, VerificationVerdict.APPROVED)
        self.assertIsNotNone(result.dossier)

    # -------------------------------------------------------------------------
    # Test 5: Infinite Loop Prevention & Max Replans Boundary Guard
    # -------------------------------------------------------------------------

    def test_infinite_loop_prevention_max_replans(self) -> None:
        """Orchestrator must respect max_replans budget and abort gracefully if issues persist."""
        mock_critic = MagicMock(spec=CriticAgent)
        mock_critic.agent_id = "critic"
        mock_critic.name = "Critic / Verifier Agent"
        mock_critic.tools = MagicMock()

        # Always reject
        audit_fatal = VerificationAudit(
            task="Audit unresolvable report",
            verdict=VerificationVerdict.REJECTED,
            confidence_score=0.20,
            issues=[
                CriticIssue(
                    severity="CRITICAL",
                    category="UNGROUNDED_CLAIM",
                    description="Persistent hallucination defect.",
                    location="Research Dossier",
                    remediation_advice="Requires human intervention.",
                )
            ],
            remediation_feedback="Fatal citation deficit.",
        )
        mock_critic.audit.return_value = audit_fatal

        orchestrator = MasterOrchestrator(
            pipeline=self.pipeline,
            graph=self.kg,
            market_feed=self.market_feed,
            critic=mock_critic,
        )

        query = "Investigate ungrounded quantum claim"
        # Permitted at most 1 replan
        result = orchestrator.run(query, max_replans=1)

        self.assertFalse(result.success)
        self.assertEqual(result.replan_count, 1)
        self.assertEqual(result.audit.verdict, VerificationVerdict.REJECTED)

    # -------------------------------------------------------------------------
    # Test 6: Input Boundaries & Edge Case Resilience
    # -------------------------------------------------------------------------

    def test_empty_query_and_edge_cases(self) -> None:
        """MasterOrchestrator must reject empty queries and handle unindexed entities gracefully."""
        orchestrator = MasterOrchestrator(
            pipeline=self.pipeline,
            graph=self.kg,
            market_feed=self.market_feed,
        )

        # Empty string
        with self.assertRaises(ValueError):
            orchestrator.run("")

        # Whitespace only
        with self.assertRaises(ValueError):
            orchestrator.run("   \n\t  ")

        # Unknown entity query should complete gracefully without unhandled exceptions
        unknown_result = orchestrator.run("Analyze QuantumFusionX HyperDrive technologies")
        self.assertIsNotNone(unknown_result)
        self.assertGreater(unknown_result.total_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
