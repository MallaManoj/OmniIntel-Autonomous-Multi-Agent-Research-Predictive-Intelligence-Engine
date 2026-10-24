"""Unit test suite for MasterOrchestrator, parallel DAG execution, and mission streaming."""

import asyncio
import unittest
from unittest.mock import MagicMock

from src.agents.base import AgentState
from src.agents.critic import CriticAgent, VerificationVerdict
from src.agents.orchestrator import MasterOrchestrator, OrchestratorResult
from src.agents.planner import PlannerAgent
from src.agents.quant import QuantAgent
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.core.dsa.dag import TaskState
from src.core.events import Event, EventBus, EventType
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.pipeline import (
    HybridRAGPipeline,
    PipelineContextItem,
    RetrievalPipelineOutput,
)
from src.knowledge_graph.schema import Entity, EntityType, FactTriple, Relation


class TestMasterOrchestrator(unittest.TestCase):
    """Test suite verifying end-to-end multi-agent orchestration, Kahn batching, and event streaming."""

    def setUp(self) -> None:
        self.event_bus = EventBus()

        # Build Knowledge Graph fixture
        self.graph = KnowledgeGraph()
        e_nvda = Entity(entity_id="nvidia", name="NVIDIA Corporation", entity_type=EntityType.ORGANIZATION)
        e_tsmc = Entity(entity_id="tsmc", name="TSMC", entity_type=EntityType.ORGANIZATION)
        self.graph.add_entity(e_nvda)
        self.graph.add_entity(e_tsmc)

        r_supp = Relation(relation_id="rel_supp", relation_type="supplier_to", weight=1.0)
        self.graph.add_triple(FactTriple(triple_id="t1", subject=e_tsmc, relation=r_supp, object=e_nvda, confidence=0.98))

        # Build Mock Hybrid RAG Pipeline
        self.mock_pipeline = MagicMock(spec=HybridRAGPipeline)
        mock_output = RetrievalPipelineOutput(
            query="NVIDIA AI computing",
            items=[
                PipelineContextItem(
                    doc_id="doc_nvda_1",
                    text="NVIDIA Blackwell architecture features 208 billion transistors manufactured on TSMC 4NP.",
                    final_score=0.97,
                    stage1_rank=1,
                    stage1_score=0.91,
                    stage2_rank=1,
                    stage2_score=0.97,
                    matched_entities=["nvidia", "tsmc"],
                )
            ],
            grounded_context_str="Sample context",
            total_candidates_retrieved=10,
            total_reranked=5,
            latency_stage1_ms=1.2,
            latency_stage2_ms=2.1,
            total_latency_ms=3.3,
        )
        self.mock_pipeline.query.return_value = mock_output
        self.mock_pipeline.retrieve.return_value = mock_output

        self.orchestrator = MasterOrchestrator(
            event_bus=self.event_bus,
            pipeline=self.mock_pipeline,
            graph=self.graph,
        )

    def test_orchestrator_initialization(self) -> None:
        """MasterOrchestrator must initialize all specialized agents and agent pool."""
        self.assertIsInstance(self.orchestrator.planner, PlannerAgent)
        self.assertIsInstance(self.orchestrator.researcher, ResearcherAgent)
        self.assertIsInstance(self.orchestrator.quant, QuantAgent)
        self.assertIsInstance(self.orchestrator.critic, CriticAgent)
        self.assertIsInstance(self.orchestrator.synthesizer, SynthesizerAgent)
        self.assertEqual(len(self.orchestrator.agent_pool), 5)

    def test_end_to_end_mission_execution(self) -> None:
        """Orchestrator must execute full research mission across parallel DAG stages to completion."""
        query = "Evaluate NVIDIA competitive moat, TSMC supplier dependency, and gross margins"
        result: OrchestratorResult = self.orchestrator.run(query)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.dossier)
        self.assertGreaterEqual(result.total_stages_executed, 3)
        self.assertGreater(result.total_latency_ms, 0.0)
        self.assertEqual(len(result.errors), 0)

        # Verify dossier contents
        dossier = result.dossier
        self.assertIn("NVIDIA", dossier.target_entities)
        self.assertEqual(dossier.audit_verdict, "APPROVED")
        self.assertGreaterEqual(dossier.overall_confidence, 0.70)
        self.assertGreaterEqual(len(dossier.qualitative_findings), 1)

        # Check DAG node states
        for task_id, node in result.plan.dag.nodes.items():
            self.assertEqual(node.state, TaskState.COMPLETED, f"Task '{task_id}' was not completed.")
            self.assertIsNotNone(node.result, f"Task '{task_id}' has no result payload.")

    def test_event_bus_streaming_telemetry(self) -> None:
        """Mission execution must stream lifecycle events to the EventBus."""
        recorded_events: list[Event] = []

        async def capture_event(event: Event) -> None:
            recorded_events.append(event)

        # Subscribe to all events via wildcard
        self.event_bus.subscribe("*", capture_event)

        query = "Analyze Apple financial margins and supplier dependencies"
        result = self.orchestrator.run(query)

        self.assertTrue(result.success)
        event_types = [e.event_type for e in recorded_events]

        # Verify key system events fired
        self.assertIn(EventType.SYSTEM_STARTED.value, event_types)
        self.assertIn(EventType.TASK_SCHEDULED.value, event_types)
        self.assertIn(EventType.TASK_STARTED.value, event_types)
        self.assertIn(EventType.TASK_COMPLETED.value, event_types)
        self.assertIn(EventType.REPORT_GENERATED.value, event_types)

    def test_orchestrator_empty_query_error(self) -> None:
        """Empty or whitespace query must raise ValueError."""
        with self.assertRaises(ValueError):
            self.orchestrator.run("")
        with self.assertRaises(ValueError):
            self.orchestrator.run("   \n\t  ")

    def test_orchestrator_result_serialization(self) -> None:
        """OrchestratorResult to_dict and to_markdown must output complete structured schemas."""
        query = "Evaluate Microsoft and AMD cloud margins"
        result = self.orchestrator.run(query)

        md = result.to_markdown()
        self.assertIn("Strategic Intelligence Dossier", md)
        self.assertIn("OmniIntel Autonomous Multi-Agent", md)

        d = result.to_dict()
        self.assertEqual(d["query"], query)
        self.assertTrue(d["success"])
        self.assertIn("plan", d)
        self.assertIn("dossier", d)
        self.assertIn("total_latency_ms", d)


if __name__ == "__main__":
    unittest.main()
