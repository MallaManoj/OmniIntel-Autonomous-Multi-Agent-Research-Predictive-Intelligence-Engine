"""Unit test suite for Closed-Loop Self-Correction & Dynamic Replanning in MasterOrchestrator."""

import asyncio
import unittest
from unittest.mock import MagicMock

from src.agents.base import AgentState
from src.agents.critic import (
    CriticAgent,
    CriticIssue,
    VerificationAudit,
    VerificationVerdict,
)
from src.agents.orchestrator import MasterOrchestrator, OrchestratorResult
from src.agents.planner import PlannerAgent
from src.agents.quant import QuantAgent
from src.agents.researcher import CitationItem, ResearchDossier, ResearcherAgent
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


class TestSelfCorrectionLoop(unittest.TestCase):
    """Test suite verifying dynamic graph surgery, Critic rejection handling, and retry loops."""

    def setUp(self) -> None:
        self.event_bus = EventBus()

        # Build Knowledge Graph fixture
        self.graph = KnowledgeGraph()
        e_nvda = Entity(entity_id="nvidia", name="NVIDIA Corporation", entity_type=EntityType.ORGANIZATION)
        self.graph.add_entity(e_nvda)

        # Build Mock Hybrid RAG Pipeline
        self.mock_pipeline = MagicMock(spec=HybridRAGPipeline)
        mock_output = RetrievalPipelineOutput(
            query="NVIDIA Blackwell",
            items=[
                PipelineContextItem(
                    doc_id="doc_nvda_1",
                    text="NVIDIA Blackwell architecture features verified 75.1% gross margin.",
                    final_score=0.96,
                    stage1_rank=1,
                    stage1_score=0.90,
                    stage2_rank=1,
                    stage2_score=0.96,
                    matched_entities=["nvidia"],
                )
            ],
            grounded_context_str="Sample context",
            total_candidates_retrieved=10,
            total_reranked=5,
            latency_stage1_ms=1.0,
            latency_stage2_ms=2.0,
            total_latency_ms=3.0,
        )
        self.mock_pipeline.query.return_value = mock_output
        self.mock_pipeline.retrieve.return_value = mock_output

    def test_self_correction_dynamic_replanning_loop(self) -> None:
        """When Critic rejects on cycle 0, Orchestrator must dynamically replan and succeed on cycle 1."""
        # Create custom mock Critic that rejects on first call and approves on second call
        mock_critic = MagicMock(spec=CriticAgent)
        mock_critic.agent_id = "critic"
        mock_critic.name = "Critic / Verifier Agent"
        mock_critic.tools = MagicMock()

        audit_rejected = VerificationAudit(
            task="Audit initial draft",
            verdict=VerificationVerdict.REJECTED,
            confidence_score=0.45,
            issues=[
                CriticIssue(
                    severity="CRITICAL",
                    category="UNGROUNDED_CLAIM",
                    description="Initial draft lacked empirical CoWoS packaging citations.",
                    location="Research Dossier",
                    remediation_advice="Re-query knowledge base for advanced packaging CoWoS capacity.",
                )
            ],
            critique_summary="Rejected initial draft due to ungrounded packaging claims.",
            remediation_feedback="Re-query knowledge base for advanced packaging CoWoS capacity.",
        )

        audit_approved = VerificationAudit(
            task="Audit remediated draft",
            verdict=VerificationVerdict.APPROVED,
            confidence_score=0.95,
            passed_checks=["Packaging citations verified", "Numerical consistency confirmed"],
            critique_summary="Remediated research dossier cleared all verification checks.",
            remediation_feedback=None,
        )

        # First audit call returns REJECTED, subsequent call returns APPROVED
        mock_critic.audit.side_effect = [audit_rejected, audit_approved]

        orchestrator = MasterOrchestrator(
            event_bus=self.event_bus,
            pipeline=self.mock_pipeline,
            graph=self.graph,
            critic=mock_critic,
        )

        recorded_events: list[Event] = []
        self.event_bus.subscribe("*", lambda e: recorded_events.append(e))

        query = "Evaluate NVIDIA advanced packaging dependencies"
        result: OrchestratorResult = orchestrator.run(query, max_replans=2)

        # 1. Mission should ultimately succeed
        self.assertTrue(result.success)
        self.assertIsNotNone(result.dossier)

        # 2. Replan count should be exactly 1
        self.assertEqual(result.replan_count, 1)
        self.assertEqual(len(result.remediation_history), 1)

        remedy_entry = result.remediation_history[0]
        self.assertEqual(remedy_entry["replan_index"], 1)
        self.assertIn("advanced packaging", remedy_entry["feedback"])

        # 3. Final audit on the dossier should be APPROVED
        self.assertEqual(result.dossier.audit_verdict, "APPROVED")

        # 4. EventBus should have recorded CRITIC_REJECTED followed by subsequent TASK_SCHEDULED
        event_types = [e.event_type for e in recorded_events]
        self.assertIn(EventType.CRITIC_REJECTED.value, event_types)
        scheduled_events = [e for e in recorded_events if e.event_type == EventType.TASK_SCHEDULED.value]
        self.assertGreaterEqual(len(scheduled_events), 2)  # Initial plan + replanned DAG

    def test_max_replans_exhaustion_safety(self) -> None:
        """If Critic persistently rejects, Orchestrator must stop after max_replans without deadlock."""
        always_reject_critic = MagicMock(spec=CriticAgent)
        always_reject_critic.agent_id = "critic"
        always_reject_critic.name = "Critic / Verifier Agent"
        always_reject_critic.tools = MagicMock()

        audit_persistently_rejected = VerificationAudit(
            task="Persistently failing",
            verdict=VerificationVerdict.REJECTED,
            confidence_score=0.3,
            issues=[
                CriticIssue(
                    severity="CRITICAL",
                    category="MISSING_EVIDENCE",
                    description="Unrecoverable missing data.",
                    location="Research Dossier",
                    remediation_advice="External feed unavailable.",
                )
            ],
            remediation_feedback="External feed unavailable.",
        )
        always_reject_critic.audit.return_value = audit_persistently_rejected

        orchestrator = MasterOrchestrator(
            event_bus=self.event_bus,
            pipeline=self.mock_pipeline,
            graph=self.graph,
            critic=always_reject_critic,
        )

        query = "Investigate unindexed speculative asset"
        result: OrchestratorResult = orchestrator.run(query, max_replans=1)

        # Replanning should halt once max_replans (1) is hit
        self.assertEqual(result.replan_count, 1)
        self.assertGreaterEqual(result.total_stages_executed, 2)


if __name__ == "__main__":
    unittest.main()
