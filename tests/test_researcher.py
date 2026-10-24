"""Unit test suite for ResearcherAgent, ResearchDossier, and citation grounding."""

import asyncio
import unittest
from unittest.mock import MagicMock

from src.agents.base import AgentState, ToolResult
from src.agents.researcher import (
    CitationItem,
    GraphProvenance,
    ResearchDossier,
    ResearcherAgent,
)
from src.core.events import EventBus
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.pipeline import (
    HybridRAGPipeline,
    PipelineContextItem,
    RetrievalPipelineOutput,
)
from src.knowledge_graph.schema import Entity, EntityType, FactTriple, Relation


class TestResearcherAgent(unittest.TestCase):
    """Test suite verifying qualitative evidence gathering, Knowledge Graph path discovery, and ReAct loop."""

    def setUp(self) -> None:
        self.event_bus = EventBus()

        # Build sample knowledge graph
        self.graph = KnowledgeGraph()
        e_nvda = Entity(entity_id="nvidia", name="NVIDIA", entity_type=EntityType.ORGANIZATION)
        e_tsmc = Entity(entity_id="tsmc", name="TSMC", entity_type=EntityType.ORGANIZATION)
        e_asml = Entity(entity_id="asml", name="ASML", entity_type=EntityType.ORGANIZATION)
        self.graph.add_entity(e_nvda)
        self.graph.add_entity(e_tsmc)
        self.graph.add_entity(e_asml)

        r_supp = Relation(relation_id="rel_supp", relation_type="supplier_to", weight=1.0)
        self.graph.add_triple(FactTriple(triple_id="t1", subject=e_tsmc, relation=r_supp, object=e_nvda, confidence=0.98))
        self.graph.add_triple(FactTriple(triple_id="t2", subject=e_asml, relation=r_supp, object=e_tsmc, confidence=0.95))

        # Build mock pipeline
        self.mock_pipeline = MagicMock(spec=HybridRAGPipeline)
        self.mock_output = RetrievalPipelineOutput(
            query="TSMC manufacturing moat",
            items=[
                PipelineContextItem(
                    doc_id="doc_tsmc_1",
                    text="TSMC produces over 90% of global advanced AI accelerators using CoWoS packaging.",
                    final_score=0.962,
                    stage1_rank=1,
                    stage1_score=0.89,
                    stage2_rank=1,
                    stage2_score=0.962,
                    matched_entities=["tsmc", "cowos"],
                )
            ],
            grounded_context_str="Sample grounded context",
            total_candidates_retrieved=10,
            total_reranked=5,
            latency_stage1_ms=1.1,
            latency_stage2_ms=2.5,
            total_latency_ms=3.6,
        )
        self.mock_pipeline.query.return_value = self.mock_output
        self.mock_pipeline.retrieve.return_value = self.mock_output

    def test_researcher_auto_registers_tools(self) -> None:
        """ResearcherAgent must automatically register retrieve_knowledge and traverse_knowledge_graph."""
        agent = ResearcherAgent(
            pipeline=self.mock_pipeline,
            graph=self.graph,
            event_bus=self.event_bus,
        )
        self.assertIn("retrieve_knowledge", agent.tools.tool_names)
        self.assertIn("traverse_knowledge_graph", agent.tools.tool_names)

    def test_conduct_research_with_rag_and_graph(self) -> None:
        """ResearcherAgent conduct_research must harvest RAG citations and graph paths."""
        agent = ResearcherAgent(
            pipeline=self.mock_pipeline,
            graph=self.graph,
            event_bus=self.event_bus,
        )

        dossier = asyncio.run(
            agent.conduct_research(
                task="Investigate TSMC manufacturing moat and supplier dependencies",
                primary_entity="TSMC",
            )
        )

        self.assertIsInstance(dossier, ResearchDossier)
        self.assertEqual(dossier.primary_entity, "TSMC")
        self.assertGreaterEqual(dossier.total_citations, 1)
        self.assertGreaterEqual(dossier.confidence_score, 0.5)

        # Verify citation content
        first_cit = dossier.citations[0]
        self.assertEqual(first_cit.doc_id, "doc_tsmc_1")
        self.assertIn("CoWoS packaging", first_cit.text)
        self.assertAlmostEqual(first_cit.relevance_score, 0.962, places=3)

        # Verify graph path: ASML / NVIDIA -> TSMC
        self.assertGreaterEqual(dossier.total_graph_paths, 1)
        path_sources = {p.source_entity for p in dossier.graph_paths}
        self.assertTrue("nvidia" in path_sources or "asml" in path_sources)
        for p in dossier.graph_paths:
            self.assertEqual(p.target_entity, "TSMC")

    def test_researcher_react_lifecycle(self) -> None:
        """ResearcherAgent run() must cycle through ReAct states and output Markdown report."""
        agent = ResearcherAgent(
            pipeline=self.mock_pipeline,
            graph=self.graph,
            event_bus=self.event_bus,
        )

        response = agent.run("Analyze NVIDIA GPU supply chain dependencies on TSMC and ASML")

        self.assertTrue(response.success)
        self.assertEqual(response.state, AgentState.COMPLETED)
        self.assertIn("# Qualitative Research Dossier", response.output)
        self.assertIn("Executive Summary", response.output)
        self.assertIn("Key Analytical Findings", response.output)

        # Check scratchpad variable
        dossier = agent.memory.get_variable("latest_dossier")
        self.assertIsInstance(dossier, ResearchDossier)
        self.assertEqual(dossier.primary_entity, "NVIDIA")

    def test_researcher_reflection_validation(self) -> None:
        """Reflection must accept grounded dossiers and reject empty/low-confidence ones."""
        agent = ResearcherAgent(
            pipeline=self.mock_pipeline,
            graph=self.graph,
            event_bus=self.event_bus,
        )

        # 1. Valid dossier
        dossier = asyncio.run(agent.conduct_research("Analyze Apple supply chain", primary_entity="Apple"))
        agent.memory.set_variable("latest_dossier", dossier)
        is_valid, msg = asyncio.run(agent.reflect("task", "thoughts", []))
        self.assertTrue(is_valid)
        self.assertIn("Research verified", msg)

        # 2. Missing dossier
        agent.memory.clear()
        is_valid, msg = asyncio.run(agent.reflect("task", "thoughts", []))
        self.assertFalse(is_valid)
        self.assertIn("failed to compile", msg)

        # 3. Low confidence dossier (< 0.2)
        dossier_low = ResearchDossier(
            task="Unknown query",
            primary_entity="GhostCorp",
            executive_summary="Empty",
            key_findings=["Finding"],
            confidence_score=0.1,
        )
        agent.memory.set_variable("latest_dossier", dossier_low)
        is_valid, msg = asyncio.run(agent.reflect("task", "thoughts", []))
        self.assertFalse(is_valid)
        self.assertIn("below minimum threshold", msg)

    def test_dossier_markdown_and_serialization(self) -> None:
        """ResearchDossier to_markdown and to_dict must produce complete structured schemas."""
        dossier = ResearchDossier(
            task="Evaluate ASML extreme ultraviolet lithography moat",
            primary_entity="ASML",
            executive_summary="ASML holds an effective monopoly in commercial High-NA and Low-NA EUV scanners.",
            key_findings=[
                "Sole commercial manufacturer of 0.33 NA and 0.55 NA EUV systems.",
                "Zeiss optics and Cymer CO2 laser light source provide high technical barrier to entry.",
            ],
            citations=[
                CitationItem(
                    doc_id="sec_10k_asml",
                    text="ASML ships EUV systems priced at >$200M each.",
                    relevance_score=0.9512,
                    matched_entities=["asml", "euv"],
                )
            ],
            graph_paths=[
                GraphProvenance(
                    source_entity="asml",
                    target_entity="tsmc",
                    reasoning_path="ASML --[supplier_to]--> TSMC",
                    confidence=0.98,
                    hop_count=1,
                )
            ],
            confidence_score=0.92,
        )

        # Test Markdown
        md = dossier.to_markdown()
        self.assertIn("# Qualitative Research Dossier: ASML", md)
        self.assertIn("ASML holds an effective monopoly", md)
        self.assertIn("Knowledge Graph Provenance", md)
        self.assertIn("Grounded Evidence & Citations", md)

        # Test Dictionary serialization
        d_dict = dossier.to_dict()
        self.assertEqual(d_dict["primary_entity"], "ASML")
        self.assertEqual(len(d_dict["citations"]), 1)
        self.assertEqual(len(d_dict["graph_paths"]), 1)
        self.assertEqual(d_dict["confidence_score"], 0.92)


if __name__ == "__main__":
    unittest.main()
