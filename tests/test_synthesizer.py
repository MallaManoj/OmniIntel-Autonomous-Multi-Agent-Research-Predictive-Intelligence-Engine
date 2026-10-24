"""Unit test suite for SynthesizerAgent, ExecutiveDossier, and final report compilation."""

import asyncio
import unittest

from src.agents.base import AgentState
from src.agents.critic import CriticIssue, VerificationAudit, VerificationVerdict
from src.agents.quant import ComputedMetric, EntityFinancialProfile, QuantReport
from src.agents.researcher import CitationItem, GraphProvenance, ResearchDossier
from src.agents.synthesizer import (
    ExecutiveDossier,
    StrategicRecommendation,
    SynthesizerAgent,
)
from src.core.events import EventBus, EventType


class TestSynthesizerAgent(unittest.TestCase):
    """Test suite verifying executive dossier synthesis, multi-agent consolidation, and Markdown generation."""

    def setUp(self) -> None:
        self.event_bus = EventBus()
        self.synthesizer = SynthesizerAgent(event_bus=self.event_bus)

        # 1. Research Dossier
        self.dossier = ResearchDossier(
            task="Evaluate NVIDIA AI moat and TSMC dependency",
            primary_entity="NVIDIA",
            executive_summary="NVIDIA possesses extensive software and hardware lock-in via CUDA.",
            key_findings=[
                "CUDA software ecosystem prevents competitive displacement by AMD or Intel.",
                "TSMC represents a critical single-source risk for advanced CoWoS packaging.",
            ],
            citations=[
                CitationItem(
                    doc_id="doc_10k_nvda",
                    text="NVIDIA datacenter acceleration revenue reached $47.5B in FY2024.",
                    relevance_score=0.965,
                    matched_entities=["nvidia", "cuda"],
                )
            ],
            graph_paths=[
                GraphProvenance(
                    source_entity="tsmc",
                    target_entity="nvidia",
                    reasoning_path="TSMC --[supplier_to]--> NVIDIA",
                    confidence=0.98,
                    hop_count=1,
                )
            ],
            confidence_score=0.88,
        )

        # 2. Quant Report
        self.quant_report = QuantReport(
            task="NVIDIA valuation multiples",
            entities_analyzed=["NVDA"],
            profiles={
                "NVDA": EntityFinancialProfile(
                    ticker="NVDA",
                    name="NVIDIA Corporation",
                    metrics={
                        "Gross Margin": ComputedMetric(
                            name="Gross Margin",
                            value=0.751,
                            formatted_value="75.10%",
                            category="Profitability",
                            formula="Gross Profit / Revenue",
                        ),
                        "Debt-to-Equity": ComputedMetric(
                            name="Debt-to-Equity",
                            value=0.18,
                            formatted_value="0.18",
                            category="Leverage",
                            formula="Debt / Equity",
                        ),
                    },
                )
            },
            comparative_summary="NVIDIA exhibits elite gross margins of 75.10% and low leverage.",
            calculations_performed=["gross_margin(75.1, 100)"],
        )

        # 3. Critic Audit
        self.audit = VerificationAudit(
            task="Audit NVIDIA intelligence",
            verdict=VerificationVerdict.APPROVED,
            confidence_score=0.95,
            passed_checks=["Factual citations verified", "Numerical consistency confirmed"],
            critique_summary="All integrity checks cleared successfully.",
        )

    def test_full_synthesis_integration(self) -> None:
        """Synthesizer must combine research, quant, and audit into a comprehensive ExecutiveDossier."""
        dossier = self.synthesizer.synthesize(
            query="Evaluate NVIDIA AI compute dominance and supply chain resilience",
            research_dossier=self.dossier,
            quant_report=self.quant_report,
            audit=self.audit,
        )

        self.assertIsInstance(dossier, ExecutiveDossier)
        self.assertIn("NVIDIA", dossier.target_entities)
        self.assertEqual(dossier.audit_verdict, "APPROVED")
        self.assertGreaterEqual(dossier.overall_confidence, 0.8)

        # Check sections
        self.assertGreaterEqual(len(dossier.qualitative_findings), 2)
        self.assertGreaterEqual(len(dossier.supply_chain_links), 1)
        self.assertIn("75.10%", dossier.financial_table_markdown)
        self.assertGreaterEqual(len(dossier.recommendations), 1)

        # Recommendation for high margin + low leverage should be ACCUMULATE
        rec = dossier.recommendations[0]
        self.assertEqual(rec.action, "ACCUMULATE")
        self.assertIn("Gross Margin >= 50%", rec.rationale)
        self.assertTrue(any("TSMC" in risk for risk in rec.key_risks))

    def test_partial_inputs_handling(self) -> None:
        """Synthesizer must gracefully handle missing quant or research inputs."""
        dossier = self.synthesizer.synthesize(
            query="Analyze TSMC geopolitical risks",
            research_dossier=self.dossier,
            quant_report=None,
            audit=None,
        )

        self.assertEqual(dossier.audit_verdict, "NOT_AUDITED")
        self.assertGreaterEqual(len(dossier.qualitative_findings), 1)
        self.assertGreaterEqual(dossier.overall_confidence, 0.5)

    def test_synthesizer_react_lifecycle_execution(self) -> None:
        """SynthesizerAgent run() must cycle through ReAct states and emit REPORT_GENERATED event."""
        context = {
            "research_dossier": self.dossier,
            "quant_report": self.quant_report,
            "audit": self.audit,
        }

        response = self.synthesizer.run(
            "Compile executive briefing on NVIDIA",
            context=context,
        )

        self.assertTrue(response.success)
        self.assertEqual(response.state, AgentState.COMPLETED)
        self.assertIn("# Strategic Intelligence Dossier", response.output)
        self.assertIn("Executive Summary & Strategic Thesis", response.output)
        self.assertIn("Quantitative Financial Modeling", response.output)
        self.assertIn("Strategic Recommendations & Risk Matrix", response.output)

        # Check scratchpad variable
        cached_dossier = self.synthesizer.memory.get_variable("latest_dossier")
        self.assertIsInstance(cached_dossier, ExecutiveDossier)

    def test_synthesizer_reflection_validation(self) -> None:
        """Reflection must accept sound dossiers and reject empty or low-confidence ones."""
        # 1. Valid dossier
        dossier = self.synthesizer.synthesize("Test query", research_dossier=self.dossier, quant_report=self.quant_report)
        self.synthesizer.memory.set_variable("latest_dossier", dossier)
        is_valid, msg = asyncio.run(self.synthesizer.reflect("task", "thoughts", []))
        self.assertTrue(is_valid)
        self.assertIn("Synthesis verified", msg)

        # 2. Missing dossier
        self.synthesizer.memory.clear()
        is_valid, msg = asyncio.run(self.synthesizer.reflect("task", "thoughts", []))
        self.assertFalse(is_valid)
        self.assertIn("failed to retain", msg)

    def test_executive_dossier_markdown_and_serialization(self) -> None:
        """ExecutiveDossier to_markdown and to_dict must output complete structured schemas."""
        dossier = self.synthesizer.synthesize(
            query="Evaluate ASML lithography monopoly",
            research_dossier=self.dossier,
            quant_report=self.quant_report,
            audit=self.audit,
        )

        md = dossier.to_markdown()
        self.assertIn("Strategic Intelligence Dossier", md)
        self.assertIn("**QA Audit Status:** `APPROVED`", md)
        self.assertIn("OmniIntel Autonomous Multi-Agent", md)

        d = dossier.to_dict()
        self.assertEqual(d["audit_verdict"], "APPROVED")
        self.assertIn("strategic_thesis", d)
        self.assertIn("recommendations", d)


if __name__ == "__main__":
    unittest.main()
