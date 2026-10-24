"""Unit test suite for CriticAgent, VerificationAudit, and adversarial cross-examination."""

import asyncio
import unittest

from src.agents.base import AgentState
from src.agents.critic import (
    CriticAgent,
    CriticIssue,
    VerificationAudit,
    VerificationVerdict,
)
from src.agents.quant import ComputedMetric, EntityFinancialProfile, QuantReport
from src.agents.researcher import CitationItem, GraphProvenance, ResearchDossier
from src.core.events import EventBus, EventType


class TestCriticAgent(unittest.TestCase):
    """Test suite verifying adversarial hallucination detection, numerical reconciliation, and replan triggers."""

    def setUp(self) -> None:
        self.event_bus = EventBus()
        self.critic = CriticAgent(event_bus=self.event_bus)

        # Baseline sound ResearchDossier
        self.valid_dossier = ResearchDossier(
            task="Investigate NVIDIA AI computing moat",
            primary_entity="NVIDIA",
            executive_summary="NVIDIA maintains strong leadership in datacenter acceleration.",
            key_findings=[
                "NVIDIA Blackwell GPUs offer up to 4x training acceleration.",
                "Gross margin is approximately 75% driven by premium pricing.",
            ],
            citations=[
                CitationItem(
                    doc_id="sec_10k_nvda",
                    text="Datacenter revenue surged 427% year-over-year.",
                    relevance_score=0.96,
                    matched_entities=["nvidia"],
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
            confidence_score=0.85,
        )

        # Baseline sound QuantReport
        self.valid_profile = EntityFinancialProfile(
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
        self.valid_quant = QuantReport(
            task="NVIDIA financial model",
            entities_analyzed=["NVDA"],
            profiles={"NVDA": self.valid_profile},
            comparative_summary="Strong profitability and low balance sheet leverage.",
            calculations_performed=["gross_margin(75, 100)"],
        )

    def test_critic_approval_on_valid_inputs(self) -> None:
        """Critic must approve fully grounded research and verified quant models."""
        audit = self.critic.audit(
            task="Audit NVIDIA intelligence package",
            research_dossier=self.valid_dossier,
            quant_report=self.valid_quant,
        )

        self.assertIsInstance(audit, VerificationAudit)
        self.assertEqual(audit.verdict, VerificationVerdict.APPROVED)
        self.assertFalse(audit.has_critical_issues)
        self.assertGreaterEqual(len(audit.passed_checks), 3)
        self.assertGreaterEqual(audit.confidence_score, 0.9)
        self.assertIsNone(audit.remediation_feedback)

    def test_critic_rejection_missing_citations(self) -> None:
        """Dossier with zero citations must be rejected for ungrounded claims."""
        ungrounded_dossier = ResearchDossier(
            task="Unbacked analysis",
            primary_entity="PhantomCorp",
            executive_summary="Unsupported claims made without any citations.",
            key_findings=["PhantomCorp will double revenue next year."],
            citations=[],  # Empty citations
            confidence_score=0.25,
        )

        audit = self.critic.audit(
            task="Audit ungrounded dossier",
            research_dossier=ungrounded_dossier,
        )

        self.assertEqual(audit.verdict, VerificationVerdict.REJECTED)
        self.assertTrue(audit.has_critical_issues)

        # Verify UNGROUNDED_CLAIM issue flagged
        issue_categories = [i.category for i in audit.issues]
        self.assertIn("UNGROUNDED_CLAIM", issue_categories)
        self.assertIsNotNone(audit.remediation_feedback)
        self.assertIn("Re-index", audit.remediation_feedback)

    def test_critic_rejection_numerical_discrepancy(self) -> None:
        """Discrepancy between qualitative text claim and quant model must trigger rejection."""
        conflicting_dossier = ResearchDossier(
            task="Investigate NVIDIA margins",
            primary_entity="NVIDIA",
            executive_summary="Conflicting claims report.",
            key_findings=[
                "NVIDIA reported unprecedented 95.0% gross margin in Q3.",  # Claims 95% vs Quant 75.1%
            ],
            citations=[
                CitationItem(doc_id="d1", text="Sample", relevance_score=0.9)
            ],
            confidence_score=0.8,
        )

        audit = self.critic.audit(
            task="Audit conflicting margins",
            research_dossier=conflicting_dossier,
            quant_report=self.valid_quant,
        )

        self.assertEqual(audit.verdict, VerificationVerdict.REJECTED)
        self.assertTrue(audit.has_critical_issues)

        mismatch_issues = [i for i in audit.issues if i.category == "NUMERICAL_MISMATCH"]
        self.assertGreaterEqual(len(mismatch_issues), 1)
        self.assertIn("95.0%", mismatch_issues[0].description)
        self.assertIn("75.1%", mismatch_issues[0].description)

    def test_critic_rejection_out_of_bounds_metrics(self) -> None:
        """Financial metrics exceeding physical bounds (e.g. gross margin > 100%) must trigger rejection."""
        corrupt_profile = EntityFinancialProfile(
            ticker="BROKEN",
            name="Broken Corp",
            metrics={
                "Gross Margin": ComputedMetric(
                    name="Gross Margin",
                    value=2.45,  # 245% is impossible
                    formatted_value="245.00%",
                    category="Profitability",
                    formula="Gross Profit / Revenue",
                )
            },
        )
        corrupt_quant = QuantReport(
            task="Corrupt model",
            entities_analyzed=["BROKEN"],
            profiles={"BROKEN": corrupt_profile},
            comparative_summary="Corrupt",
        )

        audit = self.critic.audit(
            task="Audit out-of-bounds metrics",
            quant_report=corrupt_quant,
        )

        self.assertEqual(audit.verdict, VerificationVerdict.REJECTED)
        self.assertTrue(any("exceeds theoretical bounds" in i.description for i in audit.issues))

    def test_critic_react_lifecycle_execution(self) -> None:
        """CriticAgent run() must cycle through ReAct states and format Markdown audit memo."""
        context = {
            "research_dossier": self.valid_dossier,
            "quant_report": self.valid_quant,
        }

        response = self.critic.run("Audit consolidated NVIDIA intelligence", context=context)

        self.assertTrue(response.success)
        self.assertEqual(response.state, AgentState.COMPLETED)
        self.assertIn("# Adversarial Audit Memo", response.output)
        self.assertIn("APPROVED", response.output)
        self.assertIn("Cleared Integrity Checks", response.output)

        # Check scratchpad variable
        audit_res = self.critic.memory.get_variable("latest_audit")
        self.assertIsInstance(audit_res, VerificationAudit)
        self.assertEqual(audit_res.verdict, VerificationVerdict.APPROVED)

    def test_critic_reflection_invariants(self) -> None:
        """Reflection must verify audit presence and enforce critical issue rejection consistency."""
        # 1. Valid approved audit
        audit = self.critic.audit("task", research_dossier=self.valid_dossier, quant_report=self.valid_quant)
        self.critic.memory.set_variable("latest_audit", audit)
        is_valid, msg = asyncio.run(self.critic.reflect("task", "thoughts", []))
        self.assertTrue(is_valid)
        self.assertIn("Verdict is APPROVED", msg)

        # 2. Invariant breach: Critical issue present but verdict marked APPROVED
        tampered_audit = VerificationAudit(
            task="Tampered",
            verdict=VerificationVerdict.APPROVED,  # Corrupt
            confidence_score=0.9,
            issues=[
                CriticIssue(
                    severity="CRITICAL",
                    category="HALLUCINATION",
                    description="Fabricated data",
                    location="Test",
                    remediation_advice="Fix",
                )
            ],
        )
        self.critic.memory.set_variable("latest_audit", tampered_audit)
        is_valid, msg = asyncio.run(self.critic.reflect("task", "thoughts", []))
        self.assertFalse(is_valid)
        self.assertIn("Integrity Violation", msg)

    def test_audit_markdown_and_serialization(self) -> None:
        """VerificationAudit to_markdown and to_dict must output complete structured schemas."""
        audit = self.critic.audit("Test", research_dossier=self.valid_dossier, quant_report=self.valid_quant)
        md = audit.to_markdown()
        self.assertIn("# Adversarial Audit Memo", md)
        self.assertIn("Executive Critique Summary", md)

        d = audit.to_dict()
        self.assertEqual(d["verdict"], "APPROVED")
        self.assertIn("passed_checks", d)
        self.assertIn("issues", d)


if __name__ == "__main__":
    unittest.main()
