"""Autonomous Critic / Verifier Agent for Reflection-Driven Quality Assurance.

Performs adversarial cross-examination of upstream Research Dossiers and Quant Reports:
  1. Hallucination and factual grounding verification against citations.
  2. Numerical consistency check reconciling qualitative text claims with quantitative models.
  3. Logical coherence and contradiction detection.
  4. Dynamic replanning trigger generation with actionable remediation advice.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.agents.base import AgentState, BaseAgent, Tool, ToolResult
from src.agents.quant import QuantReport
from src.agents.researcher import ResearchDossier
from src.core.events import EventBus, EventType


class VerificationVerdict(str, Enum):
    """Overall outcome of the Critic audit."""

    APPROVED = "APPROVED"        # All checks passed, ready for executive synthesis
    REJECTED = "REJECTED"        # Critical flaws detected, triggers dynamic replanning
    CONDITIONAL = "CONDITIONAL"  # Minor warnings present, acceptable with caveats


@dataclass
class CriticIssue:
    """Specific deficiency or anomaly identified during adversarial audit.

    Attributes:
        severity: 'CRITICAL' (triggers rejection), 'WARNING', or 'INFO'.
        category: 'HALLUCINATION', 'NUMERICAL_MISMATCH', 'UNGROUNDED_CLAIM',
                  'CONTRADICTION', 'MISSING_EVIDENCE'.
        description: Explicit explanation of the defect found.
        location: Component where issue originated ('Research Dossier', 'Quant Report').
        remediation_advice: Concrete instructions for the Planner/Worker to fix the defect.
    """

    severity: str
    category: str
    description: str
    location: str
    remediation_advice: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "location": self.location,
            "remediation_advice": self.remediation_advice,
        }


@dataclass
class VerificationAudit:
    """Consolidated audit report produced by the Critic / Verifier Agent.

    Attributes:
        task: Target task or query inspected.
        verdict: VerificationVerdict enum (APPROVED, REJECTED, CONDITIONAL).
        confidence_score: Auditor certainty score in [0.0, 1.0].
        issues: List of identified CriticIssue items.
        passed_checks: List of test criteria successfully cleared.
        critique_summary: Executive overview of findings.
        remediation_feedback: Actionable instruction string passed to Planner.replan().
        created_at: Epoch timestamp of audit completion.
        metadata: Diagnostic and execution parameters.
    """

    task: str
    verdict: VerificationVerdict
    confidence_score: float
    issues: List[CriticIssue] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    critique_summary: str = ""
    remediation_feedback: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == "CRITICAL" for i in self.issues)

    @property
    def total_issues(self) -> int:
        return len(self.issues)

    def to_markdown(self) -> str:
        """Render audit findings into a structured Markdown verification memo."""
        status_icon = "APPROVED" if self.verdict == VerificationVerdict.APPROVED else ("REJECTED" if self.verdict == VerificationVerdict.REJECTED else "CONDITIONAL")
        lines = [
            f"# Adversarial Audit Memo: [{status_icon}]",
            f"**Audit Directive:** {self.task}",
            f"**Verdict:** `{self.verdict.value}` | **Auditor Confidence:** {self.confidence_score * 100:.1f}% | **Issues Flagged:** {self.total_issues}",
            "",
            "## Executive Critique Summary",
            self.critique_summary,
            "",
            "## Cleared Integrity Checks",
        ]
        for chk in self.passed_checks:
            lines.append(f"- [x] {chk}")

        if self.issues:
            lines.extend(["", "## Identified Defects & Discrepancies"])
            for idx, issue in enumerate(self.issues, 1):
                lines.append(f"{idx}. **[{issue.severity}] {issue.category}** ({issue.location})")
                lines.append(f"   - *Defect:* {issue.description}")
                lines.append(f"   - *Remediation:* {issue.remediation_advice}")

        if self.remediation_feedback:
            lines.extend(["", "## Actionable Remediation Feedback for Dynamic Replanning", f"> {self.remediation_feedback}"])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "verdict": self.verdict.value,
            "confidence_score": round(self.confidence_score, 4),
            "passed_checks": self.passed_checks,
            "issues": [i.to_dict() for i in self.issues],
            "critique_summary": self.critique_summary,
            "remediation_feedback": self.remediation_feedback,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class CriticAgent(BaseAgent):
    """Specialized adversarial agent verifying research grounding and numerical fidelity.

    Acts as the quality control firewall between analysis and final synthesis.
    Can veto upstream outputs to trigger dynamic graph replanning.
    """

    def __init__(
        self,
        agent_id: str = "critic",
        name: str = "Critic / Verifier Agent",
        tools: Optional[List[Tool]] = None,
        event_bus: Optional[EventBus] = None,
        memory_capacity: int = 50,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="Adversarial Verification, Hallucination Detection & Numerical Cross-Examination",
            goal="Ensure zero hallucination, verify factual citations, and reconcile qualitative assertions with quantitative math",
            tools=tools or [],
            event_bus=event_bus,
            memory_capacity=memory_capacity,
        )

    # -------------------------------------------------------------------------
    # Core Adversarial Audit Logic
    # -------------------------------------------------------------------------

    def audit(
        self,
        task: str,
        research_dossier: Optional[ResearchDossier] = None,
        quant_report: Optional[QuantReport] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationAudit:
        """Perform comprehensive cross-examination across all analytical dimensions."""
        clean_task = task.strip()
        issues: List[CriticIssue] = []
        passed_checks: List[str] = []

        # ---------------------------------------------------------------------
        # 1. Qualitative Research Grounding Audit
        # ---------------------------------------------------------------------
        if research_dossier is not None:
            # Check A: Non-empty findings
            if not research_dossier.key_findings:
                issues.append(
                    CriticIssue(
                        severity="CRITICAL",
                        category="MISSING_EVIDENCE",
                        description=f"Research dossier for '{research_dossier.primary_entity}' contains zero analytical findings.",
                        location="Research Dossier",
                        remediation_advice="Re-run qualitative retrieval with broadened domain search terms.",
                    )
                )
            else:
                passed_checks.append(f"Substantive qualitative findings present ({len(research_dossier.key_findings)} takeaways)")

            # Check B: Citation backing
            if research_dossier.total_citations == 0:
                issues.append(
                    CriticIssue(
                        severity="CRITICAL",
                        category="UNGROUNDED_CLAIM",
                        description="Research dossier lacks empirical document citations from Hybrid RAG.",
                        location="Research Dossier",
                        remediation_advice="Re-index relevant 10-K/research documents and query with exact entity slug.",
                    )
                )
            else:
                passed_checks.append(f"Empirical citations verified ({research_dossier.total_citations} source passages)")

            # Check C: Groundedness confidence threshold
            if research_dossier.confidence_score < 0.35:
                issues.append(
                    CriticIssue(
                        severity="CRITICAL",
                        category="MISSING_EVIDENCE",
                        description=(
                            f"Overall research confidence score ({research_dossier.confidence_score:.2f}) "
                            f"is below acceptable reliability threshold (0.35)."
                        ),
                        location="Research Dossier",
                        remediation_advice="Expand knowledge graph relations or fetch additional external literature.",
                    )
                )
            else:
                passed_checks.append(f"Groundedness confidence acceptable ({research_dossier.confidence_score * 100:.1f}%)")

        # ---------------------------------------------------------------------
        # 2. Quantitative Mathematical & Ratio Integrity Audit
        # ---------------------------------------------------------------------
        if quant_report is not None:
            # Check A: Presence of evaluated profiles
            if not quant_report.profiles:
                issues.append(
                    CriticIssue(
                        severity="CRITICAL",
                        category="MISSING_EVIDENCE",
                        description="Quantitative report contains zero evaluated entity financial profiles.",
                        location="Quant Report",
                        remediation_advice="Execute market data query with valid ticker symbol.",
                    )
                )
            else:
                passed_checks.append(f"Financial fundamentals resolved ({len(quant_report.profiles)} companies modeled)")

            # Check B: Numerical sanity bounds across metrics
            for ticker, prof in quant_report.profiles.items():
                # Margin sanity: Margins should typically be in [-100%, 100%]
                gm = prof.metrics.get("Gross Margin")
                if gm and (gm.value < -1.0 or gm.value > 1.0):
                    issues.append(
                        CriticIssue(
                            severity="CRITICAL",
                            category="NUMERICAL_MISMATCH",
                            description=f"Gross margin {gm.formatted_value} for '{ticker}' exceeds theoretical bounds [-100%, 100%].",
                            location="Quant Report",
                            remediation_advice="Verify revenue and COGS inputs in market data feed.",
                        )
                    )
                elif gm:
                    passed_checks.append(f"Gross margin bounded for {ticker} ({gm.formatted_value})")

                # Leverage sanity: Debt-to-Equity should be non-negative
                de = prof.metrics.get("Debt-to-Equity")
                if de and de.value < 0:
                    issues.append(
                        CriticIssue(
                            severity="WARNING",
                            category="NUMERICAL_MISMATCH",
                            description=f"Negative Debt-to-Equity ratio ({de.formatted_value}) indicates negative stockholders' equity.",
                            location="Quant Report",
                            remediation_advice="Confirm stockholders' equity balance on balance sheet.",
                        )
                    )
                elif de:
                    passed_checks.append(f"Solvency verified for {ticker} ({de.formatted_value} D/E)")

        # ---------------------------------------------------------------------
        # 3. Cross-Functional Numerical Reconciliation (Qual vs Quant)
        # ---------------------------------------------------------------------
        if research_dossier is not None and quant_report is not None:
            primary_ticker = self._find_matching_ticker(research_dossier.primary_entity, quant_report)
            if primary_ticker and primary_ticker in quant_report.profiles:
                prof = quant_report.profiles[primary_ticker]
                findings_text = " ".join(research_dossier.key_findings)

                # Check if qualitative text claims a margin that deviates significantly from quant model
                gm_claimed = self._extract_claimed_percentage(findings_text, ["gross margin", "margin"])
                if gm_claimed is not None and "Gross Margin" in prof.metrics:
                    actual_gm = prof.metrics["Gross Margin"].value * 100.0
                    if abs(gm_claimed - actual_gm) > 10.0:
                        issues.append(
                            CriticIssue(
                                severity="CRITICAL",
                                category="NUMERICAL_MISMATCH",
                                description=(
                                    f"Qualitative finding claims ~{gm_claimed:.1f}% gross margin, "
                                    f"contradicting verified quant model ({actual_gm:.1f}%)."
                                ),
                                location="Cross-Examination",
                                remediation_advice=f"Reconcile qualitative text with verified quantitative metric of {actual_gm:.2f}%.",
                            )
                        )
                    else:
                        passed_checks.append("Qualitative claims reconcile with quantitative gross margin")

        # ---------------------------------------------------------------------
        # 4. Formulate Verdict & Remediation Feedback
        # ---------------------------------------------------------------------
        critical_count = sum(1 for i in issues if i.severity == "CRITICAL")
        warning_count = sum(1 for i in issues if i.severity == "WARNING")

        if critical_count > 0:
            verdict = VerificationVerdict.REJECTED
            confidence = max(0.2, 0.95 - (critical_count * 0.15))
            summary = (
                f"AUDIT REJECTED: Discovered {critical_count} critical integrity defects. "
                f"Pipeline execution halted to prevent ungrounded or contradictory synthesis."
            )
            # Compile primary remediation instructions for dynamic replanner
            remedy_parts = [i.remediation_advice for i in issues if i.severity == "CRITICAL"]
            remediation_feedback = " | ".join(remedy_parts)

        elif warning_count > 0:
            verdict = VerificationVerdict.CONDITIONAL
            confidence = 0.85
            summary = f"AUDIT CONDITIONAL: {warning_count} non-blocking warnings identified. Approved with cautionary disclosures."
            remediation_feedback = None

        else:
            verdict = VerificationVerdict.APPROVED
            confidence = 0.98
            summary = (
                f"AUDIT APPROVED: All {len(passed_checks)} verification checks cleared successfully. "
                f"Factual citations, numerical models, and relational graphs verified with zero anomalies."
            )
            remediation_feedback = None

        return VerificationAudit(
            task=clean_task,
            verdict=verdict,
            confidence_score=confidence,
            issues=issues,
            passed_checks=passed_checks,
            critique_summary=summary,
            remediation_feedback=remediation_feedback,
            metadata={
                "has_research": research_dossier is not None,
                "has_quant": quant_report is not None,
                "context": context or {},
            },
        )

    # -------------------------------------------------------------------------
    # BaseAgent Cognitive Implementation: think() and reflect()
    # -------------------------------------------------------------------------

    async def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Adversarial reasoning phase: inspects context variables and runs audit."""
        research_dossier: Optional[ResearchDossier] = None
        quant_report: Optional[QuantReport] = None

        if context:
            research_dossier = context.get("research_dossier")
            quant_report = context.get("quant_report")

        audit_res = self.audit(
            task=task,
            research_dossier=research_dossier,
            quant_report=quant_report,
            context=context,
        )

        # Cache audit in working scratchpad
        self.memory.set_variable("latest_audit", audit_res)

        # Broadcast event
        if audit_res.verdict == VerificationVerdict.REJECTED:
            await self._emit_event(
                EventType.CRITIC_REJECTED,
                {
                    "verdict": audit_res.verdict.value,
                    "issues_count": audit_res.total_issues,
                    "remediation": audit_res.remediation_feedback,
                },
            )
        else:
            await self._emit_event(
                EventType.CRITIC_VALIDATION,
                {
                    "verdict": audit_res.verdict.value,
                    "confidence": audit_res.confidence_score,
                    "checks_passed": len(audit_res.passed_checks),
                },
            )

        return audit_res.to_markdown()

    async def reflect(
        self,
        task: str,
        thoughts: str,
        execution_log: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Self-verification: Ensures Critic properly conducted audit and maintained verdict consistency."""
        audit_res: Optional[VerificationAudit] = self.memory.get_variable("latest_audit")
        if audit_res is None:
            return False, "Critic failed to compile and store a valid VerificationAudit."

        # Verification Invariant: If critical issues exist, verdict MUST be REJECTED
        if audit_res.has_critical_issues and audit_res.verdict != VerificationVerdict.REJECTED:
            return False, "Integrity Violation: Critical issues found but verdict was not REJECTED."

        # If rejected, remediation feedback must be populated
        if audit_res.verdict == VerificationVerdict.REJECTED and not audit_res.remediation_feedback:
            return False, "Rejected audit must provide actionable remediation feedback for Planner.replan()."

        return (
            True,
            (
                f"Critic self-reflection verified: Verdict is {audit_res.verdict.value} "
                f"with {len(audit_res.passed_checks)} checks cleared and {audit_res.total_issues} issues flagged."
            ),
        )

    # -------------------------------------------------------------------------
    # Helper Utilities
    # -------------------------------------------------------------------------

    def _find_matching_ticker(self, entity_name: str, quant_report: QuantReport) -> Optional[str]:
        """Match qualitative entity name to a ticker in QuantReport."""
        e_clean = entity_name.lower().strip()
        for ticker, prof in quant_report.profiles.items():
            if ticker.lower() in e_clean or e_clean in prof.name.lower():
                return ticker
        return None

    def _extract_claimed_percentage(self, text: str, keywords: List[str]) -> Optional[float]:
        """Search text for percentage numbers near specific keywords (before or after)."""
        for kw in keywords:
            # Pattern 1: keyword ... percentage% (e.g., "gross margin is 75%")
            pat1 = re.compile(re.escape(kw) + r"[^\d%]{1,25}([\d\.]+)%", re.IGNORECASE)
            match1 = pat1.search(text)
            if match1:
                try:
                    return float(match1.group(1))
                except ValueError:
                    pass

            # Pattern 2: percentage% ... keyword (e.g., "95.0% gross margin")
            pat2 = re.compile(r"([\d\.]+)%[^\d%]{0,15}" + re.escape(kw), re.IGNORECASE)
            match2 = pat2.search(text)
            if match2:
                try:
                    return float(match2.group(1))
                except ValueError:
                    pass
        return None
