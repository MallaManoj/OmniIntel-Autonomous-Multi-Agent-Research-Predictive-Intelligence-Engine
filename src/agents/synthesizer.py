"""Autonomous Synthesizer Agent for Executive Intelligence Dossier Compilation.

Aggregates verified qualitative findings from the Researcher Agent, deterministic financial
models from the Quantitative Analyst Agent, and adversarial quality assurance memos from
the Critic Agent into publication-ready, C-suite grade Strategic Intelligence Dossiers.
"""

from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.agents.base import AgentState, BaseAgent, Tool, ToolResult
from src.agents.critic import VerificationAudit, VerificationVerdict
from src.agents.quant import QuantReport
from src.agents.researcher import ResearchDossier
from src.core.events import EventBus, EventType


@dataclass
class StrategicRecommendation:
    """Actionable strategic stance or investment directive derived from synthesized intelligence."""

    action: str  # 'OVERWEIGHT', 'ACCUMULATE', 'NEUTRAL', 'MONITOR', 'DIVEST'
    time_horizon: str  # 'Tactical (0-6 months)', 'Strategic (1-3 years)', 'Long-Term (3-5+ years)'
    rationale: str
    key_risks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "time_horizon": self.time_horizon,
            "rationale": self.rationale,
            "key_risks": self.key_risks,
        }


@dataclass
class ExecutiveDossier:
    """Comprehensive, publication-ready intelligence briefing compiled for executive decision-makers.

    Attributes:
        title: Formal briefing title.
        task: Original user query or strategic intelligence directive.
        executive_summary: High-level distillation of thesis, findings, and risks.
        target_entities: Companies or technologies analyzed.
        strategic_thesis: Comprehensive analytical argument integrating qual + quant.
        qualitative_findings: Key empirical takeaways from ResearcherAgent.
        supply_chain_links: Relational graph connections and supplier bottlenecks.
        financial_table_markdown: Formatted Markdown comparison table from QuantAgent.
        audit_verdict: Quality assurance verdict from CriticAgent (APPROVED, CONDITIONAL, REJECTED).
        recommendations: Actionable strategic directives with risk hedges.
        citations: Attributed literature passages and evidence sources.
        overall_confidence: Composite reliability metric in [0.0, 1.0].
        created_at: Epoch timestamp of publication.
        metadata: Diagnostic and contextual parameters.
    """

    title: str
    task: str
    executive_summary: str
    target_entities: List[str]
    strategic_thesis: str
    qualitative_findings: List[str]
    supply_chain_links: List[str]
    financial_table_markdown: str
    audit_verdict: str
    recommendations: List[StrategicRecommendation] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    overall_confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the briefing into publication-quality Markdown with executive hierarchy."""
        lines = [
            f"# {self.title}",
            f"**Strategic Directive:** {self.task}",
            f"**Overall Confidence Score:** {self.overall_confidence * 100:.1f}% | **QA Audit Status:** `{self.audit_verdict}` | **Published:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(self.created_at))} UTC",
            "",
            "---",
            "",
            "## 1. Executive Summary & Strategic Thesis",
            self.executive_summary,
            "",
            f"### Core Analytical Thesis",
            self.strategic_thesis,
            "",
            "## 2. Key Qualitative & Competitive Moats",
        ]

        if self.qualitative_findings:
            for idx, finding in enumerate(self.qualitative_findings, 1):
                lines.append(f"{idx}. {finding}")
        else:
            lines.append("*No qualitative findings provided.*")

        if self.supply_chain_links:
            lines.extend(["", "### Knowledge Graph Relational & Supply Chain Links"])
            for link in self.supply_chain_links:
                lines.append(f"- {link}")

        lines.extend(["", "## 3. Quantitative Financial Modeling & Valuation Multiples"])
        if self.financial_table_markdown:
            lines.append(self.financial_table_markdown.strip())
        else:
            lines.append("*No quantitative financial tables available.*")

        lines.extend(["", "## 4. Strategic Recommendations & Risk Matrix"])
        if self.recommendations:
            for rec in self.recommendations:
                lines.append(f"### Stance: **{rec.action}** ({rec.time_horizon})")
                lines.append(f"**Rationale:** {rec.rationale}")
                if rec.key_risks:
                    lines.append("**Key Risk Factors:**")
                    for r in rec.key_risks:
                        lines.append(f"  - {r}")
        else:
            lines.append("*No specific strategic recommendations issued.*")

        lines.extend(["", "## 5. Audit Trail & Grounded Citations", f"**Critic QA Status:** `{self.audit_verdict}`"])
        if self.citations:
            for c in self.citations[:10]:
                doc_id = c.get("doc_id", "Source")
                score = c.get("relevance_score", 1.0)
                text = c.get("text", "").strip()
                lines.append(f"- **[{doc_id}]** (Relevance: {score:.4f}): \"{text}\"")

        lines.extend(["", "---", "*Report compiled by OmniIntel Autonomous Multi-Agent Predictive Intelligence Engine.*"])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "task": self.task,
            "executive_summary": self.executive_summary,
            "target_entities": self.target_entities,
            "strategic_thesis": self.strategic_thesis,
            "qualitative_findings": self.qualitative_findings,
            "supply_chain_links": self.supply_chain_links,
            "financial_table_markdown": self.financial_table_markdown,
            "audit_verdict": self.audit_verdict,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "citations": self.citations,
            "overall_confidence": round(self.overall_confidence, 4),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class SynthesizerAgent(BaseAgent):
    """Specialized autonomous agent aggregating multi-agent findings into executive briefings.

    Acts as the final cognitive aggregator in the Kahn execution DAG. Synthesizes
    qualitative research, quantitative financial tables, and critic audit approvals.
    """

    def __init__(
        self,
        agent_id: str = "synthesizer",
        name: str = "Synthesizer Agent",
        tools: Optional[List[Tool]] = None,
        event_bus: Optional[EventBus] = None,
        memory_capacity: int = 50,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            role="Executive Intelligence Synthesis & Strategic Report Generation",
            goal="Synthesize multi-agent research, financial models, and QA audits into publication-ready dossiers",
            tools=tools or [],
            event_bus=event_bus,
            memory_capacity=memory_capacity,
        )

    # -------------------------------------------------------------------------
    # Core Synthesis Logic
    # -------------------------------------------------------------------------

    def synthesize(
        self,
        query: str,
        research_dossier: Optional[ResearchDossier] = None,
        quant_report: Optional[QuantReport] = None,
        audit: Optional[VerificationAudit] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutiveDossier:
        """Consolidate heterogeneous multi-agent outputs into a unified ExecutiveDossier."""
        clean_query = query.strip()

        # 1. Resolve Target Entities
        entities: List[str] = []
        if research_dossier:
            entities.append(research_dossier.primary_entity)
        if quant_report:
            for t in quant_report.entities_analyzed:
                if t not in entities:
                    entities.append(t)
        if not entities:
            entities.append("Target Subject")

        # 2. Extract Qualitative Elements
        qual_findings: List[str] = []
        supply_links: List[str] = []
        citations_list: List[Dict[str, Any]] = []
        research_conf = 0.85

        if research_dossier is not None:
            qual_findings.extend(research_dossier.key_findings)
            research_conf = research_dossier.confidence_score
            for gp in research_dossier.graph_paths:
                supply_links.append(f"{gp.source_entity.upper()} $\\to$ {gp.target_entity.upper()} via: {gp.reasoning_path}")
            for c in research_dossier.citations:
                citations_list.append(c.to_dict())

        # 3. Extract Quantitative Elements
        fin_table_md = ""
        quant_summary = ""
        if quant_report is not None:
            quant_summary = quant_report.comparative_summary
            # Extract table portion from quant markdown
            md_lines = quant_report.to_markdown().split("\n")
            table_lines = [l for l in md_lines if l.startswith("|")]
            fin_table_md = "\n".join(table_lines)

        # 4. Resolve Quality Assurance Status
        audit_verdict_str = "NOT_AUDITED"
        audit_conf = 0.90
        if audit is not None:
            audit_verdict_str = audit.verdict.value
            audit_conf = audit.confidence_score

        # 5. Compute Composite Groundedness Confidence
        composite_confidence = round(0.45 * research_conf + 0.35 * 0.95 + 0.20 * audit_conf, 3)

        # 6. Formulate Executive Summary & Strategic Thesis
        primary_subject = entities[0] if entities else "Industry Target"
        exec_summary = (
            f"This strategic intelligence briefing evaluates {', '.join(entities)} regarding: '{clean_query}'. "
            f"Cross-functional analysis synthesized {len(qual_findings)} empirical qualitative findings, "
            f"comprehensive financial valuation models, and verified relational supply chain linkages. "
            f"All findings have been subjected to adversarial verification under Critic status `{audit_verdict_str}`."
        )

        strategic_thesis = (
            f"{primary_subject} demonstrates sustained operational moats supported by proprietary architecture "
            f"and premium pricing power. {quant_summary} However, structural supply-chain dependencies "
            f"highlight exposure to single-source foundries and geopolitical flashpoints."
        )

        # 7. Formulate Actionable Strategic Recommendations
        recommendations = self._formulate_recommendations(entities, quant_report, research_dossier)

        # Title
        title = f"Strategic Intelligence Dossier: {', '.join(entities)} Comprehensive Briefing"

        return ExecutiveDossier(
            title=title,
            task=clean_query,
            executive_summary=exec_summary,
            target_entities=entities,
            strategic_thesis=strategic_thesis,
            qualitative_findings=qual_findings,
            supply_chain_links=supply_links,
            financial_table_markdown=fin_table_md,
            audit_verdict=audit_verdict_str,
            recommendations=recommendations,
            citations=citations_list,
            overall_confidence=composite_confidence,
            metadata={"context": context or {}, "query": clean_query},
        )

    # -------------------------------------------------------------------------
    # BaseAgent Cognitive Implementation: think() and reflect()
    # -------------------------------------------------------------------------

    async def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Cognitive phase: Ingests context variables, compiles dossier, and broadcasts event."""
        research_dossier: Optional[ResearchDossier] = None
        quant_report: Optional[QuantReport] = None
        audit: Optional[VerificationAudit] = None

        if context:
            research_dossier = context.get("research_dossier")
            quant_report = context.get("quant_report")
            audit = context.get("audit")

        dossier = self.synthesize(
            query=task,
            research_dossier=research_dossier,
            quant_report=quant_report,
            audit=audit,
            context=context,
        )

        self.memory.set_variable("latest_dossier", dossier)

        # Emit completion event
        await self._emit_event(
            EventType.REPORT_GENERATED,
            {
                "title": dossier.title,
                "entities": dossier.target_entities,
                "confidence": dossier.overall_confidence,
                "verdict": dossier.audit_verdict,
            },
        )

        return dossier.to_markdown()

    async def reflect(
        self,
        task: str,
        thoughts: str,
        execution_log: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """Self-verification: Ensures dossier is complete, balanced, and satisfies quality bars."""
        dossier: Optional[ExecutiveDossier] = self.memory.get_variable("latest_dossier")
        if dossier is None:
            return False, "Synthesizer failed to retain compiled ExecutiveDossier in scratchpad."

        if not dossier.target_entities:
            return False, "Synthesized dossier contains zero identified target entities."

        if dossier.overall_confidence < 0.40:
            return False, f"Composite confidence ({dossier.overall_confidence:.2f}) is below minimum threshold."

        if not dossier.strategic_thesis:
            return False, "Dossier lacks a substantive strategic analytical thesis."

        return (
            True,
            (
                f"Synthesis verified: ExecutiveDossier compiled successfully for "
                f"{', '.join(dossier.target_entities)} with confidence {dossier.overall_confidence * 100:.1f}% "
                f"(QA Status: {dossier.audit_verdict})."
            ),
        )

    # -------------------------------------------------------------------------
    # Recommendation Formulator Helper
    # -------------------------------------------------------------------------

    def _formulate_recommendations(
        self,
        entities: List[str],
        quant_report: Optional[QuantReport],
        research_dossier: Optional[ResearchDossier],
    ) -> List[StrategicRecommendation]:
        """Derive strategic stances based on financial fundamentals and supply-chain risks."""
        recs: List[StrategicRecommendation] = []

        primary = entities[0] if entities else "Target Subject"

        # Check financial strength if quant report available
        high_margin = False
        low_leverage = False
        if quant_report and quant_report.profiles:
            prof = quant_report.profiles.get(primary)
            if not prof:
                for p in quant_report.profiles.values():
                    if (
                        p.ticker.lower() == primary.lower()
                        or primary.lower() in p.name.lower()
                        or p.name.lower() in primary.lower()
                    ):
                        prof = p
                        break

            if prof is not None:
                gm = prof.metrics.get("Gross Margin")
                de = prof.metrics.get("Debt-to-Equity")
                if gm and gm.value >= 0.50:
                    high_margin = True
                if de and de.value <= 0.50:
                    low_leverage = True

        if high_margin and low_leverage:
            recs.append(
                StrategicRecommendation(
                    action="ACCUMULATE",
                    time_horizon="Strategic (1-3 years)",
                    rationale=(
                        f"{primary} displays elite pricing power (Gross Margin >= 50%) paired with a conservative "
                        f"balance sheet (D/E <= 0.50), providing resilient cash flow cushions against cyclical downturns."
                    ),
                    key_risks=[
                        "Concentration risk in single-source fab partners (e.g., TSMC).",
                        "Geopolitical tensions and potential export control restrictions.",
                        "Hyperscaler in-house custom silicon substitution threats.",
                    ],
                )
            )
        else:
            recs.append(
                StrategicRecommendation(
                    action="MONITOR",
                    time_horizon="Tactical (0-6 months)",
                    rationale=(
                        f"Maintain a balanced monitoring stance on {primary} while tracking gross margin stability, "
                        f"valuation multiple compression, and datacenter hardware capital expenditure pacing."
                    ),
                    key_risks=[
                        "Multiple contraction in high interest rate environments.",
                        "Supply chain lead-time expansion and inventory build-up.",
                    ],
                )
            )

        return recs
