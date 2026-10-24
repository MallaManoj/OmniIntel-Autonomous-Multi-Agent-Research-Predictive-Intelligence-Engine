"""Autonomous Multi-Agent Master Orchestrator for Parallel DAG Workflow Execution.

Coordinates the end-to-end mission lifecycle:
  1. Decomposes high-level intelligence directives via PlannerAgent into an acyclic DAG.
  2. Resolves parallel execution stages using Kahn's topological batching algorithm.
  3. Dispatches concurrent agent tasks (ResearcherAgent, QuantAgent) via asyncio.gather().
  4. Channels intermediate outputs to CriticAgent for adversarial reflection.
  5. Implements closed-loop Self-Correction: triggers dynamic graph surgery via
     Planner.replan() when Critic rejects upstream outputs.
  6. Aggregates verified outputs through SynthesizerAgent into an ExecutiveDossier.
  7. Broadcasts real-time lifecycle telemetry across the asynchronous EventBus.
"""

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.agents.base import AgentResponse, AgentState, BaseAgent, ToolResult
from src.agents.critic import CriticAgent, VerificationAudit, VerificationVerdict
from src.agents.planner import ExecutionPlan, PlannedTask, PlannerAgent
from src.agents.quant import QuantAgent, QuantReport
from src.agents.researcher import ResearchDossier, ResearcherAgent
from src.agents.synthesizer import ExecutiveDossier, SynthesizerAgent
from src.core.dsa.dag import DAG, TaskNode, TaskState
from src.core.events import Event, EventBus, EventType
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.pipeline import HybridRAGPipeline


@dataclass
class OrchestratorResult:
    """Consolidated outcome of a full multi-agent research mission.

    Attributes:
        query: Original user directive or research prompt.
        success: Whether the mission completed without fatal failure.
        plan: The ExecutionPlan DAG governing task scheduling.
        dossier: Final synthesized Strategic Intelligence Dossier.
        stage_outputs: Mapping of task_id to raw agent output payload.
        total_latency_ms: Total elapsed execution time across all stages.
        total_stages_executed: Number of parallel Kahn stages processed.
        audit: VerificationAudit from the Critic agent.
        errors: List of any non-fatal errors or exceptions encountered.
        replan_count: Number of dynamic self-correction replanning cycles performed.
        remediation_history: Chronological log of graph surgeries and remediation directives.
        created_at: Epoch timestamp of mission completion.
    """

    query: str
    success: bool
    plan: ExecutionPlan
    dossier: Optional[ExecutiveDossier]
    stage_outputs: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    total_stages_executed: int = 0
    audit: Optional[VerificationAudit] = None
    errors: List[str] = field(default_factory=list)
    replan_count: int = 0
    remediation_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_markdown(self) -> str:
        """Render complete mission results including plan roadmap and executive dossier."""
        if self.dossier:
            return self.dossier.to_markdown()
        return f"# Mission Failed for Query: {self.query}\nErrors:\n" + "\n".join(f"- {e}" for e in self.errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "success": self.success,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "total_stages_executed": self.total_stages_executed,
            "replan_count": self.replan_count,
            "remediation_history": self.remediation_history,
            "plan": self.plan.to_dict(),
            "dossier": self.dossier.to_dict() if self.dossier else None,
            "audit": self.audit.to_dict() if self.audit else None,
            "errors": self.errors,
            "created_at": self.created_at,
        }


class MasterOrchestrator:
    """Central Conductor orchestrating specialized agents through DAG dependencies.

    Manages agent lifecycle dispatch, parallel stage execution, blackboard state sharing,
    event streaming, and closed-loop dynamic replanning upon Critic rejection.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        pipeline: Optional[HybridRAGPipeline] = None,
        graph: Optional[KnowledgeGraph] = None,
        market_feed: Optional[Dict[str, Dict[str, Any]]] = None,
        planner: Optional[PlannerAgent] = None,
        researcher: Optional[ResearcherAgent] = None,
        quant: Optional[QuantAgent] = None,
        critic: Optional[CriticAgent] = None,
        synthesizer: Optional[SynthesizerAgent] = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.pipeline = pipeline
        self.graph = graph
        self.market_feed = market_feed

        # Initialize or bind specialized agents
        self.planner = planner or PlannerAgent(event_bus=self.event_bus)
        self.researcher = researcher or ResearcherAgent(
            pipeline=self.pipeline,
            graph=self.graph,
            event_bus=self.event_bus,
        )
        self.quant = quant or QuantAgent(
            market_feed=self.market_feed,
            event_bus=self.event_bus,
        )
        self.critic = critic or CriticAgent(event_bus=self.event_bus)
        self.synthesizer = synthesizer or SynthesizerAgent(event_bus=self.event_bus)

        # Agent role lookup map
        self.agent_pool: Dict[str, BaseAgent] = {
            "planner": self.planner,
            "researcher": self.researcher,
            "quant": self.quant,
            "critic": self.critic,
            "synthesizer": self.synthesizer,
        }

    async def _emit_event(self, event_type: EventType, payload: Dict[str, Any]) -> None:
        """Broadcast system-level orchestration events."""
        event = Event(
            event_type=event_type,
            source="MasterOrchestrator",
            payload=payload,
        )
        await self.event_bus.publish(event)

    # -------------------------------------------------------------------------
    # Core Mission Execution Lifecycle with Self-Correction Loop
    # -------------------------------------------------------------------------

    async def run_mission(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_replans: int = 2,
    ) -> OrchestratorResult:
        """Execute end-to-end multi-agent research mission with closed-loop self-correction.

        Args:
            query: The high-level strategic intelligence prompt.
            context: Optional domain context or prior parameters.
            max_replans: Maximum self-correction iterations permitted upon Critic rejection.

        Returns:
            OrchestratorResult containing the final ExecutiveDossier and stage outputs.
        """
        start_time = time.perf_counter()
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Mission query cannot be empty.")

        await self._emit_event(EventType.SYSTEM_STARTED, {"query": clean_query})

        # ---------------------------------------------------------------------
        # Stage 0: Initial Planning & DAG Assembly
        # ---------------------------------------------------------------------
        current_plan: ExecutionPlan = self.planner.create_plan(clean_query, context=context)
        await self._emit_event(
            EventType.TASK_SCHEDULED,
            {
                "plan_id": current_plan.plan_id,
                "total_tasks": current_plan.total_tasks,
                "stages": current_plan.total_stages,
                "complexity": current_plan.complexity.value,
            },
        )

        # Shared blackboard memory across all agents in this mission
        blackboard: Dict[str, Any] = {
            "query": clean_query,
            "context": context or {},
            "task_outputs": {},
            "research_dossiers": [],
            "quant_reports": [],
            "audits": [],
            "final_dossier": None,
        }

        errors: List[str] = []
        stages_processed = 0
        replan_count = 0
        remediation_history: List[Dict[str, Any]] = []

        # ---------------------------------------------------------------------
        # Dynamic Closed-Loop Execution Engine
        # ---------------------------------------------------------------------
        while True:
            replan_triggered = False

            for stage_idx, batch_task_ids in enumerate(current_plan.batches):
                # Identify tasks in this batch that have not yet executed
                pending_batch = [
                    t_id for t_id in batch_task_ids
                    if t_id not in blackboard["task_outputs"]
                ]

                # If all tasks in batch already completed (e.g. from prior plan), skip execution
                if not pending_batch:
                    continue

                stages_processed += 1
                await self._emit_event(
                    EventType.TASK_STARTED,
                    {
                        "stage_index": stage_idx,
                        "batch_size": len(pending_batch),
                        "tasks": pending_batch,
                        "replan_cycle": replan_count,
                    },
                )

                # Concurrent execution of all independent tasks within this stage
                tasks_to_run = [
                    self._dispatch_task(task_id, current_plan, blackboard)
                    for task_id in pending_batch
                ]
                stage_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

                for task_id, res in zip(pending_batch, stage_results):
                    dag_node = current_plan.dag.get_task(task_id)
                    if isinstance(res, Exception):
                        dag_node.state = TaskState.FAILED
                        dag_node.error = str(res)
                        errors.append(f"Task '{task_id}' failed: {res}")
                        await self._emit_event(EventType.TASK_FAILED, {"task_id": task_id, "error": str(res)})
                    else:
                        dag_node.state = TaskState.COMPLETED
                        dag_node.result = res
                        blackboard["task_outputs"][task_id] = res
                        await self._emit_event(EventType.TASK_COMPLETED, {"task_id": task_id})

                # Check if Critic evaluated and issued a REJECTION
                latest_audit: Optional[VerificationAudit] = blackboard.get("latest_audit")
                critic_in_batch = any(current_plan.tasks[t].agent_role == "critic" for t in pending_batch)

                if critic_in_batch and latest_audit and latest_audit.verdict == VerificationVerdict.REJECTED:
                    await self._emit_event(
                        EventType.CRITIC_REJECTED,
                        {
                            "verdict": latest_audit.verdict.value,
                            "remediation": latest_audit.remediation_feedback,
                            "replan_count": replan_count,
                        },
                    )

                    # Trigger dynamic graph surgery if within replanning budget
                    if replan_count < max_replans:
                        failed_task_id = self._identify_failed_task(latest_audit, current_plan)
                        replan_count += 1

                        # Execute dynamic graph surgery
                        new_plan = self.planner.replan(
                            original_plan=current_plan,
                            failed_task_id=failed_task_id,
                            feedback=latest_audit.remediation_feedback or "Remediate verified defect.",
                        )

                        remediation_record = {
                            "replan_index": replan_count,
                            "failed_task_id": failed_task_id,
                            "feedback": latest_audit.remediation_feedback,
                            "new_plan_id": new_plan.plan_id,
                            "timestamp": time.time(),
                        }
                        remediation_history.append(remediation_record)

                        await self._emit_event(
                            EventType.TASK_SCHEDULED,
                            {
                                "plan_id": new_plan.plan_id,
                                "replan_count": replan_count,
                                "remediation_for": failed_task_id,
                            },
                        )

                        # Invalidate downstream critic and synthesizer in blackboard to force re-evaluation
                        for t_id in list(blackboard["task_outputs"].keys()):
                            if t_id.startswith("critic") or t_id.startswith("synthesizer"):
                                del blackboard["task_outputs"][t_id]
                        blackboard["latest_audit"] = None
                        blackboard["final_dossier"] = None

                        current_plan = new_plan
                        replan_triggered = True
                        break  # Break out of stage loop to restart execution on new_plan

            # If all stages completed without triggering a new replan, exit closed loop
            if not replan_triggered:
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        final_dossier: Optional[ExecutiveDossier] = blackboard.get("final_dossier")
        latest_audit = blackboard.get("latest_audit")

        mission_success = (final_dossier is not None) and (len(errors) == 0)

        result = OrchestratorResult(
            query=clean_query,
            success=mission_success,
            plan=current_plan,
            dossier=final_dossier,
            stage_outputs=blackboard["task_outputs"],
            total_latency_ms=elapsed_ms,
            total_stages_executed=stages_processed,
            audit=latest_audit,
            errors=errors,
            replan_count=replan_count,
            remediation_history=remediation_history,
        )

        return result

    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_replans: int = 2,
    ) -> OrchestratorResult:
        """Synchronous entry point for CLI and blocking scripts."""
        return asyncio.run(self.run_mission(query, context=context, max_replans=max_replans))

    # -------------------------------------------------------------------------
    # Internal Task Dispatcher & Failure Resolution
    # -------------------------------------------------------------------------

    async def _dispatch_task(
        self,
        task_id: str,
        plan: ExecutionPlan,
        blackboard: Dict[str, Any],
    ) -> Any:
        """Execute an individual subtask by routing it to its assigned specialized agent."""
        task: PlannedTask = plan.tasks[task_id]
        role = task.agent_role.lower()

        # Update DAG node state
        dag_node = plan.dag.get_task(task_id)
        dag_node.state = TaskState.RUNNING

        # 1. Researcher Task Dispatch
        if role == "researcher":
            target_subject = task.input_context.get("entity") or task.input_context.get("subject")
            dossier: ResearchDossier = await self.researcher.conduct_research(
                task=task.description,
                primary_entity=target_subject,
                context={**task.input_context, "query": blackboard["query"]},
            )
            blackboard["research_dossiers"].append(dossier)
            blackboard["latest_research_dossier"] = dossier
            return dossier

        # 2. Quantitative Analyst Task Dispatch
        elif role == "quant":
            target_entity = task.input_context.get("subject") or task.input_context.get("entity")
            entities = [target_entity] if target_entity else None
            report: QuantReport = await self.quant.analyze_financials(
                task=task.description,
                entities=entities,
                context={**task.input_context, "query": blackboard["query"]},
            )
            blackboard["quant_reports"].append(report)
            blackboard["latest_quant_report"] = report
            return report

        # 3. Critic / Verifier Task Dispatch
        elif role == "critic":
            dossier = blackboard.get("latest_research_dossier")
            quant_report = blackboard.get("latest_quant_report")
            audit: VerificationAudit = self.critic.audit(
                task=task.description,
                research_dossier=dossier,
                quant_report=quant_report,
                context=task.input_context,
            )
            blackboard["audits"].append(audit)
            blackboard["latest_audit"] = audit
            return audit

        # 4. Synthesizer Task Dispatch
        elif role == "synthesizer":
            dossier = blackboard.get("latest_research_dossier")
            quant_report = blackboard.get("latest_quant_report")
            audit = blackboard.get("latest_audit")

            final_briefing: ExecutiveDossier = self.synthesizer.synthesize(
                query=blackboard["query"],
                research_dossier=dossier,
                quant_report=quant_report,
                audit=audit,
                context=task.input_context,
            )
            blackboard["final_dossier"] = final_briefing
            await self._emit_event(
                EventType.REPORT_GENERATED,
                {
                    "title": final_briefing.title,
                    "entities": final_briefing.target_entities,
                    "confidence": final_briefing.overall_confidence,
                    "verdict": final_briefing.audit_verdict,
                },
            )
            return final_briefing

        else:
            raise ValueError(f"Unknown agent role '{role}' for task '{task_id}'.")

    def _identify_failed_task(
        self,
        audit: VerificationAudit,
        plan: ExecutionPlan,
    ) -> str:
        """Map Critic issue defect locations back to specific upstream task IDs."""
        # 1. Inspect issues for location or category clues
        for issue in audit.issues:
            if issue.severity == "CRITICAL":
                if issue.location == "Research Dossier" or issue.category in ("UNGROUNDED_CLAIM", "MISSING_EVIDENCE"):
                    for t_id, task in plan.tasks.items():
                        if task.agent_role == "researcher":
                            return t_id
                elif issue.location == "Quant Report" or issue.category == "NUMERICAL_MISMATCH":
                    for t_id, task in plan.tasks.items():
                        if task.agent_role == "quant":
                            return t_id

        # 2. Fallback to first non-critic upstream task
        for t_id, task in plan.tasks.items():
            if task.agent_role in ("researcher", "quant"):
                return t_id

        # 3. Default to initial task
        return plan.batches[0][0]
