"""Autonomous Multi-Agent Mission and Planning API routes."""

import time
from fastapi import APIRouter, Depends, HTTPException, status

from src.agents.orchestrator import OrchestratorResult
from src.agents.planner import ExecutionPlan
from src.api.schemas import (
    MissionRequest,
    MissionResponse,
    PlanOnlyRequest,
    PlanResponse,
)
from src.api.state import AppState, get_app_state

router = APIRouter(prefix="/api/v1/mission", tags=["Missions & Planning"])


@router.post("/plan", response_model=PlanResponse, summary="Decompose Query into DAG Plan")
async def plan_mission(
    payload: PlanOnlyRequest,
    state: AppState = Depends(get_app_state),
) -> PlanResponse:
    """Decompose high-level directive into an acyclic task DAG using Kahn's algorithm."""
    if not state.is_ready or state.orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Planning service is initializing.",
        )

    try:
        plan: ExecutionPlan = state.orchestrator.planner.create_plan(
            query=payload.query,
            context=payload.context,
        )

        tasks_dict = {
            tid: {
                "name": task.name,
                "agent_role": task.agent_role,
                "description": task.description,
                "dependencies": task.dependencies,
                "expected_output": task.expected_output,
            }
            for tid, task in plan.tasks.items()
        }

        return PlanResponse(
            plan_id=plan.plan_id,
            query=plan.query,
            complexity=plan.complexity.value,
            total_tasks=plan.total_tasks,
            total_stages=plan.total_stages,
            topological_order=plan.topological_order,
            batches=plan.batches,
            tasks=tasks_dict,
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate execution plan: {exc}",
        )


@router.post("/run", response_model=MissionResponse, summary="Execute Autonomous Research Mission")
async def execute_mission(
    payload: MissionRequest,
    state: AppState = Depends(get_app_state),
) -> MissionResponse:
    """Run full multi-agent mission across parallel stages with closed-loop self-correction."""
    if not state.is_ready or state.orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mission orchestrator is initializing.",
        )

    try:
        result: OrchestratorResult = await state.orchestrator.run_mission(
            query=payload.query,
            context=payload.context,
            max_replans=payload.max_replans,
        )

        dossier = result.dossier
        target_entities = dossier.target_entities if dossier else []
        verdict = result.audit.verdict.value if result.audit else None
        confidence = dossier.overall_confidence if dossier else 0.0
        markdown = result.to_markdown()

        return MissionResponse(
            query=result.query,
            success=result.success,
            plan_id=result.plan.plan_id,
            total_latency_ms=round(result.total_latency_ms, 2),
            total_stages_executed=result.total_stages_executed,
            replan_count=result.replan_count,
            verdict=verdict,
            confidence_score=round(confidence, 3),
            target_entities=target_entities,
            dossier_markdown=markdown,
            errors=result.errors,
            created_at=result.created_at,
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Mission execution failed: {exc}",
        )
