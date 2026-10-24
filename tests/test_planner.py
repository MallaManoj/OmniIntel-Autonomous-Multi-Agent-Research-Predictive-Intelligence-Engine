"""Unit tests for PlannerAgent, ExecutionPlan, and Dynamic Replanning."""

import asyncio
import unittest

from src.agents.base import AgentState
from src.agents.planner import (
    ExecutionPlan,
    PlanComplexity,
    PlannedTask,
    PlannerAgent,
)
from src.core.dsa.dag import CyclicDependencyError, DAG
from src.core.events import Event, EventBus, EventType


class TestPlannerAgent(unittest.TestCase):
    """Test suite verifying PlannerAgent decomposition, Kahn's algorithm batching, and replanning."""

    def setUp(self) -> None:
        self.event_bus = EventBus()
        self.planner = PlannerAgent(event_bus=self.event_bus)

    def test_plan_creation_simple_research(self) -> None:
        """Simple research query should generate research, critic, and synthesizer tasks."""
        query = "Investigate TSMC manufacturing moat and geopolitical risks"
        plan = self.planner.create_plan(query)

        self.assertIsInstance(plan, ExecutionPlan)
        self.assertGreaterEqual(plan.total_tasks, 3)
        self.assertGreaterEqual(plan.total_stages, 2)
        self.assertIn("TSMC", plan.metadata["intent"]["entities"])

        # Check topological sort validity
        self.assertEqual(len(plan.topological_order), plan.total_tasks)

        # Check that root tasks are in batch 0
        root_tasks = plan.dag.get_root_tasks()
        self.assertGreaterEqual(len(root_tasks), 1)
        for r_id in root_tasks:
            self.assertEqual(plan.get_stage_for_task(r_id), 0)

        # Terminal leaf should be synthesizer
        leaf_tasks = plan.dag.get_leaf_tasks()
        self.assertEqual(len(leaf_tasks), 1)
        self.assertEqual(plan.tasks[leaf_tasks[0]].agent_role, "synthesizer")

    def test_plan_creation_comparative_and_quant(self) -> None:
        """Comparative financial query should generate multi-entity research and quantitative tasks."""
        query = "Compare NVIDIA vs AMD gross margin trends, debt ratios, and datacenter market share"
        plan = self.planner.create_plan(query)

        intent = plan.metadata["intent"]
        self.assertTrue(intent["has_quant"])
        self.assertTrue(intent["is_comparison"])
        self.assertTrue(intent["has_research"])

        # Roles should include researcher, quant, critic, and synthesizer
        roles = {task.agent_role for task in plan.tasks.values()}
        self.assertIn("researcher", roles)
        self.assertIn("quant", roles)
        self.assertIn("critic", roles)
        self.assertIn("synthesizer", roles)

        # Stage 1 (Batch 0) should execute independent researcher and quant tasks concurrently
        batch_0 = plan.batches[0]
        self.assertGreaterEqual(len(batch_0), 2)  # At least 2 tasks running in parallel
        self.assertGreaterEqual(plan.max_parallel_width, 2)

    def test_empty_query_raises_value_error(self) -> None:
        """Empty or whitespace-only query must raise ValueError."""
        with self.assertRaises(ValueError):
            self.planner.create_plan("")
        with self.assertRaises(ValueError):
            self.planner.create_plan("   \n\t  ")

    def test_dynamic_replanning_on_failure(self) -> None:
        """Replanning must inject remediation task and rewire downstream dependencies cleanly."""
        query = "Analyze Apple supply chain dependencies"
        original_plan = self.planner.create_plan(query)

        # Pick a root research task to simulate failure
        failed_task_id = original_plan.batches[0][0]
        feedback = "Missing Tier-2 supplier data for camera sensor modules"

        new_plan = self.planner.replan(
            original_plan=original_plan,
            failed_task_id=failed_task_id,
            feedback=feedback,
        )

        remedy_id = f"remedy_{failed_task_id}"
        self.assertIn(remedy_id, new_plan.tasks)
        self.assertNotIn(failed_task_id, new_plan.tasks)

        # Verify remediation task description contains feedback
        remedy_task = new_plan.tasks[remedy_id]
        self.assertIn(feedback, remedy_task.description)

        # Verify downstream tasks now depend on remedy_task_id instead of failed_task_id
        for t_id, task in new_plan.tasks.items():
            self.assertNotIn(failed_task_id, task.dependencies)

        # Verify new plan is topologically valid without cycles
        self.assertEqual(len(new_plan.topological_order), new_plan.total_tasks)
        self.assertEqual(new_plan.metadata["remediated_task_id"], failed_task_id)

    def test_replan_invalid_task_id_raises_key_error(self) -> None:
        """Replanning with non-existent task ID must raise KeyError."""
        plan = self.planner.create_plan("Analyze Tesla battery patents")
        with self.assertRaises(KeyError):
            self.planner.replan(plan, failed_task_id="non_existent_task", feedback="error")

    def test_planner_react_lifecycle_execution(self) -> None:
        """Planner execution must transition through ReAct lifecycle and store plan in scratchpad."""
        query = "Evaluate Microsoft cloud revenue growth rate and AI capital expenditures"
        response = self.planner.run(query)

        self.assertTrue(response.success)
        self.assertEqual(response.state, AgentState.COMPLETED)
        self.assertIn("Execution Plan", response.output)

        # Verify working scratchpad retained the plan
        active_plan = self.planner.memory.get_variable("active_plan")
        self.assertIsInstance(active_plan, ExecutionPlan)
        self.assertEqual(active_plan.query, query)

        # Verify steps
        step_types = [step["step"] for step in response.execution_steps]
        self.assertIn("think", step_types)
        self.assertIn("reflect", step_types)

        # Verify reflection validity
        reflect_step = next(s for s in response.execution_steps if s["step"] == "reflect")
        self.assertTrue(reflect_step["valid"])

    def test_planner_reflection_catches_invalid_plan(self) -> None:
        """Reflection phase must reject plans that violate DAG invariants or lack synthesis."""
        # 1. Missing plan in scratchpad
        self.planner.memory.clear()
        is_valid, msg = asyncio.run(self.planner.reflect("test", "thoughts", []))
        self.assertFalse(is_valid)
        self.assertIn("failed to retain", msg)

        # 2. Plan with cyclic DAG
        plan = self.planner.create_plan("Test query")
        # Artificially inject a cycle into the plan DAG
        t1, t2 = list(plan.tasks.keys())[:2]
        plan.dag.add_dependency(t1, t2)
        try:
            plan.dag.add_dependency(t2, t1)
        except Exception:
            pass  # add_dependency might allow or we set manually
        plan.dag._nodes[t1].dependencies.add(t2)
        plan.dag._nodes[t2].dependencies.add(t1)

        self.planner.memory.set_variable("active_plan", plan)
        is_valid, msg = asyncio.run(self.planner.reflect("test", "thoughts", []))
        self.assertFalse(is_valid)
        self.assertIn("Cyclic dependency", msg)

    def test_plan_serialization_and_summary(self) -> None:
        """ExecutionPlan summary and to_dict must produce complete structured schemas."""
        query = "Evaluate Qualcomm 5G licensing margins and patent litigation"
        plan = self.planner.create_plan(query)

        summary_text = plan.summary()
        self.assertIn("Execution Plan", summary_text)
        self.assertIn("Stage 1", summary_text)

        plan_dict = plan.to_dict()
        self.assertEqual(plan_dict["query"], query)
        self.assertIn("batches", plan_dict)
        self.assertIn("tasks", plan_dict)
        self.assertEqual(plan_dict["total_tasks"], plan.total_tasks)


if __name__ == "__main__":
    unittest.main()
