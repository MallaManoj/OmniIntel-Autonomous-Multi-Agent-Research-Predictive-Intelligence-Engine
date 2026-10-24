"""Directed Acyclic Graph (DAG) data structure for Multi-Agent Task Orchestration.

Manages task dependencies, execution state transitions, and parent-child linkages.
Guarantees deterministic execution order and parallel batch discovery for agent subtasks.
"""

from enum import Enum
from typing import Any, Callable, Optional


class TaskState(str, Enum):
    """Lifecycle states of a task node within the execution graph."""

    PENDING = "PENDING"      # Waiting for dependencies to complete
    READY = "READY"          # All dependencies satisfied, ready for execution
    RUNNING = "RUNNING"      # Actively executing by an agent worker
    COMPLETED = "COMPLETED"  # Successfully finished with valid output
    FAILED = "FAILED"        # Terminated with error; dependent tasks cannot proceed
    SKIPPED = "SKIPPED"      # Bypassed due to upstream failure or conditional branching


class TaskNode:
    """Represents an atomic unit of work within the agent execution graph.

    Attributes:
        task_id: Unique identifier for the task.
        name: Human-readable name or description of the action.
        action: Optional callable agent function or tool execution logic.
        dependencies: Set of task IDs that must successfully finish before this task runs.
        dependents: Set of task IDs that wait on this task to finish.
        state: Current lifecycle state of the node.
        result: Output payload once task completes.
        error: Exception message if execution fails.
    """

    __slots__ = (
        "task_id",
        "name",
        "action",
        "dependencies",
        "dependents",
        "state",
        "result",
        "error",
    )

    def __init__(
        self,
        task_id: str,
        name: str,
        action: Optional[Callable[..., Any]] = None,
        dependencies: Optional[set[str]] = None,
    ) -> None:
        self.task_id: str = task_id
        self.name: str = name
        self.action: Optional[Callable[..., Any]] = action
        self.dependencies: set[str] = set(dependencies) if dependencies else set()
        self.dependents: set[str] = set()
        self.state: TaskState = TaskState.PENDING
        self.result: Any = None
        self.error: Optional[str] = None

    def is_ready(self, completed_task_ids: set[str]) -> bool:
        """Check if all prerequisite dependencies have successfully completed."""
        return self.dependencies.issubset(completed_task_ids)

    def __repr__(self) -> str:
        return (
            f"TaskNode(id='{self.task_id}', name='{self.name}', "
            f"state={self.state.value}, in_degree={len(self.dependencies)})"
        )


class DAG:
    """Directed Acyclic Graph modeling agent workflows using adjacency lists.

    Provides node registration, dependency edge management, and degree inspection.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, TaskNode] = {}

    @property
    def total_tasks(self) -> int:
        """Return the number of registered tasks."""
        return len(self._nodes)

    @property
    def nodes(self) -> dict[str, TaskNode]:
        """Read-only view of registered task nodes."""
        return self._nodes

    def add_task(
        self,
        task_id: str,
        name: str,
        action: Optional[Callable[..., Any]] = None,
        dependencies: Optional[list[str]] = None,
    ) -> TaskNode:
        """Add a new task node to the graph.

        Raises:
            ValueError: If task_id already exists or contains self-dependency.
        """
        if not task_id:
            raise ValueError("Task ID cannot be empty.")
        if task_id in self._nodes:
            raise ValueError(f"Task with ID '{task_id}' already exists in DAG.")

        deps = set(dependencies) if dependencies else set()
        if task_id in deps:
            raise ValueError(f"Task '{task_id}' cannot depend on itself (self-loop).")

        node = TaskNode(task_id=task_id, name=name, action=action, dependencies=deps)
        self._nodes[task_id] = node

        # Link dependencies to existing nodes
        for dep_id in deps:
            if dep_id in self._nodes:
                self._nodes[dep_id].dependents.add(task_id)

        # Update any previously registered nodes that marked this task as a dependency
        for other_node in self._nodes.values():
            if task_id in other_node.dependencies:
                node.dependents.add(other_node.task_id)

        return node

    def add_dependency(self, upstream_id: str, downstream_id: str) -> None:
        """Create a directed edge: upstream_id -> downstream_id.

        downstream_id will require upstream_id to complete first.
        """
        if upstream_id not in self._nodes:
            raise KeyError(f"Upstream task '{upstream_id}' does not exist in DAG.")
        if downstream_id not in self._nodes:
            raise KeyError(f"Downstream task '{downstream_id}' does not exist in DAG.")
        if upstream_id == downstream_id:
            raise ValueError(f"Self-dependency detected: Task '{upstream_id}' cannot depend on itself.")

        self._nodes[downstream_id].dependencies.add(upstream_id)
        self._nodes[upstream_id].dependents.add(downstream_id)

    def get_task(self, task_id: str) -> TaskNode:
        """Retrieve a task node by ID."""
        if task_id not in self._nodes:
            raise KeyError(f"Task '{task_id}' not found in DAG.")
        return self._nodes[task_id]

    def get_in_degree(self, task_id: str) -> int:
        """Return the number of incoming edges (prerequisites) for a task."""
        return len(self.get_task(task_id).dependencies)

    def get_out_degree(self, task_id: str) -> int:
        """Return the number of outgoing edges (tasks that depend on this one)."""
        return len(self.get_task(task_id).dependents)

    def get_root_tasks(self) -> list[str]:
        """Find all entry tasks with in-degree = 0 (no prerequisites).

        These tasks can start execution immediately in parallel.
        """
        return [task_id for task_id, node in self._nodes.items() if len(node.dependencies) == 0]

    def get_leaf_tasks(self) -> list[str]:
        """Find all terminal tasks with out-degree = 0 (no dependents)."""
        return [task_id for task_id, node in self._nodes.items() if len(node.dependents) == 0]

    def reset_states(self) -> None:
        """Reset all node execution states back to PENDING."""
        for node in self._nodes.values():
            node.state = TaskState.PENDING
            node.result = None
            node.error = None

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._nodes

    def __repr__(self) -> str:
        return f"DAG(total_tasks={len(self._nodes)}, roots={len(self.get_root_tasks())})"
