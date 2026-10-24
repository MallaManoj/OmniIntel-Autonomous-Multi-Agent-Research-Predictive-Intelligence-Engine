"""Directed Acyclic Graph (DAG) data structure for Multi-Agent Task Orchestration.

Manages task dependencies, execution state transitions, and parent-child linkages.
Guarantees deterministic execution order and parallel batch discovery for agent subtasks.
Includes Kahn's Algorithm for Topological Sorting and cycle detection.
"""

from collections import deque
from enum import Enum
from typing import Any, Callable, Optional


class CyclicDependencyError(Exception):
    """Raised when a circular dependency cycle is detected in the execution graph."""

    def __init__(self, cycle_path: Optional[list[str]] = None) -> None:
        self.cycle_path = cycle_path
        if cycle_path:
            formatted_path = " -> ".join(cycle_path)
            super().__init__(f"Circular dependency cycle detected: {formatted_path}")
        else:
            super().__init__("Circular dependency cycle detected in DAG.")


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

    Supports dependency management, cycle detection, and Kahn's topological sort.
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

    # -------------------------------------------------------------------------
    # Kahn's Algorithm & Topological Sorting
    # -------------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """Generate a linear ordering of tasks using Kahn's Algorithm (BFS).

        Returns:
            list[str]: Task IDs ordered such that for every dependency u -> v,
                       u appears before v.

        Raises:
            CyclicDependencyError: If a cycle exists, preventing topological sort.

        Time Complexity: O(V + E) where V is tasks count, E is dependency edges.
        Space Complexity: O(V)
        """
        # Step 1: Compute in-degree for all vertices
        in_degree: dict[str, int] = {
            task_id: len(node.dependencies) for task_id, node in self._nodes.items()
        }

        # Step 2: Initialize queue with all nodes having in-degree 0 (roots)
        queue: deque[str] = deque([task_id for task_id, deg in in_degree.items() if deg == 0])
        ordered_tasks: list[str] = []

        # Step 3: Process queue
        while queue:
            current_id = queue.popleft()
            ordered_tasks.append(current_id)

            for dependent_id in self._nodes[current_id].dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        # Step 4: Verify cycle invariant
        if len(ordered_tasks) != len(self._nodes):
            cycle = self.find_cycle()
            raise CyclicDependencyError(cycle_path=cycle)

        return ordered_tasks

    def get_execution_batches(self) -> list[list[str]]:
        """Group tasks into parallelizable execution stages using Kahn's Algorithm.

        Each batch contains tasks whose dependencies have completed in previous batches.
        All tasks within a batch can be executed concurrently via asyncio / threads.

        Returns:
            list of batches, e.g. [['fetch_a', 'fetch_b'], ['process_ab'], ['report']]

        Raises:
            CyclicDependencyError: If a cycle is present in the graph.
        """
        in_degree: dict[str, int] = {
            task_id: len(node.dependencies) for task_id, node in self._nodes.items()
        }

        # Initial frontier of ready tasks (in-degree == 0)
        current_batch: list[str] = [task_id for task_id, deg in in_degree.items() if deg == 0]
        batches: list[list[str]] = []
        processed_count = 0

        while current_batch:
            batches.append(sorted(current_batch))
            processed_count += len(current_batch)
            next_batch: list[str] = []

            for task_id in current_batch:
                for dependent_id in self._nodes[task_id].dependents:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        next_batch.append(dependent_id)

            current_batch = next_batch

        if processed_count != len(self._nodes):
            raise CyclicDependencyError(cycle_path=self.find_cycle())

        return batches

    def has_cycle(self) -> bool:
        """Check if the graph contains any circular dependency.

        Time Complexity: O(V + E)
        """
        try:
            self.topological_sort()
            return False
        except CyclicDependencyError:
            return True

    def find_cycle(self) -> Optional[list[str]]:
        """Detect and return the exact cycle path using 3-color DFS traversal.

        Colors:
            0 = WHITE (unvisited)
            1 = GRAY (currently in recursion call stack)
            2 = BLACK (fully processed)

        Returns:
            Optional[list[str]]: Path of cycle nodes [A, B, C, A] or None.
        """
        color: dict[str, int] = {task_id: 0 for task_id in self._nodes}
        parent: dict[str, Optional[str]] = {task_id: None for task_id in self._nodes}
        cycle_path: list[str] = []

        def _dfs(node_id: str) -> bool:
            color[node_id] = 1  # Mark GRAY (in stack)

            for dependent_id in self._nodes[node_id].dependents:
                if color[dependent_id] == 1:
                    # Found back-edge to an active ancestor in stack -> CYCLE!
                    curr = node_id
                    cycle_nodes = [dependent_id]
                    while curr != dependent_id and curr is not None:
                        cycle_nodes.append(curr)
                        curr = parent.get(curr)  # type: ignore[assignment]
                    cycle_nodes.append(dependent_id)
                    cycle_nodes.reverse()
                    cycle_path.extend(cycle_nodes)
                    return True

                if color[dependent_id] == 0:
                    parent[dependent_id] = node_id
                    if _dfs(dependent_id):
                        return True

            color[node_id] = 2  # Mark BLACK (done)
            return False

        for task_id in self._nodes:
            if color[task_id] == 0:
                if _dfs(task_id):
                    return cycle_path

        return None

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
