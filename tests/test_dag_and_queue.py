"""Unit tests for DAG task scheduling, Kahn's algorithm, and PriorityQueue.

Compatible with both standard library `unittest` and `pytest`.
"""

import unittest
from src.core.dsa.dag import DAG, CyclicDependencyError, TaskState
from src.core.dsa.priority_queue import PriorityQueue


class TestDAGAndScheduler(unittest.TestCase):
    """Test suite verifying DAG structure, topological sort, and cycle detection."""

    def test_add_task_validation(self) -> None:
        """DAG should reject empty IDs, duplicate tasks, and self-loops."""
        dag = DAG()
        dag.add_task("task1", "First Task")

        # Empty ID
        with self.assertRaises(ValueError):
            dag.add_task("", "Invalid")

        # Duplicate ID
        with self.assertRaises(ValueError):
            dag.add_task("task1", "Duplicate")

        # Direct self-loop in declaration
        with self.assertRaises(ValueError):
            dag.add_task("loop", "Self Loop", dependencies=["loop"])

    def test_add_dependency_validation(self) -> None:
        """Adding dependencies should validate existing nodes and reject self-dependency."""
        dag = DAG()
        dag.add_task("A", "Task A")
        dag.add_task("B", "Task B")

        dag.add_dependency("A", "B")
        self.assertEqual(dag.get_in_degree("B"), 1)
        self.assertEqual(dag.get_out_degree("A"), 1)

        # Missing upstream
        with self.assertRaises(KeyError):
            dag.add_dependency("non_existent", "B")

        # Missing downstream
        with self.assertRaises(KeyError):
            dag.add_dependency("A", "non_existent")

        # Self-dependency
        with self.assertRaises(ValueError):
            dag.add_dependency("A", "A")

    def test_root_and_leaf_discovery(self) -> None:
        """Verify correct identification of entry (root) and terminal (leaf) nodes."""
        dag = DAG()
        dag.add_task("start1", "Start 1")
        dag.add_task("start2", "Start 2")
        dag.add_task("mid", "Middle", dependencies=["start1", "start2"])
        dag.add_task("end", "End", dependencies=["mid"])

        self.assertEqual(set(dag.get_root_tasks()), {"start1", "start2"})
        self.assertEqual(dag.get_leaf_tasks(), ["end"])

    def test_topological_sort_diamond_graph(self) -> None:
        """Kahn's algorithm must order diamond dependencies correctly."""
        dag = DAG()
        dag.add_task("A", "Root")
        dag.add_task("B", "Left Branch", dependencies=["A"])
        dag.add_task("C", "Right Branch", dependencies=["A"])
        dag.add_task("D", "Merge", dependencies=["B", "C"])

        order = dag.topological_sort()
        self.assertEqual(len(order), 4)
        self.assertTrue(order.index("A") < order.index("B"))
        self.assertTrue(order.index("A") < order.index("C"))
        self.assertTrue(order.index("B") < order.index("D"))
        self.assertTrue(order.index("C") < order.index("D"))

    def test_execution_batches_concurrency(self) -> None:
        """Verify parallel execution batches for concurrent agent stages."""
        dag = DAG()
        dag.add_task("fetch_a", "Fetch A")
        dag.add_task("fetch_b", "Fetch B")
        dag.add_task("combine", "Combine A and B", dependencies=["fetch_a", "fetch_b"])
        dag.add_task("report", "Report", dependencies=["combine"])

        batches = dag.get_execution_batches()
        expected = [
            ["fetch_a", "fetch_b"],  # Stage 0: Run concurrently
            ["combine"],             # Stage 1: Waits for Stage 0
            ["report"],              # Stage 2: Waits for Stage 1
        ]
        self.assertEqual(batches, expected)

    def test_cycle_detection_direct(self) -> None:
        """A 2-node cycle A -> B -> A must raise CyclicDependencyError."""
        dag = DAG()
        dag.add_task("A", "Task A")
        dag.add_task("B", "Task B", dependencies=["A"])
        dag.add_dependency("B", "A")  # Cycle

        self.assertTrue(dag.has_cycle())
        with self.assertRaises(CyclicDependencyError):
            dag.topological_sort()
        with self.assertRaises(CyclicDependencyError):
            dag.get_execution_batches()

    def test_cycle_path_reconstruction(self) -> None:
        """Verify that find_cycle accurately extracts the 3-color DFS loop path."""
        dag = DAG()
        dag.add_task("X", "Task X")
        dag.add_task("Y", "Task Y", dependencies=["X"])
        dag.add_task("Z", "Task Z", dependencies=["Y"])
        dag.add_dependency("Z", "X")

        cycle_path = dag.find_cycle()
        self.assertIsNotNone(cycle_path)
        # Cycle must start and end at the same node
        self.assertEqual(cycle_path[0], cycle_path[-1])
        self.assertEqual(set(cycle_path), {"X", "Y", "Z"})

    def test_task_lifecycle_states(self) -> None:
        """Verify task readiness predicate and reset_states functionality."""
        dag = DAG()
        dag.add_task("p1", "Prereq 1")
        dag.add_task("p2", "Prereq 2")
        t = dag.add_task("target", "Target", dependencies=["p1", "p2"])

        self.assertFalse(t.is_ready(set()))
        self.assertFalse(t.is_ready({"p1"}))
        self.assertTrue(t.is_ready({"p1", "p2"}))

        t.state = TaskState.COMPLETED
        t.result = {"data": 123}
        dag.reset_states()
        self.assertEqual(t.state, TaskState.PENDING)
        self.assertIsNone(t.result)


class TestPriorityQueue(unittest.TestCase):
    """Test suite verifying Binary Min-Heap PriorityQueue correctness and O(log N) updates."""

    def test_min_heap_ordering(self) -> None:
        """Items must be popped in order of lowest numerical priority value."""
        pq: PriorityQueue[str] = PriorityQueue()
        pq.push("low_urgency", priority=10.0)
        pq.push("highest_urgency", priority=1.0)
        pq.push("medium_urgency", priority=5.0)

        self.assertEqual(len(pq), 3)
        self.assertEqual(pq.peek(), "highest_urgency")
        self.assertEqual(pq.pop(), "highest_urgency")
        self.assertEqual(pq.pop(), "medium_urgency")
        self.assertEqual(pq.pop(), "low_urgency")
        self.assertTrue(pq.is_empty)

    def test_fifo_stability_for_equal_priorities(self) -> None:
        """Items with identical priorities must pop in insertion sequence order."""
        pq: PriorityQueue[str] = PriorityQueue()
        pq.push("first", priority=2.0)
        pq.push("second", priority=2.0)
        pq.push("third", priority=2.0)

        self.assertEqual(pq.pop(), "first")
        self.assertEqual(pq.pop(), "second")
        self.assertEqual(pq.pop(), "third")

    def test_update_priority_decrease_key(self) -> None:
        """Decreasing an item's priority must sift it up to the top in O(log N)."""
        pq: PriorityQueue[str] = PriorityQueue()
        pq.push("task_a", priority=10.0, task_id="A")
        pq.push("task_b", priority=5.0, task_id="B")

        self.assertEqual(pq.peek(), "task_b")

        # Promote task_a to top priority (0.5)
        pq.update_priority("A", new_priority=0.5)
        self.assertEqual(pq.peek(), "task_a")
        self.assertEqual(pq.pop(), "task_a")
        self.assertEqual(pq.pop(), "task_b")

    def test_update_priority_increase_key(self) -> None:
        """Increasing an item's priority must sift it down in O(log N)."""
        pq: PriorityQueue[str] = PriorityQueue()
        pq.push("task_a", priority=1.0, task_id="A")
        pq.push("task_b", priority=5.0, task_id="B")

        # Demote task_a to priority 20.0
        pq.update_priority("A", new_priority=20.0)
        self.assertEqual(pq.peek(), "task_b")
        self.assertEqual(pq.pop(), "task_b")
        self.assertEqual(pq.pop(), "task_a")

    def test_priority_queue_error_handling(self) -> None:
        """Verify exceptions for empty pop, duplicate task_id, and unknown update."""
        pq: PriorityQueue[str] = PriorityQueue()

        # Pop from empty
        with self.assertRaises(IndexError):
            pq.pop()

        # Peek empty
        self.assertIsNone(pq.peek())
        self.assertIsNone(pq.peek_priority())

        # Duplicate task_id
        pq.push("item1", priority=1.0, task_id="id_1")
        with self.assertRaises(ValueError):
            pq.push("item2", priority=2.0, task_id="id_1")

        # Update non-existent task_id
        with self.assertRaises(KeyError):
            pq.update_priority("unknown_id", 0.1)

    def test_clear(self) -> None:
        """clear() must empty the priority queue and reset index map."""
        pq: PriorityQueue[str] = PriorityQueue()
        pq.push("item", priority=1.0, task_id="t1")
        pq.clear()
        self.assertEqual(len(pq), 0)
        self.assertTrue(pq.is_empty)
        self.assertNotIn("t1", pq)


if __name__ == "__main__":
    unittest.main()
