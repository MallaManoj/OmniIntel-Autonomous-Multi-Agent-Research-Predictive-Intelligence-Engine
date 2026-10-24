"""Binary Min-Heap Priority Queue implementation for agent task scheduling.

Provides O(log N) push and pop, O(1) peek, and O(log N) priority updates via index mapping.
Used by the orchestrator to prioritize critical agent reflection tasks and resource allocation.
"""

from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


class PriorityItem(Generic[T]):
    """Wrapper storing task item with its priority weight and insertion sequence."""

    __slots__ = ("priority", "sequence", "data", "task_id")

    def __init__(
        self,
        priority: float | int,
        sequence: int,
        data: T,
        task_id: Optional[str] = None,
    ) -> None:
        self.priority: float | int = priority
        self.sequence: int = sequence
        self.data: T = data
        self.task_id: str = task_id or str(sequence)

    def __lt__(self, other: "PriorityItem[T]") -> bool:
        # Lower numerical value = higher priority (Min-Heap)
        if self.priority != other.priority:
            return self.priority < other.priority
        # Tie-breaker: earlier insertion sequence wins (FIFO stability)
        return self.sequence < other.sequence

    def __repr__(self) -> str:
        return f"PriorityItem(task_id='{self.task_id}', priority={self.priority})"


class PriorityQueue(Generic[T]):
    """High-performance Array-based Binary Min-Heap with indexed key updates.

    Complexity:
        - push: O(log N)
        - pop: O(log N)
        - peek: O(1)
        - update_priority: O(log N) using position index map (vs O(N) in naive heaps)
    """

    def __init__(self) -> None:
        self._heap: list[PriorityItem[T]] = []
        self._entry_count: int = 0
        # Position index map: task_id -> current array index in self._heap
        self._index_map: dict[str, int] = {}

    @property
    def is_empty(self) -> bool:
        """Return True if priority queue has no items."""
        return len(self._heap) == 0

    def push(
        self,
        data: T,
        priority: float | int,
        task_id: Optional[str] = None,
    ) -> None:
        """Insert an item into the Min-Heap.

        Time Complexity: O(log N)
        """
        tid = task_id or f"task_{self._entry_count}"
        if tid in self._index_map:
            raise ValueError(f"Task with ID '{tid}' already exists in priority queue.")

        item = PriorityItem(
            priority=priority,
            sequence=self._entry_count,
            data=data,
            task_id=tid,
        )
        self._entry_count += 1

        # Append to end of array and sift up
        self._heap.append(item)
        idx = len(self._heap) - 1
        self._index_map[tid] = idx
        self._sift_up(idx)

    def pop(self) -> T:
        """Remove and return the highest priority item (lowest priority value).

        Raises:
            IndexError: If priority queue is empty.

        Time Complexity: O(log N)
        """
        if self.is_empty:
            raise IndexError("Cannot pop from an empty PriorityQueue.")

        min_item = self._heap[0]
        last_item = self._heap.pop()
        del self._index_map[min_item.task_id]

        if not self.is_empty:
            self._heap[0] = last_item
            self._index_map[last_item.task_id] = 0
            self._sift_down(0)

        return min_item.data

    def peek(self) -> Optional[T]:
        """Return the highest priority item without removing it.

        Time Complexity: O(1)
        """
        if self.is_empty:
            return None
        return self._heap[0].data

    def peek_priority(self) -> Optional[float | int]:
        """Return the priority value of the top item."""
        if self.is_empty:
            return None
        return self._heap[0].priority

    def update_priority(self, task_id: str, new_priority: float | int) -> None:
        """Update the priority weight of an existing item in O(log N).

        Time Complexity: O(log N)
        """
        if task_id not in self._index_map:
            raise KeyError(f"Task with ID '{task_id}' not found in PriorityQueue.")

        idx = self._index_map[task_id]
        old_priority = self._heap[idx].priority
        self._heap[idx].priority = new_priority

        if new_priority < old_priority:
            self._sift_up(idx)
        elif new_priority > old_priority:
            self._sift_down(idx)

    def clear(self) -> None:
        """Empty the priority queue."""
        self._heap.clear()
        self._index_map.clear()
        self._entry_count = 0

    # -------------------------------------------------------------------------
    # Binary Heap Index Arithmetic & Sifting
    # -------------------------------------------------------------------------

    @staticmethod
    def _parent(index: int) -> int:
        return (index - 1) // 2

    @staticmethod
    def _left_child(index: int) -> int:
        return 2 * index + 1

    @staticmethod
    def _right_child(index: int) -> int:
        return 2 * index + 2

    def _swap(self, i: int, j: int) -> None:
        """Swap two elements in the array and update their index mappings."""
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._index_map[self._heap[i].task_id] = i
        self._index_map[self._heap[j].task_id] = j

    def _sift_up(self, index: int) -> None:
        """Move element at index up until min-heap property is satisfied."""
        current = index
        while current > 0:
            parent = self._parent(current)
            if self._heap[current] < self._heap[parent]:
                self._swap(current, parent)
                current = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        """Move element at index down until min-heap property is satisfied."""
        current = index
        n = len(self._heap)

        while True:
            smallest = current
            left = self._left_child(current)
            right = self._right_child(current)

            if left < n and self._heap[left] < self._heap[smallest]:
                smallest = left

            if right < n and self._heap[right] < self._heap[smallest]:
                smallest = right

            if smallest != current:
                self._swap(current, smallest)
                current = smallest
            else:
                break

    def __len__(self) -> int:
        return len(self._heap)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._index_map

    def __repr__(self) -> str:
        return f"PriorityQueue(size={len(self._heap)}, top_priority={self.peek_priority()})"
