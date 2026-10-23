"""Least Recently Used (LRU) Cache implementation using a Doubly Linked List and Hash Map.

Provides O(1) time complexity for get, put, and remove operations.
Used for caching model embeddings, LLM query responses, and vector search candidates.
"""

from typing import Generic, Hashable, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class _Node(Generic[K, V]):
    """Internal node for the doubly linked list."""

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Optional[K] = None, value: Optional[V] = None) -> None:
        self.key: Optional[K] = key
        self.value: Optional[V] = value
        self.prev: Optional["_Node[K, V]"] = None
        self.next: Optional["_Node[K, V]"] = None


class LRUCache(Generic[K, V]):
    """High-performance generic LRU Cache with O(1) read/write operations.

    Maintains elements ordered by recency. When cache capacity is reached,
    the least recently accessed element is evicted in O(1) time.

    Attributes:
        capacity (int): Maximum number of elements allowed in the cache.
        hits (int): Total number of successful cache hits.
        misses (int): Total number of cache misses.
        evictions (int): Total number of items evicted due to capacity overflow.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"Capacity must be a positive integer, got {capacity}")

        self._capacity: int = capacity
        self._map: dict[K, _Node[K, V]] = {}

        # Dummy sentinel nodes to avoid edge-case checks during node insertion/removal
        self._head: _Node[K, V] = _Node()
        self._tail: _Node[K, V] = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

        # Telemetry / Monitoring statistics
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0

    @property
    def capacity(self) -> int:
        """Return the maximum capacity of the cache."""
        return self._capacity

    @property
    def size(self) -> int:
        """Return the current number of items stored in the cache."""
        return len(self._map)

    @property
    def hit_rate(self) -> float:
        """Calculate the cache hit rate ratio [0.0 - 1.0]."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, float | int]:
        """Return a snapshot of cache performance metrics."""
        return {
            "capacity": self._capacity,
            "size": len(self._map),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
        }

    def get(self, key: K) -> Optional[V]:
        """Retrieve an item by key and mark it as most recently used.

        Time Complexity: O(1)
        """
        if key not in self._map:
            self.misses += 1
            return None

        node = self._map[key]
        self._move_to_head(node)
        self.hits += 1
        return node.value

    def peek(self, key: K) -> Optional[V]:
        """Retrieve an item without updating its recency.

        Time Complexity: O(1)
        """
        node = self._map.get(key)
        return node.value if node else None

    def put(self, key: K, value: V) -> None:
        """Insert or update a key-value pair.

        If the key already exists, update its value and move to head.
        If new and capacity is exceeded, evict the least recently used node.

        Time Complexity: O(1)
        """
        if key in self._map:
            node = self._map[key]
            node.value = value
            self._move_to_head(node)
            return

        new_node = _Node(key=key, value=value)
        self._map[key] = new_node
        self._add_node(new_node)

        if len(self._map) > self._capacity:
            lru_node = self._pop_tail()
            if lru_node and lru_node.key is not None:
                del self._map[lru_node.key]
                self.evictions += 1

    def remove(self, key: K) -> bool:
        """Remove a key from the cache if present.

        Returns:
            bool: True if key was removed, False if key did not exist.

        Time Complexity: O(1)
        """
        if key not in self._map:
            return False

        node = self._map.pop(key)
        self._remove_node(node)
        return True

    def clear(self) -> None:
        """Empty the cache and reset statistics."""
        self._map.clear()
        self._head.next = self._tail
        self._tail.prev = self._head
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # -------------------------------------------------------------------------
    # Internal Doubly Linked List Operations (All O(1))
    # -------------------------------------------------------------------------

    def _add_node(self, node: _Node[K, V]) -> None:
        """Always insert a new node right after the dummy head."""
        node.prev = self._head
        node.next = self._head.next
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _remove_node(self, node: _Node[K, V]) -> None:
        """Unlink an existing node from the doubly linked list."""
        prev_node = node.prev
        next_node = node.next
        if prev_node:
            prev_node.next = next_node
        if next_node:
            next_node.prev = prev_node

    def _move_to_head(self, node: _Node[K, V]) -> None:
        """Mark a node as most recently used by moving it to right after head."""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> Optional[_Node[K, V]]:
        """Remove and return the node right before the dummy tail (least recently used)."""
        res = self._tail.prev
        if res and res != self._head:
            self._remove_node(res)
            return res
        return None

    # -------------------------------------------------------------------------
    # Dunder Methods
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key: K) -> bool:
        return key in self._map

    def __repr__(self) -> str:
        return f"LRUCache(capacity={self._capacity}, size={len(self._map)}, hit_rate={self.hit_rate:.2%})"
