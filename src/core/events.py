"""Asynchronous Event Bus implementing the Observer (Publish-Subscribe) Pattern.

Decouples agent execution, tool telemetry, streaming UI notifications, and audit logging.
Provides async broadcasting with isolated subscriber error boundaries and history replay.
"""

import asyncio
import inspect
import time
import uuid
from collections import deque
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, Union

EventHandler = Union[
    Callable[["Event"], None],
    Callable[["Event"], Coroutine[Any, Any, None]],
]


class EventType(str, Enum):
    """Standardized event types emitted throughout the OmniIntel lifecycle."""

    # System & Task Lifecycle
    SYSTEM_STARTED = "SYSTEM_STARTED"
    TASK_SCHEDULED = "TASK_SCHEDULED"
    TASK_STARTED = "TASK_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"

    # Agent Thought & Tool Actions
    AGENT_THOUGHT = "AGENT_THOUGHT"
    TOOL_CALLED = "TOOL_CALLED"
    TOOL_RETURNED = "TOOL_RETURNED"
    CRITIC_VALIDATION = "CRITIC_VALIDATION"
    CRITIC_REJECTED = "CRITIC_REJECTED"

    # Deep Learning & Retrieval
    MODEL_INFERENCE = "MODEL_INFERENCE"
    CACHE_HIT = "CACHE_HIT"
    CACHE_MISS = "CACHE_MISS"

    # Final Output
    REPORT_GENERATED = "REPORT_GENERATED"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class Event:
    """Immutable data container for system events.

    Attributes:
        event_id: Globally unique identifier for this event.
        event_type: Categorized EventType enum or string.
        source: Name of the emitting component (e.g., 'PlannerAgent', 'Reranker').
        payload: Arbitrary contextual data associated with the event.
        timestamp: Epoch timestamp when the event occurred.
    """

    __slots__ = ("event_id", "event_type", "source", "payload", "timestamp")

    def __init__(
        self,
        event_type: EventType | str,
        source: str,
        payload: Optional[dict[str, Any]] = None,
        event_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self.event_id: str = event_id or str(uuid.uuid4())
        self.event_type: str = (
            event_type.value if isinstance(event_type, EventType) else str(event_type)
        )
        self.source: str = source
        self.payload: dict[str, Any] = payload or {}
        self.timestamp: float = timestamp or time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to a JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        return (
            f"Event(type='{self.event_type}', source='{self.source}', "
            f"id='{self.event_id[:8]}...')"
        )


class EventBus:
    """Central asynchronous Event Bus managing publisher-subscriber dispatch.

    Features:
        - Supports both sync (`def`) and async (`async def`) subscriber handlers.
        - Wildcard subscriptions (`"*"`) for global listeners (e.g. loggers, WebSockets).
        - Isolated error boundaries: an exception in one handler does not halt others.
        - Fixed-capacity circular replay buffer for catching up reconnecting clients.
    """

    def __init__(self, history_capacity: int = 500) -> None:
        # Maps event_type -> set of subscriber handler callables
        self._subscribers: dict[str, set[EventHandler]] = {}
        # Fixed-size buffer storing recent events for replay
        self._history: deque[Event] = deque(maxlen=history_capacity)

    @property
    def total_subscribers(self) -> int:
        """Count total unique subscriber registrations across all event types."""
        return sum(len(handlers) for handlers in self._subscribers.values())

    @property
    def history(self) -> list[Event]:
        """Return a chronological snapshot of recently recorded events."""
        return list(self._history)

    def subscribe(self, event_type: EventType | str, handler: EventHandler) -> None:
        """Register a subscriber callback for an event type (or '*' for all events)."""
        key = event_type.value if isinstance(event_type, EventType) else str(event_type)
        if key not in self._subscribers:
            self._subscribers[key] = set()
        self._subscribers[key].add(handler)

    def unsubscribe(self, event_type: EventType | str, handler: EventHandler) -> bool:
        """Unregister a subscriber callback.

        Returns:
            bool: True if handler was found and removed, False otherwise.
        """
        key = event_type.value if isinstance(event_type, EventType) else str(event_type)
        if key in self._subscribers and handler in self._subscribers[key]:
            self._subscribers[key].remove(handler)
            if not self._subscribers[key]:
                del self._subscribers[key]
            return True
        return False

    async def publish(self, event: Event) -> None:
        """Asynchronously dispatch an event to all registered matching subscribers.

        Executes handlers concurrently while preventing exceptions from bubbling up.
        """
        self._history.append(event)

        # Collect target handlers: exact matches + wildcard '*' global listeners
        target_handlers: set[EventHandler] = set()
        if event.event_type in self._subscribers:
            target_handlers.update(self._subscribers[event.event_type])
        if "*" in self._subscribers:
            target_handlers.update(self._subscribers["*"])

        if not target_handlers:
            return

        tasks: list[Coroutine[Any, Any, None]] = []
        for handler in target_handlers:
            tasks.append(self._invoke_handler(handler, event))

        # Run all handlers concurrently with isolated error boundaries
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _invoke_handler(self, handler: EventHandler, event: Event) -> None:
        """Safely invoke an individual handler whether synchronous or coroutine."""
        try:
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                # Run synchronous blocking handler in thread pool if loop is running
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, handler, event)
        except Exception as exc:
            # Prevent subscriber errors from breaking the event bus
            print(f"[EventBus Warning] Handler {handler} failed for {event.event_type}: {exc}")

    def clear(self) -> None:
        """Clear all subscribers and historical logs."""
        self._subscribers.clear()
        self._history.clear()


# Default singleton instance for application-wide telemetry
global_event_bus = EventBus()
