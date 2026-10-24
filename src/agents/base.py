"""Autonomous Base Agent Architecture with Tool Registry, Memory, and Event Hooks.

Provides the foundational abstract base class and state machine for all specialized
OmniIntel agents (Planner, Researcher, Quantitative Analyst, Critic, Synthesizer).
Implements ReAct (Reasoning + Acting) execution loops, scratchpad working memory,
declarative tool dispatching, and asynchronous EventBus telemetry.
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import inspect
import json
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

from src.core.events import Event, EventBus, EventType


class AgentState(str, Enum):
    """Lifecycle states of an autonomous agent."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    WAITING_FOR_TOOL = "WAITING_FOR_TOOL"
    REFLECTING = "REFLECTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class AgentMessage:
    """Atomic message unit stored within an Agent's working memory."""

    role: str  # "system", "user", "assistant", "tool", "critic"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AgentMemory:
    """Manages short-term conversational history and key-value working scratchpads."""

    def __init__(self, max_messages: int = 50) -> None:
        self.max_messages = max_messages
        self._messages: List[AgentMessage] = []
        self._scratchpad: Dict[str, Any] = {}

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Append a message to working memory, enforcing sliding window capacity."""
        msg = AgentMessage(role=role, content=content, metadata=metadata or {})
        self._messages.append(msg)
        if len(self._messages) > self.max_messages:
            # Preserve system prompt if first message, trim oldest conversational turns
            if self._messages and self._messages[0].role == "system":
                self._messages = [self._messages[0]] + self._messages[-(self.max_messages - 1) :]
            else:
                self._messages = self._messages[-self.max_messages :]
        return msg

    def get_messages(self) -> List[AgentMessage]:
        """Return all active messages in chronological order."""
        return list(self._messages)

    def set_variable(self, key: str, value: Any) -> None:
        """Store intermediate variable in working scratchpad."""
        self._scratchpad[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Retrieve variable from working scratchpad."""
        return self._scratchpad.get(key, default)

    def clear(self) -> None:
        """Reset messages and scratchpad."""
        self._messages.clear()
        self._scratchpad.clear()

    def format_context_prompt(self) -> str:
        """Format working memory into a clean conversational dialogue string."""
        lines = []
        for msg in self._messages:
            lines.append(f"[{msg.role.upper()}]: {msg.content}")
        return "\n".join(lines)


@dataclass
class ToolResult:
    """Outcome of a tool execution."""

    tool_name: str
    output: Any
    success: bool
    execution_time_ms: float
    error: Optional[str] = None

    def to_str(self) -> str:
        if not self.success:
            return f"Error executing '{self.tool_name}': {self.error}"
        if isinstance(self.output, (dict, list)):
            return json.dumps(self.output, indent=2)
        return str(self.output)


class Tool:
    """Wraps an executable function with typed schema documentation for agent discovery."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters_schema: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name.strip()
        self.description = description.strip()
        self.func = func
        self.parameters_schema = parameters_schema or parameters or {}
        self.is_async = inspect.iscoroutinefunction(func)

    async def execute(self, **kwargs) -> ToolResult:
        """Execute tool safely with performance telemetry and exception isolation."""
        t0 = time.perf_counter()
        try:
            if self.is_async:
                res = await self.func(**kwargs)
            else:
                res = self.func(**kwargs)
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ToolResult(tool_name=self.name, output=res, success=True, execution_time_ms=round(elapsed, 3))
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return ToolResult(
                tool_name=self.name,
                output=None,
                success=False,
                execution_time_ms=round(elapsed, 3),
                error=str(exc),
            )

    def to_schema(self) -> Dict[str, Any]:
        """Return standardized JSON schema descriptor for LLM tool selection."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }


class ToolRegistry:
    """Catalog managing registered tools available to autonomous agents."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a new tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Look up tool by name."""
        return self._tools.get(name)

    @property
    def tool_names(self) -> List[str]:
        """Return list of all registered tool names."""
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute tool by name."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                output=None,
                success=False,
                execution_time_ms=0.0,
                error=f"Tool '{tool_name}' is not registered in ToolRegistry.",
            )
        return await tool.execute(**kwargs)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return schemas for all registered tools."""
        return [tool.to_schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)


@dataclass
class AgentResponse:
    """Final output package generated by an autonomous agent."""

    agent_id: str
    task: str
    output: str
    success: bool
    state: AgentState
    execution_steps: List[Dict[str, Any]] = field(default_factory=list)
    tools_invoked: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None


class BaseAgent(ABC):
    """Abstract Base Class for all autonomous agents in OmniIntel.

    Implements:
      - Finite State Machine (IDLE -> PLANNING -> THINKING -> EXECUTING -> REFLECTING -> COMPLETED)
      - Dynamic Tool Execution & Verification
      - Working Memory Scratchpad
      - EventBus Telemetry (Pub/Sub Observer Pattern)
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        goal: str,
        tools: Optional[List[Tool]] = None,
        event_bus: Optional[EventBus] = None,
        memory_capacity: int = 50,
    ) -> None:
        self.agent_id = agent_id.strip().lower()
        self.name = name.strip()
        self.role = role.strip()
        self.goal = goal.strip()

        self.state: AgentState = AgentState.IDLE
        self.event_bus = event_bus
        self.memory = AgentMemory(max_messages=memory_capacity)

        # Tool registry
        self.tools = ToolRegistry()
        if tools:
            for t in tools:
                self.tools.register(t)

        # Telemetry
        self.total_tasks_executed: int = 0
        self.total_execution_time_ms: float = 0.0

        # Initialize system role message
        self.memory.add_message(
            role="system",
            content=f"You are {self.name}. Role: {self.role}. Goal: {self.goal}",
        )

    async def _emit_event(self, event_type: EventType, payload: Dict[str, Any]) -> None:
        """Broadcast agent state event to EventBus."""
        if self.event_bus is not None:
            event = Event(
                event_type=event_type,
                source=self.name,
                payload={"agent_id": self.agent_id, "state": self.state.value, **payload},
            )
            await self.event_bus.publish(event)

    def transition_state(self, new_state: AgentState) -> None:
        """Transition agent state machine."""
        self.state = new_state

    @abstractmethod
    async def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Cognitive reasoning phase: analyzes task and formulates action plan."""
        pass

    async def act(self, tool_name: str, **kwargs) -> ToolResult:
        """Action execution phase: invokes registered tool via ToolRegistry."""
        self.transition_state(AgentState.WAITING_FOR_TOOL)
        await self._emit_event(EventType.TOOL_CALLED, {"tool_name": tool_name, "arguments": kwargs})

        res = await self.tools.execute(tool_name, **kwargs)

        await self._emit_event(
            EventType.TOOL_RETURNED,
            {
                "tool_name": tool_name,
                "success": res.success,
                "execution_time_ms": res.execution_time_ms,
            },
        )
        return res

    @abstractmethod
    async def reflect(self, task: str, thoughts: str, execution_log: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Reflection / Critic phase: verifies whether the formulated response satisfies the goal."""
        pass

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Full ReAct execution lifecycle: Think -> Act -> Reflect."""
        start_time = time.perf_counter()
        self.transition_state(AgentState.PLANNING)
        await self._emit_event(EventType.TASK_STARTED, {"task": task})

        self.memory.add_message(role="user", content=task)
        execution_steps: List[Dict[str, Any]] = []
        tools_invoked: List[str] = []

        try:
            # 1. Think phase
            self.transition_state(AgentState.THINKING)
            thoughts = await self.think(task, context=context)
            execution_steps.append({"step": "think", "content": thoughts})
            await self._emit_event(EventType.AGENT_THOUGHT, {"thoughts": thoughts})

            # 2. Reflection phase
            self.transition_state(AgentState.REFLECTING)
            is_valid, critique = await self.reflect(task, thoughts, execution_steps)
            execution_steps.append({"step": "reflect", "valid": is_valid, "critique": critique})

            if not is_valid:
                await self._emit_event(EventType.CRITIC_REJECTED, {"critique": critique})
            else:
                await self._emit_event(EventType.CRITIC_VALIDATION, {"critique": critique})

            self.transition_state(AgentState.COMPLETED)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            self.total_tasks_executed += 1
            self.total_execution_time_ms += elapsed_ms

            final_response = AgentResponse(
                agent_id=self.agent_id,
                task=task,
                output=thoughts,
                success=True,
                state=self.state,
                execution_steps=execution_steps,
                tools_invoked=tools_invoked,
                latency_ms=round(elapsed_ms, 3),
            )
            self.memory.add_message(role="assistant", content=thoughts)
            await self._emit_event(EventType.TASK_COMPLETED, {"latency_ms": round(elapsed_ms, 3)})
            return final_response

        except Exception as exc:
            self.transition_state(AgentState.FAILED)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            await self._emit_event(EventType.TASK_FAILED, {"error": str(exc)})
            return AgentResponse(
                agent_id=self.agent_id,
                task=task,
                output="",
                success=False,
                state=self.state,
                execution_steps=execution_steps,
                tools_invoked=tools_invoked,
                latency_ms=round(elapsed_ms, 3),
                error=str(exc),
            )

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Synchronous wrapper around execute() for CLI scripts and blocking environments."""
        return asyncio.run(self.execute(task, context=context))
