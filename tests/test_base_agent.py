"""Unit tests for BaseAgent, AgentMemory, ToolRegistry, and state transitions.

Validates memory sliding window, sync/async tool execution,
ReAct lifecycle (think -> act -> reflect), EventBus hooks, and failure boundaries.
Compatible with standard library `unittest` and `pytest`.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import unittest

from src.agents.base import (
    AgentMemory,
    AgentMessage,
    AgentResponse,
    AgentState,
    BaseAgent,
    Tool,
    ToolRegistry,
    ToolResult,
)
from src.core.events import Event, EventBus, EventType


class MockAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    async def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        if "trigger_error" in task:
            raise RuntimeError("Simulated cognitive failure in thinking phase.")
        # Invoke a tool if task mentions calculate
        if "calculate" in task:
            res = await self.act("calculator", a=5, b=10)
            return f"The calculation result is {res.output}"
        return f"Processed task: '{task}' successfully."

    async def reflect(
        self,
        task: str,
        thoughts: str,
        execution_log: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        if "invalid" in task:
            return False, "Failed quality criteria validation."
        return True, "Quality verification passed."


class TestBaseAgentAndComponents(unittest.TestCase):
    """Test suite covering AgentMemory, ToolRegistry, and BaseAgent lifecycle."""

    def test_agent_memory_sliding_window(self) -> None:
        """AgentMemory must maintain system prompt while evicting oldest conversation turns."""
        memory = AgentMemory(max_messages=4)
        memory.add_message("system", "System instruction.")
        memory.add_message("user", "Turn 1")
        memory.add_message("assistant", "Turn 2")
        memory.add_message("user", "Turn 3")
        # 4 messages currently stored: [system, turn1, turn2, turn3]
        self.assertEqual(len(memory.get_messages()), 4)

        # Adding 5th message triggers sliding window: system prompt stays, turn 1 evicted
        memory.add_message("assistant", "Turn 4")
        messages = memory.get_messages()
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].content, "Turn 2")
        self.assertEqual(messages[3].content, "Turn 4")

        # Context prompt format
        prompt = memory.format_context_prompt()
        self.assertIn("[SYSTEM]: System instruction.", prompt)
        self.assertIn("[ASSISTANT]: Turn 4", prompt)

    def test_agent_memory_scratchpad(self) -> None:
        """Scratchpad must store and retrieve arbitrary working variables."""
        memory = AgentMemory()
        memory.set_variable("ticker", "NVDA")
        memory.set_variable("target_price", 145.50)

        self.assertEqual(memory.get_variable("ticker"), "NVDA")
        self.assertEqual(memory.get_variable("target_price"), 145.50)
        self.assertIsNone(memory.get_variable("missing_key"))

        memory.clear()
        self.assertIsNone(memory.get_variable("ticker"))

    def test_tool_sync_and_async_execution(self) -> None:
        """Tool must handle both synchronous functions and async coroutines safely."""
        def sync_adder(a: int, b: int) -> int:
            return a + b

        async def async_multiplier(x: int, y: int) -> int:
            await asyncio.sleep(0.001)
            return x * y

        tool_sync = Tool("adder", "Adds two numbers", sync_adder)
        tool_async = Tool("multiplier", "Multiplies two numbers", async_multiplier)

        # Run async in event loop
        loop = asyncio.new_event_loop()
        res_sync = loop.run_until_complete(tool_sync.execute(a=3, b=7))
        res_async = loop.run_until_complete(tool_async.execute(x=4, y=5))
        loop.close()

        self.assertTrue(res_sync.success)
        self.assertEqual(res_sync.output, 10)
        self.assertTrue(res_async.success)
        self.assertEqual(res_async.output, 20)

    def test_tool_exception_isolation(self) -> None:
        """Tool failure must be caught and returned as ToolResult with error message."""
        def faulty_tool(val: int) -> float:
            return 10 / val  # ZeroDivisionError if val=0

        tool = Tool("faulty", "Fails on zero", faulty_tool)
        res = asyncio.run(tool.execute(val=0))

        self.assertFalse(res.success)
        self.assertIsNone(res.output)
        self.assertIn("division by zero", res.error)
        self.assertIn("Error executing 'faulty'", res.to_str())

    def test_tool_registry(self) -> None:
        """ToolRegistry must register, list schemas, and execute tools."""
        registry = ToolRegistry()
        tool = Tool(
            name="echo",
            description="Echoes input",
            func=lambda msg: msg,
            parameters_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        )
        registry.register(tool)

        self.assertEqual(len(registry), 1)
        self.assertTrue(registry.has_tool("echo"))
        self.assertFalse(registry.has_tool("unknown"))

        schemas = registry.list_tools()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["name"], "echo")

        # Execute existing
        res = asyncio.run(registry.execute("echo", msg="Hello World"))
        self.assertTrue(res.success)
        self.assertEqual(res.output, "Hello World")

        # Execute unregistered tool
        res_unknown = asyncio.run(registry.execute("unknown"))
        self.assertFalse(res_unknown.success)
        self.assertIn("not registered", res_unknown.error)

    def test_base_agent_react_lifecycle(self) -> None:
        """BaseAgent must cycle through PLANNING, THINKING, REFLECTING, and COMPLETED states."""
        calc_tool = Tool("calculator", "Adds numbers", lambda a, b: a + b)
        agent = MockAgent(
            agent_id="analyst",
            name="Financial Analyst",
            role="Performs corporate financial analysis",
            goal="Provide accurate valuation metrics",
            tools=[calc_tool],
        )

        response = agent.run("Please calculate current operating metrics")
        self.assertIsInstance(response, AgentResponse)
        self.assertTrue(response.success)
        self.assertEqual(response.state, AgentState.COMPLETED)
        self.assertEqual(response.output, "The calculation result is 15")
        self.assertGreater(response.latency_ms, 0.0)

        # Verify execution log steps
        steps = [s["step"] for s in response.execution_steps]
        self.assertEqual(steps, ["think", "reflect"])

    def test_base_agent_event_broadcasting(self) -> None:
        """Agent must emit lifecycle events to the EventBus."""
        bus = EventBus()
        captured_events: List[Event] = []

        async def listener(event: Event) -> None:
            captured_events.append(event)

        bus.subscribe(EventType.TASK_STARTED, listener)
        bus.subscribe(EventType.AGENT_THOUGHT, listener)
        bus.subscribe(EventType.TASK_COMPLETED, listener)

        agent = MockAgent(
            agent_id="test_agent",
            name="Test Agent",
            role="Testing events",
            goal="Test event publishing",
            event_bus=bus,
        )

        async def run_test() -> None:
            await agent.execute("Run event test")

        asyncio.run(run_test())

        event_types = [e.event_type for e in captured_events]
        self.assertIn(EventType.TASK_STARTED, event_types)
        self.assertIn(EventType.AGENT_THOUGHT, event_types)
        self.assertIn(EventType.TASK_COMPLETED, event_types)

    def test_base_agent_failure_isolation(self) -> None:
        """Cognitive exceptions must transition agent to FAILED state without crashing process."""
        agent = MockAgent(
            agent_id="faulty_agent",
            name="Faulty Agent",
            role="Testing failure",
            goal="Fail safely",
        )

        response = agent.run("Please trigger_error now")
        self.assertFalse(response.success)
        self.assertEqual(response.state, AgentState.FAILED)
        self.assertIn("Simulated cognitive failure", response.error)


if __name__ == "__main__":
    unittest.main()
