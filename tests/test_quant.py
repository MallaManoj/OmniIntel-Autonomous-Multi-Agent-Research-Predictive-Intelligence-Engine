"""Unit test suite for QuantAgent, QuantReport, and deterministic financial modeling."""

import asyncio
import unittest

from src.agents.base import AgentState
from src.agents.quant import (
    ComputedMetric,
    EntityFinancialProfile,
    QuantAgent,
    QuantReport,
)
from src.core.events import EventBus


class TestQuantAgent(unittest.TestCase):
    """Test suite verifying QuantAgent financial analysis, sandboxed calculation verification, and ReAct loop."""

    def setUp(self) -> None:
        self.event_bus = EventBus()
        self.agent = QuantAgent(event_bus=self.event_bus)

    def test_quant_auto_registers_tools(self) -> None:
        """QuantAgent must automatically register python_repl and get_market_data tools."""
        self.assertIn("python_repl", self.agent.tools.tool_names)
        self.assertIn("get_market_data", self.agent.tools.tool_names)

    def test_single_entity_financial_analysis(self) -> None:
        """QuantAgent should extract fundamentals and compute verified ratios for NVDA."""
        report = asyncio.run(
            self.agent.analyze_financials(
                task="Compute gross margins, P/E ratio, and debt leverage for NVIDIA",
                entities=["NVDA"],
            )
        )

        self.assertIsInstance(report, QuantReport)
        self.assertIn("NVDA", report.profiles)

        nvda_prof = report.profiles["NVDA"]
        self.assertEqual(nvda_prof.ticker, "NVDA")
        self.assertIn("Gross Margin", nvda_prof.metrics)
        self.assertIn("Operating Margin", nvda_prof.metrics)
        self.assertIn("P/E Multiple", nvda_prof.metrics)
        self.assertIn("Debt-to-Equity", nvda_prof.metrics)

        # Check values match DEFAULT_MARKET_FEED
        self.assertAlmostEqual(nvda_prof.metrics["Gross Margin"].value, 0.751, places=3)
        self.assertAlmostEqual(nvda_prof.metrics["Operating Margin"].value, 0.618, places=3)
        self.assertAlmostEqual(nvda_prof.metrics["P/E Multiple"].value, 46.2, places=1)
        self.assertAlmostEqual(nvda_prof.metrics["Debt-to-Equity"].value, 0.18, places=2)

        # Check that sandboxed REPL calculation was logged
        self.assertGreaterEqual(len(report.calculations_performed), 1)

    def test_comparative_multi_entity_analysis(self) -> None:
        """Comparative analysis between NVDA and AAPL must generate side-by-side comparison tables."""
        report = asyncio.run(
            self.agent.analyze_financials(
                task="Compare NVIDIA vs Apple valuation multiples and operating margins",
                entities=["NVDA", "AAPL"],
            )
        )

        self.assertIn("NVDA", report.profiles)
        self.assertIn("AAPL", report.profiles)

        # Comparative narrative should identify profitability leader
        self.assertIn("Profitability Leader", report.comparative_summary)
        self.assertIn("NVIDIA", report.comparative_summary)

        # Check Markdown table rendering
        md = report.to_markdown()
        self.assertIn("| Metric | Category | NVDA | AAPL |", md)
        self.assertIn("Gross Margin", md)
        self.assertIn("75.10%", md)
        self.assertIn("46.20%", md)

    def test_quant_react_lifecycle_execution(self) -> None:
        """QuantAgent run() must cycle through ReAct states and store latest_quant_report."""
        response = self.agent.run("Evaluate Microsoft and AMD valuation multiples")

        self.assertTrue(response.success)
        self.assertEqual(response.state, AgentState.COMPLETED)
        self.assertIn("# Quantitative Financial Analysis", response.output)
        self.assertIn("Financial Metrics Comparison Table", response.output)

        # Verify scratchpad memory variable
        report = self.agent.memory.get_variable("latest_quant_report")
        self.assertIsInstance(report, QuantReport)
        self.assertTrue(any(t in report.profiles for t in ["MSFT", "AMD"]))

    def test_quant_reflection_validation(self) -> None:
        """Reflection must accept sound reports and reject NaN or infinite anomalies."""
        # 1. Valid report
        report = asyncio.run(self.agent.analyze_financials("Analyze Apple", entities=["AAPL"]))
        self.agent.memory.set_variable("latest_quant_report", report)
        is_valid, msg = asyncio.run(self.agent.reflect("task", "thoughts", []))
        self.assertTrue(is_valid)
        self.assertIn("Quantitative analysis verified", msg)

        # 2. Missing report
        self.agent.memory.clear()
        is_valid, msg = asyncio.run(self.agent.reflect("task", "thoughts", []))
        self.assertFalse(is_valid)
        self.assertIn("failed to compile", msg)

        # 3. Injected NaN metric
        nan_profile = EntityFinancialProfile(ticker="BAD", name="Bad Corp")
        nan_profile.metrics["Corrupt Ratio"] = ComputedMetric(
            name="Corrupt Ratio",
            value=float("nan"),
            formatted_value="NaN",
            category="Profitability",
            formula="0/0",
        )
        corrupted_report = QuantReport(
            task="Test",
            entities_analyzed=["BAD"],
            profiles={"BAD": nan_profile},
            comparative_summary="Corrupt",
        )
        self.agent.memory.set_variable("latest_quant_report", corrupted_report)
        is_valid, msg = asyncio.run(self.agent.reflect("task", "thoughts", []))
        self.assertFalse(is_valid)
        self.assertIn("Detected NaN value", msg)

    def test_report_serialization(self) -> None:
        """QuantReport to_dict must output complete structured serializable dictionaries."""
        metric = ComputedMetric(
            name="Operating Margin",
            value=0.45,
            formatted_value="45.00%",
            category="Profitability",
            formula="EBIT / Revenue",
        )
        prof = EntityFinancialProfile(
            ticker="TSM",
            name="TSMC",
            metrics={"Operating Margin": metric},
        )
        report = QuantReport(
            task="TSMC metrics",
            entities_analyzed=["TSM"],
            profiles={"TSM": prof},
            comparative_summary="Summary",
            calculations_performed=["operating_margin(45, 100)"],
        )

        d = report.to_dict()
        self.assertEqual(d["task"], "TSMC metrics")
        self.assertIn("TSM", d["profiles"])
        self.assertEqual(d["profiles"]["TSM"]["metrics"]["Operating Margin"]["value"], 0.45)


if __name__ == "__main__":
    unittest.main()
