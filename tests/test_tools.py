"""Unit test suite for standard agent tools, sandboxed Python REPL, and data sources."""

import asyncio
import math
import unittest
from unittest.mock import MagicMock

from src.agents.base import Tool, ToolRegistry, ToolResult
from src.agents.tools import (
    DEFAULT_MARKET_FEED,
    SandboxedREPL,
    SandboxSecurityError,
    create_agent_toolset,
    create_graph_traversal_tool,
    create_market_data_tool,
    create_python_repl_tool,
    create_rag_retrieval_tool,
    financial_cagr,
    financial_dcf_npv,
    financial_debt_to_equity,
    financial_gross_margin,
    financial_operating_margin,
    financial_pe_ratio,
)
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.pipeline import (
    HybridRAGPipeline,
    PipelineContextItem,
    RetrievalPipelineOutput,
)
from src.knowledge_graph.schema import Entity, EntityType, FactTriple, Relation


class TestAgentTools(unittest.TestCase):
    """Test suite verifying tool interfaces, security barriers, and financial formulas."""

    # -------------------------------------------------------------------------
    # 1. Sandboxed Python REPL Tests
    # -------------------------------------------------------------------------

    def test_repl_basic_math_and_numpy(self) -> None:
        """REPL should safely evaluate basic arithmetic and numpy operations."""
        repl = SandboxedREPL()

        # Expression evaluation
        res1 = repl.execute("10 * 5 + 2")
        self.assertTrue(res1["success"])
        self.assertEqual(res1["result"], 52)

        # Math module
        res2 = repl.execute("math.sqrt(144) + math.log(1)")
        self.assertTrue(res2["success"])
        self.assertAlmostEqual(res2["result"], 12.0)

        # Numpy operations
        res3 = repl.execute("np.mean([10.0, 20.0, 30.0])")
        self.assertTrue(res3["success"])
        self.assertAlmostEqual(res3["result"], 20.0)

    def test_repl_financial_formulas(self) -> None:
        """REPL should have financial calculation helpers pre-injected."""
        repl = SandboxedREPL()

        # CAGR test: 100 to 200 in 3 years -> (2)^(1/3) - 1 ~ 0.25992
        res_cagr = repl.execute("cagr(100, 200, 3)")
        self.assertTrue(res_cagr["success"])
        expected_cagr = (200.0 / 100.0) ** (1.0 / 3.0) - 1.0
        self.assertAlmostEqual(res_cagr["result"], expected_cagr, places=4)

        # P/E Ratio: Price $120, EPS $4 -> 30.0
        res_pe = repl.execute("pe_ratio(120, 4)")
        self.assertTrue(res_pe["success"])
        self.assertEqual(res_pe["result"], 30.0)

        # Debt to Equity: Debt $50M, Equity $100M -> 0.50
        res_de = repl.execute("debt_to_equity(50, 100)")
        self.assertTrue(res_de["success"])
        self.assertEqual(res_de["result"], 0.5)

        # Margins
        res_gm = repl.execute("gross_margin(75, 100)")
        self.assertEqual(res_gm["result"], 0.75)
        res_om = repl.execute("operating_margin(30, 100)")
        self.assertEqual(res_om["result"], 0.30)

        # DCF NPV: cash flows [100, 100], discount 10%
        # 100/1.1 + 100/1.21 = 90.909 + 82.644 = 173.553
        res_dcf = repl.execute("dcf_npv([100.0, 100.0], 0.10)")
        self.assertTrue(res_dcf["success"])
        self.assertAlmostEqual(res_dcf["result"], 173.5537, places=3)

    def test_repl_multi_line_script(self) -> None:
        """REPL should execute multi-line statements and return assigned result variable."""
        repl = SandboxedREPL()
        code = """
rev_2020 = 10.0
rev_2023 = 26.0
growth = cagr(rev_2020, rev_2023, 3)
result = round(growth * 100, 2)
"""
        res = repl.execute(code)
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], 37.51)

    def test_repl_sandbox_security_rejections(self) -> None:
        """Dangerous system calls and banned tokens must be rejected with SandboxSecurityError."""
        repl = SandboxedREPL()

        malicious_snippets = [
            "__import__('os').system('ls')",
            "open('/etc/passwd', 'r')",
            "import os",
            "eval('2+2')",
            "exec('x = 1')",
            "globals()",
            "locals()",
            "subprocess.run(['echo', 'hello'])",
        ]

        for code in malicious_snippets:
            with self.subTest(code=code):
                with self.assertRaises(SandboxSecurityError):
                    repl.execute(code)

    def test_python_repl_tool_integration(self) -> None:
        """Python REPL Tool registered in ToolRegistry must execute and return ToolResult."""
        tool = create_python_repl_tool()
        registry = ToolRegistry()
        registry.register(tool)

        res: ToolResult = asyncio.run(registry.execute("python_repl", code="cagr(50, 150, 4)"))
        self.assertTrue(res.success)
        self.assertIn("0.316", res.output)

    # -------------------------------------------------------------------------
    # 2. Market Fundamentals Tool Tests
    # -------------------------------------------------------------------------

    def test_market_data_lookup_success(self) -> None:
        """Market data tool should retrieve financial metrics for known entities."""
        tool = create_market_data_tool()

        # Query by ticker
        out_nvda = tool.func(entity="NVDA")
        self.assertIn("NVIDIA Corporation", out_nvda)
        self.assertIn("Gross Margin: 75.10%", out_nvda)
        self.assertIn("Key Suppliers: TSMC", out_nvda)

        # Query by lowercase name
        out_apple = tool.func(entity="apple")
        self.assertIn("Apple Inc.", out_apple)
        self.assertIn("Market Cap", out_apple)

    def test_market_data_unknown_entity(self) -> None:
        """Market data tool should informatively report unindexed companies."""
        tool = create_market_data_tool()
        out = tool.func(entity="RandomStartup123")
        self.assertIn("not found in active market feed", out)
        self.assertIn("Available entities", out)

    # -------------------------------------------------------------------------
    # 3. Knowledge Graph Traversal Tool Tests
    # -------------------------------------------------------------------------

    def test_graph_traversal_tool_operations(self) -> None:
        """Graph traversal tool should support inspect, pathfinding, and neighborhood queries."""
        graph = KnowledgeGraph()
        e_nvda = Entity(entity_id="nvidia", name="NVIDIA", entity_type=EntityType.ORGANIZATION)
        e_tsmc = Entity(entity_id="tsmc", name="TSMC", entity_type=EntityType.ORGANIZATION)
        e_asml = Entity(entity_id="asml", name="ASML", entity_type=EntityType.TECHNOLOGY)

        graph.add_entity(e_nvda)
        graph.add_entity(e_tsmc)
        graph.add_entity(e_asml)

        r_supp = Relation(relation_id="rel_supp", relation_type="supplier_to", weight=1.0)
        graph.add_triple(FactTriple(triple_id="t1", subject=e_tsmc, relation=r_supp, object=e_nvda, confidence=0.98))
        graph.add_triple(FactTriple(triple_id="t2", subject=e_asml, relation=r_supp, object=e_tsmc, confidence=0.95))

        tool = create_graph_traversal_tool(graph)

        # 1. inspect_entity
        out_inspect = tool.func(operation="inspect_entity", source_id="tsmc")
        self.assertIn("Entity: TSMC", out_inspect)
        self.assertIn("supplier_to", out_inspect)

        # 2. find_path: ASML -> TSMC -> NVIDIA
        out_path = tool.func(operation="find_path", source_id="asml", target_id="nvidia")
        self.assertIn("Most Reliable Reasoning Path", out_path)
        self.assertIn("ASML", out_path)
        self.assertIn("TSMC", out_path)
        self.assertIn("NVIDIA", out_path)

        # 3. neighbors
        out_neighbors = tool.func(operation="neighbors", source_id="tsmc", max_hops=2)
        self.assertIn("K-Hop Neighborhood", out_neighbors)

        # 4. Unknown operation error
        out_err = tool.func(operation="invalid_op", source_id="tsmc")
        self.assertIn("Unknown operation", out_err)

    # -------------------------------------------------------------------------
    # 4. Hybrid RAG Retrieval Tool Tests
    # -------------------------------------------------------------------------

    def test_rag_retrieval_tool(self) -> None:
        """RAG tool should query HybridRAGPipeline and format grounded citations."""
        mock_pipeline = MagicMock(spec=HybridRAGPipeline)
        mock_output = RetrievalPipelineOutput(
            query="NVIDIA Blackwell architecture",
            items=[
                PipelineContextItem(
                    doc_id="doc_1",
                    text="NVIDIA Blackwell B200 delivers 20 petaflops of FP4 computing.",
                    final_score=0.942,
                    stage1_rank=1,
                    stage1_score=0.88,
                    stage2_rank=1,
                    stage2_score=0.942,
                    matched_entities=["nvidia", "blackwell"],
                )
            ],
            grounded_context_str="Sample context",
            total_candidates_retrieved=10,
            total_reranked=5,
            latency_stage1_ms=1.2,
            latency_stage2_ms=3.4,
            total_latency_ms=4.6,
        )
        mock_pipeline.retrieve.return_value = mock_output

        tool = create_rag_retrieval_tool(mock_pipeline)
        result_str = tool.func(query="NVIDIA Blackwell architecture", top_k=3)

        self.assertIn("Hybrid RAG Retrieval", result_str)
        self.assertIn("Doc: doc_1", result_str)
        self.assertIn("Reranker: 0.9420", result_str)
        self.assertIn("Blackwell B200", result_str)

    # -------------------------------------------------------------------------
    # 5. Master Default Toolset Assembly
    # -------------------------------------------------------------------------

    def test_create_agent_toolset_assembly(self) -> None:
        """Master toolset factory should instantiate and package all enabled tools."""
        mock_pipeline = MagicMock(spec=HybridRAGPipeline)
        graph = KnowledgeGraph()

        tools = create_agent_toolset(pipeline=mock_pipeline, graph=graph)
        self.assertEqual(len(tools), 4)

        tool_names = {t.name for t in tools}
        self.assertEqual(
            tool_names,
            {"python_repl", "get_market_data", "retrieve_knowledge", "traverse_knowledge_graph"},
        )


if __name__ == "__main__":
    unittest.main()
