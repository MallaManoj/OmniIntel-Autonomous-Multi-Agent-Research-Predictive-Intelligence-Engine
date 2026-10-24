"""Standard Tool Interfaces for Autonomous OmniIntel Agents.

Provides sandboxed, deterministic tools:
  1. HybridRAGRetrievalTool: Multi-channel Stage 1 + Stage 2 neural reranking retrieval.
  2. KnowledgeGraphTraversalTool: Multi-hop reasoning, shortest path, and entity exploration.
  3. PythonREPLTool: Sandboxed mathematical and financial ratio computation engine.
  4. MarketDataTool: Financial fundamentals, valuation multiples, and operational metrics.
"""

from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from src.agents.base import Tool, ToolResult
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.pipeline import (
    HybridRAGPipeline,
    RetrievalPipelineConfig,
    RetrievalPipelineOutput,
)
from src.knowledge_graph.schema import slugify_id
from src.knowledge_graph.traversal import GraphTraversalEngine, PathResult, SubgraphResult


# =============================================================================
# 1. Sandboxed Python REPL for Quantitative Analysis
# =============================================================================

class SandboxSecurityError(Exception):
    """Raised when restricted builtins or unsafe modules are accessed in the REPL."""
    pass


def financial_cagr(beginning_val: float, ending_val: float, years: float) -> float:
    """Compound Annual Growth Rate (CAGR)."""
    if beginning_val <= 0 or ending_val <= 0 or years <= 0:
        raise ValueError("CAGR parameters must be strictly positive.")
    return (ending_val / beginning_val) ** (1.0 / years) - 1.0


def financial_pe_ratio(price: float, earnings_per_share: float) -> float:
    """Price-to-Earnings (P/E) Multiple."""
    if earnings_per_share <= 0:
        raise ValueError("EPS must be positive for standard P/E multiple calculation.")
    return price / earnings_per_share


def financial_debt_to_equity(total_debt: float, total_equity: float) -> float:
    """Debt-to-Equity (D/E) Ratio."""
    if total_equity == 0:
        raise ZeroDivisionError("Total equity cannot be zero.")
    return total_debt / total_equity


def financial_operating_margin(operating_income: float, revenue: float) -> float:
    """Operating Margin (EBIT / Revenue)."""
    if revenue == 0:
        raise ZeroDivisionError("Revenue cannot be zero.")
    return operating_income / revenue


def financial_gross_margin(gross_profit: float, revenue: float) -> float:
    """Gross Profit Margin."""
    if revenue == 0:
        raise ZeroDivisionError("Revenue cannot be zero.")
    return gross_profit / revenue


def financial_dcf_npv(cash_flows: List[float], discount_rate: float) -> float:
    """Discounted Cash Flow (DCF) Net Present Value (NPV)."""
    if discount_rate <= -1.0:
        raise ValueError("Discount rate must be greater than -100%.")
    return sum(cf / ((1.0 + discount_rate) ** (t + 1)) for t, cf in enumerate(cash_flows))


class SandboxedREPL:
    """Restricted execution environment for mathematical and financial calculation."""

    BANNED_TOKENS = {
        "__import__", "importlib", "eval", "exec", "open", "os", "sys",
        "subprocess", "shutil", "socket", "builtins", "getattr", "setattr",
        "delattr", "globals", "locals", "__class__", "__subclasses__", "__base__"
    }

    SAFE_BUILTINS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "range": range,
        "zip": zip,
        "enumerate": enumerate,
        "sorted": sorted,
        "map": map,
        "filter": filter,
        "pow": pow,
        "divmod": divmod,
        "print": print,
    }

    def __init__(self) -> None:
        self.base_env = {
            "__builtins__": self.SAFE_BUILTINS,
            "math": math,
            "np": np,
            "numpy": np,
            "cagr": financial_cagr,
            "pe_ratio": financial_pe_ratio,
            "debt_to_equity": financial_debt_to_equity,
            "operating_margin": financial_operating_margin,
            "gross_margin": financial_gross_margin,
            "dcf_npv": financial_dcf_npv,
        }

    def execute(self, code: str) -> Dict[str, Any]:
        """Execute sandboxed python code and return captured stdout and result variable."""
        clean_code = code.strip()
        if not clean_code:
            return {"output": "", "result": None, "success": True}

        # Check banned tokens
        for token in self.BANNED_TOKENS:
            # Check using word boundary or explicit string search
            if re.search(r"\b" + re.escape(token) + r"\b", clean_code):
                raise SandboxSecurityError(f"Security restriction: Use of '{token}' is prohibited in REPL sandbox.")

        stdout_capture = io.StringIO()
        local_scope: Dict[str, Any] = {}
        global_scope = dict(self.base_env)

        try:
            with redirect_stdout(stdout_capture):
                # Try evaluating as single expression first
                try:
                    expr_ast = compile(clean_code, "<repl>", "eval")
                    result = eval(expr_ast, global_scope, local_scope)
                    stdout_str = stdout_capture.getvalue().strip()
                    if stdout_str:
                        output_repr = f"{stdout_str}\nResult: {result}"
                    else:
                        output_repr = str(result)
                    return {"output": output_repr, "result": result, "success": True}
                except SyntaxError:
                    # Multi-statement script execution
                    exec_ast = compile(clean_code, "<repl>", "exec")
                    exec(exec_ast, global_scope, local_scope)
                    stdout_str = stdout_capture.getvalue().strip()
                    res = local_scope.get("result", local_scope.get("output", stdout_str))
                    output_repr = stdout_str if stdout_str else f"Executed successfully. State: {local_scope}"
                    return {"output": output_repr, "result": res, "success": True}

        except Exception as exc:
            return {"output": f"Execution Error: {type(exc).__name__}: {str(exc)}", "result": None, "success": False, "error": str(exc)}


def create_python_repl_tool(repl_instance: Optional[SandboxedREPL] = None) -> Tool:
    """Factory creating a sandboxed Python REPL Tool for Quantitative Agents."""
    repl = repl_instance or SandboxedREPL()

    def run_repl(code: str) -> str:
        res = repl.execute(code)
        if not res["success"]:
            raise RuntimeError(res.get("error", "REPL execution failed."))
        return res["output"]

    return Tool(
        name="python_repl",
        description=(
            "Execute mathematical equations, financial metrics (CAGR, P/E, debt-to-equity, DCF NPV, "
            "margins), and statistical computations. Pre-loaded with numpy (np), math, and financial functions."
        ),
        func=run_repl,
        parameters_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python expression or calculation block (e.g., 'cagr(100, 250, 4)' or 'np.mean([0.15, 0.18, 0.22])')",
                }
            },
            "required": ["code"],
        },
    )


# =============================================================================
# 2. Hybrid RAG Retrieval Tool
# =============================================================================

def create_rag_retrieval_tool(pipeline: HybridRAGPipeline) -> Tool:
    """Factory creating a Hybrid RAG Retrieval Tool for Researcher Agents."""

    def search_rag(query: str, top_k: int = 5, enable_reranker: bool = True) -> str:
        cfg = RetrievalPipelineConfig(
            candidate_pool_size=max(20, top_k * 4),
            final_top_k=top_k,
            enable_reranker=enable_reranker,
        )
        output: RetrievalPipelineOutput = pipeline.retrieve(query, config=cfg)
        if not output.has_results:
            return f"No factual evidence found in index for query: '{query}'."

        formatted_lines = [
            f"=== Hybrid RAG Retrieval (Top {len(output.items)} items | Latency: {output.total_latency_ms:.1f}ms) ===",
        ]
        for idx, item in enumerate(output.items, 1):
            score_str = f"Reranker: {item.final_score:.4f}" if item.stage2_score is not None else f"Stage1: {item.stage1_score:.4f}"
            ents = f" [Entities: {', '.join(item.matched_entities)}]" if item.matched_entities else ""
            formatted_lines.append(f"[{idx}] (Doc: {item.doc_id} | Score: {score_str}{ents})")
            formatted_lines.append(f"    {item.text.strip()}")
            if item.graph_paths:
                for path in item.graph_paths:
                    formatted_lines.append(f"    Graph Path: {path.to_explanation()}")
        return "\n".join(formatted_lines)

    return Tool(
        name="retrieve_knowledge",
        description=(
            "Search indexed enterprise knowledge using Stage 1 Hybrid Search (Dense vectors + Sparse BM25/TF-IDF + "
            "Knowledge Graph) cascaded with Stage 2 Neural Cross-Encoder Reranking."
        ),
        func=search_rag,
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query or factual research question."},
                "top_k": {"type": "integer", "description": "Maximum number of passages to return (default 5)."},
                "enable_reranker": {"type": "boolean", "description": "Whether to apply neural cross-encoder reranking."},
            },
            "required": ["query"],
        },
    )


# =============================================================================
# 3. Knowledge Graph Traversal Tool
# =============================================================================

def create_graph_traversal_tool(graph: KnowledgeGraph) -> Tool:
    """Factory creating a Knowledge Graph Traversal Tool for relational reasoning."""
    engine = GraphTraversalEngine(graph)

    def traverse_graph(
        operation: str,
        source_id: str,
        target_id: Optional[str] = None,
        max_hops: int = 3,
        min_confidence: float = 0.5,
    ) -> str:
        s_id = slugify_id(source_id)
        t_id = slugify_id(target_id) if target_id else None

        if operation == "inspect_entity":
            ent = graph.get_entity(s_id)
            if ent is None:
                return f"Entity '{source_id}' not found in Knowledge Graph."
            out_neighbors = graph.get_neighbors(s_id, direction="out")
            in_neighbors = graph.get_neighbors(s_id, direction="in")

            lines = [
                f"Entity: {ent.name} (ID: {ent.entity_id})",
                f"Type: {ent.entity_type.value}",
                f"Aliases: {', '.join(sorted(ent.aliases)) if ent.aliases else 'None'}",
                f"Outgoing Connections ({len(out_neighbors)}):",
            ]
            for target_ent, triple in out_neighbors[:10]:
                lines.append(f"  --[{triple.relation.relation_name} (conf: {triple.confidence:.2f})]--> {target_ent.name}")
            if in_neighbors:
                lines.append(f"Incoming Connections ({len(in_neighbors)}):")
                for src_ent, triple in in_neighbors[:10]:
                    lines.append(f"  <--[{triple.relation.relation_name} (conf: {triple.confidence:.2f})]-- {src_ent.name}")
            return "\n".join(lines)

        elif operation == "find_path":
            if not t_id:
                return "Error: operation 'find_path' requires both 'source_id' and 'target_id'."
            path_res: PathResult = engine.find_most_reliable_path(s_id, t_id, max_hops=max_hops)
            if not path_res.found:
                return f"No path found between '{source_id}' and '{target_id}' within {max_hops} hops."
            return (
                f"Most Reliable Reasoning Path (Confidence: {path_res.confidence_score:.4f} | Hops: {path_res.hop_count}):\n"
                f"{path_res.to_explanation()}"
            )

        elif operation == "neighbors":
            subgraph: SubgraphResult = engine.bfs_k_hop(s_id, max_hops=max_hops, min_confidence=min_confidence)
            return (
                f"K-Hop Neighborhood for '{source_id}' ({len(subgraph.entities)} entities, {len(subgraph.triples)} relations):\n"
                f"{subgraph.summary()}"
            )

        elif operation == "common_neighbors":
            if not t_id:
                return "Error: operation 'common_neighbors' requires both 'source_id' and 'target_id'."
            common = engine.find_common_neighbors(s_id, t_id)
            if not common:
                return f"No shared common neighbors between '{source_id}' and '{target_id}'."
            names = [ent.name for ent in common]
            return f"Shared Entities ({len(names)}): {', '.join(names)}"

        else:
            return (
                f"Unknown operation '{operation}'. Supported operations: "
                f"'inspect_entity', 'find_path', 'neighbors', 'common_neighbors'."
            )

    return Tool(
        name="traverse_knowledge_graph",
        description=(
            "Inspect structured entities, discover multi-hop causal and supplier paths between two companies/technologies, "
            "or explore k-hop neighborhoods in the Knowledge Graph."
        ),
        func=traverse_graph,
        parameters_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["inspect_entity", "find_path", "neighbors", "common_neighbors"],
                    "description": "Graph traversal operation to perform.",
                },
                "source_id": {"type": "string", "description": "Primary entity identifier (e.g. 'nvidia', 'tsmc')."},
                "target_id": {"type": "string", "description": "Destination entity identifier for path or common neighbors."},
                "max_hops": {"type": "integer", "description": "Maximum path traversal depth (default 3)."},
                "min_confidence": {"type": "number", "description": "Minimum edge confidence score threshold (default 0.5)."},
            },
            "required": ["operation", "source_id"],
        },
    )


# =============================================================================
# 4. Market Data & Fundamentals Tool
# =============================================================================

DEFAULT_MARKET_FEED: Dict[str, Dict[str, Any]] = {
    "nvda": {
        "ticker": "NVDA",
        "name": "NVIDIA Corporation",
        "market_cap_billions": 3120.0,
        "revenue_ttm_billions": 96.3,
        "gross_margin": 0.751,
        "operating_margin": 0.618,
        "pe_ratio": 46.2,
        "debt_to_equity": 0.18,
        "free_cash_flow_billions": 39.2,
        "primary_suppliers": ["TSMC", "SK Hynix", "Micron", "ASML"],
        "datacenter_share": 0.88,
    },
    "aapl": {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "market_cap_billions": 3450.0,
        "revenue_ttm_billions": 385.6,
        "gross_margin": 0.462,
        "operating_margin": 0.307,
        "pe_ratio": 33.5,
        "debt_to_equity": 1.45,
        "free_cash_flow_billions": 108.8,
        "primary_suppliers": ["TSMC", "Foxconn", "Sony", "Samsung Display"],
        "services_growth_rate": 0.14,
    },
    "tsm": {
        "ticker": "TSM",
        "name": "Taiwan Semiconductor Manufacturing Company (TSMC)",
        "market_cap_billions": 890.0,
        "revenue_ttm_billions": 76.5,
        "gross_margin": 0.543,
        "operating_margin": 0.431,
        "pe_ratio": 24.8,
        "debt_to_equity": 0.28,
        "free_cash_flow_billions": 28.5,
        "advanced_node_share_pct": 0.67,
    },
    "msft": {
        "ticker": "MSFT",
        "name": "Microsoft Corporation",
        "market_cap_billions": 3200.0,
        "revenue_ttm_billions": 245.1,
        "gross_margin": 0.697,
        "operating_margin": 0.446,
        "pe_ratio": 35.8,
        "debt_to_equity": 0.39,
        "free_cash_flow_billions": 74.1,
        "azure_growth_rate": 0.29,
    },
    "amd": {
        "ticker": "AMD",
        "name": "Advanced Micro Devices",
        "market_cap_billions": 255.0,
        "revenue_ttm_billions": 23.2,
        "gross_margin": 0.485,
        "operating_margin": 0.114,
        "pe_ratio": 115.0,
        "debt_to_equity": 0.05,
        "free_cash_flow_billions": 2.1,
        "datacenter_gpu_revenue_billions": 4.5,
    },
}


def create_market_data_tool(feed: Optional[Dict[str, Dict[str, Any]]] = None) -> Tool:
    """Factory creating a Market Data Tool for Quantitative & Research Agents."""
    market_db = feed or DEFAULT_MARKET_FEED

    def get_market_data(entity: str) -> str:
        key = slugify_id(entity)
        # Attempt match by key or ticker
        data = market_db.get(key)
        if not data:
            for item in market_db.values():
                item_ticker = item.get("ticker", "").lower()
                item_name = item.get("name", "").lower()
                if item_ticker == key or key in item_name or item_name in key:
                    data = item
                    break

        if not data:
            available = ", ".join(k.upper() for k in market_db.keys())
            return f"Financial data for '{entity}' not found in active market feed. Available entities: {available}."

        gm_raw = data.get("gross_margin", 0)
        gm_pct = gm_raw if gm_raw > 1.0 else gm_raw * 100
        om_raw = data.get("operating_margin", 0)
        om_pct = om_raw if om_raw > 1.0 else om_raw * 100

        lines = [
            f"=== Market Fundamentals: {data.get('name')} ({data.get('ticker')}) ===",
            f"Market Cap: ${data.get('market_cap_billions', 0):,.1f} Billion",
            f"TTM Revenue: ${data.get('revenue_ttm_billions', 0):,.1f} Billion",
            f"Gross Margin: {gm_pct:.2f}%",
            f"Operating Margin: {om_pct:.2f}%",
            f"P/E Multiple: {data.get('pe_ratio', 0):.1f}x",
            f"Debt-to-Equity: {data.get('debt_to_equity', 0):.2f}",
            f"Free Cash Flow: ${data.get('free_cash_flow_billions', 0):,.1f} Billion",
        ]
        if "primary_suppliers" in data:
            lines.append(f"Key Suppliers: {', '.join(data['primary_suppliers'])}")
        return "\n".join(lines)

    return Tool(
        name="get_market_data",
        description=(
            "Retrieve verified financial statements, operating margins, valuation multiples, "
            "and balance sheet metrics for market entities (e.g. NVDA, AAPL, TSM, MSFT, AMD)."
        ),
        func=get_market_data,
        parameters_schema={
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Company ticker or name (e.g. 'NVDA', 'Apple', 'TSM')."},
            },
            "required": ["entity"],
        },
    )


# =============================================================================
# 5. Master Default Toolset Factory
# =============================================================================

def create_agent_toolset(
    pipeline: Optional[HybridRAGPipeline] = None,
    graph: Optional[KnowledgeGraph] = None,
    market_feed: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Tool]:
    """Assemble standard tool suite for OmniIntel agents."""
    tools: List[Tool] = [
        create_python_repl_tool(),
        create_market_data_tool(feed=market_feed),
    ]
    if pipeline is not None:
        tools.append(create_rag_retrieval_tool(pipeline))
    if graph is not None:
        tools.append(create_graph_traversal_tool(graph))
    return tools
