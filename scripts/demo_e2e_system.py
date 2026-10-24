#!/usr/bin/env python3
"""Master End-to-End System Demonstration for OmniIntel.

Autonomous Multi-Agent Research & Predictive Intelligence Engine:
  1. Production Gateway Health & Readiness Probing.
  2. Kahn's Algorithm Execution DAG Decomposition & Graphviz DOT Rendering.
  3. Dijkstra Multi-Hop Knowledge Graph Shortest Path Reasoning.
  4. Stage-1 Hybrid Search (BM25 + Cosine + RRF) & Stage-2 Pre-LN Cross-Encoder Neural Reranker.
  5. Prefix Trie O(N) Linear-Time Entity Extraction.
  6. End-to-End Autonomous Multi-Agent Mission with Closed-Loop Critic Reflection.

Usage:
  python3 scripts/demo_e2e_system.py
"""

import asyncio
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Terminal ANSI Color & Typography Palette
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

CYAN = "\033[96m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
WHITE = "\033[97m"

BG_BLUE = "\033[44m\033[37m"
BG_GREEN = "\033[42m\033[30m"
BG_CYAN = "\033[46m\033[30m"
BG_MAGENTA = "\033[45m\033[37m"

from starlette.testclient import TestClient

from src.api.app import create_app
from src.ui.dashboard import build_dag_dot_graph, build_subgraph_dot_graph


def print_banner() -> None:
    """Print master operational banner."""
    print(f"\n{CYAN}{BOLD}" + "=" * 80 + f"{RESET}")
    print(f"{CYAN}{BOLD}  ⚡ OMNIINTEL — MASTER END-TO-END SYSTEM DEMONSTRATION ⚡{RESET}")
    print(f"{DIM}  Autonomous Multi-Agent Research & Predictive Intelligence Engine{RESET}")
    print(f"{DIM}  FastAPI Gateway | PyTorch Cross-Encoder | Kahn's DAG | Hybrid RAG | KG{RESET}")
    print(f"{CYAN}{BOLD}" + "=" * 80 + f"{RESET}\n")


def print_section(num: int, title: str, subtitle: str = "") -> None:
    """Print visually distinct section header."""
    print(f"\n{BG_CYAN}{BOLD} [STAGE {num}] {title.upper()} {RESET}")
    if subtitle:
        print(f"{DIM}  ↳ {subtitle}{RESET}\n")
    else:
        print()


def demo_gateway_health(client: TestClient) -> None:
    """Demonstrate gateway health check and telemetry probing."""
    print_section(1, "Gateway Health & Subsystem Readiness Probe", "Verifying zero-downtime service health and telemetry")

    t0 = time.perf_counter()
    resp_liveness = client.get("/health")
    resp_readiness = client.get("/health/ready")
    resp_telemetry = client.get("/health/telemetry")
    dt = (time.perf_counter() - t0) * 1000.0

    print(f"  {GREEN}✔{RESET} {BOLD}Liveness Probe (/health):{RESET} status={resp_liveness.json().get('status')} | uptime={resp_liveness.json().get('uptime_seconds')}s")
    
    ready_data = resp_readiness.json()
    print(f"  {GREEN}✔{RESET} {BOLD}Readiness Probe (/health/ready):{RESET} status={ready_data.get('status')} | is_ready={ready_data.get('is_ready')}")
    print(f"      - {CYAN}{'Knowledge Graph':<20}{RESET} : entities={ready_data.get('knowledge_graph_entities')} | triples={ready_data.get('knowledge_graph_triples')}")
    print(f"      - {CYAN}{'Vector Store':<20}{RESET} : indexed_documents={ready_data.get('vector_store_documents')}")
    print(f"      - {CYAN}{'Neural Reranker':<20}{RESET} : active={ready_data.get('reranker_active')}")
    print(f"      - {CYAN}{'Orchestrator':<20}{RESET} : active={ready_data.get('orchestrator_active')}")

    telem = resp_telemetry.json()
    print(f"  {GREEN}✔{RESET} {BOLD}Live Telemetry (/health/telemetry):{RESET} events={telem.get('total_events_dispatched')} | rag_queries={telem.get('total_rag_queries')} | uptime={telem.get('uptime_seconds')}s")
    print(f"  {DIM}  ↳ Gateway probe completed in {dt:.2f}ms{RESET}")


def demo_dag_planning_and_visualization(client: TestClient) -> None:
    """Demonstrate Kahn's topological sorting DAG planner and visual DOT rendering."""
    print_section(2, "Kahn's Algorithm Execution DAG & Graphviz DOT Rendering", "Topological sorting with dynamic batch parallelization")

    query = "Analyze AMD Instinct MI300X datacenter solutions and compute financial margins."
    print(f"  {YELLOW}Query Directive:{RESET} '{query}'")

    t0 = time.perf_counter()
    resp = client.post("/api/v1/mission/plan", json={"query": query})
    plan = resp.json()
    dt = (time.perf_counter() - t0) * 1000.0

    print(f"  {GREEN}✔{RESET} {BOLD}Kahn's DAG Scheduled:{RESET} Plan ID={plan['plan_id']} | Total Tasks={plan['total_tasks']} | Total Stages={plan['total_stages']}")
    print(f"  {BOLD}Topological Execution Schedule:{RESET}")
    tasks_map = plan.get("tasks", {})
    for idx, batch in enumerate(plan.get("batches", []), start=1):
        tasks_in_stage = ", ".join(
            f"{tid} ({tasks_map.get(tid, {}).get('agent_role', 'agent')})"
            for tid in batch
        )
        print(f"    Stage {idx} [{len(batch)} parallel tasks]: {CYAN}{tasks_in_stage}{RESET}")

    # Visual DOT Generation
    dot_source = build_dag_dot_graph(plan)
    first_lines = "\n".join("    " + line for line in dot_source.splitlines()[:10])
    print(f"\n  {GREEN}✔{RESET} {BOLD}Graphviz DOT Pipeline Generated (Sample):{RESET}")
    print(f"{DIM}{first_lines}\n    ... [Graphviz DOT specification compiled]{RESET}")
    print(f"  {DIM}  ↳ Planning and DOT rendering completed in {dt:.2f}ms{RESET}")


def demo_knowledge_graph_reasoning(client: TestClient) -> None:
    """Demonstrate Dijkstra multi-hop shortest path and egocentric subgraphs."""
    print_section(3, "Knowledge Graph Multi-Hop Reasoning & Subgraphs", "Dijkstra logarithmic confidence routing on double adjacency list")

    # 1. Dijkstra Path: NVIDIA -> ASML
    t0 = time.perf_counter()
    resp_path = client.get("/api/v1/graph/path?source=nvidia&target=asml")
    path_data = resp_path.json()
    dt_path = (time.perf_counter() - t0) * 1000.0

    print(f"  {BOLD}Shortest Path Query:{RESET} NVIDIA $\\to$ ASML")
    print(f"  {GREEN}✔{RESET} {BOLD}Path Discovered:{RESET} Found={path_data['found']} | Hops={path_data['hop_count']} | Aggregate Confidence={path_data['aggregate_confidence'] * 100:.2f}%")
    print(f"  {YELLOW}Reasoning Chain:{RESET} {path_data['path_explanation']}")

    # 2. Egocentric Subgraph Extraction: TSMC (k=1)
    resp_sub = client.post("/api/v1/graph/subgraph", json={"root_entity_id": "tsmc", "max_hops": 1})
    sub_data = resp_sub.json()
    print(f"\n  {GREEN}✔{RESET} {BOLD}Egocentric Subgraph (TSMC, k=1):{RESET} Entities={len(sub_data['entities'])} | Triples={len(sub_data['triples'])}")
    for trip in sub_data["triples"][:3]:
        sub = trip["subject"]["name"]
        rel = trip["relation"].get("relation_type", "RELATED_TO")
        obj = trip["object"]["name"]
        conf = trip["confidence"]
        print(f"    - ({sub}) --[{rel} ({conf:.2f})]--> ({obj})")

    dot_sub = build_subgraph_dot_graph(sub_data)
    print(f"  {DIM}  ↳ Multi-hop path reasoning completed in {dt_path:.2f}ms{RESET}")


def demo_hybrid_search_and_neural_reranker(client: TestClient) -> None:
    """Demonstrate Stage-1 BM25+Cosine RRF and Stage-2 PyTorch Cross-Encoder."""
    print_section(4, "Dual-Channel Hybrid RAG & Pre-LN Neural Reranking", "Sub-linear sparse + dense search fused via Reciprocal Rank Fusion")

    query = "CoWoS advanced packaging capacity TSMC NVIDIA"
    print(f"  {YELLOW}Retrieval Query:{RESET} '{query}'")

    t0 = time.perf_counter()
    resp = client.post("/api/v1/search/hybrid", json={
        "query": query,
        "top_k": 3,
        "alpha": 0.5,
        "fusion_mode": "rrf",
        "use_reranker": True,
    })
    data = resp.json()
    dt = (time.perf_counter() - t0) * 1000.0

    print(f"  {GREEN}✔{RESET} {BOLD}Stage-1 & Stage-2 Pipeline Executed:{RESET} Hits={data['total_hits']} | Neural Reranked={data['cross_encoder_reranked']}")
    for rank, hit in enumerate(data["results"], start=1):
        doc_id = hit["doc_id"]
        score = hit["score"]
        stage = hit["stage"]
        snippet = hit["text"].replace("\n", " ")[:65]
        print(f"    [{rank}] {CYAN}{doc_id:<15}{RESET} (score: {score:.4f}, stage: {stage:<13}) : {snippet}...")

    # Direct Cross-Encoder Batch Ranking Endpoint
    print(f"\n  {BOLD}Direct Neural Cross-Encoder Batch Reranking:{RESET}")
    r_resp = client.post("/api/v1/search/rerank", json={
        "query": "Who manufactures cutting-edge EUV lithography?",
        "candidates": [
            {"doc_id": "c1", "text": "ASML Holding designs extreme ultraviolet lithography systems."},
            {"doc_id": "c2", "text": "AMD produces Instinct series datacenter accelerators."},
        ],
        "return_top_k": 2,
    })
    r_data = r_resp.json()
    for item in r_data["ranked_results"]:
        print(f"    Rank #{item['rank']} | Doc ID: {item['doc_id']} | Logit Score: {item['score']:.4f} | Passages: {item['text'][:55]}...")
    print(f"  {DIM}  ↳ Dual-channel hybrid search & reranking completed in {dt:.2f}ms{RESET}")


def demo_prefix_trie_extraction(client: TestClient) -> None:
    """Demonstrate O(N) linear time entity extraction via in-memory Prefix Trie."""
    print_section(5, "Prefix Trie O(N) Linear-Time Entity Extraction", "Scanning unstructured financial filings with 26-ary character trie")

    sample_filing = (
        "NVIDIA Corporation leverages TSMC advanced CoWoS packaging for its Blackwell B200 accelerators. "
        "ASML provides EUV lithography systems while AMD competes with its Instinct architecture."
    )
    print(f"  {BOLD}Raw Text Segment ({len(sample_filing)} chars):{RESET}\n  \"{DIM}{sample_filing}{RESET}\"")

    t0 = time.perf_counter()
    resp = client.post("/api/v1/search/scan-entities", json={"text": sample_filing})
    data = resp.json()
    dt = (time.perf_counter() - t0) * 1000.0

    print(f"\n  {GREEN}✔{RESET} {BOLD}Prefix Trie Scanned in {dt:.2f}ms:{RESET} Matches Found={data['total_matches']}")
    for m in data["matches"]:
        print(f"    - Matched '{GREEN}{m['matched_text']}{RESET}' at offsets [{m.get('start_char')}:{m.get('end_char')}] (payload: {m.get('payload_value')})")


def demo_end_to_end_autonomous_mission(client: TestClient) -> None:
    """Demonstrate full multi-agent mission execution with Critic reflection."""
    print_section(6, "Autonomous Multi-Agent Mission & Reflection Self-Correction", "Closed-loop verification, AST sandbox math, and executive briefing synthesis")

    mission_query = "Compare NVIDIA and TSMC in sovereign AI infrastructure and evaluate gross margins."
    print(f"  {BOLD}Mission Directive:{RESET} '{mission_query}'")

    t0 = time.perf_counter()
    resp = client.post("/api/v1/mission/run", json={
        "query": mission_query,
        "max_replans": 2,
    })
    data = resp.json()
    dt = (time.perf_counter() - t0) * 1000.0

    print(f"\n  {GREEN}✔{RESET} {BOLD}Mission Completed:{RESET} Success={data['success']} | Latency={data['total_latency_ms']}ms | Total Stages={data['total_stages_executed']}")
    print(f"  {BOLD}Quality Assurance:{RESET} Verdict={GREEN}{data['verdict']}{RESET} | Confidence={data['confidence_score'] * 100:.1f}% | Replans={data['replan_count']}")
    print(f"  {BOLD}Analyzed Target Entities:{RESET} {', '.join(data['target_entities'])}")

    print(f"\n  {BOLD}Synthesized Strategic Intelligence Briefing (Excerpt):{RESET}")
    dossier = data["dossier_markdown"]
    for line in dossier.splitlines()[:25]:
        print(f"    {line}")
    print(f"    {DIM}... [Remaining dossier truncated for terminal clarity]{RESET}")
    print(f"\n  {DIM}  ↳ Full autonomous mission pipeline completed in {dt:.2f}ms{RESET}")


def main() -> None:
    """Main execution entrypoint."""
    print_banner()

    print(f"{YELLOW}Initializing OmniIntel FastAPI Application Context & In-Memory Subsystems...{RESET}")
    app = create_app()

    with TestClient(app) as client:
        print(f"{GREEN}✔ FastAPI Gateway Lifespan Initialized Successfully.{RESET}")

        demo_gateway_health(client)
        demo_dag_planning_and_visualization(client)
        demo_knowledge_graph_reasoning(client)
        demo_hybrid_search_and_neural_reranker(client)
        demo_prefix_trie_extraction(client)
        demo_end_to_end_autonomous_mission(client)

    print(f"\n{CYAN}{BOLD}" + "=" * 80 + f"{RESET}")
    print(f"{GREEN}{BOLD}  ✨ ALL 6 OMNIINTEL SUBSYSTEMS DEMONSTRATED SUCCESSFULLY ✨{RESET}")
    print(f"{DIM}  Ready for production containerization, CI/CD pipeline, and placement defense.{RESET}")
    print(f"{CYAN}{BOLD}" + "=" * 80 + f"{RESET}\n")


if __name__ == "__main__":
    main()
