"""OmniIntel Real-Time Autonomous Intelligence Dashboard (Streamlit).

Interactive visual interface for:
  - Strategic Mission Directive Execution & Self-Correction Feedback
  - Real-time Kahn's Algorithm DAG Task Decomposition (Graphviz)
  - Dijkstra Multi-Hop Knowledge Graph Reasoning & Egocentric Subgraph Exploration
  - Dual-Channel Hybrid Search & Neural Transformer Cross-Encoder Playground
  - Prefix Trie Fast Entity Scanning
  - System Observability, LRU Cache Metrics & WebSocket Telemetry
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional


# =============================================================================
# Pure Helper Functions (Testable & Decoupled from Streamlit runtime)
# =============================================================================

def build_dag_dot_graph(plan_data: Dict[str, Any]) -> str:
    """Generate Graphviz DOT source string representing Kahn's DAG task schedule."""
    if not plan_data:
        return 'digraph G { label="No DAG Schedule Available"; }'

    tasks = plan_data.get("tasks", {})
    batches = plan_data.get("batches", [])
    complexity = plan_data.get("complexity", "STANDARD")

    lines = [
        "digraph KahnExecutionDAG {",
        '    rankdir="TB";',
        '    graph [bgcolor="transparent", fontname="Helvetica", fontsize=11];',
        '    node [shape="box", style="filled,rounded", fontname="Helvetica", fontsize=10, penwidth=1.5];',
        '    edge [fontname="Helvetica", fontsize=9, color="#6b7280", arrowsize=0.8];',
        f'    labelloc="t"; label="Execution DAG Schedule ({complexity})"; fontcolor="#111827";',
        "",
    ]

    # Color palette by agent role
    role_colors = {
        "PlannerAgent": "#e0f2fe:#bae6fd",       # Light sky blue
        "ResearchAgent": "#dcfce7:#bbf7d0",      # Light emerald green
        "QuantAgent": "#fef3c7:#fde68a",         # Light amber
        "CriticAgent": "#fee2e2:#fecaca",        # Light rose
        "SynthesizerAgent": "#f3e8ff:#e9d5ff",   # Light purple
    }

    # Declare task nodes with metadata
    for tid, task in tasks.items():
        name = task.get("name", tid)
        role = task.get("agent_role", "Agent")
        fill_color = role_colors.get(role, "#f3f4f6:#e5e7eb")
        label = f"<{name}<br/><font point-size='8' color='#374151'><b>{role}</b></font>>"
        lines.append(f'    "{tid}" [label={label}, fillcolor="{fill_color}", color="#4b5563"];')

    lines.append("")

    # Declare dependency edges
    for tid, task in tasks.items():
        deps = task.get("dependencies", [])
        for dep in deps:
            if dep in tasks:
                lines.append(f'    "{dep}" -> "{tid}";')

    lines.append("}")
    return "\n".join(lines)


def build_subgraph_dot_graph(subgraph_data: Dict[str, Any]) -> str:
    """Generate Graphviz DOT source string representing local Knowledge Graph topology."""
    if not subgraph_data or not subgraph_data.get("entities"):
        return 'digraph G { label="No Subgraph Available"; }'

    root_id = subgraph_data.get("root_entity_id", "")
    entities = subgraph_data.get("entities", [])
    triples = subgraph_data.get("triples", [])

    lines = [
        "digraph EgocentricTopology {",
        '    rankdir="LR";',
        '    graph [bgcolor="transparent", fontname="Helvetica", fontsize=11];',
        '    node [shape="ellipse", style="filled", fontname="Helvetica", fontsize=10];',
        '    edge [fontname="Helvetica", fontsize=8, color="#4b5563"];',
        "",
    ]

    for ent in entities:
        eid = ent.get("entity_id", "")
        name = ent.get("name", eid)
        etype = ent.get("entity_type", "ENTITY")
        is_root = (eid == root_id)

        if is_root:
            lines.append(f'    "{eid}" [label="{name}\\n({etype})", fillcolor="#fde047", color="#ca8a04", penwidth=2.5];')
        else:
            lines.append(f'    "{eid}" [label="{name}\\n({etype})", fillcolor="#e0e7ff", color="#6366f1"];')

    lines.append("")
    for trip in triples:
        sub = trip.get("subject", {}).get("entity_id", "")
        obj = trip.get("object", {}).get("entity_id", "")
        rel = trip.get("relation", {}).get("relation_type") or trip.get("relation", {}).get("relation_name", "RELATED_TO")
        conf = trip.get("confidence", 1.0)
        if sub and obj:
            lines.append(f'    "{sub}" -> "{obj}" [label="{rel}\\n({conf:.2f})"];')

    lines.append("}")
    return "\n".join(lines)


# =============================================================================
# Streamlit Dashboard Presentation Layer
# =============================================================================

def run_streamlit_app() -> None:
    """Entrypoint executing the interactive Streamlit user interface."""
    try:
        import streamlit as st
        import httpx
    except ImportError:
        print("Streamlit or httpx is not installed in current environment.")
        return

    # Configuration from environment
    api_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    st.set_page_config(
        page_title="OmniIntel — Autonomous Intelligence Engine",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("⚡ OmniIntel — Autonomous Multi-Agent Intelligence Engine")
    st.markdown(
        "Production AI system integrating **Kahn's algorithm DAG planning**, "
        "**Pre-LN Transformer Cross-Encoder neural reranking**, **dual-channel hybrid RAG**, "
        "and **closed-loop reflection-driven self-correction**."
    )

    # -------------------------------------------------------------------------
    # Sidebar: Live Telemetry & System Health
    # -------------------------------------------------------------------------
    st.sidebar.header("System Health & Telemetry")

    def get_api_status() -> Optional[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{api_url}/health/ready")
                if res.status_code == 200:
                    return res.json()
        except Exception:
            pass
        return None

    def get_telemetry() -> Optional[Dict[str, Any]]:
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{api_url}/health/telemetry")
                if res.status_code == 200:
                    return res.json()
        except Exception:
            pass
        return None

    status_data = get_api_status()
    if status_data and status_data.get("is_ready"):
        st.sidebar.success("● Gateway: ONLINE & READY")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("KG Entities", status_data.get("knowledge_graph_entities", 0))
        c2.metric("KG Triples", status_data.get("knowledge_graph_triples", 0))

        c3, c4 = st.sidebar.columns(2)
        c3.metric("Vector Docs", status_data.get("vector_store_documents", 0))
        c4.metric("Cross-Encoder", "ACTIVE" if status_data.get("reranker_active") else "OFFLINE")

        telem = get_telemetry()
        if telem:
            st.sidebar.divider()
            st.sidebar.caption("Runtime Telemetry")
            st.sidebar.metric("Events Dispatched", telem.get("total_events_dispatched", 0))
            st.sidebar.metric("Cache Hit Rate", f"{telem.get('cache_hit_rate_pct', 0.0):.1f}%")
            st.sidebar.metric("Avg RAG Latency", f"{telem.get('rag_latency_avg_ms', 0.0):.2f} ms")
    else:
        st.sidebar.warning("○ Gateway: CONNECTING...")
        st.sidebar.info(f"Target Gateway: `{api_url}`")
        st.sidebar.caption("Run `make run-api` or `make compose-up` to launch backend service.")

    # -------------------------------------------------------------------------
    # Main Tabs Interface
    # -------------------------------------------------------------------------
    tab_mission, tab_dag, tab_graph, tab_search, tab_trie = st.tabs([
        "🎯 Strategic Mission Console",
        "📊 Kahn's DAG Visualizer",
        "🕸️ Knowledge Graph & Paths",
        "🔍 Dual-Channel Hybrid Search",
        "⚡ Prefix Trie Entity Scanner",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Strategic Mission Console
    # -------------------------------------------------------------------------
    with tab_mission:
        st.subheader("Autonomous Research & Self-Correction Mission")
        st.markdown(
            "Execute end-to-end multi-agent research. If the **CriticAgent** detects ungrounded "
            "claims or numerical hallucinations, the engine **dynamically replans** using Kahn's algorithm."
        )

        preset_queries = [
            "Compare NVIDIA and TSMC in sovereign AI infrastructure and evaluate gross margins.",
            "Analyze Advanced Micro Devices gross margins and Instinct MI300X AI solutions.",
            "Assess ASML EUV lithography equipment shipments to Taiwan and evaluate supply bottlenecks.",
        ]
        selected_preset = st.selectbox("Preset Directive Templates:", preset_queries)
        mission_query = st.text_area("Research Directive:", value=selected_preset, height=80)

        col_run1, col_run2, col_run3 = st.columns([1, 1, 3])
        with col_run1:
            execute_btn = st.button("🚀 Execute Mission", type="primary")
        with col_run2:
            plan_only_btn = st.button("📋 Decompose Plan Only")

        if execute_btn:
            with st.spinner("Executing autonomous agent orchestration..."):
                start_time = time.perf_counter()
                try:
                    with httpx.Client(timeout=45.0) as client:
                        resp = client.post(
                            f"{api_url}/api/v1/mission/run",
                            json={"query": mission_query, "max_replans": 2},
                        )
                    elapsed = (time.perf_counter() - start_time) * 1000.0

                    if resp.status_code == 200:
                        m_res = resp.json()
                        st.success(f"Mission Completed in {m_res.get('total_latency_ms', elapsed):.1f}ms")

                        # Telemetry Badges
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Stages Executed", m_res.get("total_stages_executed", 0))
                        m2.metric("Replans Triggered", m_res.get("replan_count", 0))
                        m3.metric("Critic Confidence", f"{m_res.get('confidence_score', 0.0) * 100:.1f}%")
                        m4.metric("Critic Verdict", m_res.get("verdict", "N/A"))

                        st.subheader("Executive Intelligence Dossier")
                        st.markdown(m_res.get("dossier_markdown", "No dossier generated."))
                    else:
                        st.error(f"Mission Execution Failed ({resp.status_code}): {resp.text}")

                except Exception as exc:
                    st.error(f"Failed to communicate with API: {exc}")

        elif plan_only_btn:
            with st.spinner("Decomposing directive using Kahn's algorithm..."):
                try:
                    with httpx.Client(timeout=10.0) as client:
                        resp = client.post(
                            f"{api_url}/api/v1/mission/plan",
                            json={"query": mission_query},
                        )
                    if resp.status_code == 200:
                        p_res = resp.json()
                        st.session_state["last_plan"] = p_res
                        st.success(f"DAG Plan Generated: {p_res.get('total_tasks', 0)} Tasks across {p_res.get('total_stages', 0)} Stages")
                        st.info("Switch to the **Kahn's DAG Visualizer** tab to inspect the interactive dependency graph.")
                    else:
                        st.error(f"Planning failed: {resp.text}")
                except Exception as exc:
                    st.error(f"Communication error: {exc}")

    # -------------------------------------------------------------------------
    # TAB 2: Kahn's DAG Visualizer
    # -------------------------------------------------------------------------
    with tab_dag:
        st.subheader("Kahn's Algorithm Task Dependency DAG")
        st.markdown(
            "Visualizes decomposed execution stages. Independent tasks run in parallel batches "
            "while respecting topological constraints."
        )

        plan = st.session_state.get("last_plan")
        if not plan:
            st.info("No plan generated in current session. Generating default plan...")
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(
                        f"{api_url}/api/v1/mission/plan",
                        json={"query": "Compare NVIDIA and TSMC gross margins and supply chain."},
                    )
                if resp.status_code == 200:
                    plan = resp.json()
                    st.session_state["last_plan"] = plan
            except Exception:
                pass

        if plan:
            c_meta1, c_meta2, c_meta3 = st.columns(3)
            c_meta1.metric("Plan Complexity", plan.get("complexity", "STANDARD"))
            c_meta2.metric("Total Tasks", plan.get("total_tasks", 0))
            c_meta3.metric("Execution Stages", plan.get("total_stages", 0))

            dot_source = build_dag_dot_graph(plan)
            st.graphviz_chart(dot_source, use_container_width=True)

            st.caption("Topological Batch Stages:")
            for s_idx, batch in enumerate(plan.get("batches", []), start=1):
                st.write(f"**Stage {s_idx} (Parallel Batch):** `{'`, `'.join(batch)}`")
        else:
            st.warning("Could not connect to API to generate DAG plan.")

    # -------------------------------------------------------------------------
    # TAB 3: Knowledge Graph & Dijkstra Path Explorer
    # -------------------------------------------------------------------------
    with tab_graph:
        st.subheader("Knowledge Graph Multi-Hop Reasoning & Subgraph Extraction")

        sub_tab1, sub_tab2 = st.tabs(["Dijkstra Path Reasoning", "Egocentric Subgraph Explorer"])

        with sub_tab1:
            st.markdown("Find the highest-confidence multi-hop reasoning path between entities using Dijkstra's algorithm.")
            col_s1, col_s2, col_s3 = st.columns([2, 2, 1])
            with col_s1:
                src_node = st.text_input("Source Entity:", value="nvidia")
            with col_s2:
                tgt_node = st.text_input("Target Entity:", value="asml")
            with col_s3:
                st.write("")
                st.write("")
                find_path_btn = st.button("Explore Path")

            if find_path_btn:
                try:
                    with httpx.Client(timeout=5.0) as client:
                        resp = client.get(f"{api_url}/api/v1/graph/path?source={src_node}&target={tgt_node}")
                    if resp.status_code == 200:
                        path_data = resp.json()
                        if path_data.get("found"):
                            st.success(f"Reasoning Path Discovered: {path_data.get('hop_count', 0)} Hops (Aggregate Confidence: {path_data.get('aggregate_confidence', 0):.4f})")
                            st.code(path_data.get("path_explanation", ""))
                        else:
                            st.warning(path_data.get("path_explanation", "No path found."))
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as exc:
                    st.error(f"Communication error: {exc}")

        with sub_tab2:
            st.markdown("Extract and visualize local topology around a focal entity node.")
            col_sub1, col_sub2, col_sub3 = st.columns([2, 1, 1])
            with col_sub1:
                root_entity = st.text_input("Root Entity Anchor:", value="nvidia")
            with col_sub2:
                max_hops = st.slider("Max Hop Depth:", min_value=1, max_value=3, value=2)
            with col_sub3:
                st.write("")
                st.write("")
                extract_subgraph_btn = st.button("Extract Subgraph")

            if extract_subgraph_btn:
                try:
                    with httpx.Client(timeout=5.0) as client:
                        resp = client.post(
                            f"{api_url}/api/v1/graph/subgraph",
                            json={"root_entity_id": root_entity, "max_hops": max_hops, "min_confidence": 0.5},
                        )
                    if resp.status_code == 200:
                        sg_data = resp.json()
                        st.success(f"Extracted {sg_data.get('total_entities', 0)} Entities and {sg_data.get('total_triples', 0)} Incident Fact Triples")
                        sub_dot = build_subgraph_dot_graph(sg_data)
                        st.graphviz_chart(sub_dot, use_container_width=True)
                    else:
                        st.error(f"Extraction failed: {resp.text}")
                except Exception as exc:
                    st.error(f"Communication error: {exc}")

    # -------------------------------------------------------------------------
    # TAB 4: Dual-Channel Hybrid Search & Neural Cross-Encoder
    # -------------------------------------------------------------------------
    with tab_search:
        st.subheader("Dual-Channel Hybrid Retrieval & Transformer Reranker")

        col_q, col_opt1, col_opt2 = st.columns([3, 1, 1])
        with col_q:
            search_query = st.text_input("Retrieval Query:", value="CoWoS advanced packaging capacity TSMC NVIDIA")
        with col_opt1:
            top_k_val = st.slider("Top K:", min_value=1, max_value=10, value=3)
        with col_opt2:
            alpha_val = st.slider("Alpha (Dense vs Sparse):", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

        col_opt3, col_opt4 = st.columns(2)
        with col_opt3:
            fusion_val = st.selectbox("Fusion Strategy:", ["rrf", "linear", "auto"])
        with col_opt4:
            rerank_toggle = st.checkbox("Enable Stage-2 Cross-Encoder Reranking", value=True)

        if st.button("🔍 Execute Hybrid Retrieval"):
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        f"{api_url}/api/v1/search/hybrid",
                        json={
                            "query": search_query,
                            "top_k": top_k_val,
                            "alpha": alpha_val,
                            "fusion_mode": fusion_val,
                            "use_reranker": rerank_toggle,
                        },
                    )
                if resp.status_code == 200:
                    search_res = resp.json()
                    st.success(f"Retrieved {search_res.get('total_hits', 0)} hits in {search_res.get('latency_ms', 0):.2f}ms (Neural Reranked: {search_res.get('cross_encoder_reranked')})")

                    for item in search_res.get("results", []):
                        with st.expander(f"[{item.get('stage')}] Doc: {item.get('doc_id')} — Relevance Score: {item.get('score'):.4f}"):
                            st.write(item.get("text"))
                            st.caption(f"Metadata: {item.get('metadata')}")
                else:
                    st.error(f"Search failed: {resp.text}")
            except Exception as exc:
                st.error(f"Communication error: {exc}")

    # -------------------------------------------------------------------------
    # TAB 5: Prefix Trie Fast Entity Scanner
    # -------------------------------------------------------------------------
    with tab_trie:
        st.subheader("O(N) Prefix Trie Entity Scanning Engine")
        st.markdown(
            "Scan arbitrary unstructured text for registered financial tickers and corporate "
            "entities in linear time without regex overhead."
        )

        sample_corpus = (
            "NVIDIA reported unprecedented demand for Hopper and Blackwell architectures. "
            "TSMC serves as premier foundry partner utilizing advanced CoWoS packaging allocations, "
            "while ASML supplies critical extreme ultraviolet lithography systems."
        )
        trie_input = st.text_area("Input Text to Scan:", value=sample_corpus, height=100)

        if st.button("⚡ Scan Entities via Trie"):
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(
                        f"{api_url}/api/v1/search/scan-entities",
                        json={"text": trie_input},
                    )
                if resp.status_code == 200:
                    t_res = resp.json()
                    st.success(f"Detected {t_res.get('total_matches', 0)} entities in {t_res.get('scan_latency_ms', 0):.3f}ms")

                    matches = t_res.get("matches", [])
                    if matches:
                        cols = st.columns(min(len(matches), 4))
                        for idx, m in enumerate(matches):
                            cols[idx % 4].info(f"**{m.get('matched_text')}** (Chars {m.get('start_char')}–{m.get('end_char')})")
                    else:
                        st.info("No matching registered entities found in provided text.")
                else:
                    st.error(f"Scan failed: {resp.text}")
            except Exception as exc:
                st.error(f"Communication error: {exc}")


# Direct execution entrypoint for Streamlit CLI
if __name__ == "__main__":
    run_streamlit_app()
