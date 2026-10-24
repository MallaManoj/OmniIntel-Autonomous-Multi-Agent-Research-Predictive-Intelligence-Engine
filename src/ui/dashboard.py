"""OmniIntel Real-Time Autonomous Intelligence Dashboard (Streamlit)."""

import os
import sys
import time

try:
    import streamlit as st
    import httpx
except ImportError:
    # Fallback message if executed directly without Streamlit installed
    print("Streamlit or httpx is not installed in current environment.")
    sys.exit(0)

# Configuration from environment
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="OmniIntel Autonomous Research Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ OmniIntel — Autonomous Multi-Agent Predictive Intelligence")
st.markdown(
    "Enterprise-grade multi-agent orchestration combining Kahn's DAG decomposition, "
    "Pre-LN Transformer Cross-Encoder neural reranking, in-memory Knowledge Graph traversal, "
    "and closed-loop dynamic self-correction."
)

# Sidebar System Health
st.sidebar.header("System Status & Telemetry")

def fetch_health():
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{API_BASE_URL}/health/ready")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        return None

health_data = fetch_health()
if health_data and health_data.get("is_ready"):
    st.sidebar.success("● API Engine: ONLINE & READY")
    st.sidebar.metric("KG Entities", health_data.get("knowledge_graph_entities", 0))
    st.sidebar.metric("KG Triples", health_data.get("knowledge_graph_triples", 0))
    st.sidebar.metric("Vector Documents", health_data.get("vector_store_documents", 0))
    st.sidebar.metric("Cross-Encoder Active", str(health_data.get("reranker_active", False)))
else:
    st.sidebar.warning("○ API Engine: Connecting...")
    st.sidebar.info(f"Target Gateway: `{API_BASE_URL}`")

# Main Interface Tabs
tab_mission, tab_graph, tab_search = st.tabs([
    "🎯 Autonomous Mission Execution",
    "🕸️ Knowledge Graph & Multi-Hop Paths",
    "🔍 Dual-Channel Hybrid Search",
])

with tab_mission:
    st.subheader("Autonomous Research Mission Directive")
    default_query = "Compare NVIDIA and TSMC in sovereign AI infrastructure and evaluate gross margins."
    query_input = st.text_area("Strategic Intelligence Directive:", value=default_query, height=80)
    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Execute Mission", type="primary")

    if run_btn:
        if len(query_input.strip()) < 3:
            st.error("Directive must be at least 3 characters.")
        else:
            with st.spinner("Executing autonomous agent orchestration..."):
                try:
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.post(
                            f"{API_BASE_URL}/api/v1/mission/run",
                            json={"query": query_input, "max_replans": 2},
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(f"Mission Completed in {data.get('total_latency_ms', 0):.1f}ms")
                            st.markdown(data.get("dossier_markdown", "No dossier produced."))
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

with tab_graph:
    st.subheader("Knowledge Graph Topology & Path Reasoning")
    st.info("Explore multi-hop supply chain connections using Dijkstra's algorithm.")
    col_a, col_b = st.columns(2)
    with col_a:
        src = st.text_input("Source Entity:", value="nvidia")
    with col_b:
        tgt = st.text_input("Target Entity:", value="asml")

    if st.button("Explore Reasoning Path"):
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{API_BASE_URL}/api/v1/graph/path?source={src}&target={tgt}")
                if resp.status_code == 200:
                    pdata = resp.json()
                    st.success(f"Path Found: {pdata.get('hop_count', 0)} Hops (Confidence: {pdata.get('aggregate_confidence', 0):.2f})")
                    st.code(pdata.get("path_explanation", ""))
                else:
                    st.warning(f"Response {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Path reasoning error: {e}")

with tab_search:
    st.subheader("Dual-Channel Hybrid Search & Neural Reranker")
    search_q = st.text_input("Retrieval Query:", value="CoWoS advanced packaging capacity TSMC")
    if st.button("Search Knowledge Base"):
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    f"{API_BASE_URL}/api/v1/search/hybrid",
                    json={"query": search_q, "top_k": 3, "use_reranker": True},
                )
                if resp.status_code == 200:
                    sdata = resp.json()
                    st.write(f"Total Hits: {sdata.get('total_hits', 0)} ({sdata.get('latency_ms', 0):.2f}ms)")
                    for hit in sdata.get("results", []):
                        with st.expander(f"[{hit.get('stage')}] Doc ID: {hit.get('doc_id')} (Score: {hit.get('score')})"):
                            st.write(hit.get("text"))
                else:
                    st.warning(f"Search failed: {resp.text}")
        except Exception as e:
            st.error(f"Search error: {e}")
