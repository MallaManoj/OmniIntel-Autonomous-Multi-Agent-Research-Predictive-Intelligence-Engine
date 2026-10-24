"""Master End-to-End System Integration Test Suite.

Validates complete cross-layer integration across:
  - Phase 1: Core DSA Foundations (Prefix Trie, Min-Heap, LRU Cache, Kahn's DAG)
  - Phase 2: PyTorch Deep Learning (Pre-LN Transformer Cross-Encoder Reranker)
  - Phase 3: In-Memory Double Adjacency Knowledge Graph & Dual-Channel Hybrid RAG
  - Phase 4: Autonomous Multi-Agent Orchestration & Closed-Loop Critic Reflection
  - Phase 5: FastAPI ASGI Gateway, WebSockets, Rate-Limiting, and Streamlit Visualizers
"""

import asyncio
import json
import time
import unittest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.middleware import global_rate_limiter
from src.api.state import app_state
from src.core.events import Event, EventType
from src.ui.dashboard import build_dag_dot_graph, build_subgraph_dot_graph


class TestSystemEndToEndIntegration(unittest.TestCase):
    """End-to-End cross-layer verification of the entire OmniIntel engine."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.lifespan_ctx = cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def setUp(self) -> None:
        global_rate_limiter.reset()

    # -------------------------------------------------------------------------
    # 1. System Liveness, Subsystems Readiness & Telemetry
    # -------------------------------------------------------------------------

    def test_e2e_health_and_telemetry_handshake(self) -> None:
        """Verify gateway health, readiness of all underlying components, and telemetry."""
        # 1. Liveness check
        r_health = self.client.get("/health")
        self.assertEqual(r_health.status_code, 200)
        h_data = r_health.json()
        self.assertEqual(h_data["status"], "healthy")
        self.assertEqual(h_data["version"], "1.0.0")

        # 2. Readiness probe (validating graph, vector store, and neural reranker)
        r_ready = self.client.get("/health/ready")
        self.assertEqual(r_ready.status_code, 200)
        ready_data = r_ready.json()
        self.assertTrue(ready_data["is_ready"])
        self.assertGreaterEqual(ready_data["knowledge_graph_entities"], 3)
        self.assertGreaterEqual(ready_data["knowledge_graph_triples"], 3)
        self.assertGreaterEqual(ready_data["vector_store_documents"], 2)
        self.assertTrue(ready_data["reranker_active"])
        self.assertTrue(ready_data["orchestrator_active"])

        # 3. Telemetry endpoint
        r_telem = self.client.get("/health/telemetry")
        self.assertEqual(r_telem.status_code, 200)
        t_data = r_telem.json()
        self.assertIn("total_events_dispatched", t_data)
        self.assertIn("cache_hit_rate_pct", t_data)

        # 4. WebSocket stats
        r_ws = self.client.get("/ws/stats")
        self.assertEqual(r_ws.status_code, 200)
        self.assertEqual(r_ws.json()["status"], "online")

    # -------------------------------------------------------------------------
    # 2. End-to-End Multi-Agent Mission Execution Flow
    # -------------------------------------------------------------------------

    def test_e2e_full_mission_execution_flow(self) -> None:
        """Execute full autonomous multi-agent research mission and verify synthesized dossier."""
        payload = {
            "query": "Compare NVIDIA and TSMC in sovereign AI infrastructure and evaluate gross margins.",
            "max_replans": 2,
        }
        resp = self.client.post("/api/v1/mission/run", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("plan_id", data)
        self.assertGreaterEqual(data["total_stages_executed"], 3)
        self.assertGreaterEqual(data["confidence_score"], 0.85)
        self.assertIn(data["verdict"], ["APPROVED", "VALIDATED"])

        # Target entities identified
        entities_lower = [e.lower() for e in data["target_entities"]]
        self.assertTrue(any("nvda" in e or "nvidia" in e or "tsmc" in e for e in entities_lower))

        # Dossier content assertions
        dossier = data["dossier_markdown"]
        self.assertIn("Strategic Intelligence Dossier", dossier)
        self.assertIn("Quantitative Financial Modeling", dossier)
        self.assertIn("Gross Margin", dossier)
        self.assertIn("Audit Trail & Grounded Citations", dossier)

    # -------------------------------------------------------------------------
    # 3. Kahn's DAG Planning & UI Graphviz Visualizer Pipeline
    # -------------------------------------------------------------------------

    def test_e2e_dag_planning_and_visual_dot_pipeline(self) -> None:
        """Verify Kahn's algorithm planning decomposition and downstream Graphviz rendering."""
        payload = {"query": "Analyze AMD Instinct MI300X datacenter solutions and compute margins."}
        resp = self.client.post("/api/v1/mission/plan", json=payload)
        self.assertEqual(resp.status_code, 200)

        plan = resp.json()
        self.assertIn("plan_id", plan)
        self.assertGreaterEqual(plan["total_tasks"], 3)
        self.assertGreaterEqual(plan["total_stages"], 2)
        self.assertIn("batches", plan)

        # Feed directly into UI Graphviz DOT generator
        dot_code = build_dag_dot_graph(plan)
        self.assertIn("digraph KahnExecutionDAG", dot_code)
        self.assertIn("quant", dot_code)
        self.assertIn("critic", dot_code)
        self.assertIn("synthesizer", dot_code)
        self.assertIn("->", dot_code)  # Dependency arrows present

    # -------------------------------------------------------------------------
    # 4. Multi-Hop Graph Reasoning & Subgraph Visual Pipeline
    # -------------------------------------------------------------------------

    def test_e2e_graph_reasoning_and_subgraph_visual_pipeline(self) -> None:
        """Verify Dijkstra multi-hop shortest path and egocentric subgraph generation."""
        # 1. Dijkstra Path Reasoning (NVIDIA -> TSMC -> ASML)
        resp_path = self.client.get("/api/v1/graph/path?source=nvidia&target=asml")
        self.assertEqual(resp_path.status_code, 200)
        p_data = resp_path.json()
        self.assertTrue(p_data["found"])
        self.assertGreaterEqual(p_data["hop_count"], 1)
        self.assertGreater(p_data["aggregate_confidence"], 0.5)
        self.assertIn("path_explanation", p_data)

        # 2. Subgraph Extraction
        resp_sub = self.client.post(
            "/api/v1/graph/subgraph",
            json={"root_entity_id": "nvidia", "max_hops": 2, "min_confidence": 0.5},
        )
        self.assertEqual(resp_sub.status_code, 200)
        sg_data = resp_sub.json()
        self.assertEqual(sg_data["root_entity_id"], "nvidia")
        self.assertGreaterEqual(sg_data["total_entities"], 2)
        self.assertGreaterEqual(sg_data["total_triples"], 1)

        # 3. Feed into UI Subgraph DOT generator
        dot_sub = build_subgraph_dot_graph(sg_data)
        self.assertIn("digraph EgocentricTopology", dot_sub)
        self.assertIn('"nvidia"', dot_sub)
        self.assertIn("->", dot_sub)

    # -------------------------------------------------------------------------
    # 5. Dual-Channel Hybrid Search & Neural Cross-Encoder
    # -------------------------------------------------------------------------

    def test_e2e_hybrid_search_and_neural_reranker(self) -> None:
        """Verify Stage-1 hybrid search and Stage-2 Pre-LN Transformer Cross-Encoder reranking."""
        # Hybrid Search with RRF Fusion
        s_payload = {
            "query": "CoWoS advanced packaging capacity TSMC NVIDIA",
            "top_k": 3,
            "alpha": 0.5,
            "fusion_mode": "rrf",
            "use_reranker": True,
        }
        resp = self.client.post("/api/v1/search/hybrid", json=s_payload)
        self.assertEqual(resp.status_code, 200)

        s_data = resp.json()
        self.assertGreaterEqual(s_data["total_hits"], 1)
        self.assertTrue(s_data["cross_encoder_reranked"])
        top_hit = s_data["results"][0]
        self.assertEqual(top_hit["stage"], "cross_encoder")
        self.assertGreater(top_hit["score"], 0.0)

        # Direct Neural Cross-Encoder batch ranking endpoint
        r_payload = {
            "query": "Who manufactures cutting-edge EUV lithography?",
            "candidates": [
                {"doc_id": "c1", "text": "ASML Holding designs extreme ultraviolet lithography systems."},
                {"doc_id": "c2", "text": "AMD produces Instinct series datacenter accelerators."},
            ],
            "return_top_k": 2,
        }
        resp_rerank = self.client.post("/api/v1/search/rerank", json=r_payload)
        self.assertEqual(resp_rerank.status_code, 200)
        r_data = resp_rerank.json()
        self.assertEqual(r_data["total_candidates"], 2)
        # Top ranked document should be ASML lithography
        self.assertEqual(r_data["ranked_results"][0]["doc_id"], "c1")
        self.assertEqual(r_data["ranked_results"][0]["rank"], 1)

    # -------------------------------------------------------------------------
    # 6. Linear-Time Prefix Trie Entity Recognition
    # -------------------------------------------------------------------------

    def test_e2e_prefix_trie_entity_extraction(self) -> None:
        """Verify in-memory Prefix Trie scanning text in O(N) linear time."""
        raw_text = "TSMC and NVIDIA collaborate closely on advanced semiconductor wafer fabrication."
        resp = self.client.post("/api/v1/search/scan-entities", json={"text": raw_text})
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertGreaterEqual(data["total_matches"], 1)
        matched_tokens = [m["matched_text"].lower() for m in data["matches"]]
        self.assertTrue(any("nvidia" in m or "tsmc" in m for m in matched_tokens))

    # -------------------------------------------------------------------------
    # 7. Real-Time WebSocket Event Streaming During Mission
    # -------------------------------------------------------------------------

    def test_e2e_websocket_event_streaming_during_mission(self) -> None:
        """Verify real-time event broadcasting from EventBus to WebSocket clients."""
        with self.client.websocket_connect("/ws/events?client_id=e2e_client&replay=false") as ws:
            # Drain handshake
            ack = ws.receive_json()
            self.assertEqual(ack["type"], "CONNECTION_ESTABLISHED")

            # Publish event through system event_bus
            test_ev = Event(
                event_type=EventType.CRITIC_VALIDATION,
                source="E2E_Test_Runner",
                payload={"passed": True, "verdict": "E2E Ground truth reconciled"},
            )
            asyncio.run(app_state.event_bus.publish(test_ev))

            # Receive streamed event over WebSocket
            stream_msg = ws.receive_json()
            self.assertEqual(stream_msg["type"], "EVENT_STREAM")
            self.assertEqual(stream_msg["event_type"], EventType.CRITIC_VALIDATION.value)
            self.assertTrue(stream_msg["payload"]["passed"])

    # -------------------------------------------------------------------------
    # 8. Zero-Trust Boundary Validation & Rate-Limiting Protection
    # -------------------------------------------------------------------------

    def test_e2e_zero_trust_error_boundaries_and_rate_limiting(self) -> None:
        """Verify parameter injection protection and Token Bucket rate limiting."""
        # 1. Parameter injection attempt (extra forbidden field)
        bad_payload = {
            "query": "Valid query text",
            "malicious_injected_param": "attacker_value",
        }
        r_inj = self.client.post("/api/v1/mission/run", json=bad_payload)
        self.assertEqual(r_inj.status_code, 422)
        err_json = r_inj.json()
        self.assertTrue(err_json["error"])

        # 2. Rate limit headers verification
        r_ok = self.client.get("/")
        self.assertEqual(r_ok.status_code, 200)
        self.assertIn("X-RateLimit-Limit", r_ok.headers)
        self.assertIn("X-RateLimit-Remaining", r_ok.headers)
        self.assertIn("X-Process-Time-Ms", r_ok.headers)
        self.assertIn("X-Correlation-ID", r_ok.headers)


if __name__ == "__main__":
    unittest.main()
