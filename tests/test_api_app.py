"""Unit and integration test suite for OmniIntel FastAPI application and REST endpoints."""

import unittest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.state import AppState, app_state


class TestFastAPIApplication(unittest.TestCase):
    """Test suite verifying REST endpoints, lifespan setup, middleware, and schemas."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        # Enter lifespan context to initialize knowledge graph and pipelines
        cls.lifespan_ctx = cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    # -------------------------------------------------------------------------
    # Gateway & Health Probes
    # -------------------------------------------------------------------------

    def test_root_gateway(self) -> None:
        """Root endpoint must return online service status and documentation endpoints."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("docs", data)
        self.assertIn("OmniIntel", data["service"])

    def test_health_liveness(self) -> None:
        """Liveness probe /health must report healthy status and uptime."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertGreaterEqual(data["uptime_seconds"], 0.0)
        self.assertEqual(data["version"], "1.0.0")

    def test_health_readiness(self) -> None:
        """Readiness probe /health/ready must verify subsystem initialization."""
        res = self.client.get("/health/ready")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["is_ready"])
        self.assertEqual(data["status"], "ready")
        self.assertGreaterEqual(data["knowledge_graph_entities"], 4)
        self.assertGreaterEqual(data["knowledge_graph_triples"], 4)
        self.assertTrue(data["reranker_active"])
        self.assertTrue(data["orchestrator_active"])

    def test_health_telemetry(self) -> None:
        """Telemetry endpoint /health/telemetry must report live telemetry counters."""
        res = self.client.get("/health/telemetry")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_events_dispatched", data)
        self.assertIn("event_types_recorded", data)
        self.assertIn("cache_hit_rate_pct", data)

    # -------------------------------------------------------------------------
    # Middleware Tests
    # -------------------------------------------------------------------------

    def test_middleware_correlation_id_and_timing(self) -> None:
        """Every request must be stamped with X-Correlation-ID and X-Process-Time-Ms."""
        # 1. Automatic UUID generation
        res1 = self.client.get("/health")
        self.assertIn("X-Correlation-ID", res1.headers)
        self.assertIn("X-Process-Time-Ms", res1.headers)

        # 2. Passthrough of client-provided correlation ID
        custom_id = "trace-client-abc-1234"
        res2 = self.client.get("/health", headers={"X-Correlation-ID": custom_id})
        self.assertEqual(res2.headers["X-Correlation-ID"], custom_id)

    # -------------------------------------------------------------------------
    # Knowledge Graph Endpoints
    # -------------------------------------------------------------------------

    def test_graph_stats_and_entity_retrieval(self) -> None:
        """Knowledge Graph inspection endpoints must accurately report topology and entities."""
        # Stats
        res_stats = self.client.get("/api/v1/graph/stats")
        self.assertEqual(res_stats.status_code, 200)
        stats = res_stats.json()
        self.assertGreaterEqual(stats["num_entities"], 4)
        self.assertGreaterEqual(stats["num_triples"], 4)
        self.assertIn("ORGANIZATION", stats["entity_types"])

        # Entities list
        res_entities = self.client.get("/api/v1/graph/entities")
        self.assertEqual(res_entities.status_code, 200)
        entities = res_entities.json()
        self.assertGreaterEqual(len(entities), 4)

        # Filtered by type
        res_orgs = self.client.get("/api/v1/graph/entities?entity_type=ORGANIZATION")
        self.assertEqual(res_orgs.status_code, 200)
        orgs = res_orgs.json()
        for org in orgs:
            self.assertEqual(org["entity_type"], "ORGANIZATION")

        # Entity detail
        res_nvda = self.client.get("/api/v1/graph/entity/nvidia")
        self.assertEqual(res_nvda.status_code, 200)
        nvda = res_nvda.json()
        self.assertEqual(nvda["entity_id"], "nvidia")
        self.assertIn("NVIDIA", nvda["name"])
        self.assertGreaterEqual(len(nvda["incoming_relations"]) + len(nvda["outgoing_relations"]), 1)

    def test_graph_entity_not_found(self) -> None:
        """Querying a non-existent entity must return 404."""
        res = self.client.get("/api/v1/graph/entity/non_existent_corp_123")
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertTrue(data["error"])
        self.assertIn("not found", data["message"])

    def test_graph_dijkstra_path_reasoning(self) -> None:
        """Path reasoning endpoint must find most reliable multi-hop path."""
        res = self.client.get("/api/v1/graph/path?source=asml&target=nvidia")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["found"])
        self.assertEqual(data["hop_count"], 2)
        self.assertGreaterEqual(data["aggregate_confidence"], 0.90)
        self.assertIn("ASML", data["path_explanation"])
        self.assertIn("TSMC", data["path_explanation"])
        self.assertIn("NVIDIA", data["path_explanation"])

    # -------------------------------------------------------------------------
    # Mission & Planning Endpoints
    # -------------------------------------------------------------------------

    def test_mission_planning_only(self) -> None:
        """POST /api/v1/mission/plan must return Kahn DAG schedule without executing research."""
        payload = {"query": "Compare NVIDIA and TSMC gross margins and supply chain reliance."}
        res = self.client.post("/api/v1/mission/plan", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("plan_id", data)
        self.assertGreaterEqual(data["total_tasks"], 3)
        self.assertGreaterEqual(data["total_stages"], 2)
        self.assertIn(data["complexity"], ["LOW", "MEDIUM", "HIGH"])
        self.assertGreaterEqual(len(data["batches"]), 2)

    def test_mission_planning_validation_error(self) -> None:
        """Submitting an empty or too short query must trigger 422 validation failure."""
        res = self.client.post("/api/v1/mission/plan", json={"query": "a"})
        self.assertEqual(res.status_code, 422)
        data = res.json()
        self.assertTrue(data["error"])
        self.assertIn("validation failed", data["message"])

    def test_mission_run_end_to_end(self) -> None:
        """POST /api/v1/mission/run must execute multi-agent mission and return Executive Dossier."""
        payload = {
            "query": "Analyze NVIDIA Blackwell architecture gross margins and TSMC supply chain.",
            "max_replans": 2,
        }
        res = self.client.post("/api/v1/mission/run", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["verdict"], "APPROVED")
        self.assertGreaterEqual(data["confidence_score"], 0.70)
        self.assertGreater(data["total_latency_ms"], 0.0)
        self.assertIn("# Strategic Intelligence Dossier", data["dossier_markdown"])
        self.assertIn("NVIDIA", data["target_entities"])


if __name__ == "__main__":
    unittest.main()
