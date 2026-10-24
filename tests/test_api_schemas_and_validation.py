"""Unit and integration tests for Pydantic v2 schemas, strict validation contracts, and OpenAPI generation."""

import unittest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.app import create_app
from src.api.schemas import (
    HybridSearchRequest,
    MissionRequest,
    PathQueryRequest,
    RerankCandidateItem,
    RerankRequest,
    SubgraphRequest,
)


class TestApiSchemasAndValidation(unittest.TestCase):
    """Test suite covering Pydantic v2 schema constraints, zero-trust contracts, and endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.lifespan_ctx = cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    # -------------------------------------------------------------------------
    # Unit Tests: Pydantic v2 Zero-Trust Validation Constraints
    # -------------------------------------------------------------------------

    def test_extra_fields_forbidden(self) -> None:
        """Extra or misspelled parameters must trigger strict ValidationError."""
        with self.assertRaises(ValidationError) as ctx:
            MissionRequest(
                query="Analyze semiconductor gross margins",
                rogue_injected_field="malicious_payload",  # type: ignore[call-arg]
            )
        errors = ctx.exception.errors()
        self.assertTrue(any(e["type"] == "extra_forbidden" for e in errors))

    def test_whitespace_stripping_and_sanitization(self) -> None:
        """String fields must automatically trim whitespace and reject control characters."""
        req = MissionRequest(query="   Analyze TSMC CoWoS capacity   ")
        self.assertEqual(req.query, "Analyze TSMC CoWoS capacity")

        # Prohibited control character \x00
        with self.assertRaises(ValidationError):
            MissionRequest(query="Analyze NVIDIA \x00 margin")

        # Whitespace-only string must fail min_length validator
        with self.assertRaises(ValidationError):
            MissionRequest(query="      ")

    def test_range_constraints(self) -> None:
        """Numeric fields outside permitted ranges must be rejected."""
        # max_replans > 5
        with self.assertRaises(ValidationError):
            MissionRequest(query="Valid query text", max_replans=10)

        # alpha > 1.0 or < 0.0
        with self.assertRaises(ValidationError):
            HybridSearchRequest(query="Valid query text", alpha=1.5)

        with self.assertRaises(ValidationError):
            HybridSearchRequest(query="Valid query text", alpha=-0.1)

        # Subgraph max_hops > 4
        with self.assertRaises(ValidationError):
            SubgraphRequest(root_entity_id="nvidia", max_hops=10)

    def test_cross_field_model_validator(self) -> None:
        """PathQueryRequest must reject trivial identical source and target self-loops."""
        with self.assertRaises(ValidationError) as ctx:
            PathQueryRequest(source_entity="nvidia", target_entity="nvidia")
        self.assertIn("cannot be identical", str(ctx.exception))

        # Different entities must succeed
        valid_path = PathQueryRequest(source_entity="nvidia", target_entity="tsmc")
        self.assertEqual(valid_path.source_entity, "nvidia")
        self.assertEqual(valid_path.target_entity, "tsmc")

    # -------------------------------------------------------------------------
    # OpenAPI Specification Integrity
    # -------------------------------------------------------------------------

    def test_openapi_schema_generation(self) -> None:
        """OpenAPI schema generation must produce compliant specification with all schemas."""
        schema = self.app.openapi()
        self.assertIn("openapi", schema)
        self.assertEqual(schema["info"]["title"], "OmniIntel Autonomous Multi-Agent Predictive Intelligence API")

        components = schema.get("components", {}).get("schemas", {})
        expected_models = [
            "MissionRequest",
            "MissionResponse",
            "HybridSearchRequest",
            "HybridSearchResponse",
            "RerankRequest",
            "RerankResponse",
            "EntityScanRequest",
            "EntityScanResponse",
            "GraphStatsResponse",
            "PathQueryResponse",
            "SubgraphRequest",
            "SubgraphResponse",
        ]
        for model in expected_models:
            self.assertIn(model, components, f"Missing {model} in OpenAPI components")

    # -------------------------------------------------------------------------
    # REST Integration: Search & Neural Rerank Endpoints
    # -------------------------------------------------------------------------

    def test_search_hybrid_valid_request(self) -> None:
        """POST /api/v1/search/hybrid should execute dual-channel search and Cross-Encoder."""
        payload = {
            "query": "TSMC 3nm packaging capacity",
            "top_k": 3,
            "alpha": 0.5,
            "fusion_mode": "rrf",
            "use_reranker": True,
        }
        resp = self.client.post("/api/v1/search/hybrid", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["query"], "TSMC 3nm packaging capacity")
        self.assertGreater(data["total_hits"], 0)
        self.assertTrue(data["cross_encoder_reranked"])
        self.assertIn("results", data)
        self.assertIn("score", data["results"][0])

    def test_search_hybrid_extra_field_rejection(self) -> None:
        """POST /api/v1/search/hybrid must return 422 if client submits undeclared fields."""
        payload = {
            "query": "Valid query text",
            "unexpected_parameter": "should_fail",
        }
        resp = self.client.post("/api/v1/search/hybrid", json=payload)
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertTrue(data["error"])
        self.assertEqual(data["status_code"], 422)

    def test_search_rerank_endpoint(self) -> None:
        """POST /api/v1/search/rerank should score and rank passages using Cross-Encoder."""
        payload = {
            "query": "Who manufactures cutting-edge EUV lithography?",
            "candidates": [
                {"doc_id": "c1", "text": "ASML Holding designs extreme ultraviolet lithography systems."},
                {"doc_id": "c2", "text": "AMD produces Instinct series datacenter accelerators."},
            ],
            "return_top_k": 2,
        }
        resp = self.client.post("/api/v1/search/rerank", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_candidates"], 2)
        self.assertEqual(len(data["ranked_results"]), 2)
        # ASML document should rank #1
        self.assertEqual(data["ranked_results"][0]["doc_id"], "c1")
        self.assertEqual(data["ranked_results"][0]["rank"], 1)

    def test_search_scan_entities_endpoint(self) -> None:
        """POST /api/v1/search/scan-entities should scan arbitrary text via Prefix Trie."""
        payload = {"text": "NVIDIA and TSMC are leading semiconductor innovation."}
        resp = self.client.post("/api/v1/search/scan-entities", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["total_matches"], 1)
        matched_texts = [m["matched_text"].lower() for m in data["matches"]]
        self.assertTrue(any("nvidia" in m or "tsmc" in m for m in matched_texts))

    def test_graph_subgraph_extraction_endpoint(self) -> None:
        """POST /api/v1/graph/subgraph should extract egocentric subgraph around root entity."""
        payload = {
            "root_entity_id": "nvidia",
            "max_hops": 2,
            "min_confidence": 0.5,
        }
        resp = self.client.post("/api/v1/graph/subgraph", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["root_entity_id"], "nvidia")
        self.assertGreaterEqual(data["total_entities"], 2)
        self.assertGreaterEqual(data["total_triples"], 1)
        self.assertEqual(data["hop_depth"], 2)

    def test_graph_subgraph_not_found(self) -> None:
        """POST /api/v1/graph/subgraph must return 404 for unknown root entity."""
        payload = {
            "root_entity_id": "non_existent_corporation",
            "max_hops": 1,
        }
        resp = self.client.post("/api/v1/graph/subgraph", json=payload)
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
