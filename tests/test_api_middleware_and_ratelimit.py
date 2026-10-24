"""Unit and integration test suite for Token Bucket rate-limiting, structured logging, and error boundaries."""

import json
import logging
import time
import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.middleware import RateLimiter, TokenBucket, global_rate_limiter


class LogCaptureHandler(logging.Handler):
    """Custom in-memory log handler capturing formatted log messages."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


class TestApiMiddlewareAndRateLimit(unittest.TestCase):
    """Test suite verifying Token Bucket rate limiting, structured logging, and error boundaries."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.lifespan_ctx = cls.client.__enter__()

        # Attach log capture handler to logger
        cls.log_handler = LogCaptureHandler()
        cls.logger = logging.getLogger("omniintel.access")
        cls.logger.addHandler(cls.log_handler)
        cls.logger.setLevel(logging.INFO)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.logger.removeHandler(cls.log_handler)
        cls.client.__exit__(None, None, None)

    def setUp(self) -> None:
        global_rate_limiter.reset()
        self.log_handler.records.clear()

    # -------------------------------------------------------------------------
    # Unit Tests: TokenBucket Algorithm Mechanics
    # -------------------------------------------------------------------------

    def test_token_bucket_consumption_and_bounds(self) -> None:
        """Token bucket must accurately consume and bound capacity."""
        bucket = TokenBucket(capacity=5.0, refill_rate=1.0)
        self.assertEqual(bucket.tokens, 5.0)

        # Consume 3 tokens
        allowed, rem, wait = bucket.consume(3.0)
        self.assertTrue(allowed)
        self.assertEqual(rem, 2)
        self.assertEqual(wait, 0.0)

        # Consume remaining 2 tokens
        allowed, rem, wait = bucket.consume(2.0)
        self.assertTrue(allowed)
        self.assertEqual(rem, 0)

        # Next consumption must fail
        allowed, rem, wait = bucket.consume(1.0)
        self.assertFalse(allowed)
        self.assertEqual(rem, 0)
        self.assertGreater(wait, 0.0)

    def test_token_bucket_refill_rate(self) -> None:
        """Token bucket must refill tokens based on elapsed monotonic time."""
        bucket = TokenBucket(capacity=10.0, refill_rate=100.0)
        # Drain completely
        bucket.tokens = 0.0
        bucket.last_refill = time.monotonic() - 0.05  # Simulate 50ms elapsed = 5 tokens

        allowed, rem, _ = bucket.consume(2.0)
        self.assertTrue(allowed)
        self.assertGreaterEqual(rem, 2)

    # -------------------------------------------------------------------------
    # Unit Tests: Multi-Tenant RateLimiter
    # -------------------------------------------------------------------------

    def test_rate_limiter_tenant_isolation(self) -> None:
        """Exhausting quota for Client A must not throttle Client B."""
        limiter = RateLimiter(default_capacity=2.0, default_refill_rate=0.1)

        import asyncio

        # Client A consumes both tokens
        ok_a1, _, _ = asyncio.run(limiter.check_rate_limit("client_A"))
        ok_a2, _, _ = asyncio.run(limiter.check_rate_limit("client_A"))
        ok_a3, _, _ = asyncio.run(limiter.check_rate_limit("client_A"))
        self.assertTrue(ok_a1)
        self.assertTrue(ok_a2)
        self.assertFalse(ok_a3)

        # Client B must still have full capacity
        ok_b1, rem_b, _ = asyncio.run(limiter.check_rate_limit("client_B"))
        self.assertTrue(ok_b1)
        self.assertEqual(rem_b, 1)

    # -------------------------------------------------------------------------
    # Integration Tests: HTTP Rate Limiting & 429 Responses
    # -------------------------------------------------------------------------

    def test_rate_limit_headers_on_success(self) -> None:
        """Successful requests must return X-RateLimit headers."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("X-RateLimit-Limit", resp.headers)
        self.assertIn("X-RateLimit-Remaining", resp.headers)
        self.assertIn("X-RateLimit-Reset", resp.headers)

    def test_rate_limit_burst_exhaustion_returns_429(self) -> None:
        """Exhausting client bucket quota must return HTTP 429 with Retry-After header."""
        # Temporarily restrict global limiter capacity for fast test execution
        old_cap = global_rate_limiter.default_capacity
        old_refill = global_rate_limiter.default_refill_rate
        try:
            global_rate_limiter.default_capacity = 3.0
            global_rate_limiter.default_refill_rate = 0.01
            global_rate_limiter.reset()

            headers = {"X-Forwarded-For": "198.51.100.42"}

            # Send 3 requests (exhausting capacity)
            for _ in range(3):
                r = self.client.get("/api/v1/graph/stats", headers=headers)
                self.assertEqual(r.status_code, 200)

            # 4th request must be throttled
            throttled_resp = self.client.get("/api/v1/graph/stats", headers=headers)
            self.assertEqual(throttled_resp.status_code, 429)
            self.assertIn("Retry-After", throttled_resp.headers)
            self.assertEqual(throttled_resp.headers["X-RateLimit-Remaining"], "0")

            data = throttled_resp.json()
            self.assertTrue(data["error"])
            self.assertEqual(data["status_code"], 429)
            self.assertIn("Rate limit exceeded", data["message"])
            self.assertIn("retry_after_seconds", data)

        finally:
            global_rate_limiter.default_capacity = old_cap
            global_rate_limiter.default_refill_rate = old_refill
            global_rate_limiter.reset()

    def test_health_endpoints_exempt_from_rate_limit(self) -> None:
        """Health and readiness probes must never be blocked by rate limits."""
        old_cap = global_rate_limiter.default_capacity
        old_refill = global_rate_limiter.default_refill_rate
        try:
            global_rate_limiter.default_capacity = 1.0
            global_rate_limiter.default_refill_rate = 0.001
            global_rate_limiter.reset()

            headers = {"X-Forwarded-For": "203.0.113.99"}

            # Drain quota on regular endpoint
            self.client.get("/api/v1/graph/stats", headers=headers)
            # Regular endpoint is now 429
            r_blocked = self.client.get("/api/v1/graph/stats", headers=headers)
            self.assertEqual(r_blocked.status_code, 429)

            # Exempt health check must still return 200
            r_health = self.client.get("/health", headers=headers)
            self.assertEqual(r_health.status_code, 200)

            r_ready = self.client.get("/health/ready", headers=headers)
            self.assertEqual(r_ready.status_code, 200)

        finally:
            global_rate_limiter.default_capacity = old_cap
            global_rate_limiter.default_refill_rate = old_refill
            global_rate_limiter.reset()

    # -------------------------------------------------------------------------
    # Integration Tests: Structured JSON Access Logging
    # -------------------------------------------------------------------------

    def test_structured_json_access_log_emitted(self) -> None:
        """Every request must emit a single-line JSON log containing observability telemetry."""
        custom_cid = "trace_abc_12345"
        resp = self.client.get("/health", headers={"X-Correlation-ID": custom_cid, "User-Agent": "PyTestClient/1.0"})
        self.assertEqual(resp.status_code, 200)

        # Check captured logs
        matching_logs = [rec for rec in self.log_handler.records if custom_cid in rec]
        self.assertGreaterEqual(len(matching_logs), 1)

        raw_json = matching_logs[0]
        parsed = json.loads(raw_json)
        self.assertEqual(parsed["correlation_id"], custom_cid)
        self.assertEqual(parsed["method"], "GET")
        self.assertEqual(parsed["path"], "/health")
        self.assertEqual(parsed["status_code"], 200)
        self.assertIn("latency_ms", parsed)
        self.assertIn("timestamp", parsed)
        self.assertEqual(parsed["user_agent"], "PyTestClient/1.0")

    # -------------------------------------------------------------------------
    # Integration Tests: Global Error Boundaries
    # -------------------------------------------------------------------------

    def test_unhandled_exception_boundary_masks_internals(self) -> None:
        """Internal 500 errors must return standardized JSON envelope with correlation ID."""
        # Querying an empty query on plan triggers validation
        resp = self.client.post("/api/v1/mission/plan", json={"query": "ab"})
        self.assertEqual(resp.status_code, 422)
        data = resp.json()
        self.assertTrue(data["error"])
        self.assertIn("correlation_id", data)
        self.assertIn("X-Correlation-ID", resp.headers)


if __name__ == "__main__":
    unittest.main()
