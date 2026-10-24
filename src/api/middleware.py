"""Production ASGI Middleware: Token Bucket Rate-Limiting and Structured JSON Access Logging."""

import asyncio
from datetime import datetime, timezone
import json
import logging
import math
import time
from typing import Any, Callable, Dict, Optional, Set
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = logging.getLogger("omniintel.access")


# =============================================================================
# 1. Token Bucket Rate Limiting Engine
# =============================================================================

class TokenBucket:
    """Thread-safe Token Bucket rate-limiter algorithm implementation.

    Attributes:
        capacity: Maximum token burst size allowed.
        refill_rate: Number of tokens replenished per second.
        tokens: Current available token balance.
        last_refill: Timestamp of the most recent token replenishment.
    """

    __slots__ = ("capacity", "refill_rate", "tokens", "last_refill")

    def __init__(self, capacity: float = 30.0, refill_rate: float = 10.0) -> None:
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def consume(self, amount: float = 1.0) -> tuple[bool, int, float]:
        """Attempt to consume tokens from the bucket.

        Returns:
            Tuple of:
                - allowed (bool): True if tokens were available and deducted.
                - remaining (int): Integer count of tokens remaining in bucket.
                - reset_time_seconds (float): Estimated seconds until next token is available.
        """
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.last_refill = now

        # Replenish tokens proportional to elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

        if self.tokens >= amount:
            self.tokens -= amount
            remaining = int(self.tokens)
            return True, remaining, 0.0

        # Bucket exhausted: calculate wait time for required tokens
        deficit = amount - self.tokens
        reset_time = deficit / self.refill_rate if self.refill_rate > 0 else 1.0
        return False, 0, reset_time


class RateLimiter:
    """Multi-tenant IP and API-Key rate limiter with automated stale bucket eviction."""

    def __init__(
        self,
        default_capacity: float = 30.0,
        default_refill_rate: float = 10.0,
        max_tracked_clients: int = 10000,
    ) -> None:
        self.default_capacity = default_capacity
        self.default_refill_rate = default_refill_rate
        self.max_tracked_clients = max_tracked_clients
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self.exempt_paths: Set[str] = {
            "/health",
            "/health/ready",
            "/health/telemetry",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/ws/stats",
        }

    async def check_rate_limit(self, client_key: str) -> tuple[bool, int, float]:
        """Thread-safe rate limit evaluation for a specific client identifier."""
        async with self._lock:
            if client_key not in self._buckets:
                # Evict oldest entry if tracking table exceeds memory safety threshold
                if len(self._buckets) >= self.max_tracked_clients:
                    oldest_key = next(iter(self._buckets))
                    del self._buckets[oldest_key]

                self._buckets[client_key] = TokenBucket(
                    capacity=self.default_capacity,
                    refill_rate=self.default_refill_rate,
                )

            bucket = self._buckets[client_key]
            return bucket.consume(1.0)

    def reset(self) -> None:
        """Clear all client tracking buckets."""
        self._buckets.clear()


# Global rate limiter instance
global_rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI Middleware enforcing per-client token bucket rate limiting on HTTP routes."""

    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        super().__init__(app)
        self.limiter = rate_limiter or global_rate_limiter

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Exempt health probes and documentation schemas
        path = request.url.path
        if path in self.limiter.exempt_paths:
            return await call_next(request)

        # Extract client IP address or forwarding header
        forwarded = request.headers.get("X-Forwarded-For")
        client_ip = forwarded.split(",")[0].strip() if forwarded else (
            request.client.host if request.client else "127.0.0.1"
        )
        api_key = request.headers.get("X-API-Key")
        client_key = f"{client_ip}:{api_key}" if api_key else client_ip

        allowed, remaining, reset_time = await self.limiter.check_rate_limit(client_key)
        correlation_id = getattr(request.state, "correlation_id", None) or request.headers.get("X-Correlation-ID") or str(uuid.uuid4())

        if not allowed:
            retry_after_ceil = max(1, math.ceil(reset_time))
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "status_code": 429,
                    "message": "Rate limit exceeded. Too many requests.",
                    "retry_after_seconds": round(reset_time, 2),
                    "correlation_id": correlation_id,
                },
                headers={
                    "Retry-After": str(retry_after_ceil),
                    "X-RateLimit-Limit": str(int(self.limiter.default_capacity)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": f"{reset_time:.2f}",
                    "X-Correlation-ID": correlation_id,
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(int(self.limiter.default_capacity))
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = "0.00"
        return response


# =============================================================================
# 2. Structured JSON Access Logging Middleware
# =============================================================================

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """ASGI Middleware emitting single-line structured JSON access logs to standard logger."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()

        # Correlation ID management
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        client_ip = request.headers.get("X-Forwarded-For") or (
            request.client.host if request.client else "unknown"
        )
        method = request.method
        path = request.url.path
        query = str(request.query_params) if request.query_params else None
        user_agent = request.headers.get("User-Agent", "unknown")

        try:
            response = await call_next(request)
            process_time_ms = (time.perf_counter() - start_time) * 1000.0

            # Attach standard observability headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

            # Emit structured JSON audit log entry
            log_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO" if response.status_code < 400 else "WARNING",
                "correlation_id": correlation_id,
                "method": method,
                "path": path,
                "query": query,
                "status_code": response.status_code,
                "latency_ms": round(process_time_ms, 2),
                "client_ip": client_ip,
                "user_agent": user_agent,
            }
            logger.info(json.dumps(log_record))
            return response

        except Exception as exc:
            process_time_ms = (time.perf_counter() - start_time) * 1000.0
            error_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "ERROR",
                "correlation_id": correlation_id,
                "method": method,
                "path": path,
                "query": query,
                "status_code": 500,
                "latency_ms": round(process_time_ms, 2),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            logger.error(json.dumps(error_record))
            raise exc
