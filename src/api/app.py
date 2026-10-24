"""FastAPI Application Factory with Lifespan, Middleware, and Router Mounting."""

from contextlib import asynccontextmanager
import time
from typing import Any, AsyncGenerator, Callable
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.middleware import (
    RateLimitMiddleware,
    StructuredLoggingMiddleware,
    global_rate_limiter,
)
from src.api.routes import (
    graph_router,
    health_router,
    mission_router,
    search_router,
    websocket_router,
)
from src.api.state import app_state, initialize_knowledge_base
from src.api.websocket import ws_manager
from src.core.events import Event, EventType


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Modern lifespan context manager managing startup initialization and shutdown cleanup."""
    # --- Startup Lifecycle ---
    start_boot = time.perf_counter()
    state = initialize_knowledge_base()

    # Wire WebSocket broadcast bridge to EventBus
    state.event_bus.subscribe("*", ws_manager.broadcast)

    boot_latency_ms = (time.perf_counter() - start_boot) * 1000.0

    # Publish lifecycle event
    await state.event_bus.publish(
        Event(
            event_type=EventType.SYSTEM_STARTED,
            source="FastAPI_Lifespan",
            payload={
                "version": state.version,
                "boot_latency_ms": round(boot_latency_ms, 2),
                "entities_loaded": state.knowledge_graph.num_entities if state.knowledge_graph else 0,
            },
        )
    )

    yield

    # --- Shutdown Lifecycle ---
    state.event_bus.unsubscribe("*", ws_manager.broadcast)
    state.is_ready = False
    state.event_bus.clear()


def create_app() -> FastAPI:
    """Production application factory creating configured FastAPI service."""
    app = FastAPI(
        title="OmniIntel Autonomous Multi-Agent Predictive Intelligence API",
        description=(
            "Enterprise-grade REST & WebSocket API orchestrating autonomous research agents, "
            "Kahn's algorithm DAG decomposition, dual-channel hybrid RAG (Cross-Encoder), "
            "and closed-loop dynamic self-correction."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # -------------------------------------------------------------------------
    # Middleware Pipeline
    # Execution Order (Onion model):
    #   Request -> StructuredLogging -> CORS -> RateLimit -> Router
    #   Response <- StructuredLogging <- CORS <- RateLimit <- Router
    # -------------------------------------------------------------------------
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Correlation-ID",
            "X-Process-Time-Ms",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "Retry-After",
        ],
    )
    app.add_middleware(StructuredLoggingMiddleware)

    # -------------------------------------------------------------------------
    # Global Exception Handlers & Error Boundaries
    # -------------------------------------------------------------------------
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "status_code": exc.status_code,
                "message": exc.detail,
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        formatted_errors = [
            {"loc": list(err.get("loc", [])), "msg": err.get("msg"), "type": err.get("type")}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": True,
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "message": "Request payload validation failed",
                "details": formatted_errors,
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": True,
                "status_code": status.HTTP_400_BAD_REQUEST,
                "message": str(exc),
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": True,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "Internal server error occurred.",
                "detail": str(exc),
                "correlation_id": correlation_id,
            },
            headers={"X-Correlation-ID": correlation_id},
        )

    # -------------------------------------------------------------------------
    # Route Mounts
    # -------------------------------------------------------------------------
    app.include_router(health_router)
    app.include_router(mission_router)
    app.include_router(graph_router)
    app.include_router(search_router)
    app.include_router(websocket_router)

    # -------------------------------------------------------------------------
    # Root Endpoint
    # -------------------------------------------------------------------------
    @app.get("/", tags=["Root"], summary="Root API Gateway Index")
    async def root_index() -> dict[str, str]:
        return {
            "service": "OmniIntel Autonomous Multi-Agent Predictive Intelligence Engine",
            "version": "1.0.0",
            "status": "online",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
        }

    return app


# Default ASGI application instance for uvicorn
app = create_app()
