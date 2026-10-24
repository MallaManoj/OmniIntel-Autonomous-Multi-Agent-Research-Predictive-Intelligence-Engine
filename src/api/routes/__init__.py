"""FastAPI modular router exports."""

from src.api.routes.graph import router as graph_router
from src.api.routes.health import router as health_router
from src.api.routes.mission import router as mission_router
from src.api.routes.websocket import router as websocket_router

__all__ = ["graph_router", "health_router", "mission_router", "websocket_router"]
