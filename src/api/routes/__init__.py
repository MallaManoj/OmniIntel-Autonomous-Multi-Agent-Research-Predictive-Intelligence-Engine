"""FastAPI modular router exports."""

from src.api.routes.graph import router as graph_router
from src.api.routes.health import router as health_router
from src.api.routes.mission import router as mission_router
from src.api.routes.search import router as search_router
from src.api.routes.ui import router as ui_router
from src.api.routes.websocket import router as websocket_router

__all__ = ["graph_router", "health_router", "mission_router", "search_router", "ui_router", "websocket_router"]
