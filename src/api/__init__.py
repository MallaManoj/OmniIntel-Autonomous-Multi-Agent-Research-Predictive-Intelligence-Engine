"""FastAPI service, routes, middleware, and application lifecycle."""

from src.api.app import app, create_app
from src.api.state import AppState, app_state, get_app_state

__all__ = ["AppState", "app", "app_state", "create_app", "get_app_state"]
