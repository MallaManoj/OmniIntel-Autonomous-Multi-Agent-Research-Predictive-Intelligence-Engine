"""WebSocket streaming endpoints for real-time multi-agent telemetry."""

import json
import logging
import time
from typing import Optional, Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from src.api.state import get_app_state
from src.api.websocket import get_ws_manager

logger = logging.getLogger("omniintel.api.websocket")

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/events")
async def websocket_events_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None, description="Unique client identifier"),
    replay: bool = Query(True, description="Replay recent historical events upon connection"),
    history_limit: int = Query(50, ge=1, le=500, description="Max history events to replay"),
    filter_types: Optional[str] = Query(
        None, description="Comma-separated event types to filter (e.g. 'AGENT_THOUGHT,TOOL_CALLED')"
    ),
) -> None:
    """Real-time bi-directional WebSocket streaming endpoint for agent telemetry and events.

    Clients receive real-time updates as agents think, tools execute, and the Critic
    validates findings. Supports live subscription filtering and event replay.
    """
    manager = get_ws_manager()
    state = get_app_state()

    # Parse subscription filter set
    filter_set: Optional[Set[str]] = None
    if filter_types:
        parsed = {ft.strip() for ft in filter_types.split(",") if ft.strip()}
        if parsed:
            filter_set = parsed

    cid = await manager.connect(websocket, client_id=client_id, filter_types=filter_set)
    logger.info("WebSocket client connected: %s (active: %d)", cid, manager.active_count)

    # Replay historical events if requested and available
    if replay and state.event_bus:
        history = state.event_bus.history
        if history:
            await manager.replay_history(websocket, history, limit=history_limit)

    try:
        while True:
            raw_text = await websocket.receive_text()
            if not raw_text:
                continue

            # Handle text-only ping
            if raw_text.strip().upper() == "PING":
                await manager.send_json(
                    websocket,
                    {"type": "PONG", "timestamp": time.time(), "client_id": cid},
                )
                continue

            # Parse JSON control packets
            try:
                msg = json.loads(raw_text)
            except json.JSONDecodeError:
                await manager.send_json(
                    websocket,
                    {"type": "ERROR", "message": "Invalid JSON frame received", "timestamp": time.time()},
                )
                continue

            msg_type = str(msg.get("type", "")).upper()
            action = str(msg.get("action", "")).lower()

            # Process control commands
            if msg_type == "PING" or action == "ping":
                await manager.send_json(
                    websocket,
                    {
                        "type": "PONG",
                        "timestamp": time.time(),
                        "echo": msg.get("data"),
                        "client_id": cid,
                    },
                )
            elif action == "update_subscriptions" or msg_type == "UPDATE_SUBSCRIPTIONS":
                new_subs = msg.get("subscriptions", ["*"])
                await manager.update_subscriptions(websocket, new_subs)
            elif action == "get_stats" or msg_type == "GET_STATS":
                await manager.send_json(
                    websocket,
                    {
                        "type": "STATS",
                        "active_connections": manager.active_count,
                        "client_id": cid,
                        "timestamp": time.time(),
                    },
                )
            else:
                await manager.send_json(
                    websocket,
                    {
                        "type": "ACK",
                        "message": f"Command '{msg_type or action}' received",
                        "timestamp": time.time(),
                    },
                )

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info("WebSocket client disconnected normally: %s", cid)
    except Exception as exc:
        logger.warning("WebSocket client %s error: %s", cid, exc)
        await manager.disconnect(websocket)


@router.get("/ws/stats", summary="WebSocket Active Connections and Health")
async def websocket_stats() -> dict:
    """Return active WebSocket connection count and streaming bridge health."""
    manager = get_ws_manager()
    return {
        "status": "online",
        "endpoint": "/ws/events",
        "active_connections": manager.active_count,
        "timestamp": time.time(),
    }
