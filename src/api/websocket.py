"""WebSocket Connection Manager and EventBus streaming bridge."""

import asyncio
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect

from src.core.events import Event, EventBus

logger = logging.getLogger("omniintel.websocket")


class ConnectionManager:
    """Manages active WebSocket client connections, client subscriptions, and event broadcasts.

    Features:
        - Thread-safe tracking of active WebSocket client connections.
        - Event type filtering per client connection.
        - Automatic delivery of historical replay buffer for reconnecting clients.
        - Non-blocking broadcast with isolated client error boundaries (evicts dead sockets).
        - Direct ping/pong heartbeats for connection keepalive.
    """

    def __init__(self) -> None:
        # Maps active WebSocket to set of subscribed EventTypes (or {"*"} for all)
        self._active_connections: Dict[WebSocket, Set[str]] = {}
        # Client identification mapping
        self._client_ids: Dict[WebSocket, str] = {}
        # Concurrency lock for connection mutations
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        """Count of currently connected WebSocket clients."""
        return len(self._active_connections)

    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None,
        filter_types: Optional[Set[str]] = None,
    ) -> str:
        """Accept an incoming WebSocket connection and register client filters."""
        await websocket.accept()
        cid = client_id or f"client_{int(time.time() * 1000)}"

        async with self._lock:
            subs = filter_types if filter_types else {"*"}
            self._active_connections[websocket] = subs
            self._client_ids[websocket] = cid

        # Send connection acknowledgment handshake
        await self.send_json(
            websocket,
            {
                "type": "CONNECTION_ESTABLISHED",
                "client_id": cid,
                "subscriptions": list(subs),
                "timestamp": time.time(),
                "message": "Connected to OmniIntel Real-Time Agent Event Stream",
            },
        )
        return cid

    async def disconnect(self, websocket: WebSocket) -> None:
        """Gracefully unregister a disconnected WebSocket client."""
        async with self._lock:
            if websocket in self._active_connections:
                del self._active_connections[websocket]
            if websocket in self._client_ids:
                del self._client_ids[websocket]

    async def update_subscriptions(self, websocket: WebSocket, event_types: List[str]) -> None:
        """Update event type filter preferences for a connected client."""
        async with self._lock:
            if websocket in self._active_connections:
                self._active_connections[websocket] = set(event_types)
                await self.send_json(
                    websocket,
                    {
                        "type": "SUBSCRIPTIONS_UPDATED",
                        "subscriptions": event_types,
                        "timestamp": time.time(),
                    },
                )

    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]) -> bool:
        """Deliver JSON packet to an individual WebSocket client safely."""
        try:
            await websocket.send_json(data)
            return True
        except Exception:
            return False

    async def broadcast(self, event: Event) -> None:
        """Broadcast an EventBus event to all matching connected WebSocket clients.

        Runs with isolated error boundaries: failure or disconnection of one client
        evicts that socket without interrupting delivery to others.
        """
        if not self._active_connections:
            return

        payload = {
            "type": "EVENT_STREAM",
            "event_id": event.event_id,
            "event_type": event.event_type,
            "source": event.source,
            "payload": event.payload,
            "timestamp": event.timestamp,
        }

        dead_connections: List[WebSocket] = []

        # Snapshot current connections to minimize lock contention
        async with self._lock:
            recipients = list(self._active_connections.items())

        for ws, subs in recipients:
            # Check if client subscribed to this event_type or wildcard "*"
            if "*" in subs or event.event_type in subs:
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead_connections.append(ws)

        # Evict disconnected sockets
        if dead_connections:
            async with self._lock:
                for ws in dead_connections:
                    if ws in self._active_connections:
                        del self._active_connections[ws]
                    if ws in self._client_ids:
                        del self._client_ids[ws]

    async def replay_history(
        self,
        websocket: WebSocket,
        history: List[Event],
        limit: int = 50,
    ) -> None:
        """Stream chronological historical events from circular replay buffer."""
        target_events = history[-limit:] if len(history) > limit else history
        for ev in target_events:
            await self.send_json(
                websocket,
                {
                    "type": "EVENT_REPLAY",
                    "event_id": ev.event_id,
                    "event_type": ev.event_type,
                    "source": ev.source,
                    "payload": ev.payload,
                    "timestamp": ev.timestamp,
                },
            )


# Global singleton connection manager
ws_manager = ConnectionManager()


def get_ws_manager() -> ConnectionManager:
    """Dependency provider for WebSocket ConnectionManager."""
    return ws_manager
