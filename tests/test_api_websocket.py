"""Unit and integration test suite for OmniIntel WebSocket live streaming bridge (/ws/events)."""

import asyncio
import time
import unittest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.state import app_state
from src.api.websocket import ws_manager
from src.core.events import Event, EventType


class TestWebSocketStreaming(unittest.TestCase):
    """Test suite covering the real-time WebSocket event streaming bridge."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.client = TestClient(cls.app)
        cls.lifespan_ctx = cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_websocket_stats_rest_endpoint(self) -> None:
        """Verify REST telemetry endpoint reporting WebSocket status."""
        response = self.client.get("/ws/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["endpoint"], "/ws/events")
        self.assertIn("active_connections", data)

    def test_websocket_connection_handshake(self) -> None:
        """Verify connection handshake sends CONNECTION_ESTABLISHED packet."""
        with self.client.websocket_connect("/ws/events?client_id=test_client_001") as ws:
            ack = ws.receive_json()
            self.assertEqual(ack["type"], "CONNECTION_ESTABLISHED")
            self.assertEqual(ack["client_id"], "test_client_001")
            self.assertEqual(ack["subscriptions"], ["*"])
            self.assertIn("timestamp", ack)

    def test_websocket_ping_pong_protocol(self) -> None:
        """Verify bi-directional ping/pong heartbeat handling."""
        with self.client.websocket_connect("/ws/events?client_id=ping_client&replay=false") as ws:
            # Drain handshake
            _ = ws.receive_json()

            # Test JSON Ping
            ws.send_json({"type": "PING", "data": "heartbeat_probe"})
            resp = ws.receive_json()
            self.assertEqual(resp["type"], "PONG")
            self.assertEqual(resp["echo"], "heartbeat_probe")
            self.assertEqual(resp["client_id"], "ping_client")

            # Test plain-text Ping
            ws.send_text("PING")
            resp2 = ws.receive_json()
            self.assertEqual(resp2["type"], "PONG")
            self.assertEqual(resp2["client_id"], "ping_client")

    def test_websocket_control_commands(self) -> None:
        """Verify control commands like get_stats, subscription updates, and error handling."""
        with self.client.websocket_connect("/ws/events?client_id=cmd_client&replay=false") as ws:
            _ = ws.receive_json()  # Handshake

            # Request stats
            ws.send_json({"action": "get_stats"})
            stats_msg = ws.receive_json()
            self.assertEqual(stats_msg["type"], "STATS")
            self.assertGreaterEqual(stats_msg["active_connections"], 1)

            # Update subscriptions
            ws.send_json({
                "action": "update_subscriptions",
                "subscriptions": ["AGENT_THOUGHT", "TOOL_CALLED"],
            })
            sub_msg = ws.receive_json()
            self.assertEqual(sub_msg["type"], "SUBSCRIPTIONS_UPDATED")
            self.assertEqual(set(sub_msg["subscriptions"]), {"AGENT_THOUGHT", "TOOL_CALLED"})

            # Send malformed JSON
            ws.send_text("INVALID_NON_JSON")
            err_msg = ws.receive_json()
            self.assertEqual(err_msg["type"], "ERROR")
            self.assertIn("Invalid JSON", err_msg["message"])

    def test_websocket_history_replay_toggle(self) -> None:
        """Verify historical events are replayed on connection when requested."""
        # Seed an event into event_bus history
        test_event = Event(
            event_type=EventType.AGENT_THOUGHT,
            source="ReplaySeed",
            payload={"thought": "Pre-existing test event"},
        )
        asyncio.run(app_state.event_bus.publish(test_event))

        # Connect with replay=true
        with self.client.websocket_connect("/ws/events?client_id=replay_on&replay=true&history_limit=5") as ws:
            ack = ws.receive_json()
            self.assertEqual(ack["type"], "CONNECTION_ESTABLISHED")

            # Should receive replay event
            replay_packet = ws.receive_json()
            self.assertIn(replay_packet["type"], ("EVENT_REPLAY", "EVENT_STREAM"))
            self.assertIn("event_id", replay_packet)

        # Connect with replay=false
        with self.client.websocket_connect("/ws/events?client_id=replay_off&replay=false") as ws:
            ack = ws.receive_json()
            self.assertEqual(ack["type"], "CONNECTION_ESTABLISHED")
            # Send ping to confirm connection is idle with no unrequested replay packets
            ws.send_text("PING")
            pong = ws.receive_json()
            self.assertEqual(pong["type"], "PONG")

    def test_websocket_live_event_broadcasting(self) -> None:
        """Verify that EventBus broadcasts live events directly to connected WebSockets."""
        with self.client.websocket_connect("/ws/events?client_id=live_receiver&replay=false") as ws:
            ack = ws.receive_json()
            self.assertEqual(ack["type"], "CONNECTION_ESTABLISHED")

            # Publish event through app_state event_bus
            live_event = Event(
                event_type=EventType.CRITIC_VALIDATION,
                source="CriticAgent",
                payload={"passed": True, "verdict": "Ground truth matched", "confidence": 0.98},
            )
            asyncio.run(app_state.event_bus.publish(live_event))

            # Receive streamed event over websocket
            msg = ws.receive_json()
            self.assertEqual(msg["type"], "EVENT_STREAM")
            self.assertEqual(msg["event_type"], EventType.CRITIC_VALIDATION.value)
            self.assertEqual(msg["source"], "CriticAgent")
            self.assertTrue(msg["payload"]["passed"])
            self.assertEqual(msg["payload"]["confidence"], 0.98)

    def test_websocket_event_type_filtering(self) -> None:
        """Verify client filter excludes events they didn't subscribe to."""
        # Connect subscribed only to CRITIC_REJECTED
        with self.client.websocket_connect(
            "/ws/events?client_id=filtered_client&replay=false&filter_types=CRITIC_REJECTED"
        ) as ws:
            ack = ws.receive_json()
            self.assertEqual(ack["type"], "CONNECTION_ESTABLISHED")
            self.assertEqual(ack["subscriptions"], ["CRITIC_REJECTED"])

            # Publish AGENT_THOUGHT (should be ignored by this client)
            ignored_event = Event(
                event_type=EventType.AGENT_THOUGHT,
                source="ResearchAgent",
                payload={"thought": "ignored event"},
            )
            asyncio.run(app_state.event_bus.publish(ignored_event))

            # Publish CRITIC_REJECTED (should be received)
            accepted_event = Event(
                event_type=EventType.CRITIC_REJECTED,
                source="CriticAgent",
                payload={"reason": "Hallucination detected in claim"},
            )
            asyncio.run(app_state.event_bus.publish(accepted_event))

            msg = ws.receive_json()
            self.assertEqual(msg["type"], "EVENT_STREAM")
            self.assertEqual(msg["event_type"], EventType.CRITIC_REJECTED.value)
            self.assertEqual(msg["payload"]["reason"], "Hallucination detected in claim")

    def test_websocket_disconnect_eviction(self) -> None:
        """Verify connection manager cleans up disconnected client sockets."""
        initial_count = ws_manager.active_count
        with self.client.websocket_connect("/ws/events?client_id=temp_client&replay=false") as ws:
            _ = ws.receive_json()
            self.assertEqual(ws_manager.active_count, initial_count + 1)

        # After exiting context manager, client is disconnected
        time.sleep(0.05)
        self.assertEqual(ws_manager.active_count, initial_count)


if __name__ == "__main__":
    unittest.main()
