from __future__ import annotations

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.simulation_service import service

logger = logging.getLogger("api_websocket")
router = APIRouter()


@router.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket):
    """Real-time live telemetry stream.

    Streams state updates upon each simulation step with zero polling overhead.
    """
    await ws.accept()
    service.clients.add(ws)
    try:
        # Send initial full state immediately upon connect
        if service.latest_telemetry:
            await ws.send_json({"event_type": "SNAPSHOT", "payload": service.latest_telemetry})

        while True:
            # Keep-alive receive loop
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WebSocket client disconnected or error: %s", exc)
    finally:
        service.clients.discard(ws)
