"""Live dashboard feed.

Authentication over WebSocket uses a token query parameter, since browsers cannot set
headers on a WebSocket handshake. The token is the same short-lived access token.
"""
import asyncio
import uuid

import jwt as pyjwt
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.db.base import get_session_factory
from app.db.models import User
from app.security.auth import decode_token
from app.services.events import bus

router = APIRouter(tags=["events"])

HEARTBEAT_SECONDS = 25


async def _authenticate(token: str) -> uuid.UUID | None:
    try:
        payload = decode_token(token, expected_type="access")
    except pyjwt.PyJWTError:
        return None
    async with get_session_factory()() as db:
        user = (await db.execute(select(User).where(
            User.id == uuid.UUID(payload["sub"])))).scalar_one_or_none()
        if user is None or not user.is_active:
            return None
        return user.tenant_id


@router.websocket("/ws/dashboard")
async def dashboard_feed(websocket: WebSocket, token: str = Query(...)):
    tenant_id = await _authenticate(token)
    if tenant_id is None:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    queue = bus.subscribe(tenant_id)
    try:
        await websocket.send_json({"event": "connected",
                                   "recent": [e.as_json() for e in bus.recent(tenant_id)]})
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                await websocket.send_json(event.as_json())
            except asyncio.TimeoutError:
                # Keeps proxies from dropping an idle connection.
                await websocket.send_json({"event": "heartbeat"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        bus.unsubscribe(tenant_id, queue)
