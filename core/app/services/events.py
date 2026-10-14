"""In-process event bus for dashboard live updates.

Publishers call emit(); WebSocket connections subscribe. Events are also persisted as
analytics_events so the dashboard can backfill on connect and KPIs have a source.

Scope note: this is per-process. With multiple core replicas each serves the dashboards
connected to it; a Redis pub/sub fan-out is the drop-in upgrade and touches only this file.
"""
import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

MAX_QUEUE = 100
RECENT_BUFFER = 50


@dataclass
class Event:
    event_type: str
    tenant_id: uuid.UUID
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_json(self) -> dict:
        return {"event": self.event_type, "at": self.occurred_at.isoformat(),
                **self.payload}


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[uuid.UUID, set[asyncio.Queue]] = {}
        self._recent: dict[uuid.UUID, deque] = {}

    def subscribe(self, tenant_id: uuid.UUID) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE)
        self._subscribers.setdefault(tenant_id, set()).add(q)
        return q

    def unsubscribe(self, tenant_id: uuid.UUID, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(tenant_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._subscribers.pop(tenant_id, None)

    def emit(self, event: Event) -> None:
        self._recent.setdefault(event.tenant_id, deque(maxlen=RECENT_BUFFER)).append(event)
        for q in list(self._subscribers.get(event.tenant_id, ())):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # A slow dashboard must never block the engine; it will resync on reconnect.
                pass

    def recent(self, tenant_id: uuid.UUID, limit: int = 20) -> list[Event]:
        return list(self._recent.get(tenant_id, ()))[-limit:]

    def subscriber_count(self, tenant_id: uuid.UUID) -> int:
        return len(self._subscribers.get(tenant_id, ()))


bus = EventBus()


def publish(tenant_id: uuid.UUID, event_type: str, **payload: Any) -> None:
    bus.emit(Event(event_type=event_type, tenant_id=tenant_id, payload=payload))
