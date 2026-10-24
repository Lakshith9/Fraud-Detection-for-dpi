"""WebSocket fan-out: a background thread subscribes to Redis pub/sub and forwards messages to all clients."""

from __future__ import annotations

import asyncio
import threading
import time

import orjson
import redis
from fastapi import WebSocket

from core.logging import get_logger
from graph.worker import RING_CHANNEL
from pipeline.scorer import ALERT_CHANNEL

log = get_logger("ws-hub")


class Hub:
    def __init__(self, r: redis.Redis, loop: asyncio.AbstractEventLoop):
        self.r = r
        self.loop = loop
        self.clients: set[WebSocket] = set()
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._listen, name="pubsub", daemon=True)
        self.t.start()

    async def register(self, ws: WebSocket):
        self.clients.add(ws)

    def unregister(self, ws: WebSocket):
        self.clients.discard(ws)

    async def _broadcast(self, text: str):
        dead = []
        for ws in list(self.clients):
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=2)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def _listen(self):
        while not self._stop.is_set():
            try:
                ps = self.r.pubsub(ignore_subscribe_messages=True)
                ps.subscribe(ALERT_CHANNEL, RING_CHANNEL)
                while not self._stop.is_set():
                    msg = ps.get_message(timeout=1.0)
                    if not msg or not self.clients:
                        continue
                    kind = "alert" if msg["channel"] == ALERT_CHANNEL else "ring"
                    data = msg["data"]
                    text = orjson.dumps({"type": kind, "data": orjson.loads(data)}).decode()
                    asyncio.run_coroutine_threadsafe(self._broadcast(text), self.loop)
            except Exception as e:
                log.error("pubsub listener error; reconnecting", error=str(e)[:200])
                time.sleep(1)

    def stop(self):
        self._stop.set()
