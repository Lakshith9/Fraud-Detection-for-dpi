"""FastAPI application: REST + WebSocket for the operator dashboard and pre-settlement scoring.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
OpenAPI docs: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import orjson
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from api import metrics as metrics_mod
from api.auth import operator, ws_operator
from api.hub import Hub
from api.routes import cases, feedback, investigation, scoring
from api.state import STATE
from core import db
from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.redis_client import get_redis
from pipeline.scorer import ModelManager, Scorer
from pipeline.writer import DBWriter

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s.log_level)
    db.init_schema()
    r = get_redis(timeout=5)
    STATE.redis = r
    STATE.writer = DBWriter()
    STATE.scorer = Scorer(r, writer=STATE.writer, models=ModelManager(r), source="api")
    STATE.hub = Hub(get_redis(timeout=5), asyncio.get_running_loop())
    log.info("api started", model=STATE.scorer.models.bundle.version)
    yield
    STATE.hub.stop()
    STATE.writer.stop()
    db.close_pool()


class ORJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return orjson.dumps(content, default=str, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)


app = FastAPI(
    title="DPI Real-Time Fraud Detection Engine",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
    description="Real-time ML + graph fraud scoring for Mojaloop-style payment rails "
    "(simulated Mojaloop-compatible events; not a live Mojaloop integration).",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().api_cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)
for r in (scoring.router, investigation.router, cases.router, feedback.router):
    app.include_router(r)


@app.get("/healthz", tags=["system"], summary="Liveness/readiness of Redis, PostgreSQL and the model")
def healthz():
    out = {"status": "ok", "checks": {}}
    t = time.perf_counter()
    try:
        STATE.redis.ping()
        out["checks"]["redis"] = {"ok": True, "ms": round((time.perf_counter() - t) * 1000, 2)}
    except Exception as e:
        out["checks"]["redis"] = {"ok": False, "error": str(e)[:200]}
    t = time.perf_counter()
    try:
        db.fetch_one("SELECT 1 AS ok")
        out["checks"]["postgres"] = {"ok": True, "ms": round((time.perf_counter() - t) * 1000, 2)}
    except Exception as e:
        out["checks"]["postgres"] = {"ok": False, "error": str(e)[:200]}
    b = STATE.scorer.models.bundle if STATE.scorer else None
    out["checks"]["model"] = {
        "ok": b is not None,
        "version": b.version if b else None,
        "sequence_loaded": bool(b and b.seq is not None),
    }
    if not all(c["ok"] for c in out["checks"].values()):
        out["status"] = "degraded"
    return ORJSONResponse(out, status_code=200 if out["status"] == "ok" else 503)


@app.get("/v1/metrics", tags=["system"], summary="Throughput, latency percentiles, alert rate, drift, feedback")
def metrics(_op: str = Depends(operator)):
    return metrics_mod.collect(STATE.redis, STATE.scorer, STATE.cache)


@app.get("/metrics", tags=["system"], response_class=PlainTextResponse, summary="Prometheus exposition")
def prometheus():
    m = metrics_mod.collect(STATE.redis, STATE.scorer, STATE.cache)
    lines = [
        f"fraud_tps {m['throughput']['tps_last_10s']}",
        f"fraud_alert_rate {m['alert_rate']}",
        f"fraud_degraded_rate {m['degraded_rate']}",
    ]
    for q in ("p50", "p95", "p99"):
        if m["latency_ms"][q] is not None:
            lines.append(f'fraud_score_latency_ms{{quantile="{q}"}} {m["latency_ms"][q]}')
    for k, v in m["counts"].items():
        lines.append(f'fraud_scored_total{{decision="{k}"}} {v}')
    return "\n".join(lines) + "\n"


@app.websocket("/ws/alerts")
async def ws_alerts(ws: WebSocket):
    op = ws_operator(ws)
    if op is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    recent = await asyncio.to_thread(db.fetch_all, "SELECT * FROM alerts ORDER BY alert_id DESC LIMIT 50")
    await ws.send_text(orjson.dumps({"type": "snapshot", "data": recent}, default=str).decode())
    await STATE.hub.register(ws)
    try:
        while True:
            msg = await ws.receive_text()  # keep-alive pings from the client
            if msg == "ping":
                await ws.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        pass
    finally:
        STATE.hub.unregister(ws)
