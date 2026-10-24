"""Asynchronous graph worker: re-detects fraud rings on the trailing 24h EVENT-time window every N seconds.

* reads the window from PostgreSQL (transactions table, incl. bootstrap history)
* runs the NetworkX detectors, keeps ring ids stable, publishes node -> ring score hashes to Redis (g:<node>)
  which the hot scoring path reads with a single pipelined lookup
* upserts rings with score >= 0.5 into PostgreSQL and pushes ring updates to the dashboard (Redis pub/sub)
* optional: mirrors rings into Neo4j when GRAPH_BACKEND=neo4j (graph/neo4j_sink.py)

Usage: python -m graph.worker [--once]
"""

from __future__ import annotations

import argparse
import time

import orjson
import pandas as pd

from core import db
from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.redis_client import get_redis
from graph.rings import RingTracker, run_graph_cycle
from pipeline.state_store import RedisStore

log = get_logger("graph-worker")
RING_CHANNEL = "rings:live"
REPORT_MIN_SCORE = 0.5


def load_window(window_h: float) -> tuple[pd.DataFrame, int]:
    row = db.fetch_one("SELECT max(ts) AS t FROM transactions")
    if not row or row["t"] is None:
        return pd.DataFrame(columns=["ts", "payer_id", "payee_id", "amount", "device_id", "ip"]), 0
    now = int(row["t"])
    rows = db.fetch_all(
        "SELECT ts, payer_id, payee_id, amount, device_id, ip FROM transactions WHERE ts > %s ORDER BY ts",
        (now - int(window_h * 3_600_000),),
    )
    return pd.DataFrame(rows, columns=["ts", "payer_id", "payee_id", "amount", "device_id", "ip"]), now


def persist_rings(rings, r) -> int:
    keep = [x for x in rings if x.score >= REPORT_MIN_SCORE]
    with db.get_pool().connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute("UPDATE rings SET active = FALSE WHERE active")
        for x in keep:
            d = x.to_dict()
            cur.execute(
                """INSERT INTO rings (ring_id, ring_type, ring_score, members, devices, evidence, edges, first_seen,
                                      last_seen, active, updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,now())
                   ON CONFLICT (ring_id) DO UPDATE SET ring_score = EXCLUDED.ring_score, members = EXCLUDED.members,
                     devices = EXCLUDED.devices, evidence = EXCLUDED.evidence, edges = EXCLUDED.edges,
                     first_seen = LEAST(rings.first_seen, EXCLUDED.first_seen), last_seen = EXCLUDED.last_seen,
                     active = TRUE, updated_at = now()""",
                (
                    d["ring_id"],
                    d["ring_type"],
                    d["ring_score"],
                    db.Jsonb(d["members"]),
                    db.Jsonb(d["devices"]),
                    db.Jsonb(d["evidence"]),
                    db.Jsonb(d["edges"]),
                    d["first_seen"],
                    d["last_seen"],
                ),
            )
    for x in keep:
        try:
            r.publish(RING_CHANNEL, orjson.dumps({k: v for k, v in x.to_dict().items() if k != "edges"}))
        except Exception:
            pass
    return len(keep)


def run_once(r, tracker: RingTracker, prev_nodes: set[str] | None, sink=None) -> tuple[set[str], dict]:
    s = get_settings()
    t = time.perf_counter()
    df, now = load_window(s.graph_window_hours)
    reported = set(r.smembers("reported") or ())
    store = RedisStore(r)
    rings, nodes = run_graph_cycle(df, reported, now, store, tracker, prev_nodes)
    r.hset("g:meta", mapping={"wall": f"{time.time():.3f}", "window_rows": len(df)})
    n_pub = persist_rings(rings, r)
    if sink is not None:
        sink.write(rings)
    stats = {
        "rows": len(df),
        "rings": len(rings),
        "reported_rings": n_pub,
        "nodes": len(nodes),
        "ms": round((time.perf_counter() - t) * 1000, 1),
        "event_now": now,
    }
    return nodes, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    s = get_settings()
    setup_logging(s.log_level)
    db.init_schema()
    r = get_redis(timeout=10)
    sink = None
    if s.graph_backend == "neo4j":
        from graph.neo4j_sink import Neo4jSink

        sink = Neo4jSink(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
    tracker = RingTracker()
    nodes = None
    while True:
        try:
            nodes, stats = run_once(r, tracker, nodes, sink)
            log.info("graph cycle", **stats)
        except Exception as e:
            log.error("graph cycle failed; scorers fall back to non-graph combo when stale", error=str(e)[:300])
        if a.once:
            break
        time.sleep(s.graph_interval_s)


if __name__ == "__main__":
    main()
