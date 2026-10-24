"""System / model health metrics: throughput, latency percentiles, alert rate, drift (PSI), feedback counts."""

from __future__ import annotations

import time

import numpy as np

from core import db
from core.config import get_settings
from models.metrics import psi
from pipeline.scorer import COUNT_KEY, E2E_KEY, LAT_KEY, PERSIST_STREAM


def _pcts(vals: list[str]) -> dict:
    if not vals:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "max": None}
    a = np.asarray([float(v) for v in vals])
    return {
        "n": int(len(a)),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
    }


# Cyclical time-of-day is excluded: a recent-traffic sample always covers only a few hours of the day, so its PSI
# against the full-day training distribution is meaningless (it would need a same-hour baseline).
PSI_EXCLUDE = {"hour_of_day"}


def drift(bundle, cache: dict, n: int = 20_000) -> dict:
    hit = cache.get("drift")
    if hit and time.time() - hit[0] < 30:
        return hit[1]
    ref = bundle.meta.get("psi_reference", {})
    rows = db.fetch_all(
        """SELECT features, components FROM transactions
           WHERE features IS NOT NULL ORDER BY scored_at DESC LIMIT %s""",
        (n,),
    )
    out = {"n": len(rows), "features": {}}
    if rows:
        for f, r in ref.items():
            if f in PSI_EXCLUDE:
                continue
            if f == "__score__":
                m = np.array([(x["components"] or {}).get("xgb_margin", np.nan) for x in rows], dtype=float)
                vals = 1 / (1 + np.exp(-m))
                name = "xgb_score"
            else:
                vals = np.array([(x["features"] or {}).get(f) for x in rows], dtype=float)
                name = f
            v = psi(np.array(r["props"]), vals, np.array(r["edges"]))
            out["features"][name] = {
                "psi": round(v, 4),
                "status": "stable" if v < 0.1 else "moderate" if v < 0.25 else "shift",
            }
        out["max_psi"] = max((x["psi"] for x in out["features"].values()), default=0.0)
    cache["drift"] = (time.time(), out)
    return out


def collect(r, scorer, cache: dict) -> dict:
    s = get_settings()
    now = int(time.time())
    pipe = r.pipeline(transaction=False)
    pipe.lrange(LAT_KEY, 0, 4999)
    pipe.lrange(E2E_KEY, 0, 4999)
    pipe.hgetall(COUNT_KEY)
    pipe.mget([f"metrics:tps:{t}" for t in range(now - 60, now)])
    pipe.hgetall("g:meta")
    for p in range(s.stream_partitions):
        pipe.hgetall(f"worker:{p}")
        pipe.xlen(f"{s.stream_prefix}:{p}")
        pipe.xpending(f"{s.stream_prefix}:{p}", s.consumer_group)
    pipe.hgetall("persister")
    res = pipe.execute()
    lat, e2e, counts, tps_raw, gmeta = res[:5]
    pers = res[-1] or {}
    try:
        lag = int(r.xinfo_groups(PERSIST_STREAM)[0].get("lag") or 0)
    except Exception:
        lag = None
    persister = {
        "alive": (time.time() - float(pers.get("heartbeat", 0) or 0)) < 10,
        "written": int(pers.get("written", 0) or 0),
        "explained_async": int(pers.get("explained", 0) or 0),
        "explain_queue": int(pers.get("explain_queue", 0) or 0),
        "explain_skipped": int(pers.get("explain_skipped", 0) or 0),
        "lag": lag,
    }
    workers = []
    for p in range(s.stream_partitions):
        w, xlen, pend = res[5 + 3 * p : 8 + 3 * p]
        hb = float(w.get("heartbeat", 0) or 0)
        workers.append(
            {
                "partition": p,
                "alive": (time.time() - hb) < 10,
                "processed": int(w.get("processed", 0) or 0),
                "errors": int(w.get("errors", 0) or 0),
                "model": w.get("model"),
                "db_backlog": (lag or 0) + int(w.get("backlog_db", 0) or 0),
                "db_healthy": persister["alive"] and w.get("db_healthy") == "1",
                "stream_len": xlen,
                "pending": (pend or {}).get("pending", 0) if isinstance(pend, dict) else 0,
            }
        )
    tps_series = [int(x or 0) for x in tps_raw]
    counts = {k: int(v) for k, v in (counts or {}).items()}
    total = counts.get("total", 0)
    fb = db.fetch_all("SELECT kind, count(*) AS n FROM feedback GROUP BY kind")
    bundle = scorer.models.bundle
    g_age = time.time() - float(gmeta.get("wall", 0) or 0) if gmeta else None
    return {
        "time": now,
        "throughput": {
            "tps_last_10s": round(sum(tps_series[-11:-1]) / 10, 1),
            "tps_last_60s": round(sum(tps_series) / 60, 1),
            "series": tps_series,
        },
        "latency_ms": _pcts(lat),
        "e2e_latency_ms": _pcts(e2e),
        "counts": counts,
        "alert_rate": round((counts.get("REVIEW", 0) + counts.get("BLOCK", 0)) / total, 5) if total else 0.0,
        "block_rate": round(counts.get("BLOCK", 0) / total, 5) if total else 0.0,
        "degraded_rate": round(counts.get("degraded", 0) / total, 5) if total else 0.0,
        "model": {
            "version": bundle.version,
            "thresholds": {k: {"review": c.review, "block": c.block} for k, c in bundle.combos.items()},
        },
        "graph": {
            "last_run_age_s": round(g_age, 1) if g_age is not None else None,
            "stale": g_age is None or g_age > s.graph_stale_after_s,
            "rings": int(gmeta.get("n_rings", 0) or 0),
            "nodes": int(gmeta.get("n_nodes", 0) or 0),
            "run_ms": float(gmeta.get("run_ms", 0) or 0),
            "window_rows": int(gmeta.get("window_rows", 0) or 0),
        },
        "workers": workers,
        "persister": persister,
        "feedback": {x["kind"]: x["n"] for x in fb},
        "drift": drift(bundle, cache),
    }
