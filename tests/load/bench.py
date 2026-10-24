"""Load / latency benchmark against the RUNNING system (API + scoring workers + graph worker).

Phases (each uses fresh, not-yet-replayed transfers starting at --from-day, so nothing is deduplicated):
  1. stream @ 200 TPS for 120 s   (the Definition-of-Done operating point)
  2. stream @ stress TPS for 60 s (default 1000)
  3. synchronous POST /v1/score over HTTP (client-observed latency, sequential requests)

Reported per phase: achieved TPS, scoring latency p50/p95/p99/max (in-worker: feature fetch -> decision ->
state write), end-to-end latency (simulator publish -> scored, includes queueing), error rate, backlog.
Results are written to docs/benchmarks.json and printed as a markdown table.

Usage: python -m tests.load.bench [--tps 200 --seconds 120 --stress-tps 1000 --stress-seconds 60]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from core.config import ROOT, get_settings
from core.redis_client import get_redis
from pipeline.scorer import E2E_KEY, LAT_KEY
from simulator.simulator import run as simulate

DAY = 86_400_000


def pcts(vals) -> dict:
    a = np.asarray(vals, dtype=float)
    if not len(a):
        return {"n": 0}
    return {
        "n": int(len(a)),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
        "max": round(float(a.max()), 2),
        "mean": round(float(a.mean()), 2),
    }


def worker_totals(r, n_part) -> tuple[int, int, int]:
    processed = errors = dlq = 0
    s = get_settings()
    for p in range(n_part):
        w = r.hgetall(f"worker:{p}")
        processed += int(w.get("processed", 0) or 0)
        errors += int(w.get("errors", 0) or 0)
        dlq += r.xlen(f"{s.stream_prefix}:{p}:dlq")
    return processed, errors, dlq


def stream_phase(
    name: str, tps: float, seconds: float, skip: int, from_day: int = 53, drain_timeout: float = 300, lap: int = 0
) -> dict:
    s = get_settings()
    r = get_redis(timeout=10)
    r.delete(LAT_KEY, E2E_KEY)
    p0, e0, d0 = worker_totals(r, s.stream_partitions)
    n = int(tps * seconds)
    t0 = time.time()
    summ = simulate(
        tps=tps,
        from_day=from_day,
        to_day=60,
        limit=n,
        loop=False,
        dup_rate=0.0,
        disorder_rate=0.0,
        data_dir=s.data_dir,
        ground_truth=True,
        skip=skip,
        lap0=lap,
    )
    # wait for all events (transfers + interleaved reports) to be consumed
    target = summ["sent"]
    t_send_end = time.time()
    while time.time() - t_send_end < drain_timeout:
        p, _, _ = worker_totals(r, s.stream_partitions)
        if p - p0 >= target:
            break
        time.sleep(0.5)
    drain_s = time.time() - t_send_end
    p1, e1, d1 = worker_totals(r, s.stream_partitions)
    lat = [float(x) for x in r.lrange(LAT_KEY, 0, -1)]
    e2e = [float(x) for x in r.lrange(E2E_KEY, 0, -1)]
    processed = p1 - p0
    return {
        "phase": name,
        "target_tps": tps,
        "duration_s": seconds,
        "sent_events": target,
        "achieved_publish_tps": summ["achieved_tps"],
        "processed_events": processed,
        "processing_tps": round(processed / (time.time() - t0), 1),
        "drain_after_publish_s": round(drain_s, 1),
        "errors": (e1 - e0) + (d1 - d0),
        "error_rate": round(((e1 - e0) + (d1 - d0)) / max(1, processed), 6),
        "scoring_latency_ms": pcts(lat),
        "e2e_latency_ms": pcts(e2e),
        "p99_under_100ms": bool(lat and np.percentile(lat, 99) < 100),
    }


def api_phase(n: int, skip: int, from_day: int = 53, lap: int = 0) -> dict:
    s = get_settings()
    key = next(iter(s.api_keys()), None)
    tx = pd.read_parquet(s.data_dir / "transactions.parquet")
    start = int(tx["ts"].min()) // DAY * DAY
    tx = tx[tx["ts"] >= start + from_day * DAY].sort_values("ts", kind="stable")
    span = int(tx["ts"].max()) - int(tx["ts"].min()) + DAY
    tx = tx.iloc[skip : skip + n]
    cols = [
        "transfer_id",
        "quote_id",
        "ts",
        "payer_id",
        "payer_alias",
        "payer_fsp",
        "payer_type",
        "payee_id",
        "payee_alias",
        "payee_fsp",
        "payee_type",
        "amount",
        "currency",
        "scenario",
        "initiator",
        "channel",
        "device_id",
        "ip",
        "lat",
        "lon",
        "device_bound_at",
        "merchant_category",
        "original_transfer_id",
    ]
    client_ms, server_ms, errors = [], [], 0
    with httpx.Client(base_url="http://127.0.0.1:8000", headers={"X-API-Key": key}, timeout=10) as c:
        for rec in tx[cols].to_dict("records"):
            rec = {k: (None if isinstance(v, float) and v != v else v) for k, v in rec.items()}
            if lap:  # same time shift as the stream laps, distinct id suffix -> fresh, never-scored transfers
                rec["ts"] = int(rec["ts"]) + lap * span
                rec["transfer_id"] = f"{rec['transfer_id'][:-4]}a{lap:03d}"
            if rec["scenario"] not in ("TRANSFER", "PAYMENT", "REFUND", "REVERSAL"):
                continue
            t = time.perf_counter()
            res = c.post("/v1/score", json=rec)
            client_ms.append((time.perf_counter() - t) * 1000)
            if res.status_code != 200:
                errors += 1
            else:
                server_ms.append(res.json()["latency_ms"])
    return {
        "phase": "sync POST /v1/score (HTTP, sequential)",
        "requests": len(client_ms),
        "errors": errors,
        "client_latency_ms": pcts(client_ms),
        "server_scoring_latency_ms": pcts(server_ms),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tps", type=float, default=200)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--stress-tps", type=float, default=1000)
    ap.add_argument("--stress-seconds", type=float, default=60)
    ap.add_argument("--api-requests", type=int, default=2000)
    ap.add_argument("--skip", type=int, default=0, help="offset into the live window (avoid re-sent events)")
    ap.add_argument("--from-day", type=int, default=53, help="first event day replayed (after the feedback window)")
    ap.add_argument("--lap", type=int, default=0, help="replay lap (shifted times, fresh ids) so events are never dups")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "benchmarks.json")
    a = ap.parse_args(argv)
    s = get_settings()
    results = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "partitions": s.stream_partitions,
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phases": [],
    }
    skip = a.skip
    ph = stream_phase(f"stream @ {a.tps:.0f} TPS x {a.seconds:.0f}s", a.tps, a.seconds, skip, a.from_day, lap=a.lap)
    results["phases"].append(ph)
    print(json.dumps(ph, indent=1), flush=True)
    skip += int(a.tps * a.seconds)
    if a.stress_tps > 0:
        ph = stream_phase(
            f"stress @ {a.stress_tps:.0f} TPS x {a.stress_seconds:.0f}s",
            a.stress_tps,
            a.stress_seconds,
            skip,
            a.from_day,
            lap=a.lap,
        )
        results["phases"].append(ph)
        print(json.dumps(ph, indent=1), flush=True)
        skip += int(a.stress_tps * a.stress_seconds)
    if a.api_requests > 0:
        ph = api_phase(a.api_requests, skip, a.from_day, a.lap)
        results["phases"].append(ph)
        print(json.dumps(ph, indent=1), flush=True)
    results["next_skip"] = skip + a.api_requests
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=1))
    print("\n| phase | achieved TPS | p50 ms | p95 ms | p99 ms | max ms | e2e p99 ms | error rate |")
    print("|---|---|---|---|---|---|---|---|")
    for p in results["phases"]:
        lat = p.get("scoring_latency_ms") or p.get("server_scoring_latency_ms")
        e2e = (p.get("e2e_latency_ms") or p.get("client_latency_ms") or {}).get("p99")
        tps = p.get("processing_tps", "-")
        err = p.get("error_rate", round(p.get("errors", 0) / max(1, p.get("requests", 1)), 6))
        print(
            f"| {p['phase']} | {tps} | {lat.get('p50')} | {lat.get('p95')} | {lat.get('p99')} | {lat.get('max')} | "
            f"{e2e} | {err} |"
        )


if __name__ == "__main__":
    main()
