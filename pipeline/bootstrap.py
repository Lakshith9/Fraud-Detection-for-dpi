"""Bootstrap the online system from the offline replay snapshot.

1. create the PostgreSQL schema
2. (--reset) clear engine keys in Redis and transactional tables in Postgres
3. load the replay snapshot (state at the start of the live window) into Redis: the online feature store then
   holds EXACTLY the state the offline features were computed from (train/serve parity)
4. register reported accounts (graph evidence) and insert the last 24h of history into Postgres so the graph
   worker has a full window from the first cycle
5. register + promote the latest trained model if the registry has no production version
6. create stream consumer groups

Usage: python -m pipeline.bootstrap --features data/features --data data --reset
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd

from core import db
from core.config import get_settings
from core.redis_client import get_redis, stream_key
from models import registry
from pipeline.worker import ensure_group

ENGINE_PREFIXES = (
    "ph:",
    "zt:",
    "in:",
    "out:",
    "dacc:",
    "pdev:",
    "iacc:",
    "inf:",
    "cnt:",
    "g:",
    "fspn:",
    "fspr:",
    "idem:",
    "metrics:",
    "worker:",
    "reported",
    "model:",
    "fsp:transfers",
    "retrain:",
    "persist:",
    "persister",
)
TX_TABLES = (
    "transactions",
    "alerts",
    "feedback",
    "case_notes",
    "case_audit",
    "case_alerts",
    "cases",
    "rings",
    "retrain_runs",
    "sim_ground_truth",
)


def reset(r) -> None:
    for p in ENGINE_PREFIXES:
        batch = []
        for k in r.scan_iter(match=p + "*", count=5000):
            batch.append(k)
            if len(batch) >= 5000:
                r.delete(*batch)
                batch = []
        if batch:
            r.delete(*batch)
    with db.get_pool().connection() as conn:
        conn.execute("TRUNCATE " + ", ".join(TX_TABLES) + " RESTART IDENTITY CASCADE")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--snapshot-day", type=int, default=46)
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args(argv)
    s = get_settings()
    t = time.time()
    db.init_schema()
    r = get_redis(timeout=60)
    if a.reset:
        reset(r)
        print("reset redis engine keys and postgres tables")
    snap_path = a.features / f"state_day{a.snapshot_day}.pkl"
    with open(snap_path, "rb") as f:
        snap = pickle.load(f)
    n = snap["store"].dump_to_redis(r)
    snapshot_ts = int(snap["snapshot_ts"])
    print(f"loaded {n:,} state keys into redis from {snap_path.name}")
    r.hset("g:meta", mapping={"ts": snapshot_ts, "wall": f"{time.time():.3f}"})

    rep = pd.read_parquet(a.data / "fraud_reports.parquet")
    rep = rep[rep["report_ts"] <= snapshot_ts]
    if len(rep):
        r.sadd("reported", *rep["account_id"].unique().tolist())

    tx = pd.read_parquet(a.data / "transactions.parquet")
    hist = tx[(tx["ts"] < snapshot_ts) & (tx["ts"] >= snapshot_ts - int(s.graph_window_hours * 3_600_000))]
    rows = [
        (
            t.transfer_id,
            int(t.ts),
            t.payer_id,
            t.payee_id,
            t.payer_fsp,
            t.payee_fsp,
            float(t.amount),
            t.currency,
            t.scenario,
            t.channel,
            t.device_id,
            t.ip,
        )
        for t in hist.itertuples()
    ]
    with db.get_pool().connection() as conn, conn.cursor() as cur:
        for i in range(0, len(rows), 5000):
            cur.executemany(
                """INSERT INTO transactions (transfer_id, ts, payer_id, payee_id, payer_fsp, payee_fsp, amount,
                       currency, scenario, channel, device_id, ip, source)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'bootstrap') ON CONFLICT DO NOTHING""",
                rows[i : i + 5000],
            )
    print(f"inserted {len(rows):,} history transactions (last {s.graph_window_hours:.0f}h) for the graph worker")

    if registry.current() is None:
        versions = sorted(
            [p for p in s.artifacts_dir.iterdir() if p.is_dir() and (p / "bundle.json").exists()],
            key=lambda p: p.stat().st_mtime,
        )
        if not versions:
            raise SystemExit("no trained model found; run `python -m models.train` first")
        v = versions[-1].name
        metrics = json.loads((versions[-1] / "metrics.json").read_text())
        bundle = json.loads((versions[-1] / "bundle.json").read_text())
        registry.register(v, metrics, bundle["data_hash"], notes="bootstrap")
        registry.promote(v)
        print(f"registered and promoted {v}")
    else:
        registry.sync_current_to_redis()
        print(f"production model: {registry.current()}")
    for p in range(s.stream_partitions):
        ensure_group(r, stream_key(p), s.consumer_group)
    print(f"bootstrap done in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
