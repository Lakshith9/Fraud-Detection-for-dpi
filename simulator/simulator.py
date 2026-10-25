"""Mojaloop-style transaction event simulator.

Replays the generated dataset's live window (default: days [46, 60), i.e. AFTER the bootstrap snapshot) as
FSPIOP transfer-prepare events onto the partitioned Redis stream at a target TPS. Fraud-registry reports are
interleaved at their (event-time) report timestamps. Optional fault injection: duplicate deliveries and
out-of-order events. Ground truth is written to sim_ground_truth for the demo/evaluation scripts only.

Usage: python -m simulator.simulator --tps 200 [--from-day 46 --to-day 60] [--limit N] [--loop]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.redis_client import get_redis, partition_of, stream_key
from pipeline.mojaloop_adapter import to_fspiop

log = get_logger("simulator")
DAY = 86_400_000
TX_COLS = [
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


def build_timeline(data_dir: Path, from_day: int, to_day: int, limit: int | None, skip: int = 0):
    """Load only the needed rows/columns (parquet predicate push-down) to keep the simulator's footprint small."""
    path = data_dir / "transactions.parquet"
    start = int(pd.read_parquet(path, columns=["ts"])["ts"].min()) // DAY * DAY
    lo, hi = start + from_day * DAY, start + to_day * DAY
    cols = TX_COLS + ["is_fraud", "fraud_type", "case_id"]
    tx = pd.read_parquet(path, columns=cols, filters=[("ts", ">=", lo), ("ts", "<", hi)])
    tx = tx.sort_values("ts", kind="stable")
    tx = tx.iloc[skip:]
    if limit:
        tx = tx.iloc[:limit]
    lo = int(tx["ts"].min()) if len(tx) else lo
    rep = pd.read_parquet(data_dir / "fraud_reports.parquet")
    end = int(tx["ts"].max()) if len(tx) else hi
    rep = rep[(rep["report_ts"] >= lo) & (rep["report_ts"] <= end)]
    return tx, rep


def write_ground_truth(tx: pd.DataFrame) -> None:
    from core import db

    db.init_schema()
    rows = list(zip(tx["transfer_id"], tx["is_fraud"].astype(int), tx["fraud_type"], tx["case_id"]))
    with db.get_pool().connection() as conn, conn.cursor() as cur:
        for i in range(0, len(rows), 5000):
            cur.executemany(
                """INSERT INTO sim_ground_truth (transfer_id, is_fraud, fraud_type, case_id)
                               VALUES (%s,%s,%s,%s) ON CONFLICT (transfer_id) DO NOTHING""",
                rows[i : i + 5000],
            )


def run(
    tps: float,
    from_day: int,
    to_day: int,
    limit: int | None,
    loop: bool,
    dup_rate: float,
    disorder_rate: float,
    data_dir: Path,
    ground_truth: bool,
    seed: int = 7,
    skip: int = 0,
    lap0: int = 0,
) -> dict:
    s = get_settings()
    setup_logging(s.log_level)
    r = get_redis(timeout=10)
    r.ping()  # connect before the pacing clock starts: connection setup must not be counted as pipeline latency
    rng = np.random.default_rng(seed)
    tx, rep = build_timeline(data_dir, from_day, to_day, limit, skip)
    if ground_truth:
        write_ground_truth(tx)
    span = (int(tx["ts"].max()) - int(tx["ts"].min()) + DAY) if len(tx) else DAY
    # events reference row indices; transfer dicts are rendered lazily per chunk (bounded memory)
    colv = {c: tx[c].tolist() for c in TX_COLS}
    n_tx = len(tx)
    events = [(colv["ts"][i], 1, i) for i in range(n_tx)] + [
        (int(a), 0, {"type": "fraud_report", "account_id": acc}) for acc, a in zip(rep["account_id"], rep["report_ts"])
    ]
    events.sort(key=lambda e: (e[0], e[1]))
    n_part = s.stream_partitions
    log.info("simulator starting", events=len(events), transfers=n_tx, reports=len(rep), tps=tps, partitions=n_part)
    sent = dups = swapped = 0
    t_start = time.perf_counter()
    lap = lap0  # lap > 0 replays with shifted event times and suffixed transfer ids (fresh events)
    tick = 0.01
    while True:
        i = 0
        while i < len(events):
            target = int((time.perf_counter() - t_start) * tps) + 1
            n = max(0, min(target - sent, len(events) - i))
            if n == 0:
                time.sleep(tick / 2)
                continue
            chunk = events[i : i + n]
            if disorder_rate > 0 and len(chunk) > 1:
                for k in range(len(chunk) - 1):
                    if rng.random() < disorder_rate:
                        chunk[k], chunk[k + 1] = chunk[k + 1], chunk[k]
                        swapped += 1
            pipe = r.pipeline(transaction=False)
            now_ms = time.time() * 1000
            for _ts, kind, ev in chunk:
                if kind == 0:
                    payload = ev
                    part = partition_of(ev["account_id"], n_part)
                else:
                    t = {c: colv[c][ev] for c in TX_COLS}
                    if lap:
                        t["ts"] = int(t["ts"]) + lap * span
                        t["transfer_id"] = f"{t['transfer_id'][:-4]}{lap:04d}"
                    if isinstance(t.get("lat"), float) and np.isnan(t["lat"]):
                        t["lat"] = t["lon"] = None
                    payload = to_fspiop(t)
                    part = partition_of(t["payer_id"], n_part)
                fields = {"event": orjson.dumps(payload), "sent_at": f"{now_ms:.3f}"}
                pipe.xadd(stream_key(part), fields, maxlen=s.stream_maxlen, approximate=True)
                if kind == 1 and dup_rate > 0 and rng.random() < dup_rate:
                    pipe.xadd(stream_key(part), fields, maxlen=s.stream_maxlen, approximate=True)
                    dups += 1
            pipe.execute()
            i += n
            sent += n
            if sent % max(1, int(tps * 10)) < n:
                el = time.perf_counter() - t_start
                log.info(
                    "simulator progress",
                    sent=sent,
                    rate=round(sent / el, 1),
                    duplicates=dups,
                    reordered=swapped,
                    lap=lap,
                )
        if not loop:
            break
        lap += 1
    el = time.perf_counter() - t_start
    summary = {
        "sent": sent,
        "seconds": round(el, 1),
        "achieved_tps": round(sent / el, 1),
        "duplicates": dups,
        "reordered": swapped,
    }
    log.info("simulator finished", **summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tps", type=float, default=200)
    ap.add_argument("--from-day", type=int, default=46)
    ap.add_argument("--to-day", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N transfers of the window")
    ap.add_argument("--lap", type=int, default=0, help="start at replay lap N (shifted times, fresh transfer ids)")
    ap.add_argument("--loop", action="store_true", help="replay forever (timestamps shifted, ids suffixed)")
    ap.add_argument("--dup-rate", type=float, default=0.0)
    ap.add_argument("--disorder-rate", type=float, default=0.0)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--no-ground-truth", action="store_true")
    a = ap.parse_args(argv)
    run(
        a.tps,
        a.from_day,
        a.to_day,
        a.limit,
        a.loop,
        a.dup_rate,
        a.disorder_rate,
        a.data or get_settings().data_dir,
        not a.no_ground_truth,
        skip=a.skip,
        lap0=a.lap,
    )


if __name__ == "__main__":
    main()
