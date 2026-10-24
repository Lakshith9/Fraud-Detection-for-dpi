"""Offline feature computation by event-time replay through the SAME feature engine used online.

For every transaction (in event-time order):
  * fraud-registry reports with report_ts <= ts are applied first,
  * the graph worker logic runs every --graph-step-hours of event time on the trailing 24h window,
  * features are computed from the read plan results, then the write plan is applied.

Outputs (in --out):
  features.parquet   transfer_id, ts, labels, tabular + graph features
  seq.npy            float16 [N, SEQ_LEN, SEQ_DIM] sequence-model inputs (row-aligned with features.parquet)
  state_<day>.pkl    snapshot of the online state at the start of the live period (bootstraps Redis)

Usage: python -m pipeline.replay --data data --out data/features --snapshot-day 46
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import get_settings
from graph.rings import RingTracker, run_graph_cycle
from pipeline.features import GRAPH_FEATURES, SEQ_DIM, SEQ_LEN, TABULAR_FEATURES, FeatureEngine
from pipeline.state_store import InMemoryStore

HOUR = 3_600_000
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
GRAPH_COLS = ["ts", "payer_id", "payee_id", "amount", "device_id", "ip"]


class GraphRefresher:
    """Runs the graph worker logic on a schedule of EVENT time (mirrors graph/worker.py)."""

    def __init__(self, tx: pd.DataFrame, reports: pd.DataFrame, store, step_h: float, window_h: float):
        self.tx = tx[GRAPH_COLS]
        self.ts = tx["ts"].to_numpy()
        self.rep_ts = reports["report_ts"].to_numpy()
        self.rep_acc = reports["account_id"].to_numpy()
        self.store = store
        self.step = int(step_h * HOUR)
        self.window = int(window_h * HOUR)
        self.next_run = int(self.ts.min()) + self.step if len(self.ts) else 0
        self.tracker = RingTracker()
        self.nodes: set[str] = set()
        self.runs = 0
        self.ms = 0.0

    def maybe_run(self, now: int):
        while now >= self.next_run:
            t = self.next_run
            lo = np.searchsorted(self.ts, t - self.window, "left")
            hi = np.searchsorted(self.ts, t, "left")  # strictly before t: no peeking at the current event
            reported = set(self.rep_acc[self.rep_ts <= t])
            t0 = time.perf_counter()
            _, self.nodes = run_graph_cycle(self.tx.iloc[lo:hi], reported, t, self.store, self.tracker, self.nodes)
            self.ms += (time.perf_counter() - t0) * 1000
            self.runs += 1
            self.next_run += self.step


def replay(
    data_dir: Path,
    out_dir: Path,
    snapshot_day: int | None,
    graph_step_h: float = 3.0,
    window_h: float = 24.0,
    limit: int | None = None,
) -> dict:
    s = get_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    if limit:
        tx = tx.iloc[:limit]
    reports = pd.read_parquet(data_dir / "fraud_reports.parquet")
    n = len(tx)
    engine = FeatureEngine(s.settlement_cutoff_hours)
    store = InMemoryStore()
    graph = GraphRefresher(tx, reports, store, graph_step_h, window_h)
    start_ms = int(tx["ts"].min()) // DAY * DAY
    snapshot_ts = start_ms + snapshot_day * DAY if snapshot_day is not None else None
    snap_done = snapshot_ts is None

    tab = np.full((n, len(TABULAR_FEATURES)), np.nan, dtype=np.float32)
    gra = np.zeros((n, len(GRAPH_FEATURES)), dtype=np.float32)
    seq = np.lib.format.open_memmap(out_dir / "seq.npy", mode="w+", dtype=np.float16, shape=(n, SEQ_LEN, SEQ_DIM))
    rep_ts = reports["report_ts"].to_numpy()
    rep_acc = reports["account_id"].to_numpy()
    rp = 0
    t_start = time.time()
    i = 0
    chunk = 50_000
    for c0 in range(0, n, chunk):
        recs = tx.iloc[c0 : c0 + chunk][TX_COLS].to_dict("records")
        seq_buf = np.zeros((len(recs), SEQ_LEN, SEQ_DIM), dtype=np.float32)
        for j, t in enumerate(recs):
            ts = int(t["ts"])
            if not snap_done and ts >= snapshot_ts:
                graph.maybe_run(snapshot_ts)
                with open(out_dir / f"state_day{snapshot_day}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "store": store,
                            "graph_nodes": graph.nodes,
                            "snapshot_ts": snapshot_ts,
                            "next_report_index": rp,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                snap_done = True
            while rp < len(rep_ts) and rep_ts[rp] <= ts:
                store.execute(FeatureEngine.report_plan(rep_acc[rp]))
                rp += 1
            graph.maybe_run(ts)
            if isinstance(t.get("lat"), float) and np.isnan(t["lat"]):
                t["lat"] = t["lon"] = None
            res = store.execute(engine.read_plan(t))
            fv = engine.compute(t, res)
            tab[i] = fv.tabular
            gra[i] = fv.graph
            seq_buf[j] = fv.seq
            store.execute(engine.write_plan(t))
            i += 1
        seq[c0 : c0 + len(recs)] = seq_buf.astype(np.float16)
        el = time.time() - t_start
        print(
            f"  replayed {i:>9,}/{n:,}  {i / el:,.0f} tx/s  graph runs={graph.runs} "
            f"({graph.ms / max(graph.runs, 1):.0f} ms avg)",
            flush=True,
        )
    seq.flush()
    feats = pd.DataFrame(tab, columns=TABULAR_FEATURES)
    for k, name in enumerate(GRAPH_FEATURES):
        feats[name] = gra[:, k]
    meta = tx[
        ["transfer_id", "ts", "label", "is_fraud", "fraud_type", "case_id", "payer_id", "payee_id", "payer_fsp"]
    ].reset_index(drop=True)
    out = pd.concat([meta, feats], axis=1)
    out.to_parquet(out_dir / "features.parquet", index=False)
    summary = {
        "rows": n,
        "seconds": round(time.time() - t_start, 1),
        "graph_runs": graph.runs,
        "graph_avg_ms": round(graph.ms / max(graph.runs, 1), 1),
        "snapshot_day": snapshot_day,
        "snapshot_ts": snapshot_ts,
    }
    (out_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/features"))
    ap.add_argument("--snapshot-day", type=int, default=46)
    ap.add_argument("--graph-step-hours", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    print(json.dumps(replay(a.data, a.out, a.snapshot_day, a.graph_step_hours, limit=a.limit), indent=2))


if __name__ == "__main__":
    main()
