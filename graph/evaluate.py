"""Ring-level precision / recall of the graph detectors against rings planted by the generator.

A detected ring MATCHES a planted case when >= 50% of its accounts belong to that case (and >= 2 accounts
overlap). Planted rings are the COLLUSIVE typologies (merchant_fraud_ring, transaction_laundering,
settlement_manipulation) with >= 3 accounts. identity_spoofing cases (one victim + 1-3 mule payees) are account
takeovers rather than rings; they are reported separately as "ato_cases_touched" for transparency and are the
job of the tabular + sequence models. Precision = matched detected rings / detected rings; recall = planted
cases (>= 3 accounts) matched by at least one detected ring. Detection runs on sliding 24h windows every
--step-hours of event time, exactly like the online graph worker.

Usage: python -m graph.evaluate --data data --out models/artifacts/graph_eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from graph.detectors import DetectorConfig, detect_rings
from graph.rings import RingTracker

HOUR = 3_600_000
RING_TYPOLOGIES = ("merchant_fraud_ring", "transaction_laundering", "settlement_manipulation")


def sliding_rings(
    tx: pd.DataFrame, reports: pd.DataFrame, window_h=24.0, step_h=3.0, cfg=None, progress=False
) -> tuple[list, dict]:
    cfg = cfg or DetectorConfig()
    tracker = RingTracker()
    ts = tx["ts"].to_numpy()
    t0, t1 = int(ts.min()), int(ts.max())
    rep_ts = reports["report_ts"].to_numpy() if len(reports) else np.array([], dtype=np.int64)
    rep_acc = reports["account_id"].to_numpy() if len(reports) else np.array([], dtype=object)
    all_rings: dict[str, object] = {}
    timings = []
    now = t0 + int(step_h * HOUR)
    cols = ["ts", "payer_id", "payee_id", "amount", "device_id", "ip"]
    while now <= t1 + int(step_h * HOUR):
        lo = np.searchsorted(ts, now - int(window_h * HOUR), "left")
        hi = np.searchsorted(ts, now, "right")
        win = tx.iloc[lo:hi][cols]
        reported = set(rep_acc[rep_ts <= now])
        t = time.perf_counter()
        rings = tracker.assign(detect_rings(win, reported, now, cfg))
        timings.append((time.perf_counter() - t) * 1000)
        for r in rings:
            all_rings[r.ring_id] = r
        now += int(step_h * HOUR)
        if progress and len(timings) % 50 == 0:
            print(
                f"  graph windows: {len(timings)}  rings so far: {len(all_rings)}  last run {timings[-1]:.0f} ms",
                flush=True,
            )
    stats = {
        "runs": len(timings),
        "run_ms_p50": float(np.percentile(timings, 50)),
        "run_ms_p95": float(np.percentile(timings, 95)),
        "run_ms_max": float(np.max(timings)),
    }
    return list(all_rings.values()), stats


def evaluate(rings: list, cases: list[dict], min_score: float = 0.0) -> dict:
    planted = [c for c in cases if len(set(c["members"])) >= 3 and c["type"] in RING_TYPOLOGIES]
    ato = [c for c in cases if c["type"] == "identity_spoofing"]
    member_case = defaultdict(set)
    for i, c in enumerate(planted):
        for m in c["members"]:
            member_case[m].add(i)
    rings = [r for r in rings if r.score >= min_score]
    matched_cases = set()
    tp_rings = 0
    per_type_det = defaultdict(lambda: [0, 0])
    for r in rings:
        counts = defaultdict(int)
        for m in r.members:
            for ci in member_case.get(m, ()):
                counts[ci] += 1
        hit = [ci for ci, k in counts.items() if k >= 2 and k / len(r.members) >= 0.5]
        per_type_det[r.ring_type][1] += 1
        if hit:
            tp_rings += 1
            per_type_det[r.ring_type][0] += 1
            matched_cases.update(hit)
    per_typology = defaultdict(lambda: [0, 0])
    for i, c in enumerate(planted):
        per_typology[c["type"]][1] += 1
        if i in matched_cases:
            per_typology[c["type"]][0] += 1
    ring_accounts = {m for r in rings for m in r.members}
    ato_touched = sum(1 for c in ato if set(c["members"]) & ring_accounts)
    return {
        "min_score": min_score,
        "ato_cases_touched": f"{ato_touched}/{len(ato)}",
        "detected_rings": len(rings),
        "planted_rings": len(planted),
        "ring_precision": round(tp_rings / len(rings), 4) if rings else None,
        "ring_recall": round(len(matched_cases) / len(planted), 4) if planted else None,
        "precision_by_detector": {
            k: {"matched": v[0], "detected": v[1], "precision": round(v[0] / v[1], 4)} for k, v in per_type_det.items()
        },
        "recall_by_typology": {
            k: {"matched": v[0], "planted": v[1], "recall": round(v[0] / v[1], 4)} for k, v in per_typology.items()
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--step-hours", type=float, default=3.0)
    a = ap.parse_args(argv)
    tx = pd.read_parquet(a.data / "transactions.parquet")
    reports = pd.read_parquet(a.data / "fraud_reports.parquet")
    cases = json.loads((a.data / "rings_truth.json").read_text())
    t = time.time()
    rings, stats = sliding_rings(tx, reports, step_h=a.step_hours, progress=True)
    res = {
        "timing": stats,
        "wall_seconds": round(time.time() - t, 1),
        "thresholds": [evaluate(rings, cases, s) for s in (0.35, 0.5, 0.6, 0.7)],
    }
    print(json.dumps(res, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
