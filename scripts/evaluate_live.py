"""Score a live run: the engine's stored decisions vs the replay's ground truth (sim_ground_truth).

Works for any run that wrote ground truth: the direct simulator (``simulator.simulator``) or traffic through a
Mojaloop switch (``simulator.mojaloop_sender``). Waits until the persister has stored every replayed transfer.

Usage: python -m scripts.evaluate_live --label mojaloop-device --out docs/live_eval/mojaloop_device.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import orjson

from core import db
from core.logging import get_logger, setup_logging
from models.metrics import recall_at_fpr

log = get_logger("evaluate_live")

QUERY = """SELECT g.is_fraud, g.fraud_type, t.risk_score, t.decision, t.latency_ms, t.combo
           FROM sim_ground_truth g JOIN transactions t USING (transfer_id)"""


def wait_until_stored(expected: int, timeout: float) -> int:
    deadline, last, stable_since = time.time() + timeout, -1, time.time()
    while time.time() < deadline:
        n = db.fetch_all("SELECT count(*) AS n FROM sim_ground_truth g JOIN transactions t USING (transfer_id)")[0]["n"]
        if n >= expected:
            return n
        if n != last:
            last, stable_since = n, time.time()
        elif time.time() - stable_since > 60:  # nothing new for a minute: report what arrived
            return n
        time.sleep(2)
    return last


def _confusion(y: np.ndarray, flagged: np.ndarray) -> dict:
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    fn, tn = int((~flagged & (y == 1)).sum()), int((~flagged & (y == 0)).sum())
    return {
        "flagged": tp + fp,
        "frauds_caught": tp,
        "frauds_missed": fn,
        "false_alarms": fp,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "false_alarm_rate": round(fp / (fp + tn), 6) if fp + tn else None,
    }


def evaluate(label: str, timeout: float = 600) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    expected = db.fetch_all("SELECT count(*) AS n FROM sim_ground_truth")[0]["n"]
    stored = wait_until_stored(expected, timeout)
    rows = db.fetch_all(QUERY)
    y = np.array([r["is_fraud"] for r in rows], dtype=int)
    p = np.array([r["risk_score"] for r in rows], dtype=float)
    dec = np.array([r["decision"] for r in rows])
    lat = np.array([r["latency_ms"] for r in rows if r["latency_ms"] is not None], dtype=float)
    ftype = np.array([r["fraud_type"] or "" for r in rows])
    review_or_block = np.isin(dec, ["REVIEW", "BLOCK"])
    per_type = {}
    for t in sorted(set(ftype[y == 1])):
        m = (ftype == t) & (y == 1)
        per_type[str(t)] = {
            "frauds": int(m.sum()),
            "caught": int(review_or_block[m].sum()),
            "recall": round(float(review_or_block[m].mean()), 4),
        }
    out = {
        "label": label,
        "transfers_expected": expected,
        "transfers_scored": stored,
        "frauds": int(y.sum()),
        "pr_auc": round(float(average_precision_score(y, p)), 4) if y.sum() else None,
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None,
        "recall_at_1pct_fpr": round(recall_at_fpr(y, p, 0.01), 4) if y.sum() else None,
        "alerts_review_or_block": _confusion(y, review_or_block),
        "blocks": _confusion(y, dec == "BLOCK"),
        "per_fraud_type": per_type,
        "decisions": {d: int((dec == d).sum()) for d in ("ALLOW", "REVIEW", "BLOCK")},
        "scoring_latency_ms": {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
        }
        if len(lat)
        else None,
        "degraded_combos": {str(c): n for c, n in _counts([r["combo"] for r in rows]).items() if c != "full"},
    }
    return out


def _counts(xs) -> dict:
    out: dict = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--timeout", type=float, default=600, help="max seconds to wait for the persister")
    a = ap.parse_args(argv)
    setup_logging()
    res = evaluate(a.label, a.timeout)
    text = orjson.dumps(res, option=orjson.OPT_INDENT_2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_bytes(text)
    print(text.decode())


if __name__ == "__main__":
    main()
