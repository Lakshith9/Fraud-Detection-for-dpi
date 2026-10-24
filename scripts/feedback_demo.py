"""Feedback-loop demonstration against the RUNNING system.

1. replay the 'feedback window' (days [46, 53)) through the live stream -> workers score it, alerts are raised
2. simulated operators review EVERY alert through the REST API (True Fraud / False Positive, using the simulator's
   ground truth as the operators' judgement) and report a share of missed frauds as false negatives (customer
   complaints / chargebacks arrive for only part of them)
3. trigger retraining via POST /v1/models/retrain and wait for the gated result
4. print + save before/after metrics on the held-out window (days [53, 60)) -> docs/feedback_demo.json

Usage: python -m scripts.feedback_demo [--tps 1000] [--fn-report-rate 0.6]
"""

from __future__ import annotations

import argparse
import json
import time

import httpx
import numpy as np

from core import db
from core.config import ROOT, get_settings
from core.redis_client import get_redis
from simulator.simulator import run as simulate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tps", type=float, default=1000)
    ap.add_argument("--from-day", type=int, default=46)
    ap.add_argument("--to-day", type=int, default=53)
    ap.add_argument("--fn-report-rate", type=float, default=0.6)
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)
    s = get_settings()
    key = next(iter(s.api_keys()))
    rng = np.random.default_rng(a.seed)
    api = httpx.Client(base_url="http://127.0.0.1:8000", headers={"X-API-Key": key}, timeout=60)
    out: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out["model_before"] = api.get("/v1/models").json()["current"]

    if not a.skip_replay:
        r = get_redis(timeout=10)
        before = sum(int(r.hget(f"worker:{p}", "processed") or 0) for p in range(s.stream_partitions))
        summ = simulate(
            tps=a.tps,
            from_day=a.from_day,
            to_day=a.to_day,
            limit=None,
            loop=False,
            dup_rate=0.0,
            disorder_rate=0.0,
            data_dir=s.data_dir,
            ground_truth=True,
        )
        print(f"published {summ['sent']:,} events; waiting for the workers to drain...", flush=True)
        t = time.time()
        while time.time() - t < 1800:
            done = sum(int(r.hget(f"worker:{p}", "processed") or 0) for p in range(s.stream_partitions)) - before
            if done >= summ["sent"]:
                break
            time.sleep(2)
        # wait until the persister has written every scored row to PostgreSQL
        while time.time() - t < 1800:
            try:
                if int(r.xinfo_groups("persist:rows")[0].get("lag") or 0) == 0:
                    break
            except Exception:
                break
            time.sleep(2)
        time.sleep(2)
        out["replay"] = summ

    win = db.fetch_one("SELECT min(ts) AS lo, max(ts) AS hi, count(*) AS n FROM transactions WHERE source='stream'")
    alerts = db.fetch_all(
        """SELECT a.transfer_id, g.is_fraud FROM alerts a JOIN sim_ground_truth g USING (transfer_id)
           WHERE NOT EXISTS (SELECT 1 FROM feedback f WHERE f.transfer_id = a.transfer_id)"""
    )
    missed = db.fetch_all(
        """SELECT t.transfer_id FROM transactions t JOIN sim_ground_truth g USING (transfer_id)
           WHERE g.is_fraud = 1 AND t.decision = 'ALLOW' AND t.source = 'stream'
             AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.transfer_id = t.transfer_id)"""
    )
    kinds: dict[str, int] = {}
    for al in alerts:
        res = api.post(
            "/v1/feedback",
            json={
                "transfer_id": al["transfer_id"],
                "label": "fraud" if al["is_fraud"] else "legit",
                "notes": "demo operator review",
            },
        )
        res.raise_for_status()
        kinds[res.json()["kind"]] = kinds.get(res.json()["kind"], 0) + 1
    reported = [m for m in missed if rng.random() < a.fn_report_rate]
    for m in reported:
        res = api.post(
            "/v1/feedback",
            json={"transfer_id": m["transfer_id"], "label": "fraud", "notes": "customer complaint (missed fraud)"},
        )
        res.raise_for_status()
        kinds[res.json()["kind"]] = kinds.get(res.json()["kind"], 0) + 1
    out["live_window"] = {
        "scored_transfers": win["n"],
        "alerts_reviewed": len(alerts),
        "missed_frauds_in_window": len(missed),
        "missed_frauds_reported": len(reported),
        "feedback_by_kind": kinds,
    }
    print(json.dumps(out["live_window"], indent=1), flush=True)

    last = db.fetch_one("SELECT max(id) AS id FROM retrain_runs")["id"] or 0
    print("triggering retraining via POST /v1/models/retrain ...", flush=True)
    api.post("/v1/models/retrain", json={"reason": "feedback demo"}).raise_for_status()
    t = time.time()
    run = None
    while time.time() - t < 1800:
        run = db.fetch_one("SELECT * FROM retrain_runs WHERE id > %s ORDER BY id DESC LIMIT 1", (last,))
        if run and run["status"] != "running":
            break
        time.sleep(3)
    if not run or run["status"] != "done":
        raise SystemExit(f"retraining did not finish successfully: {run}")
    b, af = run["before_metrics"], run["after_metrics"]

    def row(name, m):
        t_ = m["truth"]
        return {
            "model": name,
            "pr_auc_observed": m["observed"]["pr_auc"],
            "pr_auc_truth": t_["pr_auc"],
            "roc_auc": t_["roc_auc"],
            "recall_at_1pct_fpr": t_["recall_at_1pct_fpr"],
            "review_precision": t_["review"]["precision"],
            "review_recall": t_["review"]["recall"],
            "block_precision": t_["block"]["precision"],
            "block_recall": t_["block"]["recall"],
            "drift_variant_recall_review": t_.get("drift_variant_recall_review"),
            "per_fraud_type_recall_review": {k: v["recall_review"] for k, v in t_.get("per_fraud_type", {}).items()},
        }

    out["retrain"] = {
        "run_id": run["id"],
        "base_version": run["base_version"],
        "candidate_version": run["candidate_version"],
        "promoted": run["promoted"],
        "reason": run["reason"],
        "n_feedback": run["n_feedback"],
        "holdout_days": [53, 60],
        "before": row(run["base_version"], b),
        "after": row(run["candidate_version"], af),
    }
    time.sleep(s.model_poll_interval_s + 2)
    out["model_after"] = api.get("/v1/models").json()["current"]
    out["workers_model_after"] = [get_redis().hget(f"worker:{p}", "model") for p in range(s.stream_partitions)]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "feedback_demo.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["retrain"], indent=1, default=str))
    print(
        f"production model: {out['model_before']} -> {out['model_after']}; workers now on {out['workers_model_after']}"
    )


if __name__ == "__main__":
    main()
