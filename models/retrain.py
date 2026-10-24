"""Feedback-driven retraining with a promotion gate.

Training data for the candidate = original training split (observed labels, overridden by operator feedback where
present) + live-scored transactions outside the holdout window, using the EXACT features stored at scoring time,
labelled by operator feedback (True Fraud / False Positive / reported False Negative); unlabelled live rows are
treated as legitimate (no fraud reported). The sequence model is reused from the production version.

Gate: candidate and current production model are evaluated on the same held-out future window ('holdout' split,
observed labels). The candidate is promoted only if PR-AUC and recall@1%FPR do not drop by more than
RETRAIN_TOLERANCE; otherwise it is rejected and the reason is logged in retrain_runs.

Usage:
  python -m models.retrain --trigger manual          # one run
  python -m models.retrain --schedule                # loop: run every RETRAIN_INTERVAL_HOURS if enough feedback
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from core import db
from core.config import get_settings
from core.logging import get_logger, setup_logging
from models import registry
from models.ensemble import graph_transform  # noqa: F401  (documented transform, used via meta_inputs)
from models.metrics import evaluate_scores
from models.train import DAY, SPLITS, combo_probs, split_masks, train
from pipeline.features import GRAPH_FEATURES, TABULAR_FEATURES

log = get_logger("retrain")


def latest_feedback() -> pd.DataFrame:
    rows = db.fetch_all(
        """SELECT DISTINCT ON (transfer_id) transfer_id, label, kind, operator_id, created_at
           FROM feedback ORDER BY transfer_id, created_at DESC"""
    )
    return pd.DataFrame(rows, columns=["transfer_id", "label", "kind", "operator_id", "created_at"])


def live_rows(holdout_start_ms: int) -> pd.DataFrame:
    rows = db.fetch_all(
        """SELECT transfer_id, ts, features, graph, components FROM transactions
           WHERE features IS NOT NULL AND source <> 'bootstrap' AND ts < %s""",
        (holdout_start_ms,),
    )
    return pd.DataFrame(rows, columns=["transfer_id", "ts", "features", "graph", "components"])


def offline_scores(version_dir: Path, df: pd.DataFrame, seq_logit: np.ndarray, combo: str = "full") -> np.ndarray:
    bundle = json.loads((version_dir / "bundle.json").read_text())
    b = xgb.Booster()
    b.load_model(str(version_dir / "xgb.json"))
    m = b.predict(
        xgb.DMatrix(df[TABULAR_FEATURES].to_numpy(np.float32), feature_names=TABULAR_FEATURES), output_margin=True
    )
    return combo_probs(bundle["combos"], combo, m, seq_logit, df[GRAPH_FEATURES].to_numpy(np.float32))


def gate(cur: dict, cand: dict, tol: float) -> tuple[bool, str]:
    checks = []
    for k in ("pr_auc", "recall_at_1pct_fpr"):
        if cand[k] < cur[k] - tol:
            checks.append(f"{k} dropped {cur[k]:.4f} -> {cand[k]:.4f} (tolerance {tol})")
    if checks:
        return False, "; ".join(checks)
    return True, (
        f"pr_auc {cur['pr_auc']:.4f} -> {cand['pr_auc']:.4f}, recall@1%FPR "
        f"{cur['recall_at_1pct_fpr']:.4f} -> {cand['recall_at_1pct_fpr']:.4f} within tolerance {tol}"
    )


def run_retrain(trigger: str = "manual", features_dir: Path | None = None, data_dir: Path | None = None) -> dict:
    s = get_settings()
    features_dir = features_dir or s.data_dir / "features"
    data_dir = data_dir or s.data_dir
    db.init_schema()
    base = registry.current()
    if base is None:
        raise RuntimeError("no production model to retrain from")
    run_id = db.fetch_one(
        "INSERT INTO retrain_runs (trigger, base_version) VALUES (%s,%s) RETURNING id", (trigger, base)
    )["id"]
    t0 = time.time()
    try:
        fb = latest_feedback()
        if len(fb) < 1:
            raise RuntimeError("no operator feedback available")
        base_dir = registry.artifact_path(base)
        df = pd.read_parquet(features_dir / "features.parquet")
        start_ms = int(df["ts"].min()) // DAY * DAY
        holdout_start = start_ms + SPLITS["holdout"][0] * DAY
        fb_label = dict(zip(fb["transfer_id"], (fb["label"] == "fraud").astype(int)))

        # live rows with served features; feedback overrides, unlabelled => legit
        live = live_rows(holdout_start)
        extra = None
        if len(live):
            tab = np.array(
                [[np.nan if r.get(f) is None else r[f] for f in TABULAR_FEATURES] for r in live["features"]],
                dtype=np.float32,
            )
            gra = np.array([[g.get(f, 0.0) for f in GRAPH_FEATURES] for g in live["graph"]], dtype=np.float32)
            y = np.array([fb_label.get(t, 0) for t in live["transfer_id"]], dtype=int)
            extra = {"tab": tab, "graph": gra, "y": y, "weight": 1.0}
        # feedback on rows inside the original offline training data overrides their labels
        override = pd.Series({t: v for t, v in fb_label.items()})
        version = registry.next_version()
        parent_bundle = json.loads((base_dir / "bundle.json").read_text())
        best = parent_bundle["xgb"]["params"]
        log.info(
            "retraining",
            run_id=run_id,
            base=base,
            candidate=version,
            feedback=len(fb),
            live_rows=0 if extra is None else len(extra["y"]),
            live_fraud=0 if extra is None else int(extra["y"].sum()),
        )
        res = train(
            features_dir,
            data_dir,
            version,
            s.artifacts_dir,
            seq_from=base_dir,
            label_override=override,
            extra=extra,
            notes=f"retrain from {base} ({trigger}); {len(fb)} feedback labels",
            grid=[best],
        )

        # ---- gate on the holdout window, identical rows for both models
        m = split_masks(df["ts"].to_numpy(), start_ms)["holdout"]
        hold = df[m]
        y_obs = hold["label"].to_numpy().astype(int)
        y_true = hold["is_fraud"].to_numpy().astype(int)
        seq_cur = np.load(base_dir / "seq_logits.npy")[m]
        seq_new = np.load(registry.artifact_path(version) / "seq_logits.npy")[m]
        p_cur = offline_scores(base_dir, hold, seq_cur)
        p_new = offline_scores(registry.artifact_path(version), hold, seq_new)
        cb = parent_bundle["combos"]["full"]["thresholds"]
        nb = json.loads((registry.artifact_path(version) / "bundle.json").read_text())["combos"]["full"]["thresholds"]
        ftype = hold["fraud_type"].fillna("").to_numpy()
        days = SPLITS["holdout"][1] - SPLITS["holdout"][0]
        before = {
            "observed": evaluate_scores(y_obs, p_cur, cb["review"], cb["block"]),
            "truth": evaluate_scores(y_true, p_cur, cb["review"], cb["block"], ftype, days),
        }
        after = {
            "observed": evaluate_scores(y_obs, p_new, nb["review"], nb["block"]),
            "truth": evaluate_scores(y_true, p_new, nb["review"], nb["block"], ftype, days),
        }
        drift_ids = {
            c["case_id"] for c in json.loads((data_dir / "rings_truth.json").read_text()) if c.get("drift_variant")
        }
        drift = np.isin(hold["case_id"].to_numpy(), list(drift_ids))
        if drift.any():
            before["truth"]["drift_variant_recall_review"] = round(float((p_cur[drift] >= cb["review"]).mean()), 4)
            after["truth"]["drift_variant_recall_review"] = round(float((p_new[drift] >= nb["review"]).mean()), 4)
        ok, reason = gate(before["observed"], after["observed"], s.retrain_tolerance)
        registry.register(
            version,
            {**res["metrics"], "gate": {"before": before, "after": after}},
            res["data_hash"],
            parent=base,
            notes=reason,
        )
        if ok:
            registry.promote(version)
            db.execute("UPDATE feedback SET used_in_version = %s WHERE used_in_version IS NULL", (version,))
        else:
            registry.reject(version, reason)
        db.execute(
            """UPDATE retrain_runs SET finished_at = now(), status = 'done', candidate_version = %s, promoted = %s,
                   reason = %s, n_feedback = %s, before_metrics = %s, after_metrics = %s WHERE id = %s""",
            (version, ok, reason, len(fb), db.Jsonb(before), db.Jsonb(after), run_id),
        )
        out = {
            "run_id": run_id,
            "base_version": base,
            "candidate_version": version,
            "promoted": ok,
            "reason": reason,
            "n_feedback": len(fb),
            "seconds": round(time.time() - t0, 1),
            "before": before,
            "after": after,
        }
        log.info("retrain finished", promoted=ok, candidate=version, reason=reason)
        return out
    except Exception as e:
        db.execute(
            "UPDATE retrain_runs SET finished_at = now(), status = 'failed', reason = %s WHERE id = %s",
            (str(e)[:1000], run_id),
        )
        log.error("retrain failed", error=str(e)[:500])
        raise


def due_for_retrain() -> tuple[bool, str]:
    s = get_settings()
    last = db.fetch_one("SELECT max(started_at) AS t FROM retrain_runs WHERE status = 'done'")
    n = db.fetch_one("SELECT count(*) AS n FROM feedback WHERE used_in_version IS NULL")["n"]
    if n < s.retrain_min_feedback:
        return False, f"only {n} new feedback labels (< {s.retrain_min_feedback})"
    if last and last["t"] is not None:
        age_h = db.fetch_one("SELECT extract(epoch FROM now() - %s) / 3600 AS h", (last["t"],))["h"]
        if float(age_h) < s.retrain_interval_hours:
            return False, f"last retrain {float(age_h):.1f}h ago (< {s.retrain_interval_hours}h)"
    return True, f"{n} new feedback labels"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", default="manual")
    ap.add_argument("--schedule", action="store_true")
    ap.add_argument("--check-every-s", type=float, default=300)
    a = ap.parse_args(argv)
    setup_logging(get_settings().log_level)
    if not a.schedule:
        print(json.dumps(run_retrain(a.trigger), indent=1, default=str))
        return
    while True:
        try:
            ok, why = due_for_retrain()
            log.info("retrain schedule check", due=ok, why=why)
            if ok:
                run_retrain("schedule")
        except Exception as e:
            log.error("scheduled retrain error", error=str(e)[:300])
        time.sleep(a.check_every_s)


if __name__ == "__main__":
    main()
