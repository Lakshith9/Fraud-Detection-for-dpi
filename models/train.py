"""Train the fraud ensemble with a TIME-BASED split and write a registry version.

Splits (days since dataset start, event time):
  train    [0, 36)    XGBoost + LSTM fitting
  valid_a  [36, 41)   XGBoost hyper-parameter selection / early stopping, LSTM epoch selection
  valid_b  [41, 46)   stacked meta-learner (= calibration) + decision thresholds
  test     [46, 60)   held-out future window (reported metrics); live demo replays this window
           [46, 53)   'feedback window' (operators label alerts here in the demo)
           [53, 60)   'holdout' used to gate retrained models

Thresholds are chosen on valid_b from an operational false-positive budget:
  REVIEW: FPR <= 0.5% of legitimate traffic,  BLOCK: FPR <= 0.05%.

Usage: python -m models.train --features data/features --data data [--register --promote]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from models.ensemble import COMBOS, meta_feature_names, meta_inputs, sigmoid
from models.metrics import evaluate_scores, reference_bins, threshold_for_fpr
from pipeline.features import GRAPH_FEATURES, TABULAR_FEATURES

DAY = 86_400_000
SPLITS = {
    "train": (0, 36),
    "valid_a": (36, 41),
    "valid_b": (41, 46),
    "test": (46, 60),
    "feedback_window": (46, 53),
    "holdout": (53, 60),
}
REVIEW_FPR = 0.005
BLOCK_FPR = 0.0005
SEED = 42
PSI_FEATURES = [  # monitored for drift (cyclical hour_of_day is excluded: needs a same-hour baseline)
    "amount_log",
    "payer_tx_24h",
    "device_accounts",
    "new_payee",
    "pass_through_ratio_24h",
    "payee_in_degree",
    "geo_distance_km",
    "secs_since_last_tx_log",
    "device_binding_age_hours_log",
]
XGB_GRID = [
    {"max_depth": 4, "eta": 0.05, "min_child_weight": 1, "spw": "sqrt"},
    {"max_depth": 6, "eta": 0.05, "min_child_weight": 1, "spw": "sqrt"},
    {"max_depth": 6, "eta": 0.05, "min_child_weight": 5, "spw": "full"},
    {"max_depth": 8, "eta": 0.05, "min_child_weight": 3, "spw": "sqrt"},
    {"max_depth": 6, "eta": 0.1, "min_child_weight": 1, "spw": "sqrt"},
]


def log(msg):
    print(f"[train {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def split_masks(ts: np.ndarray, start_ms: int) -> dict[str, np.ndarray]:
    day = (ts - start_ms) // DAY
    return {k: (day >= a) & (day < b) for k, (a, b) in SPLITS.items()}


def xgb_params(g: dict, pos: int, neg: int) -> dict:
    spw = float(np.sqrt(neg / max(pos, 1))) if g["spw"] == "sqrt" else float(neg / max(pos, 1))
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": g["max_depth"],
        "eta": g["eta"],
        "min_child_weight": g["min_child_weight"],
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "scale_pos_weight": spw,
        "seed": SEED,
        "nthread": -1,
        "max_bin": 256,
    }


def fit_xgb(Xtr, ytr, Xva, yva, grid=XGB_GRID, wtr=None) -> tuple[xgb.Booster, dict, list]:
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr, feature_names=TABULAR_FEATURES)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=TABULAR_FEATURES)
    pos, neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    results, best = [], None
    for g in grid:
        t = time.time()
        params = xgb_params(g, pos, neg)
        b = xgb.train(
            params, dtr, num_boost_round=1500, evals=[(dva, "valid")], early_stopping_rounds=80, verbose_eval=False
        )
        ap = float(b.best_score)
        results.append(
            {
                **g,
                "best_iteration": int(b.best_iteration),
                "valid_pr_auc": round(ap, 4),
                "seconds": round(time.time() - t, 1),
            }
        )
        log(f"  xgb {g} -> valid PR-AUC {ap:.4f} @ {b.best_iteration} ({time.time() - t:.0f}s)")
        if best is None or ap > best[0]:
            best = (ap, b, g)
    booster = best[1]
    booster = booster[: booster.best_iteration + 1]
    return booster, best[2], results


def fit_combos(margin, seq_logit, graph, y) -> dict:
    out = {}
    for combo in COMBOS:
        X = meta_inputs(combo, margin, seq_logit, graph)
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(X, y)
        p = lr.predict_proba(X)[:, 1]
        out[combo] = {
            "features": meta_feature_names(combo),
            "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
            "means": X.mean(axis=0).tolist(),
            "thresholds": {"review": threshold_for_fpr(y, p, REVIEW_FPR), "block": threshold_for_fpr(y, p, BLOCK_FPR)},
        }
    return out


def combo_probs(combos: dict, combo: str, margin, seq_logit, graph) -> np.ndarray:
    c = combos[combo]
    X = meta_inputs(combo, margin, seq_logit, graph)
    return sigmoid(X @ np.array(c["coef"]) + c["intercept"])


def fit_typology(X, ft) -> tuple[xgb.Booster, list[str]]:
    classes = sorted(set(ft))
    yk = np.array([classes.index(t) for t in ft])
    d = xgb.DMatrix(X, label=yk)
    b = xgb.train(
        {
            "objective": "multi:softprob",
            "num_class": len(classes),
            "max_depth": 4,
            "eta": 0.1,
            "seed": SEED,
            "nthread": -1,
        },
        d,
        num_boost_round=150,
    )
    return b, classes


def psi_reference(df: pd.DataFrame, scores: np.ndarray) -> dict:
    ref = {}
    for f in PSI_FEATURES:
        edges, props = reference_bins(df[f].to_numpy())
        ref[f] = {"edges": edges, "props": props}
    edges, props = reference_bins(scores)
    ref["__score__"] = {"edges": edges, "props": props}
    return ref


def evaluate_all(combos, margin, seq_logit, graph, y_true, y_obs, ftype, days) -> dict:
    out = {}
    for combo in COMBOS:
        p = combo_probs(combos, combo, margin, seq_logit, graph)
        th = combos[combo]["thresholds"]
        out[combo] = evaluate_scores(y_true, p, th["review"], th["block"], ftype, days)
        out[combo]["observed_labels"] = {
            k: v
            for k, v in evaluate_scores(y_obs, p, th["review"], th["block"]).items()
            if k in ("pr_auc", "roc_auc", "recall_at_1pct_fpr")
        }
    return out


def load_features(features_dir: Path):
    df = pd.read_parquet(features_dir / "features.parquet")
    seq = np.load(features_dir / "seq.npy", mmap_mode="r")
    return df, seq


def train(
    features_dir: Path,
    data_dir: Path,
    version: str,
    out_root: Path,
    seq_from: Path | None = None,
    label_override: pd.Series | None = None,
    extra: dict | None = None,
    notes: str = "",
    grid: list[dict] | None = None,
) -> dict:
    """Train a full bundle. If seq_from is given, reuse that version's sequence model (retraining path).

    label_override: Series indexed by transfer_id -> 0/1 replacing observed labels (operator feedback).
    extra: additional live rows {"tab": [n,F], "graph": [n,G], "seq_logit": [n], "y": [n]} added to training.
    """
    t_all = time.time()
    out = out_root / version
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    df, seq = load_features(features_dir)
    start_ms = int(df["ts"].min()) // DAY * DAY
    m = split_masks(df["ts"].to_numpy(), start_ms)
    y_obs = df["label"].to_numpy().astype(int)
    if label_override is not None and len(label_override):
        lo = df["transfer_id"].map(label_override)
        hit = lo.notna().to_numpy()
        y_obs = np.where(hit, lo.fillna(0).to_numpy(), y_obs).astype(int)
        log(f"label overrides applied to {int(hit.sum())} offline rows")
    y_true = df["is_fraud"].to_numpy().astype(int)
    ftype = df["fraud_type"].fillna("").to_numpy()
    X = df[TABULAR_FEATURES].to_numpy(dtype=np.float32)
    G = df[GRAPH_FEATURES].to_numpy(dtype=np.float32)
    for k in ("train", "valid_a", "valid_b", "test"):
        log(f"split {k:8s} rows={int(m[k].sum()):>8,} observed positives={int(y_obs[m[k]].sum()):>5}")
        if k != "test" and y_obs[m[k]].sum() == 0:
            raise ValueError(f"time split '{k}' {SPLITS[k]} has no observed fraud; generate more data or days")

    # ---------------------------------------------------------------- XGBoost
    Xtr, ytr = X[m["train"]], y_obs[m["train"]]
    wtr = None
    if extra is not None and len(extra["y"]):
        Xtr = np.vstack([Xtr, extra["tab"].astype(np.float32)])
        ytr = np.concatenate([ytr, extra["y"].astype(int)])
        wtr = np.concatenate([np.ones(int(m["train"].sum())), np.full(len(extra["y"]), extra.get("weight", 1.0))])
        log(f"added {len(extra['y'])} live feedback-window rows ({int(extra['y'].sum())} labelled fraud)")
    log("fitting XGBoost grid (time-based validation on valid_a)")
    booster, best_g, grid_res = fit_xgb(Xtr, ytr, X[m["valid_a"]], y_obs[m["valid_a"]], grid or XGB_GRID, wtr)
    booster.save_model(str(out / "xgb.json"))
    margin = booster.predict(xgb.DMatrix(X, feature_names=TABULAR_FEATURES), output_margin=True)

    # ---------------------------------------------------------------- sequence model
    if seq_from is None:
        from models.sequence_model import OnnxSequenceScorer, export_onnx, predict_logits, train_sequence_model

        log("training LSTM sequence model")
        tr_idx = np.where(m["train"])[0]
        va_idx = np.where(m["valid_a"])[0]
        Xs_tr = np.asarray(seq[tr_idx[0] : tr_idx[-1] + 1])
        Xs_va = np.asarray(seq[va_idx[0] : va_idx[-1] + 1], dtype=np.float32)
        model, seq_ap = train_sequence_model(Xs_tr, y_obs[tr_idx], Xs_va, y_obs[va_idx], log=log)
        del Xs_tr
        export_onnx(model, out / "seq.onnx")
        log("scoring all rows with the sequence model")
        seq_logit = np.concatenate(
            [
                predict_logits(model, np.asarray(seq[i : i + 50_000], dtype=np.float32))
                for i in range(0, len(df), 50_000)
            ]
        )
        onnx = OnnxSequenceScorer(out / "seq.onnx")
        chk = np.asarray(seq[:2000], dtype=np.float32)
        onnx_diff = float(np.abs(onnx.logits(chk) - seq_logit[:2000]).max())
        log(f"ONNX vs torch max |diff| = {onnx_diff:.2e}")
        seq_info = {"valid_pr_auc": round(seq_ap, 4), "onnx_max_abs_diff": onnx_diff, "trained_in": version}
    else:
        shutil.copy(seq_from / "seq.onnx", out / "seq.onnx")
        seq_logit = np.load(seq_from / "seq_logits.npy")
        seq_info = json.loads((seq_from / "bundle.json").read_text())["sequence"]
        seq_info = {**seq_info, "reused_from": seq_from.name}
    np.save(out / "seq_logits.npy", seq_logit.astype(np.float32))

    # ---------------------------------------------------------------- stacking / calibration / thresholds
    log("fitting stacked meta-learners on valid_b")
    vb = m["valid_b"]
    combos = fit_combos(margin[vb], seq_logit[vb], G[vb], y_obs[vb])

    # ---------------------------------------------------------------- typology classifier
    fr = (y_obs == 1) & (ftype != "") & (m["train"] | m["valid_a"] | m["valid_b"])
    typ, classes = fit_typology(np.hstack([X[fr], G[fr]]), ftype[fr])
    typ.save_model(str(out / "typology.json"))
    te_fr = m["test"] & (y_true == 1)
    tp = typ.predict(xgb.DMatrix(np.hstack([X[te_fr], G[te_fr]])))
    typ_acc = float((np.array(classes)[tp.argmax(1)] == ftype[te_fr]).mean()) if te_fr.any() else None

    # ---------------------------------------------------------------- evaluation
    log("evaluating on the held-out future window")
    metrics = {"splits": {k: [int(a), int(b)] for k, (a, b) in SPLITS.items()}}
    for split in ("valid_b", "test", "holdout"):
        s = m[split]
        days = SPLITS[split][1] - SPLITS[split][0]
        metrics[split] = evaluate_all(combos, margin[s], seq_logit[s], G[s], y_true[s], y_obs[s], ftype[s], days)
    full_test = combo_probs(combos, "full", margin[m["test"]], seq_logit[m["test"]], G[m["test"]])
    case_test = df["case_id"].to_numpy()[m["test"]]
    drift_ids = {
        c["case_id"] for c in json.loads((data_dir / "rings_truth.json").read_text()) if c.get("drift_variant")
    }
    drift = np.isin(case_test, list(drift_ids))
    if drift.any():
        th = combos["full"]["thresholds"]["review"]
        metrics["test"]["drift_variants"] = {
            "n": int(drift.sum()),
            "recall_review_full": round(float((full_test[drift] >= th).mean()), 4),
        }
    metrics["typology_accuracy_test"] = round(typ_acc, 4) if typ_acc is not None else None
    imp = booster.get_score(importance_type="gain")
    metrics["xgb_top_gain"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15])

    bundle = {
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_hash": manifest["data_hash"],
        "tabular_features": TABULAR_FEATURES,
        "graph_features": GRAPH_FEATURES,
        "graph_transform": "max_ring_size -> log1p",
        "combos": combos,
        "fpr_budget": {"review": REVIEW_FPR, "block": BLOCK_FPR},
        "xgb": {"params": best_g, "grid": grid_res, "best_iteration": int(booster.num_boosted_rounds())},
        "sequence": seq_info,
        "typology_classes": classes,
        "psi_reference": psi_reference(df[m["train"]], sigmoid(margin[m["train"]])),
        "start_ms": start_ms,
        "notes": notes,
    }
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    log(f"done in {time.time() - t_all:.0f}s -> {out}")
    return {"version": version, "path": str(out), "metrics": metrics, "data_hash": manifest["data_hash"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--version", default=None)
    ap.add_argument("--register", action="store_true", help="register in the Postgres model registry")
    ap.add_argument("--promote", action="store_true", help="promote to production after registering")
    a = ap.parse_args(argv)
    from core.config import get_settings
    from models import registry

    root = get_settings().artifacts_dir
    version = a.version or registry.next_version()
    res = train(a.features, a.data, version, root)
    t = res["metrics"]["test"]
    log("TEST (future window) summary:")
    for combo in COMBOS:
        r = t[combo]
        log(
            f"  {combo:9s} PR-AUC={r['pr_auc']} ROC-AUC={r['roc_auc']} R@1%FPR={r['recall_at_1pct_fpr']} "
            f"review P/R={r['review']['precision']}/{r['review']['recall']} block P/R={r['block']['precision']}/"
            f"{r['block']['recall']}"
        )
    if a.register:
        from core import db

        db.init_schema()
        registry.register(version, res["metrics"], res["data_hash"], notes="initial training")
        if a.promote:
            registry.promote(version)
            log(f"promoted {version} to production")


if __name__ == "__main__":
    main()
