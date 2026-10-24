"""Evaluation metrics (no fabricated numbers: everything here is computed from predictions)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def recall_at_fpr(y, p, fpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_for_fpr(y, p, fpr_target: float) -> float:
    """Smallest threshold whose false-positive rate on (y, p) is <= fpr_target."""
    neg = np.sort(np.asarray(p)[np.asarray(y) == 0])[::-1]
    if len(neg) == 0:
        return 0.5
    k = int(np.floor(fpr_target * len(neg)))
    k = min(k, len(neg) - 1)
    return float(np.nextafter(neg[k], 1.0))


def at_threshold(y, p, t: float) -> dict:
    y = np.asarray(y)
    pred = np.asarray(p) >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    return {
        "threshold": round(float(t), 6),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "fpr": round(fp / (fp + tn), 6) if fp + tn else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "flagged": tp + fp,
    }


def evaluate_scores(y, p, review_t: float, block_t: float, fraud_type=None, days: float | None = None) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "pr_auc": round(float(average_precision_score(y, p)), 4) if y.sum() else None,
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None,
        "recall_at_1pct_fpr": round(recall_at_fpr(y, p, 0.01), 4),
        "recall_at_0.1pct_fpr": round(recall_at_fpr(y, p, 0.001), 4),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "review": at_threshold(y, p, review_t),
        "block": at_threshold(y, p, block_t),
    }
    if days:
        out["review"]["alerts_per_day"] = round(out["review"]["flagged"] / days, 1)
        out["block"]["blocks_per_day"] = round(out["block"]["flagged"] / days, 1)
    if fraud_type is not None:
        ft = np.asarray(fraud_type)
        per = {}
        for t in sorted(set(ft[y == 1])):
            m = (ft == t) & (y == 1)
            per[t] = {
                "n": int(m.sum()),
                "recall_review": round(float((p[m] >= review_t).mean()), 4),
                "recall_block": round(float((p[m] >= block_t).mean()), 4),
            }
        out["per_fraud_type"] = per
    return out


def psi(expected_props: np.ndarray, actual: np.ndarray, edges: np.ndarray) -> float:
    """Population stability index of `actual` against reference bin proportions."""
    actual = np.asarray(actual, dtype=np.float64)
    actual = actual[~np.isnan(actual)]
    if len(actual) == 0:
        return 0.0
    counts = np.histogram(actual, bins=edges)[0].astype(np.float64)
    a = counts / counts.sum() if counts.sum() else counts
    e = np.asarray(expected_props, dtype=np.float64)
    a = np.clip(a, 1e-4, None)
    e = np.clip(e, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def reference_bins(x: np.ndarray, n_bins: int = 10) -> tuple[list[float], list[float]]:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        qs = np.array([x.min() - 0.5, x.min() + 0.5, x.max() + 1.0]) if len(x) else np.array([0.0, 1.0, 2.0])
    edges = qs.copy()
    edges[0], edges[-1] = -1e12, 1e12
    props = np.histogram(x, bins=edges)[0] / max(len(x), 1)
    return edges.tolist(), props.tolist()
