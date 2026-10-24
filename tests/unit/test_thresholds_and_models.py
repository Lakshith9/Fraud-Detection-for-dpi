import json
from pathlib import Path

import numpy as np
import pytest

from models.metrics import at_threshold, evaluate_scores, psi, recall_at_fpr, reference_bins, threshold_for_fpr
from models.retrain import gate
from models.train import fit_combos

ART = Path(__file__).resolve().parents[2] / "models" / "artifacts"


def test_threshold_respects_fpr_budget():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(10_000), np.ones(100)]
    p = np.r_[rng.random(10_000) * 0.8, 0.5 + rng.random(100) * 0.5]
    for budget in (0.01, 0.005, 0.0005):
        t = threshold_for_fpr(y, p, budget)
        assert at_threshold(y, p, t)["fpr"] <= budget


def test_at_threshold_counts():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    r = at_threshold(y, p, 0.5)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_recall_at_fpr_perfect_separation():
    y = np.array([0] * 100 + [1] * 10)
    p = np.r_[np.linspace(0, 0.4, 100), np.linspace(0.6, 1, 10)]
    assert recall_at_fpr(y, p, 0.01) == 1.0
    e = evaluate_scores(y, p, 0.5, 0.9)
    assert e["pr_auc"] == 1.0 and e["roc_auc"] == 1.0


def test_meta_learner_combos_and_thresholds_ordered():
    rng = np.random.default_rng(1)
    n = 5000
    y = (rng.random(n) < 0.05).astype(int)
    margin = y * 3 + rng.normal(0, 1, n)
    seq = y * 1.5 + rng.normal(0, 1, n)
    graph = np.column_stack([y * rng.random(n)] + [rng.random(n) for _ in range(6)])
    combos = fit_combos(margin, seq, graph, y)
    assert set(combos) == {"xgb", "xgb_seq", "xgb_graph", "full"}
    for c in combos.values():
        assert c["thresholds"]["block"] >= c["thresholds"]["review"]
        assert c["coef"][0] > 0  # higher xgb margin => higher risk


def test_gate_promotes_only_within_tolerance():
    cur = {"pr_auc": 0.80, "recall_at_1pct_fpr": 0.90}
    ok, _ = gate(cur, {"pr_auc": 0.795, "recall_at_1pct_fpr": 0.92}, 0.01)
    assert ok
    ok, why = gate(cur, {"pr_auc": 0.70, "recall_at_1pct_fpr": 0.95}, 0.01)
    assert not ok and "pr_auc" in why


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 20_000)
    edges, props = reference_bins(x)
    assert psi(np.array(props), rng.normal(0, 1, 20_000), np.array(edges)) < 0.01
    assert psi(np.array(props), rng.normal(1.5, 1, 20_000), np.array(edges)) > 0.25


def _latest_bundle():
    if not ART.exists():
        return None
    vs = sorted([p for p in ART.iterdir() if (p / "bundle.json").exists()], key=lambda p: p.stat().st_mtime)
    return vs[-1] if vs else None


def test_explanation_is_exactly_additive_in_logit_space():
    path = _latest_bundle()
    if path is None:
        pytest.skip("no trained model bundle available")
    from models.ensemble import ModelBundle
    from models.explain import explain
    from pipeline.features import GRAPH_FEATURES, TABULAR_FEATURES

    b = ModelBundle(path)
    rng = np.random.default_rng(3)
    for combo in ("full", "xgb", "xgb_seq", "xgb_graph"):
        tab = rng.normal(0, 1, len(TABULAR_FEATURES)).astype(np.float32)
        tab[1] = np.nan
        graph = np.abs(rng.normal(0, 0.3, len(GRAPH_FEATURES))).astype(np.float32)
        margin = b.xgb_margin(tab)
        seq_logit = float(rng.normal())
        p, z, _, meta_x = b.combine(combo, margin, seq_logit, graph)
        e = explain(b, combo, tab, graph, meta_x, {"amount": 100.0}, seq_logit, top_k=5)
        total = e["base_logit"] + sum(c["contribution"] for c in e["contributions"])
        assert total == pytest.approx(z, abs=2e-3), combo
        assert len(e["top_reasons"]) <= 5 and all(r["reason"] for r in e["top_reasons"])
    meta = json.loads((path / "bundle.json").read_text())
    assert meta["combos"]["full"]["thresholds"]["block"] >= meta["combos"]["full"]["thresholds"]["review"]
