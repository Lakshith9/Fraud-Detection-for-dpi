"""Model bundle: XGBoost (tabular) + LSTM/ONNX (sequence) + graph features, stacked by logistic regression.

Combos (all fitted on the calibration window; also the ablation set):
  xgb        Platt-calibrated XGBoost only              (degraded fallback)
  xgb_seq    XGBoost + sequence                          (graph unavailable/stale)
  xgb_graph  XGBoost + graph                             (sequence timed out / failed)
  full       XGBoost + sequence + graph                  (normal operation)

The meta-learner is linear in [xgb_margin, seq_logit, graph features], so final log-odds decompose exactly into
w_xgb * SHAP_j (per tabular feature) + w_seq * (seq_logit - E[seq_logit]) + w_g * (g - E[g]) + base value.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pipeline.features import GRAPH_FEATURES, TABULAR_FEATURES

COMBOS = {
    "xgb": {"seq": False, "graph": False},
    "xgb_seq": {"seq": True, "graph": False},
    "xgb_graph": {"seq": False, "graph": True},
    "full": {"seq": True, "graph": True},
}


def graph_transform(g: np.ndarray) -> np.ndarray:
    """Graph features as fed to the meta-learner (ring size log-scaled)."""
    g = np.asarray(g, dtype=np.float64).copy()
    k = GRAPH_FEATURES.index("max_ring_size")
    g[..., k] = np.log1p(g[..., k])
    return g


def meta_inputs(combo: str, xgb_margin, seq_logit, graph) -> np.ndarray:
    cols = [np.atleast_1d(np.asarray(xgb_margin, dtype=np.float64))]
    if COMBOS[combo]["seq"]:
        cols.append(np.atleast_1d(np.asarray(seq_logit, dtype=np.float64)))
    X = np.column_stack(cols)
    if COMBOS[combo]["graph"]:
        X = np.column_stack([X, np.atleast_2d(graph_transform(graph))])
    return X


def meta_feature_names(combo: str) -> list[str]:
    names = ["xgb_margin"]
    if COMBOS[combo]["seq"]:
        names.append("sequence_logit")
    if COMBOS[combo]["graph"]:
        names += GRAPH_FEATURES
    return names


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


@dataclass
class Combo:
    name: str
    coef: np.ndarray
    intercept: float
    review: float
    block: float
    means: np.ndarray  # background means of meta inputs (for additive explanations)

    def logit(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coef + self.intercept


class ModelBundle:
    """Loads a registry version directory and scores single transactions (serving) or batches (evaluation)."""

    def __init__(self, path: Path, load_sequence: bool = True):
        import shap
        import xgboost as xgb

        self.path = Path(path)
        self.meta = json.loads((self.path / "bundle.json").read_text())
        self.version = self.meta["version"]
        self.booster = xgb.Booster()
        self.booster.load_model(str(self.path / "xgb.json"))
        self.booster.set_param({"nthread": 1})
        self.typology = None
        if (self.path / "typology.json").exists():
            self.typology = xgb.Booster()
            self.typology.load_model(str(self.path / "typology.json"))
            self.typology.set_param({"nthread": 1})
        self.typology_classes = self.meta.get("typology_classes", [])
        self.combos = {
            k: Combo(
                k,
                np.array(v["coef"]),
                float(v["intercept"]),
                float(v["thresholds"]["review"]),
                float(v["thresholds"]["block"]),
                np.array(v["means"]),
            )
            for k, v in self.meta["combos"].items()
        }
        self.seq = None
        if load_sequence and (self.path / "seq.onnx").exists():
            from models.sequence_model import OnnxSequenceScorer

            self.seq = OnnxSequenceScorer(self.path / "seq.onnx")
        self.explainer = shap.TreeExplainer(self.booster)
        # Base value (bias) of the additive decomposition. Taken from XGBoost's own TreeSHAP (pred_contribs) because
        # shap's TreeExplainer reports expected_value = 0 for XGBoost >= 3 models (base_score stored as a vector);
        # its per-feature SHAP values are exact, so the two are combined and verified below.
        probe = np.zeros((1, len(TABULAR_FEATURES)), dtype=np.float32)
        contribs = self.booster.predict(xgb.DMatrix(probe, feature_names=TABULAR_FEATURES), pred_contribs=True)[0]
        self.xgb_base = float(contribs[-1])
        check = float(np.asarray(self.explainer.shap_values(probe))[0].sum()) + self.xgb_base
        if abs(check - self.xgb_margin(probe[0])) > 1e-3:
            raise RuntimeError(f"SHAP additivity check failed for model {self.version}")
        self.n_tab = len(TABULAR_FEATURES)
        # warm-up
        x = np.zeros((1, self.n_tab), dtype=np.float32)
        for _ in range(10):
            self.booster.inplace_predict(x, predict_type="margin")
        self.explainer.shap_values(x)

    # ------------------------------------------------------------------ single-row hot path pieces
    def xgb_margin(self, tab_row: np.ndarray) -> float:
        return float(self.booster.inplace_predict(tab_row.reshape(1, -1), predict_type="margin")[0])

    def seq_logit(self, seq: np.ndarray) -> float:
        if self.seq is None:
            raise RuntimeError("sequence model not loaded")
        return self.seq.logit(seq)

    def combine(self, combo: str, margin: float, seq_logit: float | None, graph: np.ndarray):
        c = self.combos[combo]
        x = meta_inputs(combo, margin, seq_logit if seq_logit is not None else 0.0, np.asarray(graph)[None, :])[0]
        z = float(c.logit(x))
        p = 1.0 / (1.0 + math.exp(-z))
        decision = "BLOCK" if p >= c.block else "REVIEW" if p >= c.review else "ALLOW"
        return p, z, decision, x

    def typology_of(self, tab_row: np.ndarray, graph: np.ndarray) -> dict:
        if self.typology is None:
            return {}
        x = np.concatenate([tab_row, graph]).reshape(1, -1).astype(np.float32)
        probs = self.typology.inplace_predict(x)[0]
        return {c: round(float(p), 4) for c, p in zip(self.typology_classes, probs)}

    def shap_tabular(self, tab_row: np.ndarray) -> np.ndarray:
        return np.asarray(self.explainer.shap_values(tab_row.reshape(1, -1).astype(np.float32)))[0]

    # ------------------------------------------------------------------ batch (evaluation / retraining)
    def batch_scores(self, tab: np.ndarray, seq: np.ndarray | None, graph: np.ndarray, combo: str = "full"):
        import xgboost as xgb

        m = self.booster.predict(xgb.DMatrix(tab, nthread=-1), output_margin=True)
        s = self.seq.logits(seq) if (self.seq is not None and seq is not None) else np.zeros(len(tab))
        x = meta_inputs(combo, m, s, graph)
        return sigmoid(self.combos[combo].logit(x)), m, s
