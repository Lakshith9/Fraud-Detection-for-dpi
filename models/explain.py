"""Additive explanations in the ensemble's log-odds space + plain-language reasons for operators.

final_logit = base
            + sum_j w_xgb * SHAP_j(x)                       (TreeExplainer on the XGBoost margin, per feature)
            + w_seq * (seq_logit - E[seq_logit])             (sequence model, one pseudo-feature)
            + sum_k w_g_k * (g_k - E[g_k])                   (graph features, exact for the linear meta-learner)
"""

from __future__ import annotations

import math

import numpy as np

from models.ensemble import COMBOS, ModelBundle, meta_feature_names
from pipeline.features import TABULAR_FEATURES


def _fmt_amt(x):
    return f"{x:,.0f}"


def reason_text(name: str, value: float, ctx: dict) -> str:
    v = value
    nan = v is None or (isinstance(v, float) and math.isnan(v))
    if name == "amount_ratio_mean_30d" and not nan:
        return f"amount {v:.1f}x this payer's 30-day average"
    if name == "amount_ratio_max_30d" and not nan:
        return f"amount {v:.1f}x this payer's 30-day maximum"
    if name == "amount_zscore_payer" and not nan:
        return f"amount is {v:+.1f} std-devs from this payer's usual"
    if name == "amount_log":
        return f"transfer amount {_fmt_amt(ctx.get('amount', 0))}"
    if name == "payer_hist_count_30d":
        return "payer has no recent history (cold start)" if v == 0 else f"payer made {int(v)} transfers in 30 days"
    if name == "device_accounts":
        return f"device shared by {int(v)} accounts"
    if name == "ip_accounts":
        return f"IP address used by {int(v)} accounts"
    if name == "new_device_for_payer":
        return "first time this payer uses this device" if v else "known device for this payer"
    if name == "device_binding_age_hours_log" and not nan:
        h = math.expm1(v)
        return f"device bound to account {h:.1f} h ago" if h < 72 else f"device bound {h / 24:.0f} days ago"
    if name == "new_payee":
        return "first transfer to this payee" if v else "payer has paid this payee before"
    if name == "geo_distance_km" and not nan:
        return f"{v:,.0f} km from payer's previous transaction location"
    if name == "geo_speed_kmh" and not nan:
        return f"implied travel speed {v:,.0f} km/h since last transaction"
    if name == "is_night":
        return "initiated at night (00-05h)" if v else "daytime transaction"
    if name == "hour_deviation" and not nan:
        return f"{v:.1f} h away from payer's usual time of day"
    if name == "payee_age_hours_log":
        h = math.expm1(v)
        return f"payee first seen {h:.0f} h ago" if h < 72 else f"payee active for {h / 24:.0f} days"
    if name == "payer_age_hours_log":
        h = math.expm1(v)
        return f"payer first seen {h:.0f} h ago" if h < 72 else f"payer active for {h / 24:.0f} days"
    if name == "pass_through_ratio_24h":
        return f"outflow is {v:.0%} of inflow in last 24h (pass-through)"
    if name == "inflow_24h_log":
        return f"received {_fmt_amt(math.expm1(v))} in last 24h"
    if name == "mins_since_last_inflow_log" and not nan:
        return f"funds received {math.expm1(v):.0f} min before this transfer"
    if name in ("near_settlement_cutoff", "mins_to_settlement_cutoff"):
        return f"{ctx.get('mins_to_cutoff', 0):.0f} min before a settlement cutoff"
    if name == "near_threshold":
        return "amount just below a reporting threshold (structuring)" if v else "amount not near a threshold"
    if name in ("is_round_amount", "round_level"):
        return "round amount" if v else "non-round amount"
    if name == "fsp_rhv_zscore":
        return f"payer FSP round high-value volume z-score {v:.1f}"
    if name == "fsp_volume_zscore":
        return f"payer FSP volume z-score {v:.1f} vs last 24h"
    if name == "counterparty_risk_score":
        return f"counterparty risk score {v:.2f}"
    if name == "payee_report_count":
        return f"payee reported for fraud {int(v)} time(s)" if v else "payee never reported"
    if name == "payee_reversal_rate":
        return f"payee reversal rate {v:.1%}"
    if name == "payer_reversal_rate":
        return f"payer reversal rate {v:.1%}"
    if name == "is_reversal":
        return "reversal of a prior transfer" if v else "not a reversal"
    if name in ("payer_tx_1m", "payer_tx_1h", "payer_tx_24h"):
        w = name.rsplit("_", 1)[1]
        return f"payer made {int(v)} transfers in last {w}"
    if name in ("payee_tx_1h", "payee_tx_24h"):
        return f"payee received {int(v)} transfers in last {name.rsplit('_', 1)[1]}"
    if name in ("device_tx_1h", "device_tx_24h"):
        return f"device used for {int(v)} transfers in last {name.rsplit('_', 1)[1]}"
    if name == "payee_in_degree":
        return f"payee received from {int(v)} distinct payers"
    if name == "payer_out_degree":
        return f"payer has paid {int(v)} distinct payees"
    if name == "secs_since_last_tx_log" and not nan:
        return f"{math.expm1(v) / 60:.1f} min since payer's previous transfer"
    if name == "sequence_logit":
        return (
            "recent transaction sequence looks anomalous"
            if ctx.get("seq_dir", 1) > 0
            else "recent transaction sequence looks normal"
        )
    if name in ("payer_ring_score", "payee_ring_score", "device_ring_score", "in_cycle", "max_ring_size"):
        ring = ctx.get("payer_ring") or ctx.get("payee_ring") or ctx.get("device_ring")
        if ring and ring.get("rid"):
            size = int(float(ring.get("sz") or 0))
            cyc = int(float(ring.get("cyc") or 0)) if ring.get("cyc") else 0
            if cyc:
                return f"part of a {cyc}-node circular flow (ring {ring['rid']})"
            return f"linked to suspected fraud ring {ring['rid']} ({size} accounts, score {float(ring['rs']):.2f})"
        return "no fraud-ring linkage"
    if name in ("payer_fraud_neighbor_ratio", "payee_fraud_neighbor_ratio"):
        who = "payer" if name.startswith("payer") else "payee"
        return f"{v:.0%} of {who}'s counterparties are flagged/reported"
    pretty = name.replace("_", " ")
    return f"{pretty} = {v:.3g}" if not nan else f"{pretty} unavailable"


def explain(
    bundle: ModelBundle,
    combo: str,
    tab_row: np.ndarray,
    graph_row: np.ndarray,
    meta_x: np.ndarray,
    ctx: dict,
    seq_logit: float | None,
    top_k: int = 5,
    shap_vals: np.ndarray | None = None,
) -> dict:
    """Return the full additive explanation and the top-k operator reasons.

    `shap_vals` may be precomputed (batched TreeSHAP, see `explain_rows`); otherwise it is computed for this row.
    """
    c = bundle.combos[combo]
    names = meta_feature_names(combo)
    w_xgb = float(c.coef[0])
    if shap_vals is None:
        shap_vals = bundle.shap_tabular(tab_row)
    contribs = []
    for j, fname in enumerate(TABULAR_FEATURES):
        v = float(shap_vals[j]) * w_xgb
        if abs(v) > 1e-9:
            contribs.append((fname, float(tab_row[j]), v, "tabular"))
    base = c.intercept + w_xgb * bundle.xgb_base
    k = 1
    if COMBOS[combo]["seq"]:
        v = float(c.coef[1]) * (float(meta_x[1]) - float(c.means[1]))
        base += float(c.coef[1]) * float(c.means[1])
        contribs.append(("sequence_logit", float(meta_x[1]), v, "sequence"))
        k = 2
    if COMBOS[combo]["graph"]:
        for gi, gname in enumerate(names[k:]):
            idx = k + gi
            v = float(c.coef[idx]) * (float(meta_x[idx]) - float(c.means[idx]))
            base += float(c.coef[idx]) * float(c.means[idx])
            raw = float(graph_row[gi])
            if abs(v) > 1e-9:
                contribs.append((gname, raw, v, "graph"))
    contribs.sort(key=lambda t: abs(t[2]), reverse=True)
    top = []
    for fname, val, v, src in contribs[:top_k]:
        cx = dict(ctx)
        cx["seq_dir"] = 1 if v > 0 else -1
        top.append(
            {
                "feature": fname,
                "value": None if isinstance(val, float) and math.isnan(val) else round(val, 4),
                "contribution": round(v, 4),
                "direction": "increases_risk" if v > 0 else "decreases_risk",
                "source": src,
                "reason": reason_text(fname, val, cx),
            }
        )
    return {
        "combo": combo,
        "base_logit": round(base, 4),
        "contributions": [
            {
                "feature": f,
                "value": None if isinstance(val, float) and math.isnan(val) else round(val, 4),
                "contribution": round(v, 5),
                "source": s,
            }
            for f, val, v, s in contribs
        ],
        "top_reasons": top,
    }


def _row_tab(row: dict) -> np.ndarray:
    feats = row.get("features") or {}
    return np.array([np.nan if feats.get(f) is None else feats[f] for f in TABULAR_FEATURES], dtype=np.float32)


def explain_rows(bundle: ModelBundle, rows: list[dict]) -> list[dict]:
    """Batched explanations: ONE vectorised TreeSHAP call for all rows, then the cheap per-row assembly."""
    if not rows:
        return []
    X = np.vstack([_row_tab(r) for r in rows])
    sv = np.asarray(bundle.explainer.shap_values(X))
    return [explain_row(bundle, r, shap_vals=sv[i]) for i, r in enumerate(rows)]


def explain_row(bundle: ModelBundle, row: dict, shap_vals: np.ndarray | None = None) -> dict:
    """Explanation for a persisted scored-transaction row (stored features + model components).

    Used for asynchronous ALLOW-path explanations (pipeline/persister.py) and on-demand API requests; it reproduces
    exactly the meta-learner inputs used at scoring time.
    """
    from pipeline.features import GRAPH_FEATURES

    comp = row.get("components") or {}
    gdict = row.get("graph") or {}
    combo = comp.get("combo", "xgb")
    tab = _row_tab(row)
    graph = np.array([gdict.get(f, 0.0) for f in GRAPH_FEATURES], dtype=np.float32)
    margin = comp.get("xgb_margin")
    if margin is None:
        margin = bundle.xgb_margin(tab)
    _, _, _, meta_x = bundle.combine(combo, margin, comp.get("seq_logit"), graph)
    ctx = {"amount": row.get("amount"), **(gdict.get("context") or {}), **(comp.get("context") or {})}
    e = explain(bundle, combo, tab, graph, meta_x, ctx, comp.get("seq_logit"), shap_vals=shap_vals)
    e["context"] = comp.get("context") or {}
    return e
