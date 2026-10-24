"""Alerts, transaction investigation, explanations and fraud rings."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import operator
from api.state import STATE
from core import db
from models.ensemble import ModelBundle
from models.explain import explain_row

router = APIRouter(prefix="/v1", tags=["investigation"])


@router.get("/alerts", summary="List alerts (newest first)")
def alerts(
    decision: str | None = Query(None, pattern="^(REVIEW|BLOCK)$"),
    fraud_type: str | None = None,
    fsp: str | None = None,
    status: str | None = None,
    min_score: float = 0.0,
    limit: int = Query(100, ge=1, le=1000),
    before_id: int | None = None,
    _op: str = Depends(operator),
):
    where, params = ["risk_score >= %s"], [min_score]
    if decision:
        where.append("decision = %s")
        params.append(decision)
    if fraud_type:
        where.append("predicted_type = %s")
        params.append(fraud_type)
    if fsp:
        where.append("(payer_fsp = %s OR payee_fsp = %s)")
        params += [fsp, fsp]
    if status:
        where.append("status = %s")
        params.append(status)
    if before_id:
        where.append("alert_id < %s")
        params.append(before_id)
    rows = db.fetch_all(
        f"SELECT * FROM alerts WHERE {' AND '.join(where)} ORDER BY alert_id DESC LIMIT %s", (*params, limit)
    )
    return {"items": rows, "count": len(rows)}


def _tx(transfer_id: str) -> dict:
    row = db.fetch_one("SELECT * FROM transactions WHERE transfer_id = %s", (transfer_id,))
    if not row:
        raise HTTPException(404, "transaction not found")
    return row


@router.get("/transactions/{transfer_id}", summary="Full transaction details for investigation")
def transaction(transfer_id: str, _op: str = Depends(operator)):
    row = _tx(transfer_id)
    hist_cols = "transfer_id, ts, payer_id, payee_id, amount, scenario, risk_score, decision, device_id"
    payer_hist = db.fetch_all(
        f"SELECT {hist_cols} FROM transactions WHERE payer_id = %s OR payee_id = %s ORDER BY ts DESC LIMIT 30",
        (row["payer_id"], row["payer_id"]),
    )
    payee_hist = db.fetch_all(
        f"SELECT {hist_cols} FROM transactions WHERE payer_id = %s OR payee_id = %s ORDER BY ts DESC LIMIT 30",
        (row["payee_id"], row["payee_id"]),
    )
    device_accounts = []
    if row.get("device_id"):
        device_accounts = [
            r["payer_id"]
            for r in db.fetch_all(
                "SELECT DISTINCT payer_id FROM transactions WHERE device_id = %s LIMIT 50", (row["device_id"],)
            )
        ]
    rings = db.fetch_all(
        """SELECT ring_id, ring_type, ring_score, jsonb_array_length(members) AS size, evidence FROM rings
           WHERE members ? %s OR members ? %s ORDER BY ring_score DESC LIMIT 10""",
        (row["payer_id"], row["payee_id"]),
    )
    alert = db.fetch_one("SELECT * FROM alerts WHERE transfer_id = %s", (transfer_id,))
    feedback = db.fetch_all("SELECT * FROM feedback WHERE transfer_id = %s ORDER BY created_at DESC", (transfer_id,))
    cases = db.fetch_all(
        "SELECT c.case_id, c.title, c.status FROM cases c JOIN case_alerts a USING (case_id) WHERE a.transfer_id = %s",
        (transfer_id,),
    )
    return {
        "transaction": row,
        "alert": alert,
        "feedback": feedback,
        "cases": cases,
        "payer_history": payer_hist,
        "payee_history": payee_hist,
        "linked_entities": {"device_accounts": device_accounts, "rings": rings},
    }


def _bundle_for(version: str) -> ModelBundle:
    cur = STATE.scorer.models.bundle
    if cur.version == version:
        return cur
    cache = STATE.cache.setdefault("bundles", {})
    if version not in cache:
        from core.config import get_settings

        path = get_settings().artifacts_dir / version
        if not (path / "bundle.json").exists():
            return cur
        cache[version] = ModelBundle(path, load_sequence=False)
    return cache[version]


@router.get("/transactions/{transfer_id}/explanation", summary="SHAP explanation (computed on demand if pending)")
def explanation(transfer_id: str, _op: str = Depends(operator)):
    row = _tx(transfer_id)
    if row.get("explanation"):
        return {
            "transfer_id": transfer_id,
            "risk_score": row["risk_score"],
            "decision": row["decision"],
            "model_version": row["model_version"],
            "computed": "at_scoring",
            "sequence_logit": (row.get("components") or {}).get("seq_logit"),
            "graph": row.get("graph"),
            **row["explanation"],
        }
    if not row.get("features"):
        raise HTTPException(409, "transaction was not scored (bootstrap history)")
    bundle = _bundle_for(row["model_version"])
    comp = row["components"] or {}
    e = explain_row(bundle, row)
    return {
        "transfer_id": transfer_id,
        "risk_score": row["risk_score"],
        "decision": row["decision"],
        "model_version": row["model_version"],
        "computed": "on_demand",
        "sequence_logit": comp.get("seq_logit"),
        "graph": row.get("graph"),
        **e,
    }


@router.get("/rings", summary="Detected fraud rings")
def rings(
    active: bool = True,
    ring_type: str | None = None,
    min_score: float = 0.0,
    limit: int = Query(100, ge=1, le=500),
    _op: str = Depends(operator),
):
    where, params = ["ring_score >= %s"], [min_score]
    if active:
        where.append("active")
    if ring_type:
        where.append("ring_type = %s")
        params.append(ring_type)
    rows = db.fetch_all(
        f"""SELECT ring_id, ring_type, ring_score, members, devices, evidence, first_seen, last_seen, active,
                   updated_at FROM rings WHERE {" AND ".join(where)} ORDER BY ring_score DESC, last_seen DESC
            LIMIT %s""",
        (*params, limit),
    )
    return {"items": rows, "count": len(rows)}


@router.get("/rings/{ring_id}", summary="Ring details incl. edges for graph visualisation")
def ring(ring_id: str, _op: str = Depends(operator)):
    row = db.fetch_one("SELECT * FROM rings WHERE ring_id = %s", (ring_id,))
    if not row:
        raise HTTPException(404, "ring not found")
    members = row["members"]
    alerts = db.fetch_all(
        """SELECT transfer_id, decision, risk_score, payer_id, payee_id, amount FROM alerts
           WHERE payer_id = ANY(%s) OR payee_id = ANY(%s) ORDER BY alert_id DESC LIMIT 100""",
        (members, members),
    )
    return {**row, "alerts": alerts}
