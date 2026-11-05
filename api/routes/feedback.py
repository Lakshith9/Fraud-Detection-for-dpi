"""Operator feedback and model lifecycle endpoints."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import threading

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import operator
from api.schemas import FeedbackIn, RetrainIn
from api.state import STATE
from core import db
from core.config import ROOT
from models import registry
from pipeline.features import FeatureEngine

router = APIRouter(prefix="/v1", tags=["feedback & models"])


@router.post("/feedback", status_code=201, summary="Mark an alert True Fraud / False Positive or report a missed fraud")
def feedback(body: FeedbackIn, op: str = Depends(operator)):
    tx = db.fetch_one(
        "SELECT transfer_id, payee_id, decision FROM transactions WHERE transfer_id = %s", (body.transfer_id,)
    )
    if not tx:
        raise HTTPException(404, "unknown transfer_id")
    alerted = tx["decision"] in ("REVIEW", "BLOCK")
    kind = {
        (True, "fraud"): "true_positive",
        (True, "legit"): "false_positive",
        (False, "fraud"): "false_negative",
        (False, "legit"): "true_negative",
    }[(alerted, body.label)]
    row = db.fetch_one(
        """INSERT INTO feedback (transfer_id, label, kind, operator_id, notes) VALUES (%s,%s,%s,%s,%s)
           RETURNING id, created_at""",
        (body.transfer_id, body.label, kind, op, body.notes),
    )
    if alerted:
        db.execute(
            "UPDATE alerts SET status = %s WHERE transfer_id = %s",
            ("confirmed_fraud" if body.label == "fraud" else "false_positive", body.transfer_id),
        )
    if body.label == "fraud":
        # confirmed fraud feeds the counterparty-risk registry used by the online features and graph worker
        with contextlib.suppress(Exception):
            STATE.scorer.store.execute(
                FeatureEngine.report_plan(tx["payee_id"]) + [("sadd", "reported", tx["payee_id"])]
            )
    return {
        "id": row["id"],
        "transfer_id": body.transfer_id,
        "label": body.label,
        "kind": kind,
        "operator_id": op,
        "created_at": row["created_at"],
    }


@router.get("/feedback", summary="Recent feedback and counts")
def list_feedback(limit: int = Query(100, ge=1, le=1000), _op: str = Depends(operator)):
    return {
        "counts": {r["kind"]: r["n"] for r in db.fetch_all("SELECT kind, count(*) AS n FROM feedback GROUP BY kind")},
        "items": db.fetch_all("SELECT * FROM feedback ORDER BY created_at DESC LIMIT %s", (limit,)),
    }


@router.get("/models", summary="Model registry, production version and retraining runs")
def models(_op: str = Depends(operator)):
    return {
        "current": registry.current(),
        "loaded": STATE.scorer.models.bundle.version,
        "versions": registry.list_versions(),
        "retrain_runs": db.fetch_all("SELECT * FROM retrain_runs ORDER BY id DESC LIMIT 20"),
    }


_retrain_lock = threading.Lock()
_retrain_proc: subprocess.Popen | None = None


@router.post("/models/retrain", status_code=202, summary="Trigger feedback retraining (runs in a separate process)")
def retrain(body: RetrainIn | None = None, op: str = Depends(operator)):
    global _retrain_proc
    # The run only appears in retrain_runs once the child has started, so repeated calls in that window
    # would each start a full training process; the lock and the child handle close that gap.
    with _retrain_lock:
        if _retrain_proc is not None and _retrain_proc.poll() is None:
            raise HTTPException(409, f"retrain process {_retrain_proc.pid} already in progress")
        running = db.fetch_one(
            "SELECT id FROM retrain_runs WHERE status = 'running' AND started_at > now() - interval '2 hours'"
        )
        if running:
            raise HTTPException(409, f"retrain run {running['id']} already in progress")
        log_dir = ROOT / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "retrain.log", "ab") as out:
            proc = _retrain_proc = subprocess.Popen(
                [sys.executable, "-m", "models.retrain", "--trigger", f"api:{op}"],
                cwd=str(ROOT),
                stdout=out,
                stderr=subprocess.STDOUT,
            )
    return {"status": "started", "pid": proc.pid, "trigger": f"api:{op}", "reason": body.reason if body else None}


@router.post("/models/{version}/promote", summary="Promote a registered version to production (hot reload)")
def promote(version: str, op: str = Depends(operator)):
    try:
        registry.promote(version)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(404, str(e)) from e
    db.execute(
        "UPDATE model_registry SET notes = coalesce(notes, '') || %s WHERE version = %s",
        (f" | promoted by {op}", version),
    )
    return {"promoted": version}


@router.post("/models/rollback", summary="Roll back to the previously promoted version")
def rollback(_op: str = Depends(operator)):
    try:
        return {"promoted": registry.rollback()}
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
