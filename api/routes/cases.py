"""Case management with a full audit trail."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import operator
from api.schemas import CaseIn, CasePatch
from core import db

router = APIRouter(prefix="/v1/cases", tags=["cases"])


def _audit(cur, case_id: int, actor: str, action: str, detail: dict):
    cur.execute(
        "INSERT INTO case_audit (case_id, actor, action, detail) VALUES (%s,%s,%s,%s)",
        (case_id, actor, action, db.Jsonb(detail)),
    )


@router.post("", status_code=201, summary="Create a case from alerts / a ring")
def create_case(body: CaseIn, op: str = Depends(operator)):
    with db.get_pool().connection() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """INSERT INTO cases (title, priority, assignee, created_by, ring_id)
                       VALUES (%s,%s,%s,%s,%s) RETURNING case_id""",
            (body.title, body.priority, body.assignee, op, body.ring_id),
        )
        case_id = cur.fetchone()[0]
        for t in body.transfer_ids:
            cur.execute(
                "INSERT INTO case_alerts (case_id, transfer_id) VALUES (%s,%s) ON CONFLICT DO NOTHING", (case_id, t)
            )
            cur.execute(
                "UPDATE alerts SET case_id = %s, status = CASE WHEN status = 'open' THEN 'in_case' "
                "ELSE status END WHERE transfer_id = %s",
                (case_id, t),
            )
        if body.note:
            cur.execute("INSERT INTO case_notes (case_id, author, body) VALUES (%s,%s,%s)", (case_id, op, body.note))
        _audit(cur, case_id, op, "created", body.model_dump())
    return get_case(case_id, op)


@router.get("", summary="List cases")
def list_cases(
    status: str | None = Query(None, pattern="^(open|investigating|closed)$"),
    assignee: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    _op: str = Depends(operator),
):
    where, params = ["TRUE"], []
    if status:
        where.append("status = %s")
        params.append(status)
    if assignee:
        where.append("assignee = %s")
        params.append(assignee)
    rows = db.fetch_all(
        f"""SELECT c.*, (SELECT count(*) FROM case_alerts a WHERE a.case_id = c.case_id) AS n_alerts
            FROM cases c WHERE {" AND ".join(where)} ORDER BY updated_at DESC LIMIT %s""",
        (*params, limit),
    )
    return {"items": rows, "count": len(rows)}


@router.get("/{case_id}", summary="Case detail with alerts, notes and audit trail")
def get_case(case_id: int, _op: str = Depends(operator)):
    c = db.fetch_one("SELECT * FROM cases WHERE case_id = %s", (case_id,))
    if not c:
        raise HTTPException(404, "case not found")
    c["alerts"] = db.fetch_all(
        """SELECT a.transfer_id, al.decision, al.risk_score, al.predicted_type, al.amount, al.payer_id, al.payee_id,
                  al.status FROM case_alerts a LEFT JOIN alerts al USING (transfer_id) WHERE a.case_id = %s""",
        (case_id,),
    )
    c["notes"] = db.fetch_all("SELECT * FROM case_notes WHERE case_id = %s ORDER BY created_at", (case_id,))
    c["audit"] = db.fetch_all("SELECT * FROM case_audit WHERE case_id = %s ORDER BY created_at", (case_id,))
    return c


@router.patch("/{case_id}", summary="Update status / assignee / priority, add notes or alerts")
def patch_case(case_id: int, body: CasePatch, op: str = Depends(operator)):
    if not db.fetch_one("SELECT case_id FROM cases WHERE case_id = %s", (case_id,)):
        raise HTTPException(404, "case not found")
    changes = {
        k: v
        for k, v in body.model_dump().items()
        if v is not None and k in ("status", "assignee", "priority", "resolution")
    }
    with db.get_pool().connection() as conn, conn.transaction(), conn.cursor() as cur:
        if changes:
            sets = ", ".join(f"{k} = %s" for k in changes)
            cur.execute(f"UPDATE cases SET {sets}, updated_at = now() WHERE case_id = %s", (*changes.values(), case_id))
            _audit(cur, case_id, op, "updated", changes)
        if body.note:
            cur.execute("INSERT INTO case_notes (case_id, author, body) VALUES (%s,%s,%s)", (case_id, op, body.note))
            cur.execute("UPDATE cases SET updated_at = now() WHERE case_id = %s", (case_id,))
            _audit(cur, case_id, op, "note_added", {"length": len(body.note)})
        for t in body.add_transfer_ids:
            cur.execute(
                "INSERT INTO case_alerts (case_id, transfer_id) VALUES (%s,%s) ON CONFLICT DO NOTHING", (case_id, t)
            )
            cur.execute("UPDATE alerts SET case_id = %s WHERE transfer_id = %s", (case_id, t))
        if body.add_transfer_ids:
            _audit(cur, case_id, op, "alerts_added", {"transfer_ids": body.add_transfer_ids})
    return get_case(case_id, op)
