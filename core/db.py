"""PostgreSQL access (psycopg 3 + connection pool) and schema."""

from __future__ import annotations

import threading

import orjson
from psycopg.types.json import Jsonb, set_json_dumps, set_json_loads
from psycopg_pool import ConnectionPool

from core.config import get_settings

set_json_dumps(
    lambda o: orjson.dumps(o, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS, default=str).decode()
)
set_json_loads(orjson.loads)

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    transfer_id   TEXT PRIMARY KEY,
    ts            BIGINT NOT NULL,
    payer_id      TEXT NOT NULL,
    payee_id      TEXT NOT NULL,
    payer_fsp     TEXT,
    payee_fsp     TEXT,
    amount        DOUBLE PRECISION NOT NULL,
    currency      TEXT,
    scenario      TEXT,
    channel       TEXT,
    device_id     TEXT,
    ip            TEXT,
    payload       JSONB,
    risk_score    DOUBLE PRECISION,
    decision      TEXT,
    model_version TEXT,
    combo         TEXT,
    degraded      BOOLEAN DEFAULT FALSE,
    latency_ms    DOUBLE PRECISION,
    components    JSONB,
    features      JSONB,
    graph         JSONB,
    explanation   JSONB,
    typology      JSONB,
    source        TEXT DEFAULT 'stream',
    scored_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_tx_ts ON transactions (ts DESC);
CREATE INDEX IF NOT EXISTS ix_tx_payer ON transactions (payer_id, ts DESC);
CREATE INDEX IF NOT EXISTS ix_tx_payee ON transactions (payee_id, ts DESC);
CREATE INDEX IF NOT EXISTS ix_tx_scored ON transactions (scored_at DESC);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id       BIGSERIAL PRIMARY KEY,
    transfer_id    TEXT UNIQUE NOT NULL,
    ts             BIGINT NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT now(),
    risk_score     DOUBLE PRECISION NOT NULL,
    decision       TEXT NOT NULL,
    predicted_type TEXT,
    payer_id       TEXT,
    payee_id       TEXT,
    payer_fsp      TEXT,
    payee_fsp      TEXT,
    amount         DOUBLE PRECISION,
    top_reasons    JSONB,
    model_version  TEXT,
    degraded       BOOLEAN DEFAULT FALSE,
    status         TEXT NOT NULL DEFAULT 'open',
    case_id        BIGINT
);
CREATE INDEX IF NOT EXISTS ix_alerts_created ON alerts (alert_id DESC);
CREATE INDEX IF NOT EXISTS ix_alerts_decision ON alerts (decision, alert_id DESC);

CREATE TABLE IF NOT EXISTS feedback (
    id              BIGSERIAL PRIMARY KEY,
    transfer_id     TEXT NOT NULL,
    label           TEXT NOT NULL CHECK (label IN ('fraud', 'legit')),
    kind            TEXT NOT NULL CHECK (kind IN ('true_positive', 'false_positive', 'false_negative',
                                                  'true_negative')),
    operator_id     TEXT NOT NULL,
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    used_in_version TEXT
);
CREATE INDEX IF NOT EXISTS ix_feedback_tx ON feedback (transfer_id, created_at DESC);

CREATE TABLE IF NOT EXISTS rings (
    ring_id     TEXT PRIMARY KEY,
    ring_type   TEXT NOT NULL,
    ring_score  DOUBLE PRECISION NOT NULL,
    members     JSONB NOT NULL,
    devices     JSONB,
    evidence    JSONB,
    edges       JSONB,
    first_seen  BIGINT,
    last_seen   BIGINT,
    active      BOOLEAN DEFAULT TRUE,
    updated_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_rings_score ON rings (active, ring_score DESC);

CREATE TABLE IF NOT EXISTS cases (
    case_id     BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'closed')),
    priority    TEXT NOT NULL DEFAULT 'medium',
    assignee    TEXT,
    created_by  TEXT NOT NULL,
    resolution  TEXT,
    ring_id     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS case_alerts (
    case_id     BIGINT REFERENCES cases(case_id) ON DELETE CASCADE,
    transfer_id TEXT NOT NULL,
    PRIMARY KEY (case_id, transfer_id)
);
CREATE TABLE IF NOT EXISTS case_notes (
    id         BIGSERIAL PRIMARY KEY,
    case_id    BIGINT REFERENCES cases(case_id) ON DELETE CASCADE,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS case_audit (
    id         BIGSERIAL PRIMARY KEY,
    case_id    BIGINT REFERENCES cases(case_id) ON DELETE CASCADE,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    detail     JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_registry (
    version        TEXT PRIMARY KEY,
    created_at     TIMESTAMPTZ DEFAULT now(),
    status         TEXT NOT NULL DEFAULT 'candidate'
                   CHECK (status IN ('candidate', 'production', 'archived', 'rejected')),
    data_hash      TEXT,
    metrics        JSONB,
    artifact_path  TEXT NOT NULL,
    parent_version TEXT,
    promoted_at    TIMESTAMPTZ,
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS retrain_runs (
    id                BIGSERIAL PRIMARY KEY,
    started_at        TIMESTAMPTZ DEFAULT now(),
    finished_at       TIMESTAMPTZ,
    trigger           TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'running',
    base_version      TEXT,
    candidate_version TEXT,
    promoted          BOOLEAN,
    reason            TEXT,
    n_feedback        INT,
    before_metrics    JSONB,
    after_metrics     JSONB
);

-- Simulator ground truth: used ONLY by the demo/evaluation scripts (never read by the scoring path).
CREATE TABLE IF NOT EXISTS sim_ground_truth (
    transfer_id TEXT PRIMARY KEY,
    is_fraud    SMALLINT NOT NULL,
    fraud_type  TEXT,
    case_id     TEXT
);
"""

_pool: ConnectionPool | None = None
_lock = threading.Lock()


def get_pool(min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ConnectionPool(
                get_settings().database_url,
                min_size=min_size,
                max_size=max_size,
                kwargs={"autocommit": True},
                open=True,
                timeout=10,
            )
        return _pool


def close_pool() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def init_schema() -> None:
    with get_pool().connection() as conn:
        conn.execute(SCHEMA)


def fetch_all(sql: str, params=None) -> list[dict]:
    from psycopg.rows import dict_row

    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def fetch_one(sql: str, params=None) -> dict | None:
    rows = fetch_all(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params=None) -> int:
    with get_pool().connection() as conn:
        cur = conn.execute(sql, params)
        return cur.rowcount


__all__ = ["Jsonb", "get_pool", "close_pool", "init_schema", "fetch_all", "fetch_one", "execute", "SCHEMA"]
