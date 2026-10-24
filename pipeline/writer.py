"""Batched PostgreSQL persistence of scored transactions and alerts.

`DBWriter.write_batch` is used by the persister process (pipeline/persister.py). The threaded `DBWriter` itself is
the in-process FALLBACK the scorer uses only when Redis (and therefore the persist stream) is unavailable.

* batches up to 500 rows or 200 ms, uses executemany with ON CONFLICT DO NOTHING (idempotent by transfer_id)
* on PostgreSQL outage: keeps retrying with backoff; when the in-memory buffer exceeds its cap, rows are spooled
  to data/spool/*.jsonl and replayed automatically once the database is back
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import orjson

from core import db
from core.config import get_settings
from core.logging import get_logger

log = get_logger("writer")

TX_SQL = """
INSERT INTO transactions (transfer_id, ts, payer_id, payee_id, payer_fsp, payee_fsp, amount, currency, scenario,
    channel, device_id, ip, payload, risk_score, decision, model_version, combo, degraded, latency_ms, components,
    features, graph, explanation, typology, source)
VALUES (%(transfer_id)s, %(ts)s, %(payer_id)s, %(payee_id)s, %(payer_fsp)s, %(payee_fsp)s, %(amount)s, %(currency)s,
    %(scenario)s, %(channel)s, %(device_id)s, %(ip)s, %(payload)s, %(risk_score)s, %(decision)s, %(model_version)s,
    %(combo)s, %(degraded)s, %(latency_ms)s, %(components)s, %(features)s, %(graph)s, %(explanation)s, %(typology)s,
    %(source)s)
ON CONFLICT (transfer_id) DO NOTHING
"""
ALERT_SQL = """
INSERT INTO alerts (transfer_id, ts, risk_score, decision, predicted_type, payer_id, payee_id, payer_fsp, payee_fsp,
    amount, top_reasons, model_version, degraded)
VALUES (%(transfer_id)s, %(ts)s, %(risk_score)s, %(decision)s, %(predicted_type)s, %(payer_id)s, %(payee_id)s,
    %(payer_fsp)s, %(payee_fsp)s, %(amount)s, %(top_reasons)s, %(model_version)s, %(degraded)s)
ON CONFLICT (transfer_id) DO NOTHING
"""
EXPL_SQL = "UPDATE transactions SET explanation = %(explanation)s WHERE transfer_id = %(transfer_id)s"
JSON_TX = ("payload", "components", "features", "graph", "explanation", "typology")


class DBWriter:
    def __init__(self, batch_size: int = 500, flush_ms: int = 200, buffer_cap: int = 50_000):
        self.q: queue.Queue = queue.Queue()
        self.batch_size = batch_size
        self.flush_s = flush_ms / 1000
        self.buffer_cap = buffer_cap
        self.spool_dir = get_settings().data_dir / "spool"
        self._stop = threading.Event()
        self.healthy = True
        self.written = 0
        self.spooled = 0
        self.t = threading.Thread(target=self._run, name="db-writer", daemon=True)
        self.t.start()

    def submit(self, kind: str, row: dict) -> None:
        self.q.put((kind, row))

    def backlog(self) -> int:
        return self.q.qsize()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self.t.join(timeout)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _adapt(kind: str, row: dict) -> dict:
        r = dict(row)
        if kind == "tx":
            for k in JSON_TX:
                r[k] = db.Jsonb(r.get(k)) if r.get(k) is not None else None
        elif kind == "alert":
            r["top_reasons"] = db.Jsonb(r.get("top_reasons") or [])
        elif kind == "expl":
            r["explanation"] = db.Jsonb(r["explanation"])
        return r

    @staticmethod
    def write_batch(items: list[tuple[str, dict]]) -> None:
        """Insert a batch of ("tx" | "alert" | "expl", row) items in one transaction (idempotent)."""
        tx = [DBWriter._adapt(k, r) for k, r in items if k == "tx"]
        al = [DBWriter._adapt(k, r) for k, r in items if k == "alert"]
        ex = [DBWriter._adapt(k, r) for k, r in items if k == "expl"]
        with db.get_pool().connection() as conn:
            with conn.transaction(), conn.cursor() as cur:
                if tx:
                    cur.executemany(TX_SQL, tx)
                if al:
                    cur.executemany(ALERT_SQL, al)
                if ex:
                    cur.executemany(EXPL_SQL, ex)

    def _write(self, items: list[tuple[str, dict]]) -> None:
        self.write_batch(items)
        self.written += len(items)

    def _spool(self, items: list[tuple[str, dict]]) -> None:
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        path = self.spool_dir / f"spool-{int(time.time() * 1000)}.jsonl"
        with open(path, "wb") as f:
            for k, r in items:
                f.write(orjson.dumps({"kind": k, "row": r}, option=orjson.OPT_SERIALIZE_NUMPY, default=str) + b"\n")
        self.spooled += len(items)
        log.warning("database unavailable; spooled rows to disk", rows=len(items), path=str(path))

    def _replay_spool(self) -> None:
        if not self.spool_dir.exists():
            return
        for path in sorted(self.spool_dir.glob("spool-*.jsonl")):
            items = [(d["kind"], d["row"]) for d in map(orjson.loads, Path(path).read_bytes().splitlines())]
            self._write(items)
            path.unlink()
            log.info("replayed spooled rows", rows=len(items), path=str(path))

    def _run(self) -> None:
        pending: list[tuple[str, dict]] = []
        backoff = 0.5
        last_flush = time.monotonic()
        while not (self._stop.is_set() and self.q.empty() and not pending):
            try:
                pending.append(self.q.get(timeout=self.flush_s))
                while len(pending) < self.batch_size * 4:
                    pending.append(self.q.get_nowait())
            except queue.Empty:
                pass
            if not pending or (
                len(pending) < self.batch_size
                and time.monotonic() - last_flush < self.flush_s
                and not self._stop.is_set()
            ):
                continue
            try:
                if not self.healthy:
                    self._replay_spool()
                self._write(pending)
                pending = []
                self.healthy = True
                backoff = 0.5
                last_flush = time.monotonic()
            except Exception as e:  # database outage or transient error
                if self.healthy:
                    log.error("database write failed; buffering", error=str(e)[:200], buffered=len(pending))
                self.healthy = False
                if len(pending) > self.buffer_cap:
                    self._spool(pending)
                    pending = []
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)
