"""Persister: moves scored rows from the Redis stream `persist:rows` into PostgreSQL, off the scoring processes.

* consumer group `persisters` (run one or more instances); batches of up to 1000 rows, idempotent inserts
  (ON CONFLICT DO NOTHING), XACK only after a successful commit
* PostgreSQL outage: nothing is acknowledged, rows stay in Redis (durable buffer) and are retried with backoff;
  on restart the persister first drains its own pending entries
* WRITE FIRST, EXPLAIN LATER: every batch is committed to PostgreSQL before any explanation work, so rows are
  queryable within about a second of scoring regardless of explanation load
* asynchronous SHAP explanations for ALLOW decisions (REVIEW/BLOCK are explained synchronously by the scorer) run
  afterwards in vectorised batches under a per-cycle time budget (--explain-budget-ms), only while the persister is
  caught up; rows that overflow the queue are explained on demand by the API

Usage: python -m pipeline.persister
"""

from __future__ import annotations

import os

# keep explanation work from competing with the scoring workers for every core
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import orjson  # noqa: E402
import redis  # noqa: E402

from core import db  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.logging import get_logger, setup_logging  # noqa: E402
from core.redis_client import get_redis  # noqa: E402
from models.ensemble import ModelBundle  # noqa: E402
from models.explain import explain_rows  # noqa: E402
from pipeline.scorer import PERSIST_STREAM  # noqa: E402
from pipeline.worker import ensure_group  # noqa: E402
from pipeline.writer import DBWriter  # noqa: E402

log = get_logger("persister")
GROUP = "persisters"


class BundleCache:
    def __init__(self):
        self.s = get_settings()
        self.cache: dict[str, ModelBundle] = {}

    def get(self, version: str) -> ModelBundle | None:
        if version not in self.cache:
            path = self.s.artifacts_dir / version
            if not (path / "bundle.json").exists():
                return None
            self.cache[version] = ModelBundle(path, load_sequence=False)
        return self.cache[version]


def explain_backlog(queue: collections.deque, bundles: BundleCache, budget_s: float, chunk: int = 32) -> list:
    """Explain queued ALLOW rows in vectorised chunks until the time budget is spent -> [(transfer_id, expl)]."""
    out = []
    t0 = time.perf_counter()
    while queue and time.perf_counter() - t0 < budget_s:
        rows = [queue.popleft() for _ in range(min(chunk, len(queue)))]
        by_version: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            by_version[row["model_version"]].append(row)
        for version, vrows in by_version.items():
            b = bundles.get(version)
            if b is None:
                continue
            try:
                for row, e in zip(vrows, explain_rows(b, vrows)):
                    out.append((row["transfer_id"], {**e, "computed": "asynchronous"}))
            except Exception as ex:
                log.error("async explanation failed", error=str(ex)[:200])
    return out


def run(
    batch: int = 1000,
    explain_budget_ms: float = 250,
    explain_queue: int = 20_000,
    stop_event: threading.Event | None = None,
) -> None:
    s = get_settings()
    setup_logging(s.log_level)
    db.init_schema()
    r = get_redis(timeout=10)
    ensure_group(r, PERSIST_STREAM, GROUP)
    consumer = f"{socket.gethostname()}-persister"
    bundles = BundleCache()
    stop_event = stop_event or threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    pending_expl: collections.deque = collections.deque()
    last_id = "0"
    written = explained = skipped_expl = 0
    backoff = 0.5
    last_beat = 0.0
    log.info("persister started", stream=PERSIST_STREAM)
    while not stop_event.is_set():
        try:
            # short block while explanations are queued, so they are processed between bursts
            block = 50 if pending_expl else 1000
            resp = r.xreadgroup(GROUP, consumer, {PERSIST_STREAM: last_id}, count=batch, block=block)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            log.error("redis unavailable; retrying", error=str(e)[:200])
            time.sleep(1)
            continue
        entries = resp[0][1] if resp else []
        if last_id == "0" and not entries:
            last_id = ">"
            continue
        if entries:
            items: list[tuple[str, dict]] = [(f["k"], orjson.loads(f["r"])) for _, f in entries if f]
            try:
                DBWriter.write_batch(items)  # 1) persist first
                r.xack(PERSIST_STREAM, GROUP, *[m for m, _ in entries])
                written += len(items)
                backoff = 0.5
            except Exception as e:  # PostgreSQL outage: keep entries pending, retry from the PEL
                log.error("database write failed; rows stay buffered in Redis", error=str(e)[:200], rows=len(items))
                last_id = "0"
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)
                continue
            for kind, row in items:  # 2) queue ALLOW rows for asynchronous explanation
                if kind == "tx" and row.get("explanation") is None and row.get("features"):
                    if len(pending_expl) < explain_queue:
                        pending_expl.append(row)
                    else:
                        skipped_expl += 1  # explained on demand by the API instead
        # 3) explanations only while caught up (a full batch means a backlog), within a time budget
        if pending_expl and len(entries) < batch:
            done = explain_backlog(pending_expl, bundles, explain_budget_ms / 1000)
            if done:
                try:
                    DBWriter.write_batch([("expl", {"transfer_id": t, "explanation": e}) for t, e in done])
                    explained += len(done)
                except Exception as e:
                    log.error("explanation update failed", error=str(e)[:200])
        now = time.time()
        if now - last_beat > 2:
            try:
                r.hset(
                    "persister",
                    mapping={
                        "heartbeat": now,
                        "written": written,
                        "explained": explained,
                        "explain_skipped": skipped_expl,
                        "explain_queue": len(pending_expl),
                    },
                )
            except redis.RedisError:
                pass
            last_beat = now
    log.info("persister stopped", written=written)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--explain-budget-ms", type=float, default=250, help="explanation time per cycle")
    ap.add_argument("--explain-queue", type=int, default=20_000, help="max queued ALLOW rows awaiting explanation")
    a = ap.parse_args(argv)
    run(a.batch, a.explain_budget_ms, a.explain_queue)


if __name__ == "__main__":
    main()
