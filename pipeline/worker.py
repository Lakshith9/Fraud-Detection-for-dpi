"""Scoring worker: consumes one stream partition (Redis Streams consumer group), scores every transfer.

Partitioning by payer hash keeps each payer's events ordered on a single worker (like Kafka partitions).
At-least-once delivery: entries are XACKed after scoring; on restart the worker first drains its own pending
entries (PEL). Duplicates are harmless: scoring is idempotent by transferId (Redis idem:<id> + DB PK).

Usage: python -m pipeline.worker --partition 0
"""

from __future__ import annotations

import argparse
import signal
import socket
import threading
import time

import orjson
import redis

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.redis_client import get_redis, stream_key
from pipeline.mojaloop_adapter import AdapterError, from_fspiop
from pipeline.scorer import Scorer
from pipeline.writer import DBWriter

log = get_logger("worker")


def ensure_group(r: redis.Redis, key: str, group: str) -> None:
    try:
        r.xgroup_create(key, group, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def handle(scorer: Scorer, fields: dict) -> str:
    ev = orjson.loads(fields["event"])
    sent_at = float(fields.get("sent_at") or 0) or None
    if ev.get("type") == "fraud_report":
        scorer.apply_report(ev["account_id"])
        return "report"
    tx = from_fspiop(ev)
    scorer.score(tx, sent_at_ms=sent_at)
    return "transfer"


def run(partition: int, batch: int = 200, stop_event: threading.Event | None = None) -> None:
    s = get_settings()
    setup_logging(s.log_level)
    r = get_redis(timeout=5)
    key = stream_key(partition)
    ensure_group(r, key, s.consumer_group)
    consumer = f"{socket.gethostname()}-p{partition}"
    writer = DBWriter()
    scorer = Scorer(r, writer=writer, source="stream")
    stop_event = stop_event or threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    r.hset(f"worker:{partition}", mapping={"consumer": consumer, "started": time.time()})
    log.info("worker started", partition=partition, stream=key, model=scorer.models.bundle.version)
    last_id = "0"  # drain own pending entries first
    processed = errors = 0
    last_beat = 0.0
    while not stop_event.is_set():
        try:
            resp = r.xreadgroup(s.consumer_group, consumer, {key: last_id}, count=batch, block=1000)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            log.error("redis unavailable; retrying", error=str(e)[:200])
            time.sleep(1)
            continue
        entries = resp[0][1] if resp else []
        if last_id == "0" and not entries:
            last_id = ">"
            continue
        ack = []
        for msg_id, fields in entries:
            if not fields:
                ack.append(msg_id)
                continue
            try:
                handle(scorer, fields)
                processed += 1
            except AdapterError as e:
                errors += 1
                r.xadd(f"{key}:dlq", {"event": fields.get("event", ""), "error": str(e)[:300]}, maxlen=100_000)
            except Exception as e:  # never wedge the partition on one bad event
                errors += 1
                log.error("scoring failed", error=str(e)[:300])
                r.xadd(f"{key}:dlq", {"event": fields.get("event", ""), "error": str(e)[:300]}, maxlen=100_000)
            ack.append(msg_id)
        if ack:
            try:
                r.xack(key, s.consumer_group, *ack)
            except redis.RedisError as e:
                log.error("xack failed (entries will be redelivered, idempotent)", error=str(e)[:200])
        now = time.time()
        if now - last_beat > 2:
            try:
                r.hset(
                    f"worker:{partition}",
                    mapping={
                        "heartbeat": now,
                        "processed": processed,
                        "errors": errors,
                        "backlog_db": writer.backlog(),
                        "model": scorer.models.bundle.version,
                        "db_healthy": int(writer.healthy),
                    },
                )
            except redis.RedisError:
                pass
            last_beat = now
    log.info("worker stopping; flushing writer", processed=processed)
    writer.stop()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", type=int, default=0)
    a = ap.parse_args(argv)
    run(a.partition)


if __name__ == "__main__":
    main()
