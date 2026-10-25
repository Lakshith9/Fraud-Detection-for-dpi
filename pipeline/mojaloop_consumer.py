"""Live Mojaloop connection: Kafka ``topic-transfer-prepare`` -> partitioned Redis scoring stream.

The consumer joins the switch's transfer-prepare topic with its OWN consumer group (default ``fraud-engine``),
so Kafka hands it a copy of every transfer while central-ledger keeps processing them untouched. Each event
is decoded (data-URI payload, ILP packet), validated through ``from_fspiop`` and appended to the same Redis
stream partition the simulator would use, so the scoring workers, persister and dashboard run unchanged.

Shadow mode by construction: the engine scores and alerts but never answers the switch, so it cannot delay
or block a transfer. Blocking belongs to an FSP calling ``POST /v1/score/fspiop`` before it sends.

Delivery: Kafka offsets are committed only after the batch is on the Redis stream, so a crash re-delivers
rather than drops (at-least-once). Duplicates are harmless: scoring is idempotent by transferId.
Unparseable events go to the ``mojaloop:dlq`` stream; non-prepare events (bulk, fx, ...) are skipped.

Usage:
    python -m pipeline.mojaloop_consumer                     # KAFKA_BOOTSTRAP_SERVERS from .env
    python -m pipeline.mojaloop_consumer --bootstrap kafka:29092 --from-beginning
    python -m pipeline.mojaloop_consumer --from-file captured.jsonl   # replay captured message values
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import orjson
import redis

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.redis_client import get_redis, partition_of, stream_key
from pipeline.mojaloop_adapter import AdapterError, from_fspiop, is_transfer_prepare, normalize_message

log = get_logger("mojaloop_consumer")
DLQ = "mojaloop:dlq"
STATS = "mojaloop_consumer"


class Forwarder:
    """Turns raw Kafka message values into scoring-stream entries. Kafka-free, so it is unit-testable."""

    def __init__(self, r: redis.Redis, partitions: int | None = None, maxlen: int | None = None):
        s = get_settings()
        self.r = r
        self.partitions = partitions or s.stream_partitions
        self.maxlen = maxlen or s.stream_maxlen
        self.counts = {"forwarded": 0, "skipped": 0, "invalid": 0}

    def forward(self, values: Iterable[bytes | str | None]) -> dict[str, int]:
        """Append a batch to the stream in one pipeline. Raises redis errors (caller must not commit)."""
        batch = {"forwarded": 0, "skipped": 0, "invalid": 0}
        pipe = self.r.pipeline(transaction=False)
        now_ms = time.time() * 1000
        for value in values:
            if not value:  # tombstones / empty keepalives
                batch["skipped"] += 1
                continue
            try:
                msg = orjson.loads(value)
                if not isinstance(msg, dict):
                    raise AdapterError("message is not a JSON object")
                if not is_transfer_prepare(msg):
                    batch["skipped"] += 1
                    continue
                event = normalize_message(msg)
                tx = from_fspiop(event)
            except (orjson.JSONDecodeError, AdapterError) as e:
                batch["invalid"] += 1
                raw = value if isinstance(value, str) else value.decode("utf-8", errors="replace")
                pipe.xadd(DLQ, {"event": raw[:20_000], "error": str(e)[:300]}, maxlen=100_000, approximate=True)
                continue
            fields = {"event": orjson.dumps(event), "sent_at": f"{now_ms:.3f}"}
            key = stream_key(partition_of(tx["payer_id"], self.partitions))
            pipe.xadd(key, fields, maxlen=self.maxlen, approximate=True)
            batch["forwarded"] += 1
        pipe.hincrby(STATS, "forwarded", batch["forwarded"])
        pipe.hincrby(STATS, "skipped", batch["skipped"])
        pipe.hincrby(STATS, "invalid", batch["invalid"])
        pipe.hset(STATS, "heartbeat", time.time())
        pipe.execute()
        for k, v in batch.items():
            self.counts[k] += v
        return batch


def _kafka_consumer(bootstrap: str, topic: str, group: str, offset_reset: str):
    try:
        from confluent_kafka import Consumer
    except ImportError as e:  # optional dependency: only the live connection needs it
        raise SystemExit("confluent-kafka is not installed: pip install confluent-kafka") from e
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "enable.auto.commit": False,
            "auto.offset.reset": offset_reset,
            "client.id": "dpi-fraud-engine",
        }
    )
    consumer.subscribe([topic])
    return consumer


def run_kafka(
    bootstrap: str | None = None,
    topic: str | None = None,
    group: str | None = None,
    offset_reset: str | None = None,
    batch: int = 500,
    stop_event: threading.Event | None = None,
) -> dict[str, int]:
    s = get_settings()
    bootstrap = bootstrap or s.kafka_bootstrap_servers
    topic = topic or s.mojaloop_prepare_topic
    group = group or s.mojaloop_consumer_group
    consumer = _kafka_consumer(bootstrap, topic, group, offset_reset or s.mojaloop_offset_reset)
    fwd = Forwarder(get_redis(timeout=5))
    stop_event = stop_event or threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    fwd.r.hset(STATS, mapping={"bootstrap": bootstrap, "topic": topic, "group": group, "started": time.time()})
    log.info("mojaloop consumer started", bootstrap=bootstrap, topic=topic, group=group)
    last_log = time.time()
    try:
        while not stop_event.is_set():
            msgs = consumer.consume(num_messages=batch, timeout=1.0)
            values = []
            for m in msgs:
                if m.error():
                    log.error("kafka error", error=str(m.error())[:300])
                    continue
                values.append(m.value())
            if not msgs:
                continue
            delivered = False
            while not delivered and not stop_event.is_set():
                try:
                    fwd.forward(values)
                    delivered = True
                except redis.RedisError as e:  # hold the batch (offsets uncommitted) until Redis is back
                    log.error("redis unavailable; retrying batch", error=str(e)[:200])
                    time.sleep(1)
            if not delivered:
                break  # stopping mid-outage: uncommitted, so Kafka re-delivers the batch on restart
            consumer.commit(asynchronous=False)
            if time.time() - last_log > 10:
                log.info("mojaloop consumer progress", **fwd.counts)
                last_log = time.time()
    finally:
        consumer.close()
        log.info("mojaloop consumer stopped", **fwd.counts)
    return fwd.counts


def run_file(path: Path, batch: int = 500) -> dict[str, int]:
    """Replay captured Kafka message values (one JSON message per line) through the same path."""
    fwd = Forwarder(get_redis(timeout=5))
    chunk: list[bytes] = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk.append(line)
            if len(chunk) >= batch:
                fwd.forward(chunk)
                chunk = []
    if chunk:
        fwd.forward(chunk)
    log.info("replayed file", path=str(path), **fwd.counts)
    return fwd.counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bootstrap", help="Kafka bootstrap servers (default: KAFKA_BOOTSTRAP_SERVERS)")
    ap.add_argument("--topic", help="default: MOJALOOP_PREPARE_TOPIC (topic-transfer-prepare)")
    ap.add_argument("--group", help="default: MOJALOOP_CONSUMER_GROUP (fraud-engine)")
    ap.add_argument("--from-beginning", action="store_true", help="a new group starts at the oldest event")
    ap.add_argument("--from-file", type=Path, help="replay a JSON-lines file of message values instead of Kafka")
    ap.add_argument("--batch", type=int, default=500)
    a = ap.parse_args(argv)
    setup_logging(get_settings().log_level)
    if a.from_file:
        print(run_file(a.from_file, a.batch))
        return
    run_kafka(a.bootstrap, a.topic, a.group, "earliest" if a.from_beginning else None, a.batch)


if __name__ == "__main__":
    main()
