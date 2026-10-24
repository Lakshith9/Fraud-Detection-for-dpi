"""End-to-end: generator -> replay -> train -> bootstrap -> simulator -> Redis stream -> scoring worker -> PostgreSQL
-> alerts -> REST API / WebSocket -> operator feedback -> retraining (+ gate) -> registry -> hot reload.

Runs against real Redis + PostgreSQL in ISOLATED namespaces (Redis DB 15, database `fraud_test`).
Skips automatically when either service is unreachable.
"""

from __future__ import annotations

import os
import threading
import time
from urllib.parse import urlparse, urlunparse

import pytest

pytestmark = pytest.mark.integration

API_KEY = "integration-test-key"


def _test_db_url() -> str:
    if os.environ.get("TEST_DATABASE_URL"):
        return os.environ["TEST_DATABASE_URL"]
    from core.config import Settings

    u = urlparse(Settings().database_url)
    return urlunparse(u._replace(path="/fraud_test"))


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    import psycopg
    import redis

    from tests.conftest import TEST_REDIS_URL

    db_url = _test_db_url()
    try:
        redis.Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=2).ping()
        psycopg.connect(db_url, connect_timeout=3).close()
    except Exception as e:
        pytest.skip(f"integration services unavailable: {e}")

    tmp = tmp_path_factory.mktemp("e2e")
    env = {
        "REDIS_URL": TEST_REDIS_URL,
        "DATABASE_URL": db_url,
        "DATA_DIR": str(tmp / "data"),
        "ARTIFACTS_DIR": str(tmp / "artifacts"),
        "STREAM_PREFIX": "test:transfers",
        "STREAM_PARTITIONS": "1",
        "OPERATOR_API_KEYS": f"tester:{API_KEY}",
        "MODEL_POLL_INTERVAL_S": "1",
        "RETRAIN_TOLERANCE": "1.0",
        "LOG_LEVEL": "WARNING",
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    from core import db
    from core.config import get_settings

    db.close_pool()
    get_settings.cache_clear()
    redis.Redis.from_url(TEST_REDIS_URL).flushdb()

    from data_gen.generate import GenConfig, Generator, write_outputs
    from models import registry
    from models.train import train
    from pipeline import bootstrap
    from pipeline.replay import replay

    data = tmp / "data"
    cfg = GenConfig(seed=5, n_transactions=60_000, days=60, fraud_rate=0.02)
    t = time.time()
    res = Generator(cfg).run()
    write_outputs(res, cfg, data, time.time() - t)
    replay(data, data / "features", snapshot_day=46, graph_step_h=6.0)
    db.init_schema()
    with db.get_pool().connection() as conn:
        conn.execute("TRUNCATE model_registry")
    out = train(data / "features", data, "v1", get_settings().artifacts_dir)
    registry.register("v1", out["metrics"], out["data_hash"])
    bootstrap.main(["--features", str(data / "features"), "--data", str(data), "--reset"])
    registry.promote("v1")

    from pipeline import persister, worker

    stop = threading.Event()
    th = threading.Thread(target=worker.run, args=(0,), kwargs={"stop_event": stop}, daemon=True)
    th.start()
    th_p = threading.Thread(target=persister.run, kwargs={"stop_event": stop}, daemon=True)
    th_p.start()
    yield {"data": data, "tx": res["transactions"]}
    stop.set()
    th.join(timeout=20)
    th_p.join(timeout=20)
    db.close_pool()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    get_settings.cache_clear()


def _wait(pred, timeout=120, every=0.5):
    t = time.time()
    while time.time() - t < timeout:
        if pred():
            return True
        time.sleep(every)
    return False


def test_full_pipeline(system):
    from fastapi.testclient import TestClient

    from core import db
    from graph import worker as graph_worker
    from graph.rings import RingTracker
    from simulator.simulator import run as simulate

    n = 1500
    summary = simulate(
        tps=1500,
        from_day=46,
        to_day=50,
        limit=n,
        loop=False,
        dup_rate=0.02,
        disorder_rate=0.01,
        data_dir=system["data"],
        ground_truth=True,
    )
    assert summary["sent"] >= n
    scored = lambda: db.fetch_one("SELECT count(*) AS n FROM transactions WHERE source = 'stream'")["n"]  # noqa: E731
    assert _wait(lambda: scored() >= n), f"only {scored()} of {n} transfers scored"
    time.sleep(1.0)
    # idempotency: duplicates were delivered but each transfer is stored exactly once
    assert scored() == n
    assert summary["duplicates"] > 0

    # every scored row carries model version, latency, features and decision
    row = db.fetch_one("SELECT * FROM transactions WHERE source='stream' LIMIT 1")
    assert row["model_version"] == "v1" and row["latency_ms"] > 0 and row["features"] and row["decision"]
    lat = [r["latency_ms"] for r in db.fetch_all("SELECT latency_ms FROM transactions WHERE source='stream'")]
    assert sorted(lat)[int(0.99 * len(lat))] < 100, "p99 scoring latency above the 100 ms budget"

    # graph worker cycle on the live window publishes ring scores
    import redis

    from tests.conftest import TEST_REDIS_URL

    r = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
    _, stats = graph_worker.run_once(r, RingTracker(), None)
    assert stats["rows"] > 0

    from api.main import app

    with TestClient(app) as client:
        h = {"X-API-Key": API_KEY}
        assert client.get("/v1/alerts").status_code == 401
        assert client.get("/healthz").json()["status"] == "ok"
        alerts = client.get("/v1/alerts?limit=1000", headers=h).json()["items"]
        assert alerts, "no alerts raised"
        a = alerts[0]
        d = client.get(f"/v1/transactions/{a['transfer_id']}", headers=h).json()
        assert d["transaction"]["transfer_id"] == a["transfer_id"] and d["payer_history"]
        e = client.get(f"/v1/transactions/{a['transfer_id']}/explanation", headers=h).json()
        assert len(e["top_reasons"]) == 5 and all(x["reason"] for x in e["top_reasons"])
        allow = db.fetch_one("SELECT transfer_id FROM transactions WHERE decision='ALLOW' AND source='stream' LIMIT 1")
        e2 = client.get(f"/v1/transactions/{allow['transfer_id']}/explanation", headers=h).json()
        assert e2["contributions"]

        # synchronous pre-settlement scoring + idempotency
        nxt = system["tx"].iloc[-1].to_dict()
        body = {
            k: (None if isinstance(v, float) and v != v else v)
            for k, v in nxt.items()
            if k not in ("label", "is_fraud", "fraud_type", "case_id")
        }
        s1 = client.post("/v1/score", json=body, headers=h).json()
        assert s1["decision"] in ("ALLOW", "REVIEW", "BLOCK") and 0 <= s1["risk_score"] <= 1 and s1["latency_ms"] > 0
        s2 = client.post("/v1/score", json=body, headers=h).json()
        assert s2["duplicate"] is True and s2["risk_score"] == s1["risk_score"]
        bad = dict(body, amount=-5)
        assert client.post("/v1/score", json=bad, headers=h).status_code == 422

        # cases
        c = client.post("/v1/cases", json={"title": "it case", "transfer_ids": [a["transfer_id"]]}, headers=h).json()
        c = client.patch(
            f"/v1/cases/{c['case_id']}", json={"status": "investigating", "note": "looking"}, headers=h
        ).json()
        assert c["status"] == "investigating" and c["notes"] and len(c["audit"]) == 3

        # operator feedback from ground truth: label every alert, report missed frauds
        gt = {
            x["transfer_id"]: x["is_fraud"] for x in db.fetch_all("SELECT transfer_id, is_fraud FROM sim_ground_truth")
        }
        kinds = []
        for al in alerts:
            res = client.post(
                "/v1/feedback",
                json={"transfer_id": al["transfer_id"], "label": "fraud" if gt.get(al["transfer_id"]) else "legit"},
                headers=h,
            )
            kinds.append(res.json()["kind"])
        missed = db.fetch_all("""SELECT t.transfer_id FROM transactions t JOIN sim_ground_truth g USING (transfer_id)
                                  WHERE g.is_fraud = 1 AND t.decision = 'ALLOW'""")
        for m in missed:
            kinds.append(
                client.post("/v1/feedback", json={"transfer_id": m["transfer_id"], "label": "fraud"}, headers=h).json()[
                    "kind"
                ]
            )
        assert set(kinds) <= {"true_positive", "false_positive", "false_negative"}

        with client.websocket_connect(f"/ws/alerts?api_key={API_KEY}") as ws:
            assert ws.receive_json()["type"] == "snapshot"

        m = client.get("/v1/metrics", headers=h).json()
        assert m["counts"]["total"] >= n and m["latency_ms"]["p99"] is not None and m["drift"]["features"]

    # retraining with the gate (tolerance relaxed in this test so the promotion path is exercised)
    from models.retrain import run_retrain

    out = run_retrain("integration-test")
    assert out["candidate_version"] == "v2" and out["promoted"] is True
    run = db.fetch_one("SELECT * FROM retrain_runs WHERE id = %s", (out["run_id"],))
    assert run["status"] == "done" and run["before_metrics"] and run["after_metrics"]
    # hot reload: the running worker picks up v2 without restart
    assert _wait(lambda: r.hget("worker:0", "model") == "v2", timeout=30)
