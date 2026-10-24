import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:6379/15")


def _redis_or_none():
    try:
        import redis

        r = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)
        r.ping()
        return r
    except Exception:
        return None


@pytest.fixture()
def test_redis():
    r = _redis_or_none()
    if r is None:
        pytest.skip("Redis not reachable at TEST_REDIS_URL")
    r.flushdb()
    yield r
    r.flushdb()


def base_tx(**kw) -> dict:
    tx = {
        "transfer_id": "t-0",
        "quote_id": "q-0",
        "ts": 1_767_225_600_000 + 10 * 3_600_000,
        "payer_id": "cust-1",
        "payer_alias": "+911",
        "payer_fsp": "dfsp-01",
        "payer_type": "CONSUMER",
        "payee_id": "cust-2",
        "payee_alias": "+912",
        "payee_fsp": "dfsp-02",
        "payee_type": "CONSUMER",
        "amount": 500.0,
        "currency": "INR",
        "scenario": "TRANSFER",
        "initiator": "PAYER",
        "channel": "APP",
        "device_id": "dev-1",
        "ip": "10.0.0.1",
        "lat": 19.07,
        "lon": 72.87,
        "device_bound_at": 1_767_225_600_000 - 90 * 86_400_000,
        "merchant_category": "",
        "original_transfer_id": "",
    }
    tx.update(kw)
    return tx
