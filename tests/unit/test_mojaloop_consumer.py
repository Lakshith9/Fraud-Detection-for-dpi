import base64
from pathlib import Path
from urllib.parse import quote

import orjson
import pytest

from core.redis_client import partition_of
from pipeline.mojaloop_adapter import AdapterError, from_fspiop, is_transfer_prepare, normalize_message, to_fspiop
from pipeline.mojaloop_consumer import DLQ, STATS, Forwarder
from tests.conftest import base_tx
from tests.unit.test_ilp import TRANSACTION, b64url, prepare_v4

BODY = {
    "transferId": "b51ec534-ee48-4575-b6a9-ead2955b8069",
    "payerFsp": "payerfsp",
    "payeeFsp": "payeefsp",
    "amount": {"amount": "100", "currency": "USD"},
    "ilpPacket": prepare_v4(b64url(orjson.dumps(TRANSACTION)).encode()),
    "condition": "f5sqb7tBTWPd5Y8BDFdMm9BJR_MNI4isf8p8n4D5pHA",
    "expiration": "2026-10-25T14:31:00.123Z",
}


def live_message(body=BODY, action="prepare", base64_payload=True) -> dict:
    """A transfer-prepare event as ml-api-adapter publishes it: the body is a data URI string."""
    mime = "application/vnd.interoperability.transfers+json;version=1.1"
    raw = orjson.dumps(body)
    payload = (
        f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        if base64_payload
        else f"data:{mime},{quote(raw.decode())}"
    )
    return {
        "id": body["transferId"],
        "from": "payerfsp",
        "to": "payeefsp",
        "type": "application/json",
        "content": {
            "uriParams": {"id": body["transferId"]},
            "headers": {
                "content-type": f"{mime}",
                "date": "Sun, 25 Oct 2026 14:30:00 GMT",
                "fspiop-source": "payerfsp",
                "fspiop-destination": "payeefsp",
            },
            "payload": payload,
        },
        "metadata": {
            "correlationId": body["transferId"],
            "event": {
                "id": "3920382d-f78c-4023-adf9-0d7a4a2a3a2f",
                "type": "transfer",
                "action": action,
                "createdAt": "2026-10-25T14:30:00.123Z",
                "state": {"status": "success", "code": 0},
            },
        },
    }


@pytest.mark.parametrize("base64_payload", [True, False])
def test_live_data_uri_message_maps_to_internal_schema(base64_payload):
    tx = from_fspiop(live_message(base64_payload=base64_payload))
    assert tx["transfer_id"] == BODY["transferId"]
    assert tx["ts"] == 1792938600123
    assert tx["amount"] == 100.0 and tx["currency"] == "USD"
    assert tx["payer_id"] == "MSISDN:44123456789" and tx["payee_id"] == "MSISDN:27713803912"
    assert tx["device_id"] == "" and tx["lat"] is None  # FSPIOP carries no device/geo signals by default


def test_normalize_decodes_payload_and_keeps_envelope():
    msg = live_message()
    norm = normalize_message(msg)
    assert norm["content"]["payload"] == BODY
    assert norm["metadata"] == msg["metadata"]
    assert isinstance(msg["content"]["payload"], str)  # input untouched
    assert from_fspiop(norm) == from_fspiop(msg)


def test_simulator_messages_pass_through_normalize_unchanged():
    msg = to_fspiop(base_tx())
    assert normalize_message(msg) == msg


def test_event_filter():
    assert is_transfer_prepare(live_message())
    assert not is_transfer_prepare(live_message(action="bulk-prepare"))
    assert not is_transfer_prepare(live_message(action="fx-prepare"))
    assert is_transfer_prepare(BODY)  # bare POST /transfers body


def test_bad_data_uri_raises():
    msg = live_message()
    msg["content"]["payload"] = "data:application/json;base64,@@@"
    with pytest.raises(AdapterError):
        from_fspiop(msg)


class FakePipeline:
    def __init__(self, sink):
        self.sink = sink

    def xadd(self, key, fields, maxlen=None, approximate=False):
        self.sink.setdefault(key, []).append(fields)

    def hincrby(self, key, field, n):
        h = self.sink.setdefault(key, {})
        h[field] = h.get(field, 0) + n

    def hset(self, key, field=None, value=None, mapping=None):
        self.sink.setdefault(key, {})[field] = value

    def execute(self):
        return []


class FakeRedis:
    def __init__(self):
        self.data = {}

    def pipeline(self, transaction=False):
        return FakePipeline(self.data)


def test_forwarder_routes_by_payer_and_dead_letters_bad_events():
    r = FakeRedis()
    fwd = Forwarder(r, partitions=2, maxlen=1000)
    out = fwd.forward(
        [
            orjson.dumps(live_message()),
            orjson.dumps(live_message(action="bulk-prepare")),
            b"{not json",
            orjson.dumps({"content": {"payload": "data:application/json;base64,e30"}}),  # {} -> no transferId
            None,
        ]
    )
    assert out == {"forwarded": 1, "skipped": 2, "invalid": 2}
    part = partition_of("MSISDN:44123456789", 2)
    entries = r.data[f"fsp:transfers:{part}"]
    assert len(entries) == 1
    event = orjson.loads(entries[0]["event"])
    assert event["content"]["payload"] == BODY  # workers receive the decoded shape
    assert float(entries[0]["sent_at"]) > 0
    assert len(r.data[DLQ]) == 2
    assert r.data[STATS]["forwarded"] == 1 and r.data[STATS]["invalid"] == 2


def test_forwarded_event_is_what_the_worker_expects():
    from pipeline.worker import handle

    r = FakeRedis()
    Forwarder(r, partitions=1, maxlen=1000).forward([orjson.dumps(live_message())])
    fields = r.data["fsp:transfers:0"][0]

    class StubScorer:
        def score(self, tx, sent_at_ms=None):
            self.tx, self.sent_at = tx, sent_at_ms

    s = StubScorer()
    assert handle(s, fields) == "transfer"
    assert s.tx["transfer_id"] == BODY["transferId"] and s.sent_at > 0


def captured_message() -> dict:
    """A transfer-prepare event captured from a live Mojaloop switch (ml-core-test-harness, TTK P2P test)."""
    path = Path(__file__).resolve().parent.parent / "fixtures" / "mojaloop_transfer_prepare.json"
    return orjson.loads(path.read_bytes())


def test_captured_live_event_maps_to_internal_schema():
    msg = captured_message()
    assert msg["metadata"]["event"]["type"] == "prepare"  # what ml-api-adapter really sends
    assert is_transfer_prepare(msg)
    tx = from_fspiop(msg)
    assert tx["transfer_id"] == "7cec31e8-eae8-41d6-8157-a2353c16cb55"
    assert tx["payer_id"] == "MSISDN:44123456789" and tx["payer_fsp"] == "testingtoolkitdfsp"
    assert tx["payee_id"] == "MSISDN:27713803912" and tx["payee_fsp"] == "payeefsp"
    assert tx["quote_id"] == "29e54ef9-376f-402a-bd48-949a6f6d4c26"
    assert tx["amount"] == 100.0 and tx["currency"] == "XXX" and tx["scenario"] == "TRANSFER"
    assert tx["ts"] == 1792941419265


def test_captured_live_event_is_forwarded():
    r = FakeRedis()
    out = Forwarder(r, partitions=2, maxlen=1000).forward([orjson.dumps(captured_message())])
    assert out == {"forwarded": 1, "skipped": 0, "invalid": 0}
