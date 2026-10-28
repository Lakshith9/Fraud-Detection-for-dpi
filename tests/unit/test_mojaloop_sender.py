import asyncio
import base64
import hashlib
import re
import uuid
from datetime import UTC, datetime

import httpx
import orjson
import pytest

from core.config import get_settings
from pipeline.ilp import decode_ilp_packet, encode_ilp_prepare
from pipeline.mojaloop_adapter import from_fspiop, fspiop_amount, fspiop_extensions
from simulator.mojaloop_sender import MockFsp, build_transfer, condition_for, fulfilment_for, switch_id
from tests.conftest import base_tx

# FSPIOP API data types (Mojaloop API spec): Amount, IlpCondition, IlpPacket
AMOUNT_RE = re.compile(r"^([0]|([1-9][0-9]{0,17}))([.][0-9]{0,3}[1-9])?$")
CONDITION_RE = re.compile(r"^[A-Za-z0-9-_]{43}$")
ILP_RE = re.compile(r"^[A-Za-z0-9-_]+[=]{0,2}$")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


@pytest.mark.parametrize(
    "amount,expected", [(500.0, "500"), (12.5, "12.5"), (0.01, "0.01"), (1234.56789, "1234.5679"), (100.10, "100.1")]
)
def test_fspiop_amount_format(amount, expected):
    assert fspiop_amount(amount) == expected
    assert AMOUNT_RE.match(expected)


def test_ilp_prepare_encoder_round_trip():
    trx = {"transactionId": "t", "payer": {"partyIdInfo": {"partyIdType": "MSISDN", "partyIdentifier": "1"}}}
    cond = hashlib.sha256(b"x").digest()
    pkt = decode_ilp_packet(encode_ilp_prepare(trx, 150_000, "g.dfsp-02.msisdn.9198", "20261028150000123", cond))
    assert (pkt.kind, pkt.amount, pkt.destination, pkt.expires_at) == (
        "prepare_v4",
        150_000,
        "g.dfsp-02.msisdn.9198",
        "20261028150000123",
    )
    assert pkt.transaction == trx


def test_extensions_device_toggle_and_nan():
    tx = base_tx(lat=float("nan"), merchant_category="grocery")
    keys = {e["key"] for e in fspiop_extensions(tx)}
    assert {"deviceId", "ipAddress", "channel", "deviceBoundAt", "merchantCategory"} <= keys
    assert "geoLat" not in keys  # NaN dropped
    plain = {e["key"] for e in fspiop_extensions(tx, device_signals=False)}
    assert plain == {"merchantCategory"}


@pytest.mark.parametrize("device_signals", [True, False])
def test_build_transfer_is_a_valid_fspiop_body(device_signals):
    tx = base_tx(transfer_id="tx-000123", quote_id="q-1", amount=1999.0)
    body = build_transfer(tx, run_id="r1", secret="s", device_signals=device_signals)
    assert body["transferId"] == switch_id("r1", "tx-000123")
    uuid.UUID(body["transferId"])
    assert AMOUNT_RE.match(body["amount"]["amount"]) and body["amount"]["amount"] == "1999"
    assert CONDITION_RE.match(body["condition"]) and ILP_RE.match(body["ilpPacket"])
    fulfilment = fulfilment_for(body["transferId"], "s")
    assert _b64d(body["condition"]) == condition_for(fulfilment) == hashlib.sha256(fulfilment).digest()
    assert datetime.fromisoformat(body["expiration"].replace("Z", "+00:00")) > datetime.now(UTC)
    pkt = decode_ilp_packet(body["ilpPacket"])
    assert pkt.amount == 199_900 and pkt.destination == "g.dfsp-02.msisdn.912"
    assert pkt.transaction["payer"]["partyIdInfo"]["partySubIdOrType"] == "cust-1"  # identity kept for history
    ext = {e["key"]: e["value"] for e in body["extensionList"]["extension"]}
    assert ext["eventTime"] == "2026-01-01T10:00:00.000Z"
    assert ("deviceId" in ext) is device_signals
    assert all(len(e["key"]) <= 32 and len(e["value"]) <= 128 for e in body["extensionList"]["extension"])


def test_event_time_extension_only_used_when_trusted(monkeypatch):
    body = build_transfer(base_tx(), run_id="r1", secret="s", device_signals=True)
    original_ts = base_tx()["ts"]
    monkeypatch.setattr(get_settings(), "trust_fsp_event_time", False)
    assert from_fspiop(body)["ts"] != original_ts  # falls back to the wire expiration (real time)
    monkeypatch.setattr(get_settings(), "trust_fsp_event_time", True)
    tx = from_fspiop(body)
    assert tx["ts"] == original_ts
    assert tx["payer_id"] == "cust-1" and tx["device_id"] == "dev-1"


def test_mock_fsp_fulfils_with_matching_fulfilment():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    mock = MockFsp("http://switch", secret="s")
    mock.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tid = switch_id("r1", "tx-1")
    asyncio.run(mock._fulfil(tid, "dfsp-01", "dfsp-02"))
    (req,) = seen
    assert req.method == "PUT" and str(req.url) == f"http://switch/transfers/{tid}"
    assert req.headers["fspiop-source"] == "dfsp-02" and req.headers["fspiop-destination"] == "dfsp-01"
    put = orjson.loads(req.content)
    assert put["transferState"] == "COMMITTED"
    assert hashlib.sha256(_b64d(put["fulfilment"])).digest() == condition_for(fulfilment_for(tid, "s"))
    assert mock.counts["fulfils_sent"] == 1
