import pytest

from pipeline.mojaloop_adapter import AdapterError, from_fspiop, to_fspiop
from tests.conftest import base_tx


def test_round_trip_preserves_internal_schema():
    tx = base_tx(
        payee_id="merch-000007", payee_type="MERCHANT", scenario="PAYMENT", merchant_category="grocery", amount=1234.5
    )
    back = from_fspiop(to_fspiop(tx))
    for k in (
        "transfer_id",
        "quote_id",
        "ts",
        "payer_id",
        "payee_id",
        "payer_fsp",
        "payee_fsp",
        "payer_type",
        "payee_type",
        "scenario",
        "initiator",
        "channel",
        "device_id",
        "ip",
        "device_bound_at",
        "merchant_category",
        "currency",
    ):
        assert back[k] == tx[k], k
    assert back["amount"] == pytest.approx(1234.5)
    assert back["lat"] == pytest.approx(tx["lat"]) and back["lon"] == pytest.approx(tx["lon"])


def test_fspiop_message_shape():
    msg = to_fspiop(base_tx())
    p = msg["content"]["payload"]
    assert msg["metadata"]["event"]["action"] == "prepare"
    assert msg["content"]["headers"]["fspiop-source"] == "dfsp-01"
    assert set(p) >= {"transferId", "payerFsp", "payeeFsp", "amount", "ilpPacket", "condition", "expiration"}
    assert p["amount"] == {"amount": "500", "currency": "INR"}  # FSPIOP Amount: no trailing zeros


def test_reversal_maps_to_refund_subscenario_and_back():
    tx = base_tx(scenario="REVERSAL", original_transfer_id="orig-1", initiator="PAYEE")
    msg = to_fspiop(tx)
    back = from_fspiop(msg)
    assert back["scenario"] == "REVERSAL"
    assert back["original_transfer_id"] == "orig-1"


def test_bare_transfer_body_without_ilp_party_info():
    body = {
        "transferId": "abc",
        "payerFsp": "dfsp-01",
        "payeeFsp": "dfsp-02",
        "amount": {"amount": "10", "currency": "INR"},
        "expiration": "2026-01-01T00:01:00.000Z",
    }
    out = from_fspiop(body)
    assert out["transfer_id"] == "abc"
    assert out["payer_id"] == "fsp:dfsp-01" and out["payee_id"] == "fsp:dfsp-02"
    assert out["ts"] == 1767225600000
    assert out["lat"] is None and out["device_id"] == ""


def test_invalid_message_raises():
    with pytest.raises(AdapterError):
        from_fspiop({"content": {"payload": {"amount": {"amount": "x"}}}})
