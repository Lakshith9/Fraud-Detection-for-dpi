import base64

import orjson
import pytest

from pipeline.ilp import IlpError, decode_ilp_packet
from pipeline.mojaloop_adapter import from_fspiop

TRANSACTION = {
    "transactionId": "85feac2f-39b2-491b-817e-4a03203d4f14",
    "quoteId": "7c23e80c-d078-4077-8263-2c047876fcf6",
    "payee": {"partyIdInfo": {"partyIdType": "MSISDN", "partyIdentifier": "27713803912", "fspId": "payeefsp"}},
    "payer": {
        "partyIdInfo": {"partyIdType": "MSISDN", "partyIdentifier": "44123456789", "fspId": "payerfsp"},
        "personalInfo": {"complexName": {"firstName": "Mats", "lastName": "Hagman"}},
    },
    "amount": {"amount": "100", "currency": "USD"},
    "transactionType": {"scenario": "TRANSFER", "initiator": "PAYER", "initiatorType": "CONSUMER"},
}


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def oer_var(b: bytes) -> bytes:
    """OER length-prefixed octets, written out by hand (short form < 128, else 0x8n + n length bytes)."""
    if len(b) < 0x80:
        return bytes([len(b)]) + b
    n = (len(b).bit_length() + 7) // 8
    return bytes([0x80 | n]) + len(b).to_bytes(n, "big") + b


def prepare_v4(data: bytes, amount=10_000, destination=b"g.payeefsp.msisdn.27713803912") -> str:
    body = (
        amount.to_bytes(8, "big")
        + b"20261025143000123"  # expiresAt, 17 ASCII chars
        + bytes(range(32))  # executionCondition
        + oer_var(destination)
        + oer_var(data)
    )
    return b64url(bytes([12]) + oer_var(body))


def payment_v1(data: bytes, amount=10_000, account=b"g.payeefsp.msisdn.27713803912") -> str:
    body = amount.to_bytes(8, "big") + oer_var(account) + oer_var(data) + b"\x00"
    return b64url(bytes([1]) + oer_var(body))


def test_ilpv4_prepare_with_base64url_json_data():
    pkt = decode_ilp_packet(prepare_v4(b64url(orjson.dumps(TRANSACTION)).encode()))
    assert pkt.kind == "prepare_v4"
    assert pkt.amount == 10_000
    assert pkt.destination == "g.payeefsp.msisdn.27713803912"
    assert pkt.expires_at == "20261025143000123"
    assert pkt.transaction == TRANSACTION


def test_ilpv4_prepare_with_raw_json_data_and_standard_base64():
    s = prepare_v4(orjson.dumps(TRANSACTION))
    std = base64.b64encode(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))).decode()
    assert decode_ilp_packet(std).transaction == TRANSACTION


def test_ilpv1_payment_long_form_length():
    data = b64url(orjson.dumps(TRANSACTION)).encode()
    assert len(data) > 255  # forces the two-byte OER length prefix
    pkt = decode_ilp_packet(payment_v1(data))
    assert pkt.kind == "payment_v1"
    assert pkt.destination == "g.payeefsp.msisdn.27713803912"
    assert pkt.transaction["payer"]["partyIdInfo"]["partyIdentifier"] == "44123456789"


def test_mock_packet_still_decodes():
    pkt = decode_ilp_packet(b64url(orjson.dumps(TRANSACTION)))
    assert pkt.kind == "mock_json" and pkt.transaction == TRANSACTION


@pytest.mark.parametrize(
    "packet",
    [
        "not base64 !!",
        b64url(bytes([12, 5, 1, 2])),  # truncated
        b64url(bytes([99, 1, 0])),  # unknown type
        prepare_v4(b"%%% not json %%%"),
    ],
)
def test_bad_packets_raise(packet):
    with pytest.raises(IlpError):
        decode_ilp_packet(packet)


def test_adapter_reads_parties_from_real_ilp_packet():
    body = {
        "transferId": "b51ec534-ee48-4575-b6a9-ead2955b8069",
        "payerFsp": "payerfsp",
        "payeeFsp": "payeefsp",
        "amount": {"amount": "100", "currency": "USD"},
        "ilpPacket": prepare_v4(b64url(orjson.dumps(TRANSACTION)).encode()),
        "condition": "f5sqb7tBTWPd5Y8BDFdMm9BJR_MNI4isf8p8n4D5pHA",
        "expiration": "2026-10-25T14:31:00.123Z",
    }
    tx = from_fspiop(body)
    assert tx["payer_id"] == "MSISDN:44123456789" and tx["payer_fsp"] == "payerfsp"
    assert tx["payee_id"] == "MSISDN:27713803912" and tx["payee_fsp"] == "payeefsp"
    assert tx["payer_type"] == "CONSUMER" and tx["scenario"] == "TRANSFER"
    assert tx["quote_id"] == TRANSACTION["quoteId"]


def test_adapter_degrades_to_fsp_identities_on_unreadable_packet():
    body = {
        "transferId": "x",
        "payerFsp": "payerfsp",
        "payeeFsp": "payeefsp",
        "amount": {"amount": "1", "currency": "USD"},
        "ilpPacket": b64url(bytes([99, 1, 0])),
        "expiration": "2026-10-25T14:31:00.000Z",
    }
    tx = from_fspiop(body)
    assert tx["payer_id"] == "fsp:payerfsp" and tx["payee_id"] == "fsp:payeefsp"
