"""Decoder for the ILP packets carried in Mojaloop FSPIOP ``POST /transfers`` bodies.

In FSPIOP the ``ilpPacket`` field is a base64url string. Inside is a binary, OER-encoded Interledger packet:

* **ILPv4 Prepare** (type 12), used by Mojaloop SDKs since FSPIOP 1.1::

      type(1) | len | amount UInt64 | expiresAt (17 ASCII, YYYYMMDDHHmmssfff) | executionCondition(32)
                    | destination (var IA5) | data (var octet)

* **ILPv1 Payment** (type 1), produced by older Mojaloop deployments::

      type(1) | len | amount UInt64 | account (var IA5) | data (var octet) | extensions

``data`` holds the FSPIOP ``Transaction`` object (payer, payee, transactionType) as base64url-encoded JSON;
some implementations put the JSON there directly. The simulator's mocked packet is the base64url JSON
``Transaction`` with no ILP envelope; ``decode_ilp_packet`` accepts that too.

Spec references: Interledger RFC 0027 (ILPv4), RFC 0030 (OER notes), Mojaloop FSPIOP API v1.1 section 4.5.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from typing import Any

import orjson

TYPE_ILP_PAYMENT_V1 = 1
TYPE_ILP_PREPARE_V4 = 12


class IlpError(ValueError):
    pass


@dataclass
class IlpPacket:
    kind: str  # "prepare_v4" | "payment_v1" | "mock_json"
    amount: int | None = None
    destination: str = ""
    expires_at: str = ""
    transaction: dict[str, Any] = field(default_factory=dict)


class _Reader:
    def __init__(self, buf: bytes):
        self.buf, self.pos = buf, 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise IlpError(f"truncated ILP packet (need {n} bytes at offset {self.pos}, have {len(self.buf)})")
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def length(self) -> int:
        """OER length determinant: short form (< 128) or long form (0x80 | number of length bytes)."""
        first = self.take(1)[0]
        if first < 0x80:
            return first
        n = first & 0x7F
        if n == 0 or n > 4:
            raise IlpError(f"unsupported OER length prefix 0x{first:02x}")
        return int.from_bytes(self.take(n), "big")

    def var_octets(self) -> bytes:
        return self.take(self.length())


def b64_any(s: str) -> bytes:
    """Decode base64url or standard base64, padded or not."""
    s = s.strip().replace("-", "+").replace("_", "/")
    try:
        return base64.b64decode(s + "=" * (-len(s) % 4), validate=True)
    except (binascii.Error, ValueError) as e:
        raise IlpError(f"ilpPacket is not base64: {e}") from e


def _transaction_from_data(data: bytes) -> dict[str, Any]:
    data = data.strip()
    if not data:
        return {}
    if data[:1] != b"{":
        try:
            data = b64_any(data.decode("ascii"))
        except (UnicodeDecodeError, IlpError) as e:
            raise IlpError("ILP data is neither JSON nor base64 JSON") from e
    try:
        obj = orjson.loads(data)
    except orjson.JSONDecodeError as e:
        raise IlpError(f"ILP data is not a JSON Transaction: {e}") from e
    if not isinstance(obj, dict):
        raise IlpError("ILP data is not a JSON object")
    return obj


def _parse_prepare_v4(r: _Reader) -> IlpPacket:
    amount = int.from_bytes(r.take(8), "big")
    expires = r.take(17).decode("ascii", errors="replace")
    r.take(32)  # executionCondition
    destination = r.var_octets().decode("ascii", errors="replace")
    data = r.var_octets()
    return IlpPacket("prepare_v4", amount, destination, expires, _transaction_from_data(data))


def _parse_payment_v1(r: _Reader) -> IlpPacket:
    amount = int.from_bytes(r.take(8), "big")
    account = r.var_octets().decode("ascii", errors="replace")
    data = r.var_octets()
    return IlpPacket("payment_v1", amount, account, "", _transaction_from_data(data))


def decode_ilp_packet(packet: str) -> IlpPacket:
    """Decode an FSPIOP ``ilpPacket`` string. Raises IlpError if it is not a recognised packet."""
    raw = b64_any(packet)
    if raw[:1] == b"{":
        return IlpPacket("mock_json", transaction=_transaction_from_data(raw))
    if not raw:
        raise IlpError("empty ILP packet")
    r = _Reader(raw)
    ptype = r.take(1)[0]
    body = _Reader(r.var_octets())
    if ptype == TYPE_ILP_PREPARE_V4:
        return _parse_prepare_v4(body)
    if ptype == TYPE_ILP_PAYMENT_V1:
        return _parse_payment_v1(body)
    raise IlpError(f"unsupported ILP packet type {ptype}")
