"""Thin adapter between Mojaloop FSPIOP-style transfer events and the engine's internal transaction schema.

The simulator renders internal rows as Mojaloop *transfer prepare* messages, shaped like the events the
Mojaloop central-ledger publishes on its ``topic-transfer-prepare`` Kafka topic::

    {"id", "from", "to", "type", "content": {"headers": {...}, "payload": <POST /transfers body>},
     "metadata": {"event": {"type": "transfer", "action": "prepare", "createdAt": ISO-8601}}}

In real Mojaloop the payer/payee party details and transactionType live inside the ILP packet's ``data``
field (base64url encoded FSPIOP ``Transaction`` object). Here the ILP packet is MOCKED: it is the
base64url encoded JSON Transaction object without the OER/ILP envelope. Risk signals that FSPIOP does not
carry natively (device, IP, geo, channel, device-binding time) travel in the standard ``extensionList``.

``from_fspiop`` also accepts a bare ``POST /transfers`` body (payload only), which is what a real Mojaloop
sandbox callback or a Kafka consumer on central-ledger would hand you, so a live switch can be plugged in by
pointing a consumer at its topic and calling ``from_fspiop`` on each message.

This is a Mojaloop-compatible SIMULATION + adapter, not a live Mojaloop integration.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime
from email.utils import formatdate
from typing import Any

import orjson

_EXT_KEYS = {
    "deviceId": "device_id",
    "ipAddress": "ip",
    "geoLat": "lat",
    "geoLon": "lon",
    "channel": "channel",
    "deviceBoundAt": "device_bound_at",
    "merchantCategory": "merchant_category",
}
_NUMERIC_EXT = {"lat", "lon"}
_INT_EXT = {"device_bound_at"}
_PARTY_TYPE = {"CONSUMER": "MSISDN", "MERCHANT": "BUSINESS", "BUSINESS": "BUSINESS"}


class AdapterError(ValueError):
    pass


def _b64e(obj: Any) -> str:
    return base64.urlsafe_b64encode(orjson.dumps(obj)).decode().rstrip("=")


def _b64d(s: str) -> Any:
    pad = "=" * (-len(s) % 4)
    return orjson.loads(base64.urlsafe_b64decode(s + pad))


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_iso(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def _party(pid: str, alias: str, fsp: str, kind: str, mcc: str = "") -> dict:
    party: dict[str, Any] = {
        "partyIdInfo": {
            "partyIdType": _PARTY_TYPE.get(kind, "MSISDN"),
            "partyIdentifier": alias,
            "partySubIdOrType": pid,
            "fspId": fsp,
        },
        "name": pid,
    }
    if kind == "MERCHANT":
        party["merchantClassificationCode"] = mcc or "general"
    return party


def to_fspiop(tx: dict) -> dict:
    """Render an internal transaction dict as a Mojaloop transfer-prepare event message."""
    scenario = tx["scenario"]
    ttype: dict[str, Any] = {
        "scenario": "REFUND" if scenario == "REVERSAL" else scenario,
        "initiator": tx.get("initiator", "PAYER"),
        "initiatorType": "BUSINESS" if tx.get("payer_type") in ("MERCHANT", "BUSINESS") else "CONSUMER",
    }
    if scenario == "REVERSAL":
        ttype["subScenario"] = "REVERSAL"
    if tx.get("original_transfer_id"):
        ttype["refundInfo"] = {"originalTransactionId": tx["original_transfer_id"]}
    amount = {"amount": f"{float(tx['amount']):.2f}", "currency": tx.get("currency", "INR")}
    transaction = {
        "transactionId": tx["transfer_id"],
        "quoteId": tx.get("quote_id") or tx["transfer_id"],
        "payer": _party(tx["payer_id"], tx.get("payer_alias", ""), tx["payer_fsp"], tx.get("payer_type", "CONSUMER")),
        "payee": _party(
            tx["payee_id"],
            tx.get("payee_alias", ""),
            tx["payee_fsp"],
            tx.get("payee_type", "CONSUMER"),
            tx.get("merchant_category", ""),
        ),
        "amount": amount,
        "transactionType": ttype,
    }
    ext = []
    for k_out, k_in in _EXT_KEYS.items():
        v = tx.get(k_in)
        if v is not None and v != "":
            ext.append({"key": k_out, "value": str(v)})
    ts = int(tx["ts"])
    condition = base64.urlsafe_b64encode(hashlib.sha256(tx["transfer_id"].encode()).digest()).decode().rstrip("=")
    payload = {
        "transferId": tx["transfer_id"],
        "payerFsp": tx["payer_fsp"],
        "payeeFsp": tx["payee_fsp"],
        "amount": amount,
        "ilpPacket": _b64e(transaction),  # MOCK: base64url JSON Transaction, no OER envelope
        "condition": condition,
        "expiration": _iso(ts + 60_000),
        "extensionList": {"extension": ext},
    }
    return {
        "id": tx["transfer_id"],
        "from": tx["payer_fsp"],
        "to": tx["payee_fsp"],
        "type": "application/json",
        "content": {
            "headers": {
                "content-type": "application/vnd.interoperability.transfers+json;version=1.1",
                "date": formatdate(ts / 1000, usegmt=True),
                "fspiop-source": tx["payer_fsp"],
                "fspiop-destination": tx["payee_fsp"],
            },
            "payload": payload,
        },
        "metadata": {
            "event": {
                "type": "transfer",
                "action": "prepare",
                "createdAt": _iso(ts),
                "state": {"status": "success", "code": 0},
            }
        },
    }


def _party_key(party: dict) -> tuple[str, str, str, str]:
    info = party.get("partyIdInfo") or {}
    alias = str(info.get("partyIdentifier") or "")
    ptype = str(info.get("partyIdType") or "MSISDN")
    pid = info.get("partySubIdOrType") or f"{ptype}:{alias}"
    if party.get("merchantClassificationCode"):
        kind = "MERCHANT"
    else:
        kind = "CONSUMER" if ptype == "MSISDN" else "BUSINESS"
    return str(pid), alias, str(info.get("fspId") or ""), kind


def from_fspiop(msg: dict) -> dict:
    """Map a Mojaloop transfer-prepare message (or bare POST /transfers body) to the internal schema.

    Missing optional fields are filled with neutral defaults; missing mandatory fields raise AdapterError.
    """
    if "content" in msg:
        payload = (msg.get("content") or {}).get("payload") or {}
        created = ((msg.get("metadata") or {}).get("event") or {}).get("createdAt")
    else:
        payload, created = msg, msg.get("createdAt")
    try:
        transfer_id = str(payload["transferId"])
        amount = float(payload["amount"]["amount"])
        currency = str(payload["amount"].get("currency", ""))
    except (KeyError, TypeError, ValueError) as e:
        raise AdapterError(f"invalid transfer payload: {e}") from e
    trx: dict = {}
    if payload.get("ilpPacket"):
        try:
            trx = _b64d(payload["ilpPacket"])
        except Exception:  # a real OER ILP packet can't be decoded here; fall back to header-level data
            trx = {}
    payer_id, payer_alias, payer_fsp, payer_kind = _party_key(trx.get("payer") or {})
    payee_id, payee_alias, payee_fsp, payee_kind = _party_key(trx.get("payee") or {})
    payer_fsp = payer_fsp or str(payload.get("payerFsp") or "")
    payee_fsp = payee_fsp or str(payload.get("payeeFsp") or "")
    if payer_id.endswith(":") or payee_id.endswith(":"):
        # no party info at all: identify by FSP (degraded entity resolution)
        payer_id = payer_id if not payer_id.endswith(":") else f"fsp:{payer_fsp}"
        payee_id = payee_id if not payee_id.endswith(":") else f"fsp:{payee_fsp}"
    ttype = trx.get("transactionType") or {}
    scenario = str(ttype.get("scenario") or "TRANSFER")
    if ttype.get("subScenario") == "REVERSAL":
        scenario = "REVERSAL"
    ext = {e.get("key"): e.get("value") for e in ((payload.get("extensionList") or {}).get("extension") or [])}
    out: dict[str, Any] = {
        "transfer_id": transfer_id,
        "quote_id": str(trx.get("quoteId") or transfer_id),
        "ts": _parse_iso(created)
        if created
        else _parse_iso(payload["expiration"]) - 60_000
        if payload.get("expiration")
        else 0,
        "payer_id": payer_id,
        "payer_alias": payer_alias,
        "payer_fsp": payer_fsp,
        "payer_type": payer_kind,
        "payee_id": payee_id,
        "payee_alias": payee_alias,
        "payee_fsp": payee_fsp,
        "payee_type": payee_kind,
        "amount": amount,
        "currency": currency,
        "scenario": scenario,
        "initiator": str(ttype.get("initiator") or "PAYER"),
        "original_transfer_id": str((ttype.get("refundInfo") or {}).get("originalTransactionId") or ""),
        "device_id": "",
        "ip": "",
        "lat": None,
        "lon": None,
        "channel": "",
        "device_bound_at": None,
        "merchant_category": "",
    }
    mcc = (trx.get("payee") or {}).get("merchantClassificationCode")
    if mcc and mcc != "general":
        out["merchant_category"] = str(mcc)
    for k_ext, k_in in _EXT_KEYS.items():
        if k_ext in ext and ext[k_ext] not in (None, ""):
            v = ext[k_ext]
            try:
                out[k_in] = float(v) if k_in in _NUMERIC_EXT else int(float(v)) if k_in in _INT_EXT else str(v)
            except ValueError:
                pass
    if not out["ts"]:
        raise AdapterError("event has no timestamp (metadata.event.createdAt or expiration)")
    return out
