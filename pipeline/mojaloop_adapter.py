"""Thin adapter between Mojaloop FSPIOP-style transfer events and the engine's internal transaction schema.

Mojaloop's ml-api-adapter publishes every ``POST /transfers`` as a *transfer prepare* event on the
``topic-transfer-prepare`` Kafka topic::

    {"id", "from", "to", "type", "content": {"headers": {...}, "payload": <POST /transfers body>},
     "metadata": {"event": {"type": "transfer", "action": "prepare", "createdAt": ISO-8601}}}

On a live switch ``content.payload`` is a data URI
(``data:application/vnd.interoperability.transfers+json;version=1.1;base64,<body>``); the simulator sends the
body as a JSON object. ``from_fspiop`` accepts both, and also a bare ``POST /transfers`` body.

The payer/payee party details and transactionType live inside the ILP packet's ``data`` field (FSPIOP
``Transaction`` object). ``pipeline.ilp`` decodes real OER packets (ILPv4 Prepare, ILPv1 Payment) as well as
the simulator's mocked packet (base64url JSON Transaction, no envelope). Risk signals that FSPIOP does not
carry natively (device, IP, geo, channel, device-binding time) travel in the standard ``extensionList``; on a
live switch they are empty unless the FSPs populate them.

``pipeline.mojaloop_consumer`` feeds a live switch's Kafka topic through this adapter onto the scoring stream.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime
from email.utils import formatdate
from typing import Any
from urllib.parse import unquote_to_bytes

import orjson

from core.config import get_settings
from pipeline.ilp import IlpError, b64_any, decode_ilp_packet

_EXT_KEYS = {
    "deviceId": "device_id",
    "ipAddress": "ip",
    "geoLat": "lat",
    "geoLon": "lon",
    "channel": "channel",
    "deviceBoundAt": "device_bound_at",
    "merchantCategory": "merchant_category",
}
_DEVICE_EXT = {"deviceId", "ipAddress", "geoLat", "geoLon", "channel", "deviceBoundAt"}
EVENT_TIME_EXT = "eventTime"  # original event time (ISO-8601) of a replayed transfer; see TRUST_FSP_EVENT_TIME
_NUMERIC_EXT = {"lat", "lon"}
_INT_EXT = {"device_bound_at"}
_PARTY_TYPE = {"CONSUMER": "MSISDN", "MERCHANT": "BUSINESS", "BUSINESS": "BUSINESS"}


class AdapterError(ValueError):
    pass


def _b64e(obj: Any) -> str:
    return base64.urlsafe_b64encode(orjson.dumps(obj)).decode().rstrip("=")


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


def fspiop_amount(amount: float) -> str:
    """FSPIOP Amount string: at most 4 decimals and no trailing zeros ("500", "12.5"); switches reject "500.00"."""
    return f"{round(float(amount), 4):.4f}".rstrip("0").rstrip(".")


def fspiop_transaction(tx: dict) -> dict:
    """The FSPIOP ``Transaction`` object (payer, payee, amount, type) that travels in the ILP packet's data."""
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
    amount = {"amount": fspiop_amount(tx["amount"]), "currency": tx.get("currency", "INR")}
    return {
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


def fspiop_extensions(tx: dict, device_signals: bool = True) -> list[dict]:
    """Risk signals FSPIOP does not carry natively, as ``extensionList`` entries (device ones optional)."""
    ext = []
    for k_out, k_in in _EXT_KEYS.items():
        if not device_signals and k_out in _DEVICE_EXT:
            continue
        v = tx.get(k_in)
        if v is not None and v != "" and v == v:  # v == v drops NaN
            ext.append({"key": k_out, "value": str(v)})
    return ext


def to_fspiop(tx: dict) -> dict:
    """Render an internal transaction dict as a Mojaloop transfer-prepare event message."""
    transaction = fspiop_transaction(tx)
    amount = transaction["amount"]
    ext = fspiop_extensions(tx)
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


def decode_payload(payload: Any) -> dict:
    """Return the ``POST /transfers`` body of an event as a dict (it may arrive as a data URI string)."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")
    if not isinstance(payload, str) or not payload:
        raise AdapterError("event has no payload")
    if payload.startswith("data:"):
        header, sep, data = payload.partition(",")
        if not sep:
            raise AdapterError("malformed data URI payload")
        try:
            raw = b64_any(data) if header.endswith(";base64") else unquote_to_bytes(data)
        except IlpError as e:
            raise AdapterError(f"undecodable data URI payload: {e}") from e
    else:
        raw = payload.encode()
    try:
        body = orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise AdapterError(f"payload is not JSON: {e}") from e
    if not isinstance(body, dict):
        raise AdapterError("payload is not a JSON object")
    return body


def is_transfer_prepare(msg: dict) -> bool:
    """True for transfer prepare events; False for other actions that may share a topic (bulk, fx, ...).

    ml-api-adapter tags these events ``type: "prepare"`` (older releases and the simulator use ``"transfer"``).
    """
    event = (msg.get("metadata") or {}).get("event") or {}
    if not event:
        return "content" in msg or "transferId" in msg
    return event.get("type", "prepare") in ("prepare", "transfer") and event.get("action", "prepare") == "prepare"


def normalize_message(msg: dict) -> dict:
    """Copy of a transfer-prepare event with ``content.payload`` decoded to a dict (the simulator's shape)."""
    if "content" not in msg:
        return msg
    content = dict(msg.get("content") or {})
    content["payload"] = decode_payload(content.get("payload"))
    return {**msg, "content": content}


def from_fspiop(msg: dict) -> dict:
    """Map a Mojaloop transfer-prepare message (or bare POST /transfers body) to the internal schema.

    Missing optional fields are filled with neutral defaults; missing mandatory fields raise AdapterError.
    """
    if "content" in msg:
        payload = decode_payload((msg.get("content") or {}).get("payload"))
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
            trx = decode_ilp_packet(str(payload["ilpPacket"])).transaction
        except IlpError:  # unreadable packet: fall back to header-level data (FSP-level identities)
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
    if ext.get(EVENT_TIME_EXT) and get_settings().trust_fsp_event_time:
        try:
            out["ts"] = _parse_iso(str(ext[EVENT_TIME_EXT]))
        except ValueError as e:
            raise AdapterError(f"invalid {EVENT_TIME_EXT} extension: {e}") from e
    if not out["ts"]:
        raise AdapterError("event has no timestamp (metadata.event.createdAt or expiration)")
    return out
