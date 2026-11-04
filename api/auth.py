"""Operator authentication: static API keys mapped to operator ids (configured via OPERATOR_API_KEYS).

Keys travel only in the ``X-API-Key`` header (REST) or in the first WebSocket message, never in a URL: query
strings end up in uvicorn/nginx access logs, browser history and proxies.
"""

from __future__ import annotations

import hmac

import orjson
from fastapi import Header, HTTPException, WebSocket, status

from core.config import get_settings


def _lookup(key: str | None) -> str | None:
    if not key or not isinstance(key, str):
        return None
    for k, op in get_settings().api_keys().items():
        if hmac.compare_digest(k.encode(), key.encode()):
            return op
    return None


def operator(x_api_key: str | None = Header(default=None)) -> str:
    op = _lookup(x_api_key)
    if op is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key (X-API-Key header)")
    return op


def ws_operator(ws: WebSocket) -> str | None:
    """Operator from the handshake's X-API-Key header (non-browser clients)."""
    return _lookup(ws.headers.get("x-api-key"))


def message_operator(text: str) -> str | None:
    """Operator from a ``{"type": "auth", "api_key": ...}`` message (browsers cannot set WebSocket headers)."""
    try:
        msg = orjson.loads(text)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(msg, dict) or msg.get("type") != "auth":
        return None
    return _lookup(msg.get("api_key"))
