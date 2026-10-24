"""Operator authentication: static API keys mapped to operator ids (configured via OPERATOR_API_KEYS)."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Query, WebSocket, status

from core.config import get_settings


def _lookup(key: str | None) -> str | None:
    if not key:
        return None
    for k, op in get_settings().api_keys().items():
        if hmac.compare_digest(k, key):
            return op
    return None


def operator(x_api_key: str | None = Header(default=None), api_key: str | None = Query(default=None)) -> str:
    op = _lookup(x_api_key or api_key)
    if op is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key (X-API-Key header)")
    return op


def ws_operator(ws: WebSocket) -> str | None:
    return _lookup(ws.query_params.get("api_key") or ws.headers.get("x-api-key"))
