"""Shared Redis client factory."""

from __future__ import annotations

import zlib

import redis

from core.config import get_settings


def get_redis(timeout: float = 2.0) -> redis.Redis:
    return redis.Redis.from_url(
        get_settings().redis_url,
        decode_responses=True,
        socket_timeout=timeout,
        socket_connect_timeout=timeout,
        health_check_interval=30,
    )


def stream_key(partition: int) -> str:
    return f"{get_settings().stream_prefix}:{partition}"


def partition_of(payer_id: str, n: int) -> int:
    """Stream partition for a payer (stable hash): keeps each payer's events ordered on one worker."""
    return zlib.crc32(payer_id.encode()) % n
