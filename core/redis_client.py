"""Shared Redis client factory."""

from __future__ import annotations

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
