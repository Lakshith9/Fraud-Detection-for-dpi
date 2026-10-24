"""Process-wide API state (initialised in the FastAPI lifespan)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AppState:
    redis: object = None
    scorer: object = None
    writer: object = None
    hub: object = None
    cache: dict = field(default_factory=dict)


STATE = AppState()
