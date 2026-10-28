"""Central configuration loaded from environment variables / .env (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- infrastructure -------------------------------------------------
    redis_url: str = "redis://127.0.0.1:6379/0"
    database_url: str = "postgresql://fraud:fraud@localhost:5432/fraud"

    # --- streaming --------------------------------------------------------
    stream_prefix: str = "fsp:transfers"
    stream_partitions: int = 2  # payer-hash partitions; one scoring worker per partition
    stream_maxlen: int = 50_000  # per partition; processed events are trimmed (bounded Redis memory)
    consumer_group: str = "scorers"

    # --- live Mojaloop switch (pipeline.mojaloop_consumer) ----------------
    kafka_bootstrap_servers: str = "localhost:9092"
    mojaloop_prepare_topic: str = "topic-transfer-prepare"
    mojaloop_consumer_group: str = "fraud-engine"  # own group: reads alongside central-ledger, never instead of it
    mojaloop_offset_reset: str = "latest"  # where a brand-new group starts: "latest" | "earliest"
    # Use the FSP-supplied ``eventTime`` extension as the event time instead of the switch's receive time.
    # Only for replaying historical traffic through a switch (simulator/mojaloop_sender.py); an FSP could
    # otherwise shift its own time features.
    trust_fsp_event_time: bool = False

    # --- paths ------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    artifacts_dir: Path = ROOT / "models" / "artifacts"

    # --- auth -------------------------------------------------------------
    # comma separated "operator_id:api_key" pairs; no default secrets (see .env.example)
    operator_api_keys: str = ""

    # --- scoring / latency budget ----------------------------------------
    sequence_timeout_ms: float = 25.0
    graph_stale_after_s: float = 300.0
    model_poll_interval_s: float = 5.0
    shap_sync_for_allow: bool = False  # ALLOW explanations are computed asynchronously

    # --- graph worker -----------------------------------------------------
    graph_window_hours: float = 24.0
    graph_interval_s: float = 20.0
    graph_backend: str = "networkx"  # "networkx" | "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # --- feedback / retraining -------------------------------------------
    retrain_interval_hours: float = 24.0
    retrain_min_feedback: int = 50
    retrain_tolerance: float = 0.01  # max allowed drop in PR-AUC / recall@1%FPR
    settlement_cutoff_hours: list[int] = Field(default_factory=lambda: [0, 6, 12, 18])

    # --- misc -------------------------------------------------------------
    log_level: str = "INFO"
    api_cors_origins: str = "http://localhost:5173,http://localhost:8080"

    def api_keys(self) -> dict[str, str]:
        """Return mapping api_key -> operator_id."""
        out: dict[str, str] = {}
        for pair in self.operator_api_keys.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            op, key = pair.split(":", 1)
            out[key.strip()] = op.strip()
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
