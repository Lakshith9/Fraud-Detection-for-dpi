"""Model registry: versioned artifact directories on disk + metadata/status in PostgreSQL.

Exactly one version has status 'production'. Promotion/rollback updates Postgres and publishes the new
version on Redis (key ``model:current`` + channel ``model:updates``); scoring workers poll the key and
hot-swap the bundle without downtime.

CLI:
  python -m models.registry list
  python -m models.registry promote <version>
  python -m models.registry rollback            # back to the previously promoted version
  python -m models.registry current
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import db
from core.config import get_settings

CURRENT_KEY = "model:current"
UPDATES_CHANNEL = "model:updates"


def _redis():
    import redis

    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True, socket_timeout=2)


def artifact_path(version: str) -> Path:
    return get_settings().artifacts_dir / version


def next_version() -> str:
    root = get_settings().artifacts_dir
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name[1:]) for p in root.iterdir() if p.is_dir() and p.name[1:].isdigit() and p.name[0] == "v"]
    return f"v{(max(nums) + 1) if nums else 1}"


def register(version: str, metrics: dict, data_hash: str, parent: str | None = None, notes: str = "") -> None:
    db.execute(
        """INSERT INTO model_registry (version, status, data_hash, metrics, artifact_path, parent_version, notes)
           VALUES (%s, 'candidate', %s, %s, %s, %s, %s)
           ON CONFLICT (version) DO UPDATE SET metrics = EXCLUDED.metrics, data_hash = EXCLUDED.data_hash,
                                               notes = EXCLUDED.notes""",
        (version, data_hash, db.Jsonb(metrics), str(artifact_path(version)), parent, notes),
    )


def current() -> str | None:
    row = db.fetch_one("SELECT version FROM model_registry WHERE status = 'production' LIMIT 1")
    return row["version"] if row else None


def promote(version: str, publish: bool = True) -> None:
    row = db.fetch_one("SELECT version FROM model_registry WHERE version = %s", (version,))
    if row is None:
        raise ValueError(f"unknown model version {version}")
    if not (artifact_path(version) / "bundle.json").exists():
        raise FileNotFoundError(f"artifacts for {version} missing at {artifact_path(version)}")
    with db.get_pool().connection() as conn:
        with conn.transaction():
            conn.execute("UPDATE model_registry SET status = 'archived' WHERE status = 'production'")
            conn.execute(
                "UPDATE model_registry SET status = 'production', promoted_at = now() WHERE version = %s", (version,)
            )
    if publish:
        try:
            r = _redis()
            r.set(CURRENT_KEY, version)
            r.publish(UPDATES_CHANNEL, json.dumps({"version": version}))
        except Exception:  # Redis down: workers fall back to polling Postgres via sync_current()
            pass


def reject(version: str, reason: str) -> None:
    db.execute("UPDATE model_registry SET status = 'rejected', notes = %s WHERE version = %s", (reason, version))


def rollback() -> str:
    rows = db.fetch_all(
        """SELECT version FROM model_registry WHERE promoted_at IS NOT NULL AND status <> 'production'
           ORDER BY promoted_at DESC LIMIT 1"""
    )
    if not rows:
        raise RuntimeError("no previously promoted version to roll back to")
    promote(rows[0]["version"])
    return rows[0]["version"]


def list_versions() -> list[dict]:
    return db.fetch_all(
        """SELECT version, status, created_at, promoted_at, data_hash, parent_version, notes,
                  metrics -> 'test' -> 'full' AS test_full
           FROM model_registry ORDER BY created_at DESC"""
    )


def sync_current_to_redis() -> str | None:
    v = current()
    if v:
        try:
            _redis().set(CURRENT_KEY, v)
        except Exception:
            pass
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("current")
    p = sub.add_parser("promote")
    p.add_argument("version")
    sub.add_parser("rollback")
    a = ap.parse_args(argv)
    db.init_schema()
    if a.cmd == "list":
        for r in list_versions():
            print(json.dumps(r, default=str))
    elif a.cmd == "current":
        print(current())
    elif a.cmd == "promote":
        promote(a.version)
        print(f"promoted {a.version}")
    elif a.cmd == "rollback":
        print(f"rolled back to {rollback()}")


if __name__ == "__main__":
    main()
