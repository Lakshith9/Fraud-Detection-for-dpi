"""Real-time scorer used by both the stream workers and the synchronous POST /v1/score endpoint.

Hot path per transaction (latency budget, see docs/architecture.md):
  1 Redis round trip  : idempotency check + all feature-state reads + graph lookups (pipelined)
  feature compute     : pure Python, shared with training (pipeline/features.py)
  XGBoost margin      : inplace_predict, 1 thread
  sequence model      : ONNX Runtime batch=1, guarded by a timeout -> degraded fallback
  graph               : cached node scores from the async graph worker (stale -> degraded fallback)
  meta-learner        : linear, microseconds
  SHAP                : synchronous for REVIEW/BLOCK; ALLOW explanations are computed asynchronously by the persister
  1 Redis round trip  : state updates + idempotency record + counters + alert publish (pipelined)
After the decision (not part of the measured latency): one small pipeline enqueues the scored row on the
`persist:rows` stream and records the full latency sample. PostgreSQL writes happen in a separate process
(pipeline/persister.py), so database work never competes with scoring for this process's GIL.
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from pathlib import Path

import numpy as np
import orjson
import redis

from core.config import get_settings
from core.logging import get_logger
from models.ensemble import ModelBundle
from models.explain import explain
from models.registry import CURRENT_KEY
from pipeline.features import GRAPH_FEATURES, TABULAR_FEATURES, FeatureEngine
from pipeline.state_store import InMemoryStore, RedisStore

log = get_logger("scorer")
ALERT_CHANNEL = "alerts:live"
LAT_KEY = "metrics:latency"
E2E_KEY = "metrics:e2e"
COUNT_KEY = "metrics:counts"
PERSIST_STREAM = "persist:rows"
PERSIST_MAXLEN = 200_000  # bounded buffer (~800 MB worst case) while PostgreSQL is unavailable
METRIC_WINDOW = 100_000  # latency samples kept for percentile reporting


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


class ModelManager:
    """Holds the production bundle and hot-swaps it when the registry pointer changes (no downtime)."""

    def __init__(self, r: redis.Redis | None, poll_s: float | None = None, version: str | None = None):
        self.s = get_settings()
        self.r = r
        self.bundle: ModelBundle | None = None
        self.poll_s = poll_s if poll_s is not None else self.s.model_poll_interval_s
        self._stop = threading.Event()
        self.load(version or self._target_version())
        if self.poll_s > 0:
            threading.Thread(target=self._poll, name="model-poll", daemon=True).start()

    def _target_version(self) -> str:
        v = None
        try:
            v = self.r.get(CURRENT_KEY) if self.r is not None else None
        except redis.RedisError:
            v = None
        if not v:
            try:
                from models import registry

                v = registry.current()
            except Exception:
                v = None
        if not v:
            raise RuntimeError("no production model: run training + bootstrap first (see README)")
        return v

    def load(self, version: str) -> None:
        path = self.s.artifacts_dir / version
        t = time.perf_counter()
        b = ModelBundle(Path(path))
        self.bundle = b  # atomic reference swap
        log.info("model loaded", version=version, ms=round((time.perf_counter() - t) * 1000, 1))

    def _poll(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                v = self._target_version()
                if self.bundle is None or v != self.bundle.version:
                    self.load(v)
            except Exception as e:
                log.error("model hot-reload failed; keeping current version", error=str(e)[:200])

    def stop(self):
        self._stop.set()


class Scorer:
    def __init__(
        self,
        r: redis.Redis | None = None,
        writer=None,
        models: ModelManager | None = None,
        source: str = "stream",
    ):
        self.s = get_settings()
        self.r = r
        self.store = RedisStore(r) if r is not None else None
        self.engine = FeatureEngine(self.s.settlement_cutoff_hours)
        self.models = models or ModelManager(r)
        self.writer = writer
        self.source = source
        self.seq_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="seq")
        self.seq_timeout = self.s.sequence_timeout_ms / 1000
        self._cold = InMemoryStore()
        self.redis_ok = True

    # ------------------------------------------------------------------ public
    def score(self, tx: dict, sent_at_ms: float | None = None) -> dict:
        t0 = time.perf_counter()
        bundle = self.models.bundle
        tid = tx["transfer_id"]
        plan = self.engine.read_plan(tx)
        plan_extra = [("get", f"idem:{tid}"), ("hgetall", "g:meta")]
        degraded: list[str] = []
        try:
            res = self.store.execute(plan + plan_extra)
            self.redis_ok = True
        except (redis.RedisError, OSError, AttributeError):
            # online store unavailable: score with cold-start state (stateless features only)
            res = self._cold.execute(plan) + [None, {}]
            degraded.append("feature_store_unavailable")
            self.redis_ok = False
        t_fetch = time.perf_counter()
        cached = res[-2]
        if cached:
            out = orjson.loads(cached)
            out["duplicate"] = True
            return out
        fv = self.engine.compute(tx, res[:-2])
        tab = np.asarray(fv.tabular, dtype=np.float32)
        graph = np.asarray(fv.graph, dtype=np.float32)
        t_feat = time.perf_counter()
        margin = bundle.xgb_margin(tab)
        t_xgb = time.perf_counter()

        seq_logit = None
        if bundle.seq is not None:
            try:
                seq_logit = self.seq_pool.submit(bundle.seq_logit, fv.seq).result(timeout=self.seq_timeout)
            except FutTimeout:
                degraded.append("sequence_timeout")
            except Exception as e:  # model failure must not block scoring
                degraded.append("sequence_error")
                log.error("sequence model failed", error=str(e)[:200])
        else:
            degraded.append("sequence_unavailable")
        t_seq = time.perf_counter()

        gmeta = res[-1] or {}
        graph_ok = bool(gmeta) and (time.time() - float(gmeta.get("wall", 0) or 0)) <= self.s.graph_stale_after_s
        if not graph_ok:
            degraded.append("graph_stale")
        combo = {(True, True): "full", (True, False): "xgb_seq", (False, True): "xgb_graph", (False, False): "xgb"}[
            (seq_logit is not None, graph_ok)
        ]
        p, z, decision, meta_x = bundle.combine(combo, margin, seq_logit, graph)
        t_meta = time.perf_counter()

        typology = {}
        explanation = None
        if decision != "ALLOW" or self.s.shap_sync_for_allow:
            explanation = explain(bundle, combo, tab, graph, meta_x, fv.context, seq_logit)
            typology = bundle.typology_of(tab, graph)
        t_shap = time.perf_counter()

        result = {
            "transfer_id": tid,
            "risk_score": round(p, 6),
            "decision": decision,
            "top_reasons": explanation["top_reasons"] if explanation else [],
            "explanation_pending": explanation is None,
            "model_version": bundle.version,
            "combo": combo,
            "degraded": bool(degraded),
            "degraded_reasons": degraded,
            "predicted_type": max(typology, key=typology.get) if typology else None,
        }
        now_ms = time.time() * 1000
        latency_pre_write = (t_shap - t0) * 1000
        write = self.engine.write_plan(tx)
        write.append(
            ("set", f"idem:{tid}", orjson.dumps({**result, "latency_ms": round(latency_pre_write, 3)}), 86_400)
        )
        write.append(("hincrby", COUNT_KEY, decision, 1))
        write.append(("hincrby", COUNT_KEY, "total", 1))
        if degraded:
            write.append(("hincrby", COUNT_KEY, "degraded", 1))
        write.append(("incr", f"metrics:tps:{int(now_ms // 1000)}"))
        write.append(("expire", f"metrics:tps:{int(now_ms // 1000)}", 300))
        alert = None
        if decision != "ALLOW":
            alert = {
                "transfer_id": tid,
                "ts": int(tx["ts"]),
                "risk_score": result["risk_score"],
                "decision": decision,
                "predicted_type": result["predicted_type"],
                "payer_id": tx["payer_id"],
                "payee_id": tx["payee_id"],
                "payer_fsp": tx.get("payer_fsp"),
                "payee_fsp": tx.get("payee_fsp"),
                "amount": float(tx["amount"]),
                "top_reasons": result["top_reasons"],
                "model_version": bundle.version,
                "degraded": bool(degraded),
            }
            write.append(("publish", ALERT_CHANNEL, orjson.dumps(alert, default=str)))
        if self.redis_ok:
            try:
                self.store.execute(write)
            except (redis.RedisError, OSError) as e:
                degraded.append("state_update_failed")
                log.error("state update failed", error=str(e)[:200])
        t_end = time.perf_counter()
        latency_ms = (t_end - t0) * 1000
        result["latency_ms"] = round(latency_ms, 3)
        result["timings_ms"] = {
            "feature_fetch": round((t_fetch - t0) * 1000, 3),
            "feature_compute": round((t_feat - t_fetch) * 1000, 3),
            "xgboost": round((t_xgb - t_feat) * 1000, 3),
            "sequence": round((t_seq - t_xgb) * 1000, 3),
            "graph_meta": round((t_meta - t_seq) * 1000, 3),
            "shap": round((t_shap - t_meta) * 1000, 3),
            "state_write": round((t_end - t_shap) * 1000, 3),
        }
        row = self._tx_row(tx, result, fv, margin, seq_logit, graph, p, explanation, typology)
        self._post(row, alert, latency_ms, sent_at_ms)
        return result

    def _post(self, row: dict, alert: dict | None, latency_ms: float, sent_at_ms: float | None) -> None:
        """Enqueue persistence + record latency samples (after the decision; not part of the scoring latency)."""
        plan: list[tuple] = [
            ("xadd", PERSIST_STREAM, {"k": "tx", "r": orjson.dumps(row, default=str)}, "*", PERSIST_MAXLEN, True),
            ("lpush", LAT_KEY, f"{latency_ms:.3f}"),
            ("ltrim", LAT_KEY, 0, METRIC_WINDOW - 1),
        ]
        if alert is not None:
            plan.append(
                (
                    "xadd",
                    PERSIST_STREAM,
                    {"k": "alert", "r": orjson.dumps(alert, default=str)},
                    "*",
                    PERSIST_MAXLEN,
                    True,
                )
            )
        if sent_at_ms:
            plan.append(("lpush", E2E_KEY, f"{time.time() * 1000 - sent_at_ms:.3f}"))
            plan.append(("ltrim", E2E_KEY, 0, METRIC_WINDOW - 1))
        try:
            if not self.redis_ok:
                raise redis.ConnectionError("redis unavailable")
            self.store.execute(plan)
        except (redis.RedisError, OSError):
            # Redis down: fall back to writing PostgreSQL directly from this process (buffered + disk spool)
            if self.writer is not None:
                self.writer.submit("tx", row)
                if alert is not None:
                    self.writer.submit("alert", alert)

    def apply_report(self, account_id: str) -> None:
        self.store.execute(FeatureEngine.report_plan(account_id) + [("sadd", "reported", account_id)])

    # ------------------------------------------------------------------ internals
    def _tx_row(self, tx, result, fv, margin, seq_logit, graph, p, explanation, typology) -> dict:
        return {
            "transfer_id": tx["transfer_id"],
            "ts": int(tx["ts"]),
            "payer_id": tx["payer_id"],
            "payee_id": tx["payee_id"],
            "payer_fsp": tx.get("payer_fsp"),
            "payee_fsp": tx.get("payee_fsp"),
            "amount": float(tx["amount"]),
            "currency": tx.get("currency"),
            "scenario": tx.get("scenario"),
            "channel": tx.get("channel"),
            "device_id": tx.get("device_id"),
            "ip": tx.get("ip"),
            "payload": {k: _clean(v) for k, v in tx.items()},
            "risk_score": float(p),
            "decision": result["decision"],
            "model_version": result["model_version"],
            "combo": result["combo"],
            "degraded": result["degraded"],
            "latency_ms": result["latency_ms"],
            "components": {
                "context": {k: _clean(v) for k, v in fv.context.items() if not isinstance(v, dict)},
                "xgb_margin": float(margin),
                "seq_logit": None if seq_logit is None else float(seq_logit),
                "combo": result["combo"],
                "degraded_reasons": result["degraded_reasons"],
                "timings_ms": result["timings_ms"],
            },
            "features": {k: _clean(float(v)) for k, v in zip(TABULAR_FEATURES, fv.tabular)},
            "graph": {
                **{k: float(v) for k, v in zip(GRAPH_FEATURES, graph)},
                "context": {k: fv.context.get(k) for k in ("payer_ring", "payee_ring", "device_ring")},
            },
            "explanation": {
                **explanation,
                "context": {k: _clean(v) for k, v in fv.context.items() if not isinstance(v, dict)},
            }
            if explanation
            else None,
            "typology": typology or None,
            "source": self.source,
        }
