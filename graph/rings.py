"""Turns detected rings into node-level scores consumed by the hot scoring path via Redis lookups."""

from __future__ import annotations

import pandas as pd

from graph.detectors import Ring, detect_rings, fraud_neighbor_ratio

RING_TYPE_BITS = {"cycle": 1, "dense_subgraph": 2, "mule_network": 4, "shared_device": 8, "reciprocal_burst": 16}
FLAG_SCORE = 0.5  # ring members at or above this score count as "flagged" neighbours


class RingTracker:
    """Keeps ring ids stable across runs by matching new rings to previous ones (member Jaccard >= 0.5)."""

    def __init__(self):
        self.known: dict[str, Ring] = {}

    def assign(self, rings: list[Ring]) -> list[Ring]:
        for r in rings:
            best, best_j = None, 0.0
            rs = r.key_members()
            for rid, old in self.known.items():
                if old.ring_type != r.ring_type:
                    continue
                os_ = old.key_members()
                j = len(rs & os_) / max(1, len(rs | os_))
                if j > best_j:
                    best, best_j = rid, j
            if best is not None and best_j >= 0.5:
                old = self.known[best]
                r.ring_id = best
                r.first_seen = min(r.first_seen, old.first_seen)
            self.known[r.ring_id] = r
        return rings


def node_scores(df: pd.DataFrame, rings: list[Ring], reported: set[str], now_ts: int) -> dict[str, dict]:
    """node key ("a:<acct>" / "d:<device>") -> fields written to Redis hash g:<node>."""
    out: dict[str, dict] = {}

    def upd(node: str, r: Ring, cyc: int):
        cur = out.get(node)
        if cur is None:
            cur = {"rs": 0.0, "rid": "", "rt": 0, "sz": 0, "cyc": 0, "fnr": 0.0, "ts": now_ts}
            out[node] = cur
        cur["rt"] = int(cur["rt"]) | RING_TYPE_BITS[r.ring_type]
        if r.score > cur["rs"]:
            cur["rs"], cur["rid"] = round(r.score, 4), r.ring_id
        cur["sz"] = max(int(cur["sz"]), len(r.members))
        cur["cyc"] = max(int(cur["cyc"]), cyc)

    for r in rings:
        cyc = len(r.members) if r.ring_type == "cycle" else 0
        for a in r.members:
            upd("a:" + a, r, cyc)
        for d in r.devices:
            if r.ring_type in ("shared_device", "dense_subgraph"):
                upd("d:" + d, r, 0)
    flagged = set(reported) | {k[2:] for k, v in out.items() if k.startswith("a:") and v["rs"] >= FLAG_SCORE}
    for acct, ratio in fraud_neighbor_ratio(df, flagged).items():
        node = "a:" + acct
        if node not in out:
            out[node] = {"rs": 0.0, "rid": "", "rt": 0, "sz": 0, "cyc": 0, "fnr": 0.0, "ts": now_ts}
        out[node]["fnr"] = round(ratio, 4)
    return out


def graph_write_plan(
    scores: dict[str, dict], previous_nodes: set[str], now_ts: int, n_rings: int, run_ms: float
) -> list[tuple]:
    """Replace the published graph scores atomically enough for a lookup cache (stale nodes deleted)."""
    plan: list[tuple] = []
    stale = previous_nodes - set(scores)
    if stale:
        stale_l = list(stale)
        for i in range(0, len(stale_l), 500):
            plan.append(("delete", *[f"g:{n}" for n in stale_l[i : i + 500]]))
    for node, fields in scores.items():
        plan.append(("delete", f"g:{node}"))
        plan.append(("hset", f"g:{node}", None, None, {k: str(v) for k, v in fields.items()}))
    plan.append(("delete", "g:nodes"))
    if scores:
        nodes = list(scores)
        for i in range(0, len(nodes), 1000):
            plan.append(("sadd", "g:nodes", *nodes[i : i + 1000]))
    plan.append(
        (
            "hset",
            "g:meta",
            None,
            None,
            {"ts": str(now_ts), "n_rings": str(n_rings), "n_nodes": str(len(scores)), "run_ms": f"{run_ms:.1f}"},
        )
    )
    return plan


def run_graph_cycle(
    df: pd.DataFrame,
    reported: set[str],
    now_ts: int,
    store,
    tracker: RingTracker | None = None,
    previous_nodes: set[str] | None = None,
) -> tuple[list[Ring], set[str]]:
    """Detect rings on a window, publish node scores to the given store. Returns rings and published nodes."""
    import time

    t = time.perf_counter()
    rings = detect_rings(df, reported, now_ts)
    if tracker is not None:
        rings = tracker.assign(rings)
    scores = node_scores(df, rings, reported, now_ts)
    prev = previous_nodes if previous_nodes is not None else set(store.execute([("smembers", "g:nodes")])[0] or ())
    store.execute(graph_write_plan(scores, prev, now_ts, len(rings), (time.perf_counter() - t) * 1000))
    return rings, set(scores)
