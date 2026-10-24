"""Graph-based fraud ring detection over a sliding event-time window (NetworkX).

Nodes: accounts ("a:<id>"), devices ("d:<id>"), IPs ("i:<ip>"). Edges: transfers (directed, timestamped,
amount) and shared-attribute links (account-device, account-ip).

Detectors
  (a) temporal_cycles      circular flows A->B->C->...->A with strictly increasing timestamps inside a time bound
  (b) dense_subgraphs      k-core + Louvain communities on the account/device/ip graph, scored by density,
                           mutual flows and shared attributes
  (c) mule_patterns        fan-out -> rapid pass-through -> fan-in layering structures
  (d) shared_attribute     devices / IPs used by an unusual number of accounts, expanded to connected clusters
  (e) reciprocal_bursts    short bursts of transfers that are sent back (reversals / round-tripping) between a
                           small group of accounts - settlement-window manipulation signature

No labels are used. Fraud *reports* that arrived before the window end (registry feed / operator-confirmed)
are used as evidence only, exactly as they would be in production.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

MIN = 60_000
HOUR = 3_600_000


@dataclass
class Ring:
    ring_type: str  # cycle | dense_subgraph | mule_network | shared_device
    members: list[str]  # account ids
    devices: list[str]
    score: float
    evidence: dict
    first_seen: int
    last_seen: int
    ring_id: str = ""
    edges: list[tuple] = field(default_factory=list)  # (src, dst, amount, ts) for visualisation

    def key_members(self) -> frozenset:
        return frozenset(self.members)

    def to_dict(self) -> dict:
        return {
            "ring_id": self.ring_id,
            "ring_type": self.ring_type,
            "members": self.members,
            "devices": self.devices,
            "ring_score": round(float(self.score), 4),
            "evidence": self.evidence,
            "first_seen": int(self.first_seen),
            "last_seen": int(self.last_seen),
            "edges": [list(e) for e in self.edges[:400]],
        }


@dataclass
class DetectorConfig:
    cycle_min_len: int = 3
    cycle_max_len: int = 6
    cycle_max_span_ms: int = 12 * HOUR
    cycle_max_starts: int = 20_000
    kcore_k: int = 3
    dense_min_accounts: int = 4
    dense_max_accounts: int = 60
    attr_max_accounts: int = 30  # ignore attribute nodes shared by more accounts (CGNAT, public wifi)
    device_min_accounts: int = 3
    fan_min_degree: int = 4
    pass_through_window_ms: int = 3 * HOUR
    pass_through_min_ratio: float = 0.8
    min_score: float = 0.35


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


class WindowGraph:
    """Pre-indexed transfer window."""

    def __init__(self, df: pd.DataFrame, reported: set[str] | None = None, now_ts: int | None = None):
        df = df[df["payer_id"] != df["payee_id"]]
        self.df = df
        self.reported = reported or set()
        self.src = df["payer_id"].to_numpy()
        self.dst = df["payee_id"].to_numpy()
        self.ts = df["ts"].to_numpy(dtype=np.int64)
        self.amt = df["amount"].to_numpy(dtype=np.float64)
        self.dev = df["device_id"].fillna("").to_numpy()
        self.ip = df["ip"].fillna("").to_numpy()
        self.now = int(now_ts if now_ts is not None else (self.ts.max() if len(self.ts) else 0))
        self.out_edges: dict[str, list[int]] = defaultdict(list)
        self.in_edges: dict[str, list[int]] = defaultdict(list)
        for i in range(len(self.src)):
            self.out_edges[self.src[i]].append(i)
            self.in_edges[self.dst[i]].append(i)
        self.first_seen: dict[str, int] = {}
        # account first activity inside the window (newness proxy)
        for i in range(len(self.src)):
            for a in (self.src[i], self.dst[i]):
                if a not in self.first_seen:
                    self.first_seen[a] = int(self.ts[i])
        # attribute -> accounts (payer side: device/ip belong to the initiating account)
        self.dev_accounts: dict[str, set[str]] = defaultdict(set)
        self.ip_accounts: dict[str, set[str]] = defaultdict(set)
        self.acct_devices: dict[str, set[str]] = defaultdict(set)
        for i in range(len(self.src)):
            if self.dev[i]:
                self.dev_accounts[self.dev[i]].add(self.src[i])
                self.acct_devices[self.src[i]].add(self.dev[i])
            if self.ip[i]:
                self.ip_accounts[self.ip[i]].add(self.src[i])


# ---------------------------------------------------------------------------------------------- (a) cycles
def temporal_cycles(W: WindowGraph, cfg: DetectorConfig) -> list[Ring]:
    if not len(W.src):
        return []
    g = nx.DiGraph()
    g.add_edges_from(zip(W.src, W.dst))
    scc_nodes = set()
    for comp in nx.strongly_connected_components(g):
        if len(comp) >= cfg.cycle_min_len:
            scc_nodes |= comp
    if not scc_nodes:
        return []
    idx = [i for i in range(len(W.src)) if W.src[i] in scc_nodes and W.dst[i] in scc_nodes]
    idx.sort(key=lambda i: W.ts[i])
    out_by: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        out_by[W.src[i]].append(i)
    found: dict[tuple, list[int]] = {}

    def dfs(start: str, path_nodes: list[str], path_edges: list[int], t0: int):
        last = path_edges[-1]
        node = W.dst[last]
        for e in out_by.get(node, ()):
            te = W.ts[e]
            if te <= W.ts[last]:
                continue
            if te - t0 > cfg.cycle_max_span_ms:
                break
            nxt = W.dst[e]
            if nxt == start and len(path_nodes) >= cfg.cycle_min_len:
                cyc = path_nodes
                k = cyc.index(min(cyc))
                canon = tuple(cyc[k:] + cyc[:k])
                if canon not in found:
                    found[canon] = path_edges + [e]
            elif nxt not in path_nodes and len(path_nodes) < cfg.cycle_max_len:
                dfs(start, path_nodes + [nxt], path_edges + [e], t0)

    for n, i in enumerate(idx):
        if n >= cfg.cycle_max_starts:
            break
        dfs(W.src[i], [W.src[i], W.dst[i]], [i], int(W.ts[i]))

    rings = []
    for canon, edges in found.items():
        amts = W.amt[edges]
        span = int(W.ts[edges[-1]] - W.ts[edges[0]])
        sim = float(amts.min() / max(amts.max(), 1.0))
        speed = 1.0 - span / cfg.cycle_max_span_ms
        size = _clip01(float(amts.mean()) / 50_000)
        new_share = np.mean([1.0 if W.first_seen.get(a, 0) > W.now - 24 * HOUR else 0.0 for a in canon])
        rep = sum(1 for a in canon if a in W.reported)
        score = _clip01(0.35 * sim**2 + 0.25 * speed + 0.2 * size + 0.1 * new_share + 0.25 * min(1, rep))
        rings.append(
            Ring(
                "cycle",
                list(canon),
                sorted({W.dev[e] for e in edges if W.dev[e]}),
                score,
                {
                    "cycle_path": list(canon) + [canon[0]],
                    "cycle_length": len(canon),
                    "span_minutes": round(span / MIN, 1),
                    "total_amount": round(float(amts.sum()), 2),
                    "amount_similarity": round(sim, 3),
                    "reported_members": rep,
                },
                int(W.ts[edges[0]]),
                int(W.ts[edges[-1]]),
                edges=[(W.src[e], W.dst[e], float(W.amt[e]), int(W.ts[e])) for e in edges],
            )
        )
    return _merge_overlapping(rings, "cycle")


# ---------------------------------------------------------------------------------------------- (b) dense
def dense_subgraphs(W: WindowGraph, cfg: DetectorConfig, seed: int = 7) -> list[Ring]:
    g = nx.Graph()
    pair_count: dict[tuple, int] = defaultdict(int)
    for i in range(len(W.src)):
        a, b = W.src[i], W.dst[i]
        pair_count[(a, b)] += 1
        g.add_edge("a:" + a, "a:" + b)
    for d, accts in W.dev_accounts.items():
        if 2 <= len(accts) <= cfg.attr_max_accounts:
            for a in accts:
                g.add_edge("d:" + d, "a:" + a)
    for ip, accts in W.ip_accounts.items():
        if 2 <= len(accts) <= cfg.attr_max_accounts:
            for a in accts:
                g.add_edge("i:" + ip, "a:" + a)
    if g.number_of_nodes() == 0:
        return []
    g.remove_edges_from(nx.selfloop_edges(g))
    core = nx.k_core(g, cfg.kcore_k)
    rings = []
    for comp in nx.connected_components(core):
        sub = core.subgraph(comp)
        accts = [n for n in comp if n.startswith("a:")]
        if len(accts) < cfg.dense_min_accounts:
            continue
        if len(accts) > cfg.dense_max_accounts:
            parts = nx.community.louvain_communities(sub, seed=seed, resolution=1.0)
        else:
            parts = [comp]
        for part in parts:
            acc = sorted(n[2:] for n in part if n.startswith("a:"))
            if not (cfg.dense_min_accounts <= len(acc) <= cfg.dense_max_accounts):
                continue
            accset = set(acc)
            devs = sorted(n[2:] for n in part if n.startswith("d:"))
            internal = [(a, b) for (a, b) in pair_count if a in accset and b in accset]
            if not internal:
                continue
            n_int = len(internal)
            density = n_int / (len(acc) * (len(acc) - 1) / 2)
            mutual = sum(1 for (a, b) in internal if (b, a) in pair_count) / n_int
            tx_count = sum(pair_count[p] for p in internal)
            idxs = [i for i in range(len(W.src)) if W.src[i] in accset and W.dst[i] in accset]
            total = float(W.amt[idxs].sum()) if idxs else 0.0
            shared_dev_accts = len({a for d in devs for a in W.dev_accounts.get(d, ()) if a in accset})
            new_share = np.mean([1.0 if W.first_seen.get(a, 0) > W.now - 24 * HOUR else 0.0 for a in acc])
            rep = sum(1 for a in acc if a in W.reported)
            score = _clip01(
                0.30 * min(1.0, density * 2.5)
                + 0.25 * mutual
                + 0.20 * min(1.0, shared_dev_accts / len(acc) * 1.5)
                + 0.10 * min(1.0, tx_count / (3 * len(acc)))
                + 0.05 * new_share
                + 0.25 * min(1, rep)
            )
            rings.append(
                Ring(
                    "dense_subgraph",
                    acc,
                    devs,
                    score,
                    {
                        "density": round(density, 3),
                        "mutual_share": round(mutual, 3),
                        "internal_tx": tx_count,
                        "accounts": len(acc),
                        "shared_devices": len(devs),
                        "accounts_on_shared_devices": shared_dev_accts,
                        "total_amount": round(total, 2),
                        "reported_members": rep,
                    },
                    int(W.ts[idxs].min()) if idxs else W.now,
                    int(W.ts[idxs].max()) if idxs else W.now,
                    edges=[(W.src[i], W.dst[i], float(W.amt[i]), int(W.ts[i])) for i in idxs[:400]],
                )
            )
    return rings


# ---------------------------------------------------------------------------------------------- (c) mules
def mule_patterns(W: WindowGraph, cfg: DetectorConfig) -> list[Ring]:
    """Fan-out from a source to several accounts that rapidly pass funds through to common collectors."""
    rings = []
    # pass-through accounts: money out within window after money in, ratio >= threshold
    pt: dict[str, tuple[float, list[int], list[int]]] = {}
    for a, ins in W.in_edges.items():
        outs = W.out_edges.get(a)
        if not outs:
            continue
        t_in = min(W.ts[i] for i in ins)
        in_amt = float(sum(W.amt[i] for i in ins))
        outs_after = [o for o in outs if t_in <= W.ts[o] <= t_in + cfg.pass_through_window_ms]
        if not outs_after or in_amt <= 0:
            continue
        ratio = float(sum(W.amt[o] for o in outs_after)) / in_amt
        if ratio >= cfg.pass_through_min_ratio and ratio <= 1.2:
            pt[a] = (ratio, ins, outs_after)
    if not pt:
        return []
    # group pass-through accounts by shared sources / sinks (layering network)
    ug = nx.Graph()
    for a, (_, ins, outs) in pt.items():
        for i in ins:
            ug.add_edge(a, W.src[i])
        for o in outs:
            ug.add_edge(a, W.dst[o])
    for comp in nx.connected_components(ug):
        mules = [a for a in comp if a in pt]
        if len(mules) < 2:
            continue
        sources = {W.src[i] for a in mules for i in pt[a][1]}
        sinks = {W.dst[o] for a in mules for o in pt[a][2]}
        fan_out = max((len({W.dst[o] for o in W.out_edges.get(s, ()) if W.dst[o] in comp}) for s in sources), default=0)
        fan_in = max((len({W.src[i] for i in W.in_edges.get(k, ()) if W.src[i] in comp}) for k in sinks), default=0)
        if max(fan_out, fan_in) < cfg.fan_min_degree and len(mules) < 3:
            continue
        members = sorted(comp)
        if len(members) > 80:
            continue
        mean_ratio = float(np.mean([pt[a][0] for a in mules]))
        delays = []
        for a in mules:
            t_in = min(W.ts[i] for i in pt[a][1])
            t_out = min(W.ts[o] for o in pt[a][2])
            delays.append((t_out - t_in) / MIN)
        med_delay = float(np.median(delays))
        idxs = sorted({i for a in mules for i in pt[a][1]} | {o for a in mules for o in pt[a][2]})
        new_share = np.mean([1.0 if W.first_seen.get(a, 0) > W.now - 24 * HOUR else 0.0 for a in members])
        rep = sum(1 for a in members if a in W.reported)
        score = _clip01(
            0.25 * min(1.0, len(mules) / 6)
            + 0.25 * min(1.0, max(fan_out, fan_in) / 8)
            + 0.2 * (1.0 - min(1.0, med_delay / 180))
            + 0.15 * _clip01((mean_ratio - 0.8) / 0.2)
            + 0.05 * new_share
            + 0.25 * min(1, rep)
        )
        rings.append(
            Ring(
                "mule_network",
                members,
                sorted({W.dev[i] for i in idxs if W.dev[i]})[:20],
                score,
                {
                    "pass_through_accounts": len(mules),
                    "fan_out": fan_out,
                    "fan_in": fan_in,
                    "median_pass_through_minutes": round(med_delay, 1),
                    "mean_pass_through_ratio": round(mean_ratio, 3),
                    "total_amount": round(float(W.amt[idxs].sum()), 2),
                    "reported_members": rep,
                },
                int(W.ts[idxs].min()),
                int(W.ts[idxs].max()),
                edges=[(W.src[i], W.dst[i], float(W.amt[i]), int(W.ts[i])) for i in idxs[:400]],
            )
        )
    return rings


# ---------------------------------------------------------------------------------------------- (d) shared attrs
def shared_attribute_clusters(W: WindowGraph, cfg: DetectorConfig) -> list[Ring]:
    g = nx.Graph()
    for d, accts in W.dev_accounts.items():
        if cfg.device_min_accounts <= len(accts) <= cfg.attr_max_accounts:
            for a in accts:
                g.add_edge("d:" + d, "a:" + a)
    rings = []
    for comp in nx.connected_components(g):
        acc = sorted(n[2:] for n in comp if n.startswith("a:"))
        devs = sorted(n[2:] for n in comp if n.startswith("d:"))
        if len(acc) < cfg.device_min_accounts:
            continue
        accset = set(acc)
        idxs = [i for a in acc for i in W.out_edges.get(a, ())]
        if not idxs:
            continue
        payees = [W.dst[i] for i in idxs]
        common_payee_share = 1.0 - len(set(payees)) / max(len(payees), 1)
        internal = sum(1 for p in payees if p in accset) / len(payees)
        new_share = np.mean([1.0 if W.first_seen.get(a, 0) > W.now - 24 * HOUR else 0.0 for a in acc])
        rep = sum(1 for a in acc if a in W.reported)
        amt = float(W.amt[idxs].sum())
        score = _clip01(
            0.30 * min(1.0, (len(acc) - 2) / 6)
            + 0.2 * common_payee_share
            + 0.2 * internal
            + 0.15 * _clip01(amt / max(len(acc), 1) / 20_000)
            + 0.05 * new_share
            + 0.3 * min(1, rep)
        )
        rings.append(
            Ring(
                "shared_device",
                acc,
                devs,
                score,
                {
                    "accounts": len(acc),
                    "devices": len(devs),
                    "common_payee_share": round(common_payee_share, 3),
                    "internal_share": round(internal, 3),
                    "total_amount": round(amt, 2),
                    "reported_members": rep,
                },
                int(W.ts[idxs].min()),
                int(W.ts[idxs].max()),
                edges=[(W.src[i], W.dst[i], float(W.amt[i]), int(W.ts[i])) for i in idxs[:400]],
            )
        )
    return rings


# ---------------------------------------------------------------------------------------------- (e) bursts
def reciprocal_bursts(W: WindowGraph, cfg: DetectorConfig) -> list[Ring]:
    """Groups of >= 3 accounts linked by money that flows back within a couple of hours."""
    first: dict[tuple, int] = {}
    order = np.argsort(W.ts, kind="stable")
    recip_edges: dict[tuple, list[int]] = defaultdict(list)
    for i in order:
        a, b = W.src[i], W.dst[i]
        back = first.get((b, a))
        if back is not None and W.ts[i] - W.ts[back] <= 2 * HOUR:
            key = (b, a)
            recip_edges[key].extend([int(back), int(i)])
        first.setdefault((a, b), int(i))
    if not recip_edges:
        return []
    g = nx.Graph()
    for a, b in recip_edges:
        g.add_edge(a, b)
    rings = []
    for comp in nx.connected_components(g):
        if len(comp) < 3 or len(comp) > 40:
            continue
        idxs = sorted(
            {i for (a, b), ids in recip_edges.items() if a in comp for i in ids}
            | {i for a in comp for i in W.out_edges.get(a, ()) if W.dst[i] in comp}
        )
        amts = W.amt[idxs]
        span = int(W.ts[idxs].max() - W.ts[idxs].min())
        round_share = float(
            np.mean(
                [
                    1.0
                    if (
                        a >= 5_000
                        and (a % 1_000 == 0 or 9_000 <= a < 10_000 or 45_000 <= a < 50_000 or 90_000 <= a < 100_000)
                    )
                    else 0.0
                    for a in amts
                ]
            )
        )
        burst = 1.0 - min(1.0, span / (3 * HOUR))
        rep = sum(1 for a in comp if a in W.reported)
        score = _clip01(
            0.35 * round_share
            + 0.25 * burst
            + 0.2 * min(1.0, len(idxs) / 12)
            + 0.1 * _clip01(float(amts.mean()) / 30_000)
            + 0.25 * min(1, rep)
        )
        rings.append(
            Ring(
                "reciprocal_burst",
                sorted(comp),
                sorted({W.dev[i] for i in idxs if W.dev[i]})[:20],
                score,
                {
                    "accounts": len(comp),
                    "transfers": len(idxs),
                    "round_amount_share": round(round_share, 3),
                    "span_minutes": round(span / MIN, 1),
                    "total_amount": round(float(amts.sum()), 2),
                    "reported_members": rep,
                },
                int(W.ts[idxs].min()),
                int(W.ts[idxs].max()),
                edges=[(W.src[i], W.dst[i], float(W.amt[i]), int(W.ts[i])) for i in idxs[:400]],
            )
        )
    return rings


def _merge_overlapping(rings: list[Ring], rtype: str) -> list[Ring]:
    """Merge rings of the same type that share accounts (e.g. rotations / repeats of one laundering loop)."""
    if len(rings) <= 1:
        return rings
    g = nx.Graph()
    for i, r in enumerate(rings):
        g.add_node(("r", i))
        for m in r.members:
            g.add_edge(("r", i), ("m", m))
    merged = []
    for comp in nx.connected_components(g):
        rs = [rings[n[1]] for n in comp if n[0] == "r"]
        if len(rs) == 1:
            merged.append(rs[0])
            continue
        best = max(rs, key=lambda r: r.score)
        members = sorted({m for r in rs for m in r.members})
        ev = dict(best.evidence)
        ev["merged_cycles"] = len(rs)
        ev["total_amount"] = round(sum(r.evidence.get("total_amount", 0) for r in rs), 2)
        merged.append(
            Ring(
                rtype,
                members,
                sorted({d for r in rs for d in r.devices}),
                _clip01(best.score + 0.05 * (len(rs) - 1)),
                ev,
                min(r.first_seen for r in rs),
                max(r.last_seen for r in rs),
                edges=[e for r in rs for e in r.edges],
            )
        )
    return merged


def ring_id_for(ring: Ring) -> str:
    h = hashlib.sha1((ring.ring_type + "|" + "|".join(sorted(ring.members)[:8])).encode()).hexdigest()[:10]
    return f"R-{h}"


def detect_rings(
    df: pd.DataFrame, reported: set[str] | None = None, now_ts: int | None = None, cfg: DetectorConfig | None = None
) -> list[Ring]:
    cfg = cfg or DetectorConfig()
    W = WindowGraph(df, reported, now_ts)
    rings = (
        temporal_cycles(W, cfg)
        + dense_subgraphs(W, cfg)
        + mule_patterns(W, cfg)
        + shared_attribute_clusters(W, cfg)
        + reciprocal_bursts(W, cfg)
    )
    rings = [r for r in rings if r.score >= cfg.min_score]
    for r in rings:
        r.ring_id = ring_id_for(r)
    return rings


def fraud_neighbor_ratio(df: pd.DataFrame, flagged: set[str]) -> dict[str, float]:
    """Share of each account's window counterparties that are flagged (reported or high-score ring members)."""
    if not flagged or df.empty:
        return {}
    nbrs: dict[str, set[str]] = defaultdict(set)
    for a, b in zip(df["payer_id"].to_numpy(), df["payee_id"].to_numpy()):
        if a != b:
            nbrs[a].add(b)
            nbrs[b].add(a)
    out = {}
    for a, ns in nbrs.items():
        k = sum(1 for n in ns if n in flagged)
        if k:
            out[a] = k / len(ns)
    return out
