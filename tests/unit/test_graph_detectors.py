import pandas as pd

from graph.detectors import (
    DetectorConfig,
    WindowGraph,
    dense_subgraphs,
    detect_rings,
    mule_patterns,
    reciprocal_bursts,
    shared_attribute_clusters,
    temporal_cycles,
)
from graph.rings import RingTracker, node_scores

MIN = 60_000
T0 = 1_767_225_600_000


def frame(rows):
    return pd.DataFrame(rows, columns=["ts", "payer_id", "payee_id", "amount", "device_id", "ip"])


def background(n=60):
    # unrelated legit noise: customers paying distinct merchants
    return [(T0 + i * MIN, f"c{i}", f"m{i % 7}", 300.0 + i, f"dc{i}", f"ipc{i}") for i in range(n)]


def test_temporal_cycle_detected_with_path():
    rows = background() + [
        (T0 + 100 * MIN, "A", "B", 50_000, "d1", "i1"),
        (T0 + 110 * MIN, "B", "C", 49_500, "d2", "i2"),
        (T0 + 125 * MIN, "C", "D", 49_000, "d3", "i3"),
        (T0 + 140 * MIN, "D", "A", 48_600, "d4", "i4"),
    ]
    rings = temporal_cycles(WindowGraph(frame(rows)), DetectorConfig())
    assert len(rings) == 1
    r = rings[0]
    assert set(r.members) == {"A", "B", "C", "D"}
    assert r.evidence["cycle_length"] == 4
    assert r.evidence["cycle_path"][0] == r.evidence["cycle_path"][-1]
    assert r.score > 0.5


def test_cycle_with_non_increasing_timestamps_is_not_a_flow():
    rows = [
        (T0 + 50 * MIN, "A", "B", 1000, "d1", "i1"),
        (T0 + 40 * MIN, "B", "C", 1000, "d2", "i2"),
        (T0 + 30 * MIN, "C", "A", 1000, "d3", "i3"),
    ]
    assert temporal_cycles(WindowGraph(frame(rows)), DetectorConfig()) == []


def test_dense_clique_with_shared_devices_detected():
    members = [f"r{i}" for i in range(6)]
    rows = background()
    k = 0
    for i, a in enumerate(members):
        for j, b in enumerate(members):
            if i != j:
                rows.append((T0 + k * MIN, a, b, 5_000, f"shared{i % 2}", "ipx"))
                k += 1
    rings = dense_subgraphs(WindowGraph(frame(rows)), DetectorConfig())
    assert any(set(members) <= set(r.members) and r.evidence["mutual_share"] == 1.0 for r in rings)


def test_fan_out_fan_in_mule_network():
    rows = background()
    mules = [f"mule{i}" for i in range(6)]
    for i, m in enumerate(mules):
        rows.append((T0 + i * MIN, "SRC", m, 9_000, "dsrc", "isrc"))
        rows.append((T0 + (10 + i) * MIN, m, "COLLECT", 8_700, f"dm{i}", f"im{i}"))
    rings = mule_patterns(WindowGraph(frame(rows)), DetectorConfig())
    assert rings
    r = max(rings, key=lambda x: x.score)
    assert set(mules) <= set(r.members) and "SRC" in r.members and "COLLECT" in r.members
    assert r.evidence["fan_out"] >= 6 and r.evidence["fan_in"] >= 6


def test_shared_device_cluster():
    rows = background() + [(T0 + i * MIN, f"acct{i}", "BEN", 2_000, "farm-device", f"ip{i}") for i in range(5)]
    rings = shared_attribute_clusters(WindowGraph(frame(rows)), DetectorConfig())
    assert len(rings) == 1 and set(rings[0].members) == {f"acct{i}" for i in range(5)}
    assert rings[0].devices == ["farm-device"]


def test_reciprocal_burst():
    rows = background()
    for i, (a, b) in enumerate([("X", "Y"), ("X", "Z"), ("W", "Y")]):
        rows.append((T0 + i * MIN, a, b, 49_500, "dx", "ix"))
        rows.append((T0 + (30 + i) * MIN, b, a, 49_500, "dy", "iy"))
    rings = reciprocal_bursts(WindowGraph(frame(rows)), DetectorConfig())
    assert len(rings) == 1 and set(rings[0].members) == {"W", "X", "Y", "Z"}


def test_legit_background_alone_yields_no_rings():
    assert detect_rings(frame(background(200))) == []


def test_node_scores_and_tracker_stable_ids():
    rows = background() + [
        (T0 + 100 * MIN, "A", "B", 50_000, "d1", "i1"),
        (T0 + 110 * MIN, "B", "C", 49_500, "d2", "i2"),
        (T0 + 125 * MIN, "C", "A", 49_000, "d3", "i3"),
    ]
    df = frame(rows)
    tr = RingTracker()
    r1 = tr.assign(detect_rings(df))
    r2 = tr.assign(detect_rings(df))
    assert [r.ring_id for r in r1] == [r.ring_id for r in r2]
    scores = node_scores(df, r1, set(), T0)
    assert scores["a:A"]["cyc"] == 3 and scores["a:A"]["rs"] > 0
