"""Train/serve parity: the offline replay (InMemoryStore) and the online path (RedisStore on a real Redis) must
produce IDENTICAL features for the same event stream, including fraud reports and graph-score refreshes."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_gen.generate import GenConfig, Generator
from graph.rings import RingTracker, run_graph_cycle
from pipeline.features import FeatureEngine
from pipeline.replay import GRAPH_COLS, TX_COLS
from pipeline.state_store import InMemoryStore, RedisStore

HOUR = 3_600_000


@pytest.fixture(scope="module")
def dataset():
    cfg = GenConfig(seed=11, n_transactions=20_000, days=10, drift_start_day=8)
    res = Generator(cfg).run()
    return res["transactions"], res["reports"]


def _replay(store, tx: pd.DataFrame, reports: pd.DataFrame, n: int):
    engine = FeatureEngine()
    tracker = RingTracker()
    nodes: set[str] = set()
    ts_all = tx["ts"].to_numpy()
    step = 6 * HOUR
    next_run = int(ts_all.min()) + step
    rp = 0
    rep_ts, rep_acc = reports["report_ts"].to_numpy(), reports["account_id"].to_numpy()
    out_tab, out_graph, out_seq = [], [], []
    for t in tx.iloc[:n][TX_COLS].to_dict("records"):
        ts = int(t["ts"])
        while rp < len(rep_ts) and rep_ts[rp] <= ts:
            store.execute(FeatureEngine.report_plan(rep_acc[rp]))
            rp += 1
        while ts >= next_run:
            lo = np.searchsorted(ts_all, next_run - 24 * HOUR, "left")
            hi = np.searchsorted(ts_all, next_run, "left")
            _, nodes = run_graph_cycle(
                tx.iloc[lo:hi][GRAPH_COLS], set(rep_acc[rep_ts <= next_run]), next_run, store, tracker, nodes
            )
            next_run += step
        fv = engine.compute(t, store.execute(engine.read_plan(t)))
        store.execute(engine.write_plan(t))
        out_tab.append(fv.tabular)
        out_graph.append(fv.graph)
        out_seq.append(fv.seq)
    return np.array(out_tab, dtype=np.float64), np.array(out_graph), np.array(out_seq)


def test_offline_and_online_features_identical(dataset, test_redis):
    tx, reports = dataset
    n = 6000
    t_mem, g_mem, s_mem = _replay(InMemoryStore(), tx, reports, n)
    t_red, g_red, s_red = _replay(RedisStore(test_redis), tx, reports, n)
    assert t_mem.shape == t_red.shape == (n, t_mem.shape[1])
    np.testing.assert_array_equal(np.isnan(t_mem), np.isnan(t_red))
    np.testing.assert_allclose(np.nan_to_num(t_mem), np.nan_to_num(t_red), rtol=0, atol=1e-9)
    np.testing.assert_allclose(g_mem, g_red, rtol=0, atol=1e-9)
    np.testing.assert_allclose(s_mem, s_red, rtol=0, atol=1e-7)
    # the comparison is not vacuous: graph scores and history-based features are populated
    assert (g_mem[:, 0] > 0).any() or (g_mem[:, 1] > 0).any()
    assert (~np.isnan(t_mem[:, 1])).sum() > n * 0.3


def test_replay_module_matches_online_path(tmp_path: Path, dataset, test_redis):
    """pipeline.replay (the code that builds the training matrix) == online RedisStore path."""
    from pipeline.replay import replay

    tx, reports = dataset
    d = tmp_path / "data"
    d.mkdir()
    tx.to_parquet(d / "transactions.parquet", index=False)
    reports.to_parquet(d / "fraud_reports.parquet", index=False)
    n = 3000
    replay(d, tmp_path / "feat", snapshot_day=None, graph_step_h=6.0, limit=n)
    feats = pd.read_parquet(tmp_path / "feat" / "features.parquet")
    from pipeline.features import GRAPH_FEATURES, TABULAR_FEATURES

    t_red, g_red, _ = _replay(RedisStore(test_redis), tx, reports, n)
    off = feats[TABULAR_FEATURES].to_numpy(dtype=np.float64)
    # replay stores float32: compare at float32 precision
    np.testing.assert_array_equal(np.isnan(off), np.isnan(t_red))
    np.testing.assert_allclose(np.nan_to_num(off), np.nan_to_num(t_red.astype(np.float32)), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(feats[GRAPH_FEATURES].to_numpy(), g_red.astype(np.float32), rtol=1e-6, atol=1e-6)
    assert not math.isnan(off[-1, 0])
