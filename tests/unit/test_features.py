import math

import numpy as np
import pytest

from pipeline.features import (
    GRAPH_FEATURES,
    SEQ_DIM,
    SEQ_LEN,
    TABULAR_FEATURES,
    FeatureEngine,
    haversine_km,
    is_round_high_value,
    round_level,
)
from pipeline.state_store import InMemoryStore
from tests.conftest import base_tx

MIN, HOUR, DAY = 60_000, 3_600_000, 86_400_000
F = {n: i for i, n in enumerate(TABULAR_FEATURES)}


def run(engine, store, tx):
    fv = engine.compute(tx, store.execute(engine.read_plan(tx)))
    store.execute(engine.write_plan(tx))
    return fv


def test_feature_counts():
    assert len(TABULAR_FEATURES) >= 15 and len(set(TABULAR_FEATURES)) == len(TABULAR_FEATURES)
    assert len(GRAPH_FEATURES) == 7


def test_round_level_and_structuring():
    assert round_level(500.0) == 2
    assert round_level(10_000.0) == 4
    assert round_level(123.45) == 0
    assert round_level(7.0) == 0
    assert is_round_high_value(49_500.0)  # structuring band
    assert is_round_high_value(20_000.0)
    assert not is_round_high_value(1_000.0)  # below the high-value floor
    assert not is_round_high_value(12_345.67)


def test_haversine_known_distance():
    # Mumbai -> Delhi ~ 1150 km
    assert 1100 < haversine_km(19.076, 72.8777, 28.7041, 77.1025) < 1200
    assert haversine_km(10, 10, 10, 10) == 0


def test_minutes_to_cutoff():
    e = FeatureEngine((0, 6, 12, 18))
    assert e.minutes_to_cutoff(5 * HOUR + 50 * MIN) == pytest.approx(10)
    assert e.minutes_to_cutoff(6 * HOUR) == pytest.approx(0)
    assert e.minutes_to_cutoff(23 * HOUR) == pytest.approx(60)


def test_cold_start_payer_has_nan_history_features():
    e, s = FeatureEngine(), InMemoryStore()
    fv = run(e, s, base_tx())
    tab = fv.tabular
    assert math.isnan(tab[F["amount_ratio_mean_30d"]])
    assert math.isnan(tab[F["secs_since_last_tx_log"]])
    assert tab[F["payer_hist_count_30d"]] == 0
    assert tab[F["new_payee"]] == 1
    assert tab[F["new_device_for_payer"]] == 1
    assert fv.seq.shape == (SEQ_LEN, SEQ_DIM)
    assert fv.seq[-1, -2] == 1 and fv.seq[:-1].sum() == 0  # only the current step is present


def test_velocity_amount_ratio_and_new_payee():
    e, s = FeatureEngine(), InMemoryStore()
    t0 = base_tx()["ts"]
    for i in range(5):
        run(e, s, base_tx(transfer_id=f"a{i}", ts=t0 + i * 10 * MIN, amount=100.0))
    fv = run(e, s, base_tx(transfer_id="x", ts=t0 + 40 * MIN + 30_000, amount=1000.0, payee_id="cust-9"))
    tab = fv.tabular
    assert tab[F["payer_tx_1h"]] == 5
    assert tab[F["payer_tx_1m"]] == 1  # the one 30 s earlier
    assert tab[F["amount_ratio_mean_30d"]] == pytest.approx(10.0)
    assert tab[F["new_payee"]] == 1
    assert tab[F["payer_out_degree"]] == 1
    assert tab[F["secs_since_last_tx_log"]] == pytest.approx(math.log1p(30))


def test_device_and_ip_sharing_counts_accounts():
    e, s = FeatureEngine(), InMemoryStore()
    t0 = base_tx()["ts"]
    for i in range(4):
        run(e, s, base_tx(transfer_id=f"d{i}", payer_id=f"acct-{i}", ts=t0 + i * MIN))
    fv = run(e, s, base_tx(transfer_id="d9", payer_id="acct-9", ts=t0 + 10 * MIN))
    assert fv.tabular[F["device_accounts"]] == 5
    assert fv.tabular[F["ip_accounts"]] == 5
    assert fv.tabular[F["device_tx_1h"]] == 4


def test_pass_through_ratio():
    e, s = FeatureEngine(), InMemoryStore()
    t0 = base_tx()["ts"]
    run(e, s, base_tx(transfer_id="in1", payer_id="src", payee_id="mule", ts=t0, amount=10_000.0, device_id="dz"))
    fv = run(
        e,
        s,
        base_tx(transfer_id="out1", payer_id="mule", payee_id="sink", ts=t0 + 5 * MIN, amount=9_500.0, device_id="dm"),
    )
    assert fv.tabular[F["pass_through_ratio_24h"]] == pytest.approx(0.95)
    assert fv.tabular[F["mins_since_last_inflow_log"]] == pytest.approx(math.log1p(5))


def test_out_of_order_event_ignores_future_history():
    e, s = FeatureEngine(), InMemoryStore()
    t0 = base_tx()["ts"]
    run(e, s, base_tx(transfer_id="late", ts=t0 + HOUR, amount=100.0))
    fv = run(e, s, base_tx(transfer_id="early", ts=t0, amount=100.0))  # arrives after a later event
    assert fv.tabular[F["payer_tx_24h"]] == 0
    assert fv.tabular[F["payer_hist_count_30d"]] == 0


def test_reports_feed_counterparty_risk():
    e, s = FeatureEngine(), InMemoryStore()
    s.execute(FeatureEngine.report_plan("cust-2"))
    fv = run(e, s, base_tx())
    assert fv.tabular[F["payee_report_count"]] == 1
    assert fv.tabular[F["counterparty_risk_score"]] > 0.6


def test_missing_optional_fields_do_not_crash():
    e, s = FeatureEngine(), InMemoryStore()
    fv = run(e, s, base_tx(device_id="", ip="", lat=None, lon=None, device_bound_at=None, channel=None))
    assert math.isnan(fv.tabular[F["device_binding_age_hours_log"]])
    assert np.isfinite(np.nan_to_num(np.array(fv.tabular, dtype=float))).all()


def test_graph_features_read_from_store():
    e, s = FeatureEngine(), InMemoryStore()
    s.hset("g:a:cust-2", mapping={"rs": "0.8", "rid": "R-1", "sz": "5", "cyc": "4", "fnr": "0.5", "ts": "1"})
    fv = run(e, s, base_tx())
    g = dict(zip(GRAPH_FEATURES, fv.graph))
    assert g["payee_ring_score"] == pytest.approx(0.8)
    assert g["in_cycle"] == 1 and g["max_ring_size"] == 5
    assert fv.context["payee_ring"]["rid"] == "R-1"
