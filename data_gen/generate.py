"""Synthetic, labelled, reproducible Mojaloop-style payment data generator.

Produces (in --out directory):
  transactions.parquet   internal flat schema (one row per transfer, sorted by event time)
  fraud_reports.parquet  delayed fraud reports against beneficiary accounts (registry feed)
  rings_truth.json       planted fraud cases / rings with member accounts (ground truth)
  accounts.parquet       entity table
  sample_fspiop_events.jsonl  first 1000 events rendered as Mojaloop FSPIOP-style prepare messages
  manifest.json          config + summary statistics + data hash

NOTE: this is SIMULATED data shaped like Mojaloop FSPIOP transfers. It is not a live Mojaloop integration.

Usage: python -m data_gen.generate --n-transactions 1000000 --seed 42 --out data/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_gen.entities import (
    CATEGORY_NAMES,
    CITIES,
    DAY_MS,
    HOUR_MS,
    MERCHANT_CATEGORIES,
    MIN_MS,
    Universe,
)

SCENARIOS = ("TRANSFER", "PAYMENT", "REFUND", "REVERSAL")
TYPOLOGIES = ("identity_spoofing", "settlement_manipulation", "merchant_fraud_ring", "transaction_laundering")


@dataclass
class GenConfig:
    seed: int = 42
    n_transactions: int = 1_000_000
    days: int = 60
    start: str = "2026-01-01"
    customers_per_tx: float = 0.04
    merchants_per_tx: float = 0.003
    n_fsps: int = 12
    fraud_rate: float = 0.003
    typology_mix: dict = field(
        default_factory=lambda: {
            "identity_spoofing": 0.25,
            "settlement_manipulation": 0.20,
            "merchant_fraud_ring": 0.30,
            "transaction_laundering": 0.25,
        }
    )
    label_noise_fn: float = 0.03  # share of fraud rows whose observed label is 0 (unreported fraud)
    label_noise_fp: float = 0.0001  # share of legit rows whose observed label is 1 (false reports)
    report_prob: float = 0.6  # probability a fraud beneficiary gets reported to the registry
    stealth_share: float = 0.3  # share of fraud cases that deliberately mimic legitimate behaviour
    drift_start_day: int = 48  # new fraud variants appear from this day (concept drift)
    settlement_cutoff_hours: tuple = (0, 6, 12, 18)
    festival_days: tuple = (20, 21, 22, 52, 53)
    currency: str = "INR"

    @property
    def n_customers(self) -> int:
        return max(1500, int(self.n_transactions * self.customers_per_tx))

    @property
    def n_merchants(self) -> int:
        return max(150, int(self.n_transactions * self.merchants_per_tx))

    @property
    def n_employers(self) -> int:
        return max(5, self.n_transactions // 25_000)


COLS = [
    "ts",
    "payer",
    "payee",
    "amount",
    "scenario",
    "channel",
    "device",
    "ip",
    "lat",
    "lon",
    "device_bound_ts",
    "orig_uid",
    "is_fraud",
    "fraud_type",
    "case_id",
]


class RowBuffer:
    """Accumulates rows from Python-level generators (fraud typologies, refunds)."""

    def __init__(self):
        self.rows: dict[str, list] = {c: [] for c in COLS}

    def add(
        self,
        ts,
        payer,
        payee,
        amount,
        scenario,
        channel,
        device,
        ip,
        lat,
        lon,
        device_bound_ts,
        orig_uid=-1,
        is_fraud=0,
        fraud_type="",
        case_id="",
    ):
        r = self.rows
        r["ts"].append(int(ts))
        r["payer"].append(int(payer))
        r["payee"].append(int(payee))
        r["amount"].append(float(round(amount, 2)))
        r["scenario"].append(scenario)
        r["channel"].append(channel)
        r["device"].append(int(device))
        r["ip"].append(int(ip))
        r["lat"].append(float(lat))
        r["lon"].append(float(lon))
        r["device_bound_ts"].append(int(device_bound_ts))
        r["orig_uid"].append(int(orig_uid))
        r["is_fraud"].append(int(is_fraud))
        r["fraud_type"].append(fraud_type)
        r["case_id"].append(case_id)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=COLS)


def _round_to(x, base):
    return np.maximum(base, np.round(np.asarray(x) / base) * base)


class Generator:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.start_ms = int(pd.Timestamp(cfg.start, tz="UTC").value // 1_000_000)
        self.end_ms = self.start_ms + cfg.days * DAY_MS
        self.U = Universe(cfg, self.rng, self.start_ms)
        self.cases: list[dict] = []
        self.reports: list[tuple[int, int]] = []

    # ================================================================ entities
    def build_entities(self):
        cfg, rng, U = self.cfg, self.rng, self.U
        nC = cfg.n_customers
        # 10% of customers join during the period (legit new accounts => hard negatives for "new account")
        created = self.start_ms - rng.integers(30, 2500, nC) * DAY_MS
        joiners = rng.random(nC) < 0.10
        created[joiners] = (
            self.start_ms
            + rng.integers(0, cfg.days - 2, joiners.sum()) * DAY_MS
            + rng.integers(0, DAY_MS, joiners.sum())
        )
        bound = np.where(joiners, created + rng.integers(0, 6, nC) * HOUR_MS, 0)
        bound_old = np.maximum(created, self.start_ms - rng.integers(1, 700, nC) * DAY_MS)
        bound = np.where(joiners, bound, bound_old)
        self.cust = U.add_accounts(nC, 0, created_ts=created, device_bound_ts=bound)
        # households: 5% of customers share a device and home IP with 1-3 others (legit device sharing)
        n_house = int(nC * 0.05 / 2.5)
        devices = np.array(U._cols["device"], dtype=np.int64)
        ips = np.array(U._cols["home_ip"], dtype=np.int64)
        for _ in range(n_house):
            members = rng.choice(self.cust, size=rng.integers(2, 5), replace=False)
            devices[members] = devices[members[0]]
            ips[members] = ips[members[0]]
        U._cols["device"] = devices.tolist()
        U._cols["home_ip"] = ips.tolist()

        ncat = len(CATEGORY_NAMES)
        cat_w = np.array([MERCHANT_CATEGORIES[c][0] for c in CATEGORY_NAMES])
        mcat = rng.choice(ncat, size=cfg.n_merchants, p=cat_w / cat_w.sum())
        self.merch = U.add_accounts(cfg.n_merchants, 1, category=mcat)
        self.biz = U.add_accounts(cfg.n_employers, 2)
        A = U.freeze()
        self.A = A

        # behavioural parameters per customer
        self.activity = rng.lognormal(0, 0.9, nC)
        self.scale = rng.lognormal(np.log(450), 0.7, nC)  # typical spend
        self.peak_hour = np.clip(rng.normal(14.5, 3.5, nC), 6, 23)
        self.p2m_pref = rng.beta(5, 3, nC)
        # second device (some customers use two phones)
        self.has_second = rng.random(nC) < 0.08
        self.second_dev = np.where(self.has_second, U.new_devices(nC), -1)
        # legit phone change mid-period (hard negative for identity spoofing)
        self.switch = rng.random(nC) < 0.015
        self.switch_ts = np.where(
            self.switch,
            self.start_ms + rng.integers(0, cfg.days, nC) * DAY_MS + rng.integers(0, DAY_MS, nC),
            np.iinfo(np.int64).max,
        )
        self.switch_dev = np.where(self.switch, U.new_devices(nC), -1)
        # legit app re-install / UPI re-registration: same device, fresh binding (hard negative)
        self.rebind = rng.random(nC) < 0.05
        self.rebind_ts = np.where(
            self.rebind,
            self.start_ms + rng.integers(0, cfg.days, nC) * DAY_MS + rng.integers(0, DAY_MS, nC),
            np.iinfo(np.int64).max,
        )
        # travel windows (geo jumps, hard negative)
        self.travel = rng.random(nC) < 0.05
        self.travel_start = np.where(self.travel, self.start_ms + rng.integers(0, cfg.days - 3, nC) * DAY_MS, 0)
        self.travel_end = self.travel_start + rng.integers(1, 5, nC) * DAY_MS
        self.travel_city = rng.choice(len(CITIES), size=nC, p=U.city_weights)
        # carrier-grade NAT pool shared by many customers (hard negative for IP sharing)
        self.nat_ips = U.new_ips(max(20, nC // 150))
        # social graph: contacts are mostly in the same city; reciprocity ~60%
        K = 8
        by_city: dict[int, np.ndarray] = {}
        cust_city = A.city[self.cust]
        for c in range(len(CITIES)):
            by_city[c] = self.cust[cust_city == c]
        contacts = np.empty((nC, K), dtype=np.int64)
        for i in range(nC):
            pool = by_city[cust_city[i]]
            same = rng.random(K) < 0.85
            contacts[i] = np.where(same, pool[rng.integers(0, len(pool), K)], self.cust[rng.integers(0, nC, K)])
        for i in rng.choice(nC, size=int(nC * 0.6), replace=False):
            j = contacts[i, 0] - self.cust[0]
            contacts[j, rng.integers(0, K)] = self.cust[i]
        self.contacts = contacts
        # merchant popularity (Zipf) and favourites per customer (city-local)
        mpop = rng.pareto(1.2, cfg.n_merchants) + 1
        self.merch_pop = mpop / mpop.sum()
        merch_city = A.city[self.merch]
        fav = np.empty((nC, 6), dtype=np.int64)
        merch_by_city = {c: self.merch[merch_city == c] for c in range(len(CITIES))}
        for i in range(nC):
            pool = merch_by_city.get(cust_city[i])
            if pool is None or len(pool) < 3:
                pool = self.merch
            p = self.merch_pop[pool - self.merch[0]]
            fav[i] = rng.choice(pool, size=6, p=p / p.sum())
        self.fav = fav
        # employers -> employees; landlords -> tenants
        self.salaried = rng.random(nC) < 0.4
        self.employer = rng.choice(self.biz, size=nC)
        self.salary = _round_to(rng.lognormal(np.log(35_000), 0.6, nC), 100)
        self.tenant = rng.random(nC) < 0.15
        landlords = rng.choice(self.cust, size=max(5, int(nC * 0.02)), replace=False)
        self.landlord = rng.choice(landlords, size=nC)
        self.rent = _round_to(rng.lognormal(np.log(12_000), 0.5, nC), 500)

    # ================================================================ helpers
    def _hour_of(self, ci: np.ndarray) -> np.ndarray:
        rng = self.rng
        h = (self.peak_hour[ci] + rng.normal(0, 3.2, len(ci))) % 24
        uni = rng.random(len(ci)) < 0.07
        h[uni] = rng.uniform(0, 24, uni.sum())
        return h

    def _geo_for(self, acct: np.ndarray, ts: np.ndarray):
        A = self.A
        lat = A.lat[acct] + self.rng.normal(0, 0.02, len(acct))
        lon = A.lon[acct] + self.rng.normal(0, 0.02, len(acct))
        ci = acct - self.cust[0]
        is_c = (acct >= self.cust[0]) & (acct <= self.cust[-1])
        trav = np.zeros(len(acct), dtype=bool)
        trav[is_c] = (
            self.travel[ci[is_c]] & (ts[is_c] >= self.travel_start[ci[is_c]]) & (ts[is_c] < self.travel_end[ci[is_c]])
        )
        if trav.any():
            tc = self.travel_city[ci[trav]]
            lat[trav] = self.U.city_lat[tc] + self.rng.normal(0, 0.02, trav.sum())
            lon[trav] = self.U.city_lon[tc] + self.rng.normal(0, 0.02, trav.sum())
        # cellular IP-geolocation error: occasional far-away location for legit traffic (hard negative)
        err = self.rng.random(len(acct)) < 0.03
        if err.any():
            ec = self.rng.choice(len(CITIES), size=err.sum(), p=self.U.city_weights)
            lat[err] = self.U.city_lat[ec]
            lon[err] = self.U.city_lon[ec]
        return lat, lon

    def _device_for(self, ci: np.ndarray, ts: np.ndarray):
        """Device + device binding timestamp for customers (index into customer arrays)."""
        acct = self.cust[ci]
        dev = self.A.device[acct].copy()
        bound = self.A.device_bound_ts[acct].copy()
        sec = self.has_second[ci] & (self.rng.random(len(ci)) < 0.3)
        dev[sec] = self.second_dev[ci[sec]]
        rb = ts >= self.rebind_ts[ci]
        bound[rb] = np.maximum(bound[rb], self.rebind_ts[ci[rb]])
        sw = ts >= self.switch_ts[ci]
        dev[sw] = self.switch_dev[ci[sw]]
        bound[sw] = np.maximum(bound[sw], self.switch_ts[ci[sw]])
        return dev, bound

    def _ip_for(self, acct: np.ndarray):
        nat = self.rng.random(len(acct)) < 0.35
        ip = self.A.home_ip[acct].copy()
        ip[nat] = self.rng.choice(self.nat_ips, size=nat.sum())
        return ip

    def _day_weights(self):
        cfg = self.cfg
        w = np.ones(cfg.days)
        dows = (pd.Timestamp(cfg.start).dayofweek + np.arange(cfg.days)) % 7
        w[dows >= 5] *= 1.15
        for d in cfg.festival_days:
            if d < cfg.days:
                w[d] *= 1.6
        for d in self._salary_days():
            for k in range(3):
                if d + k < cfg.days:
                    w[d + k] *= 1.3
        return w / w.sum()

    def _salary_days(self):
        days = []
        for d in range(self.cfg.days):
            if (pd.Timestamp(self.cfg.start) + pd.Timedelta(days=d)).day == 1:
                days.append(d)
        return days

    # ================================================================ legit traffic
    def gen_normal(self, n: int) -> pd.DataFrame:
        cfg, rng, A = self.cfg, self.rng, self.A
        nC = cfg.n_customers
        act = self.activity / self.activity.sum()
        ci = rng.choice(nC, size=n, p=act)
        day = rng.choice(cfg.days, size=n, p=self._day_weights())
        created_day = (self.A.created_ts[self.cust[ci]] - self.start_ms) // DAY_MS
        late = created_day >= day
        if late.any():
            cd = np.clip(created_day[late], 0, cfg.days - 1)
            day[late] = cd + (rng.random(late.sum()) * (cfg.days - cd)).astype(int)
        hour = self._hour_of(ci)
        ts = self.start_ms + day.astype(np.int64) * DAY_MS + (hour * HOUR_MS).astype(np.int64)
        ts = np.maximum(ts, self.A.created_ts[self.cust[ci]] + MIN_MS)
        ts = np.minimum(ts, self.end_ms - 1)
        payer = self.cust[ci]
        is_p2m = rng.random(n) < self.p2m_pref[ci]
        # P2M payees
        fav_pick = self.fav[ci, rng.integers(0, 6, n)]
        rnd_merch = rng.choice(self.merch, size=n, p=self.merch_pop)
        merch = np.where(rng.random(n) < 0.75, fav_pick, rnd_merch)
        # P2P payees: Zipf over contact slots, 8% to a random (new) customer
        slot = np.minimum(rng.zipf(1.6, n) - 1, 7)
        contact = self.contacts[ci, slot]
        rnd_c = self.cust[rng.integers(0, nC, n)]
        p2p = np.where(rng.random(n) < 0.92, contact, rnd_c)
        p2p = np.where(p2p == payer, rnd_c, p2p)
        payee = np.where(is_p2m, merch, p2p)
        # amounts
        cat = np.where(is_p2m, A.category[payee], 0)
        cat_mult = np.array([MERCHANT_CATEGORIES[c][1] for c in CATEGORY_NAMES])[cat]
        cat_sig = np.array([MERCHANT_CATEGORIES[c][2] for c in CATEGORY_NAMES])[cat]
        cat_round = np.array([MERCHANT_CATEGORIES[c][4] for c in CATEGORY_NAMES])[cat]
        festival = np.isin(day, cfg.festival_days) & is_p2m
        amt_m = self.scale[ci] * cat_mult * rng.lognormal(0, cat_sig) * np.where(festival, 1.5, 1.0)
        amt_p = self.scale[ci] * 1.4 * rng.lognormal(0, 0.8, n)
        amt = np.where(is_p2m, amt_m, amt_p)
        # legit large one-offs (vehicle / deposit / medical payments) to a NEW payee: hard negative for ATO
        big = (~is_p2m) & (rng.random(n) < 0.004)
        amt = np.where(big, amt * rng.uniform(4, 15, n), amt)
        payee = np.where(big, rnd_c, payee)
        rnd = np.where(is_p2m, rng.random(n) < cat_round, rng.random(n) < 0.45)
        amt = np.where(rnd & (amt >= 200), _round_to(amt, 100), np.where(rnd, _round_to(amt, 10), amt))
        amt = np.clip(np.round(amt, 2), 1.0, 100_000.0)
        dev, bound = self._device_for(ci, ts)
        ip = self._ip_for(payer)
        lat, lon = self._geo_for(payer, ts)
        ch = np.where(
            is_p2m,
            rng.choice(["QR", "APP", "WEB"], size=n, p=[0.55, 0.40, 0.05]),
            rng.choice(["APP", "USSD", "WEB"], size=n, p=[0.92, 0.06, 0.02]),
        )
        return pd.DataFrame(
            {
                "ts": ts,
                "payer": payer,
                "payee": payee,
                "amount": amt,
                "scenario": np.where(is_p2m, "PAYMENT", "TRANSFER"),
                "channel": ch,
                "device": dev,
                "ip": ip,
                "lat": lat,
                "lon": lon,
                "device_bound_ts": bound,
                "orig_uid": -1,
                "is_fraud": 0,
                "fraud_type": "",
                "case_id": "",
            }
        )

    def gen_salary_rent_b2b(self) -> pd.DataFrame:
        """Salary fan-out bursts, rent fan-in, and B2B round payments (hard negatives)."""
        cfg, rng, A = self.cfg, self.rng, self.A
        buf = RowBuffer()
        cut = np.array(cfg.settlement_cutoff_hours)
        for d in self._salary_days():
            emp_idx = np.where(self.salaried)[0]
            for ci in emp_idx:
                if A.created_ts[self.cust[ci]] > self.start_ms + d * DAY_MS:
                    continue
                emp = self.employer[ci]
                ts = self.start_ms + d * DAY_MS + int((9 + rng.random() * 3) * HOUR_MS) + int(rng.integers(0, 600_000))
                buf.add(
                    ts,
                    emp,
                    self.cust[ci],
                    self.salary[ci],
                    "TRANSFER",
                    "WEB",
                    A.device[emp],
                    A.home_ip[emp],
                    A.lat[emp],
                    A.lon[emp],
                    A.device_bound_ts[emp],
                )
            for ci in np.where(self.tenant)[0]:
                if A.created_ts[self.cust[ci]] > self.start_ms + d * DAY_MS:
                    continue
                ts = (
                    self.start_ms
                    + (d + int(rng.integers(0, 5))) * DAY_MS
                    + int(self._hour_of(np.array([ci]))[0] * HOUR_MS)
                )
                if ts >= self.end_ms:
                    continue
                p = self.cust[ci]
                dev, bound = self._device_for(np.array([ci]), np.array([ts]))
                lat, lon = self._geo_for(np.array([p]), np.array([ts]))
                buf.add(
                    ts,
                    p,
                    self.landlord[ci],
                    self.rent[ci],
                    "TRANSFER",
                    "APP",
                    dev[0],
                    self._ip_for(np.array([p]))[0],
                    lat[0],
                    lon[0],
                    bound[0],
                )
        # family relays: receive from one contact, forward 85-100% to another within 10-180 min
        relays = rng.choice(len(self.cust), size=max(5, len(self.cust) // 100), replace=False)
        for ci in relays:
            for _ in range(int(rng.integers(3, 7))):
                d = int(rng.integers(0, cfg.days))
                t_in = self.start_ms + d * DAY_MS + int((8 + rng.random() * 12) * HOUR_MS)
                t_out = t_in + int(rng.integers(10 * MIN_MS, 180 * MIN_MS))
                if t_out >= self.end_ms or A.created_ts[self.cust[ci]] > t_in:
                    continue
                me = self.cust[ci]
                src = int(self.contacts[ci, int(rng.integers(0, 8))])
                dst = int(self.contacts[ci, int(rng.integers(0, 8))])
                if src == me or dst == me or src == dst:
                    continue
                a_in = float(_round_to(rng.lognormal(np.log(12_000), 0.6), 100))
                a_out = float(round(a_in * rng.uniform(0.85, 1.0), -1))
                si = src - self.cust[0]
                d_src, b_src = self._device_for(np.array([si]), np.array([t_in]))
                buf.add(
                    t_in,
                    src,
                    me,
                    a_in,
                    "TRANSFER",
                    "APP",
                    d_src[0],
                    self._ip_for(np.array([src]))[0],
                    A.lat[src],
                    A.lon[src],
                    b_src[0],
                )
                d_me, b_me = self._device_for(np.array([ci]), np.array([t_out]))
                buf.add(
                    t_out,
                    me,
                    dst,
                    a_out,
                    "TRANSFER",
                    "APP",
                    d_me[0],
                    self._ip_for(np.array([me]))[0],
                    A.lat[me],
                    A.lon[me],
                    b_me[0],
                )
        # B2B supplier payments, round amounts, some right before settlement cutoffs
        n_b2b = max(400, cfg.n_transactions // 150)
        for _ in range(n_b2b):
            b = rng.choice(self.biz)
            m = rng.choice(self.merch, p=self.merch_pop)
            d = int(rng.integers(0, cfg.days))
            if rng.random() < 0.35:
                h = int(rng.choice(cut))
                ts = self.start_ms + d * DAY_MS + h * HOUR_MS - int(rng.integers(30_000, 15 * MIN_MS))
            else:
                ts = self.start_ms + d * DAY_MS + int((9 + rng.random() * 9) * HOUR_MS)
            if not (self.start_ms <= ts < self.end_ms):
                continue
            if rng.random() < 0.6:
                amt = float(
                    rng.choice([5_000, 9_900, 9_950, 10_000, 20_000, 25_000, 49_000, 49_500, 50_000, 75_000, 99_000])
                )
            else:
                amt = float(_round_to(rng.lognormal(np.log(30_000), 0.6), 100))
            buf.add(
                ts, b, m, amt, "PAYMENT", "WEB", A.device[b], A.home_ip[b], A.lat[b], A.lon[b], A.device_bound_ts[b]
            )
        return buf.frame()

    def gen_refunds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Legit merchant refunds of earlier P2M payments (hard negative for reversal patterns)."""
        rng, A = self.rng, self.A
        p2m = df[(df["scenario"] == "PAYMENT") & (df["payee"].isin(self.merch))]
        cat = A.category[p2m["payee"].to_numpy()]
        rate = np.array([MERCHANT_CATEGORIES[c][3] for c in CATEGORY_NAMES])[cat]
        # technical reversals (switch timeouts) of ordinary transfers, minutes later (hard negative)
        tr = df[df["scenario"].isin(["PAYMENT", "TRANSFER"])]
        tr = tr[rng.random(len(tr)) < 0.001]
        tts = tr["ts"].to_numpy() + rng.integers(2 * MIN_MS, 30 * MIN_MS, len(tr))
        tk = tts < self.end_ms
        tr, tts = tr[tk], tts[tk]
        q = tr["payee"].to_numpy()
        tech = pd.DataFrame(
            {
                "ts": tts,
                "payer": q,
                "payee": tr["payer"].to_numpy(),
                "amount": tr["amount"].to_numpy(),
                "scenario": "REVERSAL",
                "channel": "WEB",
                "device": A.device[q],
                "ip": A.home_ip[q],
                "lat": A.lat[q],
                "lon": A.lon[q],
                "device_bound_ts": A.device_bound_ts[q],
                "orig_uid": tr["uid"].to_numpy(),
                "is_fraud": 0,
                "fraud_type": "",
                "case_id": "",
            }
        )
        pick = p2m[rng.random(len(p2m)) < rate]
        delay = rng.integers(HOUR_MS, 72 * HOUR_MS, len(pick))
        ts = pick["ts"].to_numpy() + delay
        keep = ts < self.end_ms
        pick, ts = pick[keep], ts[keep]
        m = pick["payee"].to_numpy()
        refunds = pd.DataFrame(
            {
                "ts": ts,
                "payer": m,
                "payee": pick["payer"].to_numpy(),
                "amount": pick["amount"].to_numpy(),
                "scenario": "REFUND",
                "channel": "WEB",
                "device": A.device[m],
                "ip": A.home_ip[m],
                "lat": A.lat[m],
                "lon": A.lon[m],
                "device_bound_ts": A.device_bound_ts[m],
                "orig_uid": pick["uid"].to_numpy(),
                "is_fraud": 0,
                "fraud_type": "",
                "case_id": "",
            }
        )
        return pd.concat([refunds, tech], ignore_index=True)

    # ================================================================ fraud typologies
    def _new_accounts(self, n, kind=0, lax=True, created_ts=None, shared_devices=None, category=None):
        """Create fraud-controlled accounts (mules / fake merchants), returned as global account indices."""
        U, rng = self.U, self.rng
        fsp = rng.choice(U.lax_fsps + list(range(len(U.fsp_ids))), size=n) if lax else None
        if created_ts is None:
            created_ts = self.start_ms + rng.integers(0, self.cfg.days, n) * DAY_MS
        created_ts = np.broadcast_to(np.asarray(created_ts, dtype=np.int64), n).copy()
        dev = None
        if shared_devices is not None:
            dev = rng.choice(shared_devices, size=n)
        bound = created_ts + rng.integers(0, 3 * HOUR_MS, n)
        idx = U.add_accounts(
            n, kind, fsp=fsp, created_ts=created_ts, device=dev, device_bound_ts=bound, category=category
        )
        self.A = U.freeze()
        return idx

    def _mules(self, n, created_ts, shared_devices=None, recruited=0.5):
        """Mule accounts: a mix of freshly opened accounts and recruited existing customers (rent-an-account)."""
        k = int(self.rng.binomial(n, recruited))
        if n - k > 0:
            new = self._new_accounts(n - k, created_ts=created_ts, shared_devices=shared_devices)
        else:
            new = np.array([], dtype=np.int64)
        old = self.cust[self.rng.integers(0, self.cfg.n_customers, k)] if k > 0 else np.array([], dtype=np.int64)
        out = np.concatenate([new, old]).astype(np.int64)
        self.rng.shuffle(out)
        return out

    def _acct(self, i):
        """Device, IP, location (with the same jitter as legit traffic) and binding time of an account."""
        A, rng = self.A, self.rng
        return (
            A.device[i],
            A.home_ip[i],
            A.lat[i] + rng.normal(0, 0.02),
            A.lon[i] + rng.normal(0, 0.02),
            A.device_bound_ts[i],
        )

    def _report(self, acct, ts):
        if self.rng.random() < self.cfg.report_prob:
            delay = int(self.rng.lognormal(np.log(36 * HOUR_MS), 0.6))
            self.reports.append((int(acct), int(ts) + delay))

    def _add(self, buf: RowBuffer, ts, p, q, amt, sc, ch, dv, ip, lat, lon, b, ftype, case, orig_uid=-1):
        buf.add(
            ts, p, q, amt, sc, ch, dv, ip, lat, lon, b, orig_uid=orig_uid, is_fraud=1, fraud_type=ftype, case_id=case
        )

    def gen_identity_spoofing(self, buf: RowBuffer, target: int):
        """Account takeover: SIM-swap / device re-binding, geo mismatch, rapid drains to mule payees.

        Stealth cases (cfg.stealth_share) mimic the victim: cloned device, victim's location, daytime, amounts close
        to the victim's normal spend, slower pacing, recruited (established) mule accounts.
        """
        cfg, rng, A = self.cfg, self.rng, self.A
        fraud_ips = self.U.new_ips(max(5, target // 60))
        farm_devices = self.U.new_devices(max(3, target // 120))
        made, k = 0, 0
        while made < target:
            k += 1
            case = f"ATO-{k:04d}"
            ci = int(rng.integers(0, cfg.n_customers))
            victim = int(self.cust[ci])
            t0 = int(self.start_ms + rng.integers(1, cfg.days) * DAY_MS + rng.integers(0, DAY_MS))
            if t0 >= self.end_ms - 2 * HOUR_MS or A.created_ts[victim] > t0:
                continue
            after = (t0 - self.start_ms) >= cfg.drift_start_day * DAY_MS
            drift = after and rng.random() < 0.7
            stealth = rng.random() < cfg.stealth_share
            clone = drift or stealth or (not after and rng.random() < 0.25)
            if not stealth and rng.random() < 0.35:
                t0 = t0 - (t0 - self.start_ms) % DAY_MS + int(rng.uniform(0.5, 5) * HOUR_MS)  # night attack
            if clone:
                # device-fingerprint cloning -> victim's own device id & old binding (more common after drift)
                dev, bound = self._device_for(np.array([ci]), np.array([t0]))
                dev, bound = int(dev[0]), int(bound[0])
            else:
                dev = int(rng.choice(farm_devices)) if rng.random() < 0.35 else int(self.U.new_devices(1)[0])
                bound = t0 - int(rng.integers(2 * MIN_MS, 90 * MIN_MS))  # fresh SIM-swap/device binding
            ip = int(self._ip_for(np.array([victim]))[0]) if stealth else int(rng.choice(fraud_ips))
            local = stealth or rng.random() < 0.45  # attacker appears local (VPN / same-city operation)
            if local:
                lat0, lon0 = A.lat[victim], A.lon[victim]
            else:
                other_city = int(rng.choice(len(CITIES), p=self.U.city_weights))
                lat0, lon0 = self.U.city_lat[other_city], self.U.city_lon[other_city]
            n_mules = int(rng.integers(1, 4))
            mules = self._mules(n_mules, t0 - int(rng.integers(1, 20)) * DAY_MS, recruited=0.9 if stealth else 0.5)
            n_tx = int(rng.integers(1, 4)) if stealth else int(rng.integers(2, 8))
            ts = t0
            members = [victim] + [int(m) for m in mules]
            for _ in range(n_tx):
                ts += int(rng.integers(20, 240)) * MIN_MS if stealth else int(rng.integers(30_000, 25 * MIN_MS))
                if ts >= self.end_ms:
                    break
                mult = rng.uniform(0.8, 2.0) if stealth else rng.uniform(1.2, 8)
                amt = min(100_000.0, self.scale[ci] * 1.4 * mult)
                if not stealth and rng.random() < 0.4:
                    amt = float(_round_to(amt, 1000))
                m = int(rng.choice(mules))
                la, lo = lat0 + rng.normal(0, 0.02), lon0 + rng.normal(0, 0.02)
                self._add(buf, ts, victim, m, amt, "TRANSFER", "APP", dev, ip, la, lo, bound, "identity_spoofing", case)
                self._report(m, ts)
                made += 1
                # mule cashes out part quickly (pass-through), labelled as part of the same scheme
                if rng.random() < (0.15 if stealth else 0.4):
                    md, mip, mlat, mlon, mb = self._acct(m)
                    out = int(self.cust[rng.integers(0, cfg.n_customers)])
                    self._add(
                        buf,
                        ts + int(rng.integers(MIN_MS, 30 * MIN_MS)),
                        m,
                        out,
                        amt * rng.uniform(0.85, 0.98),
                        "TRANSFER",
                        "APP",
                        md,
                        mip,
                        mlat,
                        mlon,
                        mb,
                        "identity_spoofing",
                        case,
                    )
                    made += 1
            self.cases.append(
                {
                    "case_id": case,
                    "type": "identity_spoofing",
                    "members": members,
                    "devices": [dev],
                    "start_ts": t0,
                    "end_ts": ts,
                    "drift_variant": bool(drift),
                    "stealth": bool(stealth),
                }
            )

    def gen_settlement_manipulation(self, buf: RowBuffer, target: int):
        """Round-amount structuring bursts near settlement cutoffs from a rogue FSP, followed by reversals.

        Stealth cases use non-structured amounts away from cutoffs, fewer transfers and fewer reversals.
        """
        cfg, rng = self.cfg, self.rng
        made, k = 0, 0
        cut = list(cfg.settlement_cutoff_hours)
        while made < target:
            k += 1
            case = f"STL-{k:04d}"
            rogue_fsp = int(rng.choice(self.U.lax_fsps + [int(rng.integers(0, cfg.n_fsps))]))
            ctl = self._new_accounts(
                int(rng.integers(2, 5)),
                kind=2,
                lax=False,
                created_ts=self.start_ms - int(rng.integers(5, 200)) * DAY_MS,
            )
            self.U._cols["fsp"][ctl[0] : ctl[-1] + 1] = [rogue_fsp] * len(ctl)
            ben = self._mules(int(rng.integers(2, 4)), self.start_ms + int(rng.integers(0, cfg.days)) * DAY_MS)
            self.A = self.U.freeze()
            d = int(rng.integers(1, cfg.days))
            cutoff = self.start_ms + d * DAY_MS + int(rng.choice(cut)) * HOUR_MS
            if cutoff >= self.end_ms - HOUR_MS:
                continue
            stealth = rng.random() < cfg.stealth_share
            near = (not stealth) and rng.random() < 0.65
            n_tx = int(rng.integers(3, 7)) if stealth else int(rng.integers(6, 15))
            uids = []
            for _ in range(n_tx):
                if near:
                    ts = cutoff - int(rng.integers(20_000, 12 * MIN_MS))
                else:
                    ts = cutoff - int(rng.integers(15 * MIN_MS, 150 * MIN_MS))
                payer = int(rng.choice(ctl))
                payee = int(rng.choice(ben))
                if stealth:
                    amt = float(round(rng.lognormal(np.log(18_000), 0.6), 2))
                elif rng.random() < 0.7:
                    amt = float(
                        rng.choice([9_900, 9_950, 9_990, 49_000, 49_500, 49_900, 99_000, 10_000, 25_000, 50_000])
                    )
                else:
                    amt = float(min(99_000, rng.lognormal(np.log(30_000), 0.5)))
                dv, ip, lat, lon, b = self._acct(payer)
                uid = len(buf.rows["ts"])
                self._add(
                    buf, ts, payer, payee, amt, "TRANSFER", "WEB", dv, ip, lat, lon, b, "settlement_manipulation", case
                )
                uids.append((uid, payer, payee, amt))
                made += 1
                self._report(payee, ts)
            for uid, payer, payee, amt in uids:
                if rng.random() < (0.3 if stealth else 0.65):
                    rts = cutoff + int(rng.integers(2 * MIN_MS, 60 * MIN_MS))
                    dv, ip, lat, lon, b = self._acct(payee)
                    self._add(
                        buf,
                        rts,
                        payee,
                        payer,
                        amt,
                        "REVERSAL",
                        "WEB",
                        dv,
                        ip,
                        lat,
                        lon,
                        b,
                        "settlement_manipulation",
                        case,
                        orig_uid=-(uid + 2),
                    )
                    made += 1
            self.cases.append(
                {
                    "case_id": case,
                    "type": "settlement_manipulation",
                    "members": [int(x) for x in ctl] + [int(x) for x in ben],
                    "fsp": self.U.fsp_ids[rogue_fsp],
                    "start_ts": cutoff - 12 * MIN_MS,
                    "end_ts": cutoff + HOUR_MS,
                    "stealth": bool(stealth),
                }
            )

    def gen_merchant_rings(self, buf: RowBuffer, target: int):
        """Colluding merchants + payers with shared devices transacting densely and mutually.

        Stealth rings use only established merchants and recruited customers on their own phones/IPs with
        everyday amounts.
        """
        cfg, rng = self.cfg, self.rng
        made, k = 0, 0
        while made < target:
            k += 1
            case = f"RNG-{k:04d}"
            t_start = int(self.start_ms + rng.integers(1, cfg.days - 3) * DAY_MS)
            t_end = min(self.end_ms - 1, t_start + int(rng.integers(2, 7)) * DAY_MS)
            stealth = rng.random() < cfg.stealth_share
            pool = self.U.new_devices(int(rng.integers(2, 5)))
            ip_pool = self.U.new_ips(int(rng.integers(1, 4)))
            cats = [CATEGORY_NAMES.index(c) for c in ("electronics", "gaming", "travel", "ecommerce")]
            n_m = int(rng.integers(3, 7))
            n_old = n_m if stealth else int(rng.binomial(n_m, 0.5))  # pre-existing merchants that turn bad
            if n_m - n_old:
                fresh = self._new_accounts(
                    n_m - n_old,
                    kind=1,
                    created_ts=t_start - int(rng.integers(1, 30)) * DAY_MS,
                    category=int(rng.choice(cats)),
                )
            else:
                fresh = np.array([], dtype=np.int64)
            merchants = np.concatenate([fresh, rng.choice(self.merch, size=n_old, replace=False)]).astype(np.int64)
            n_new = 0 if stealth else int(rng.integers(2, 6))
            if n_new:
                payers_new = self._new_accounts(
                    n_new, created_ts=t_start - int(rng.integers(1, 10)) * DAY_MS, shared_devices=pool
                )
            else:
                payers_new = np.array([], dtype=np.int64)
            recruited = self.cust[rng.integers(0, cfg.n_customers, int(rng.integers(3, 7)))]
            payers = np.concatenate([payers_new, recruited]).astype(np.int64)
            if stealth:
                payer_dev = {int(p): int(self.A.device[p]) for p in payers}
            else:
                payer_dev = {int(p): int(rng.choice(pool)) for p in payers}
            n_tx = int(rng.integers(30, 70)) if stealth else int(rng.integers(50, 140))
            members = [int(x) for x in merchants] + [int(x) for x in payers]
            for _ in range(n_tx):
                ts = int(rng.integers(t_start, t_end))
                r = rng.random()
                if r < 0.6:  # payer -> ring merchant (fake sales / cashback abuse)
                    p, q = int(rng.choice(payers)), int(rng.choice(merchants))
                    amt = float(rng.lognormal(np.log(1_500 if stealth else 6_000), 0.7))
                    if not stealth and rng.random() < 0.35:
                        amt = float(_round_to(amt, 500))
                    sc, ch = "PAYMENT", "QR"
                    _, home_ip, lat, lon, _ = self._acct(p)
                    dv = payer_dev[p]
                    ipx = int(home_ip) if stealth else int(rng.choice(ip_pool))
                    b = int(self.A.device_bound_ts[p])
                elif r < 0.9:  # merchant -> payer payouts (mutual flows)
                    p, q = int(rng.choice(merchants)), int(rng.choice(payers))
                    amt = float(rng.lognormal(np.log(1_100 if stealth else 4_500), 0.6))
                    sc, ch = "TRANSFER", "WEB"
                    dv, ipx, lat, lon, b = self._acct(p)
                    if not stealth:
                        ipx = int(rng.choice(ip_pool))
                else:  # merchant <-> merchant
                    p, q = (int(x) for x in rng.choice(merchants, 2, replace=False))
                    amt = float(rng.lognormal(np.log(4_000 if stealth else 15_000), 0.5))
                    sc, ch = "TRANSFER", "WEB"
                    dv, ipx, lat, lon, b = self._acct(p)
                self._add(buf, ts, p, q, min(amt, 100_000.0), sc, ch, dv, ipx, lat, lon, b, "merchant_fraud_ring", case)
                made += 1
            for m in merchants:
                self._report(m, t_end)
            self.cases.append(
                {
                    "case_id": case,
                    "type": "merchant_fraud_ring",
                    "members": members,
                    "devices": [int(d) for d in pool] if not stealth else [],
                    "start_ts": t_start,
                    "end_ts": t_end,
                    "stealth": bool(stealth),
                }
            )

    def gen_laundering(self, buf: RowBuffer, target: int):
        """Circular flows and fan-out/fan-in layering through mule accounts with rapid pass-through.

        Stealth cases move smaller amounts through recruited (established) mules with hours-long hops.
        """
        cfg, rng = self.cfg, self.rng
        made, k = 0, 0
        while made < target:
            k += 1
            case = f"AML-{k:04d}"
            t0 = int(self.start_ms + rng.integers(1, cfg.days) * DAY_MS + rng.integers(0, DAY_MS))
            if t0 >= self.end_ms - 6 * HOUR_MS:
                continue
            drift = (t0 - self.start_ms) >= cfg.drift_start_day * DAY_MS and rng.random() < 0.5
            stealth = rng.random() < cfg.stealth_share
            hop = (30 * MIN_MS, 150 * MIN_MS) if drift else (2 * MIN_MS, 45 * MIN_MS)
            if stealth:
                hop = (60 * MIN_MS, 6 * HOUR_MS)
            mule_devs = self.U.new_devices(int(rng.integers(1, 4))) if (not stealth and rng.random() < 0.4) else None
            rec = 0.9 if stealth else 0.5
            members: list[int] = []
            if rng.random() < 0.4:  # circular A -> B -> C (-> D ...) -> A
                L = int(rng.integers(3, 7))
                ring = self._mules(L, t0 - int(rng.integers(1, 30)) * DAY_MS, shared_devices=mule_devs, recruited=rec)
                if rng.random() < 0.4:
                    ring[0] = int(self.cust[rng.integers(0, cfg.n_customers)])
                members = [int(x) for x in ring]
                amt = float(rng.uniform(3_000, 15_000)) if stealth else float(rng.uniform(20_000, 90_000))
                ts = t0
                for _rep in range(int(rng.integers(1, 4))):
                    a = amt
                    for i in range(L):
                        p, q = int(ring[i]), int(ring[(i + 1) % L])
                        ts += int(rng.integers(*hop))
                        if ts >= self.end_ms:
                            break
                        dv, ip, lat, lon, b = self._acct(p)
                        self._add(
                            buf, ts, p, q, a, "TRANSFER", "APP", dv, ip, lat, lon, b, "transaction_laundering", case
                        )
                        made += 1
                        a *= rng.uniform(0.97, 0.995)
                    ts += int(rng.integers(HOUR_MS, 6 * HOUR_MS))
                sub = "cycle"
            else:  # fan-out -> mules -> fan-in collectors -> cash-out
                if rng.random() < 0.5:
                    src = int(self.cust[rng.integers(0, cfg.n_customers)])
                else:
                    src = int(self._new_accounts(1, created_ts=t0 - 20 * DAY_MS)[0])
                m = int(rng.integers(3, 6)) if stealth else int(rng.integers(5, 13))
                mules = self._mules(m, t0 - int(rng.integers(1, 30)) * DAY_MS, shared_devices=mule_devs, recruited=rec)
                collectors = self._mules(
                    int(rng.integers(1, 3)), t0 - int(rng.integers(1, 30)) * DAY_MS, recruited=rec * 0.5
                )
                cashout = int(self._new_accounts(1, created_ts=t0 - 40 * DAY_MS)[0])
                members = [src] + [int(x) for x in mules] + [int(x) for x in collectors] + [cashout]
                total = float(rng.uniform(10_000, 40_000)) if stealth else float(rng.uniform(60_000, 400_000))
                coll_in = {int(c): 0.0 for c in collectors}
                last_ts = t0
                for mu in mules:
                    ts = t0 + int(rng.integers(0, (120 if stealth else 15) * MIN_MS))
                    a = min(99_000.0, total / m * rng.uniform(0.8, 1.2))
                    if not stealth and rng.random() < 0.5 and a > 10_000:
                        a = float(rng.choice([9_000, 9_500, 9_900, 49_000]))
                    dv, ip, lat, lon, b = self._acct(src)
                    self._add(
                        buf, ts, src, int(mu), a, "TRANSFER", "APP", dv, ip, lat, lon, b, "transaction_laundering", case
                    )
                    c = int(rng.choice(collectors))
                    ts2 = ts + int(rng.integers(*hop))
                    a2 = a * rng.uniform(0.9, 0.98)
                    dv, ip, lat, lon, b = self._acct(int(mu))
                    self._add(
                        buf, ts2, int(mu), c, a2, "TRANSFER", "APP", dv, ip, lat, lon, b, "transaction_laundering", case
                    )
                    coll_in[c] += a2
                    last_ts = max(last_ts, ts2)
                    made += 2
                for c, v in coll_in.items():
                    if v <= 0:
                        continue
                    ts3 = last_ts + int(rng.integers(*hop))
                    remaining = v
                    while remaining > 1:
                        a3 = min(remaining * rng.uniform(0.95, 0.99), 100_000.0)
                        dv, ip, lat, lon, b = self._acct(c)
                        self._add(
                            buf,
                            ts3,
                            c,
                            cashout,
                            a3,
                            "TRANSFER",
                            "APP",
                            dv,
                            ip,
                            lat,
                            lon,
                            b,
                            "transaction_laundering",
                            case,
                        )
                        made += 1
                        remaining -= a3 / 0.97
                        ts3 += int(rng.integers(MIN_MS, 5 * MIN_MS))
                    self._report(c, ts3)
                sub = "fan_out_fan_in"
            for mm in members[1:3]:
                self._report(mm, t0 + HOUR_MS)
            self.cases.append(
                {
                    "case_id": case,
                    "type": "transaction_laundering",
                    "subtype": sub,
                    "members": members,
                    "start_ts": t0,
                    "end_ts": t0 + 12 * HOUR_MS,
                    "drift_variant": bool(drift),
                    "stealth": bool(stealth),
                }
            )

    # ================================================================ orchestration
    def run(self) -> dict:
        cfg, rng = self.cfg, self.rng
        t_start = time.time()
        self.build_entities()
        extra = self.gen_salary_rent_b2b()
        n_fraud_target = int(cfg.n_transactions * cfg.fraud_rate)
        n_normal = cfg.n_transactions - len(extra) - n_fraud_target
        normal = self.gen_normal(int(n_normal * 0.995))
        base = pd.concat([normal, extra], ignore_index=True)
        base["uid"] = np.arange(len(base))
        refunds = self.gen_refunds(base)

        buf = RowBuffer()
        mix = cfg.typology_mix
        self.gen_identity_spoofing(buf, int(n_fraud_target * mix["identity_spoofing"]))
        self.gen_settlement_manipulation(buf, int(n_fraud_target * mix["settlement_manipulation"]))
        self.gen_merchant_rings(buf, int(n_fraud_target * mix["merchant_fraud_ring"]))
        self.gen_laundering(buf, int(n_fraud_target * mix["transaction_laundering"]))
        fraud = buf.frame()
        fraud["uid"] = np.arange(len(base), len(base) + len(fraud))
        # REVERSAL rows reference fraud rows by local index encoded as -(idx+2)
        rev = fraud["orig_uid"] <= -2
        fraud.loc[rev, "orig_uid"] = (-fraud.loc[rev, "orig_uid"] - 2) + len(base)
        refunds["uid"] = np.arange(len(base) + len(fraud), len(base) + len(fraud) + len(refunds))

        df = pd.concat([base, fraud, refunds], ignore_index=True)
        df = df[(df["ts"] >= self.start_ms) & (df["ts"] < self.end_ms) & (df["payer"] != df["payee"])]
        df = df.sort_values(["ts", "uid"], kind="stable").reset_index(drop=True)
        # GPS / IP-geolocation jitter on every event (~1 km) so no population has exact repeated coordinates
        df["lat"] = df["lat"].to_numpy() + rng.normal(0, 0.01, len(df))
        df["lon"] = df["lon"].to_numpy() + rng.normal(0, 0.01, len(df))

        # observed labels with noise
        is_fraud = df["is_fraud"].to_numpy()
        label = is_fraud.copy()
        fn = (is_fraud == 1) & (rng.random(len(df)) < cfg.label_noise_fn)
        fp = (is_fraud == 0) & (rng.random(len(df)) < cfg.label_noise_fp)
        label[fn] = 0
        label[fp] = 1
        df["label"] = label

        A = self.A = self.U.freeze()
        ids = np.array(A.ids, dtype=object)
        aliases = np.array(A.alias, dtype=object)
        kinds = np.array(["CONSUMER", "MERCHANT", "BUSINESS"], dtype=object)
        cats = np.array([""] + CATEGORY_NAMES, dtype=object)
        fsps = np.array(self.U.fsp_ids, dtype=object)
        # deterministic UUIDs
        raw = rng.bytes(16 * len(df))
        tids = [str(uuid.UUID(bytes=raw[i * 16 : (i + 1) * 16], version=4)) for i in range(len(df))]
        raw = rng.bytes(16 * len(df))
        qids = [str(uuid.UUID(bytes=raw[i * 16 : (i + 1) * 16], version=4)) for i in range(len(df))]
        uid_to_tid = dict(zip(df["uid"].to_numpy().tolist(), tids))
        payer, payee = df["payer"].to_numpy(), df["payee"].to_numpy()
        out = pd.DataFrame(
            {
                "transfer_id": tids,
                "quote_id": qids,
                "ts": df["ts"].astype(np.int64),
                "payer_id": ids[payer],
                "payer_alias": aliases[payer],
                "payer_fsp": fsps[A.fsp[payer]],
                "payer_type": kinds[A.kind[payer]],
                "payee_id": ids[payee],
                "payee_alias": aliases[payee],
                "payee_fsp": fsps[A.fsp[payee]],
                "payee_type": kinds[A.kind[payee]],
                "amount": df["amount"].round(2),
                "currency": cfg.currency,
                "scenario": df["scenario"],
                "initiator": np.where(df["scenario"].isin(["REFUND", "REVERSAL"]), "PAYEE", "PAYER"),
                "channel": df["channel"],
                "device_id": ["dev-" + format(int(d), "07x") for d in df["device"].to_numpy()],
                "ip": [f"10.{(i >> 16) & 255}.{(i >> 8) & 255}.{i & 255}" for i in df["ip"].to_numpy()],
                "lat": df["lat"].round(5),
                "lon": df["lon"].round(5),
                "device_bound_at": df["device_bound_ts"].astype(np.int64),
                "merchant_category": cats[A.category[payee] + 1],
                "original_transfer_id": [
                    uid_to_tid.get(int(u), "") if u >= 0 else "" for u in df["orig_uid"].to_numpy()
                ],
                "label": df["label"].astype(np.int8),
                "is_fraud": df["is_fraud"].astype(np.int8),
                "fraud_type": df["fraud_type"],
                "case_id": df["case_id"],
            }
        )
        reports = pd.DataFrame(self.reports, columns=["account", "report_ts"])
        reports = reports[reports["report_ts"] < self.end_ms]
        reports = pd.DataFrame(
            {"account_id": ids[reports["account"].to_numpy()], "report_ts": reports["report_ts"].astype(np.int64)}
        ).sort_values("report_ts")
        for c in self.cases:
            c["members"] = [str(ids[m]) for m in c["members"]]
            if "devices" in c:
                c["devices"] = ["dev-" + format(int(d), "07x") for d in c["devices"]]
        accounts = pd.DataFrame(
            {
                "account_id": ids,
                "alias": aliases,
                "kind": kinds[A.kind],
                "fsp": fsps[A.fsp],
                "city": [CITIES[c][0] for c in A.city],
                "created_ts": A.created_ts,
                "merchant_category": cats[A.category + 1],
            }
        )
        self.elapsed = time.time() - t_start
        return {"transactions": out, "reports": reports, "accounts": accounts, "cases": self.cases}


def data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df[["transfer_id", "ts", "amount", "label"]], index=False).values.tobytes())
    return h.hexdigest()[:16]


def write_outputs(res: dict, cfg: GenConfig, out_dir: Path, elapsed: float) -> dict:
    from pipeline.mojaloop_adapter import to_fspiop

    out_dir.mkdir(parents=True, exist_ok=True)
    tx = res["transactions"]
    tx.to_parquet(out_dir / "transactions.parquet", index=False)
    res["reports"].to_parquet(out_dir / "fraud_reports.parquet", index=False)
    res["accounts"].to_parquet(out_dir / "accounts.parquet", index=False)
    (out_dir / "rings_truth.json").write_text(json.dumps(res["cases"], indent=1))
    with open(out_dir / "sample_fspiop_events.jsonl", "w") as f:
        for rec in tx.head(1000).to_dict("records"):
            f.write(json.dumps(to_fspiop(rec)) + "\n")
    fr = tx[tx["is_fraud"] == 1]
    manifest = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "n_transactions": int(len(tx)),
        "n_accounts": int(len(res["accounts"])),
        "fraud_rows": int(len(fr)),
        "fraud_rate": round(float(len(fr) / len(tx)), 5),
        "observed_label_rate": round(float(tx["label"].mean()), 5),
        "by_typology": fr["fraud_type"].value_counts().to_dict(),
        "n_cases": len(res["cases"]),
        "n_reports": int(len(res["reports"])),
        "start_ts": int(tx["ts"].min()),
        "end_ts": int(tx["ts"].max()),
        "data_hash": data_hash(tx),
        "generation_seconds": round(elapsed, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-transactions", type=int, default=1_000_000)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--fraud-rate", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args(argv)
    cfg = GenConfig(seed=a.seed, n_transactions=a.n_transactions, days=a.days, fraud_rate=a.fraud_rate)
    if cfg.drift_start_day >= cfg.days:
        cfg.drift_start_day = int(cfg.days * 0.8)
    t = time.time()
    res = Generator(cfg).run()
    manifest = write_outputs(res, cfg, a.out, time.time() - t)
    print(json.dumps({k: v for k, v in manifest.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
