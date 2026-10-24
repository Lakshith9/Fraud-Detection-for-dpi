"""Single source of truth for feature engineering (training replay AND online serving).

Flow per transaction (identical offline and online):
    plan    = engine.read_plan(tx)          # Redis-style read commands (one pipeline round trip)
    results = store.execute(plan)           # InMemoryStore (training) or RedisStore (serving)
    fv      = engine.compute(tx, results)   # pure function -> tabular, graph and sequence features
    store.execute(engine.write_plan(tx))    # state updates (after scoring)

All windows use EVENT time (tx["ts"], epoch ms), so replaying at any speed yields identical features.
Every feature is documented in docs/feature_dictionary.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN = 60_000
HOUR = 3_600_000
DAY = 86_400_000
HIST_LEN = 50  # payer history entries kept online
SEQ_LEN = 16  # sequence model looks at the last 15 transactions + the current one
FSP_BUCKET_MS = 30 * MIN
FSP_BASELINE_BUCKETS = 48  # 24h of 30-min buckets
STRUCTURING_BANDS = ((9_000, 10_000), (45_000, 50_000), (90_000, 100_000))
CHANNELS = {"APP": 0, "QR": 1, "WEB": 2, "USSD": 3}
NAN = float("nan")

TABULAR_FEATURES = [
    # amount & payer behaviour
    "amount_log",
    "amount_zscore_payer",
    "amount_ratio_mean_30d",
    "amount_ratio_max_30d",
    "payer_hist_count_30d",
    "is_round_amount",
    "round_level",
    "near_threshold",
    # velocity
    "payer_tx_1m",
    "payer_tx_1h",
    "payer_tx_24h",
    "payer_amt_1h_log",
    "payee_tx_1h",
    "payee_tx_24h",
    "device_tx_1h",
    "device_tx_24h",
    # counterparty
    "new_payee",
    "payer_out_degree",
    "payee_in_degree",
    "payee_out_degree",
    "payee_age_hours_log",
    "payer_age_hours_log",
    "payee_report_count",
    "payer_report_count",
    "counterparty_risk_score",
    # time
    "hour_of_day",
    "is_night",
    "hour_deviation",
    "is_weekend",
    "dow_deviation",
    "secs_since_last_tx_log",
    # device / ip / geo
    "device_accounts",
    "ip_accounts",
    "new_device_for_payer",
    "payer_device_count",
    "device_binding_age_hours_log",
    "geo_distance_km",
    "geo_speed_kmh",
    # reversals / pass-through / settlement / fsp
    "payee_reversal_rate",
    "payer_reversal_rate",
    "is_reversal",
    "is_refund",
    "inflow_24h_log",
    "pass_through_ratio_24h",
    "mins_since_last_inflow_log",
    "mins_to_settlement_cutoff",
    "near_settlement_cutoff",
    "fsp_rhv_zscore",
    "fsp_volume_zscore",
    # categorical-ish
    "channel_code",
    "payee_is_merchant",
    "payer_is_business",
    "cross_fsp",
]
GRAPH_FEATURES = [
    "payer_ring_score",
    "payee_ring_score",
    "device_ring_score",
    "payer_fraud_neighbor_ratio",
    "payee_fraud_neighbor_ratio",
    "max_ring_size",
    "in_cycle",
]
SEQ_FEATURES = [
    "amt_log",
    "amt_vs_median",
    "hour_sin",
    "hour_cos",
    "gap_log",
    "new_payee_in_seq",
    "device_change",
    "geo_jump_log",
    "round_flag",
    "payee_merchant",
    "is_current",
    "present",
]
SEQ_DIM = len(SEQ_FEATURES)


@dataclass
class FeatureVector:
    tabular: list[float]
    graph: list[float]
    seq: np.ndarray  # (SEQ_LEN, SEQ_DIM) float32
    context: dict  # raw facts used for plain-language explanations


def _f(x, default=0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(a)))


def round_level(amount: float) -> int:
    if amount <= 0 or abs(amount - round(amount)) > 1e-6:
        return 0
    a, lvl = int(round(amount)), 0
    while lvl < 5 and a % 10 == 0:
        a //= 10
        lvl += 1
    return lvl


def is_round_high_value(amount: float) -> bool:
    return amount >= 5_000 and (round_level(amount) >= 3 or any(lo <= amount < hi for lo, hi in STRUCTURING_BANDS))


def _node(kind: str, ident: str) -> str:
    return f"{kind}:{ident}"


def encode_hist(tx: dict) -> str:
    lat, lon = tx.get("lat"), tx.get("lon")
    geo = (
        f"{lat:.4f}|{lon:.4f}"
        if lat is not None and lon is not None and not (isinstance(lat, float) and math.isnan(lat))
        else "|"
    )
    kind = "M" if tx.get("payee_type") == "MERCHANT" else "C"
    rev = "R" if tx.get("scenario") in ("REFUND", "REVERSAL") else "-"
    return f"{int(tx['ts'])}|{float(tx['amount']):.2f}|{tx['payee_id']}|{tx.get('device_id') or ''}|{geo}|{kind}|{rev}"


def decode_hist(s: str):
    ts, amt, payee, dev, lat, lon, kind, rev = s.split("|")
    return (int(ts), float(amt), payee, dev, float(lat) if lat else None, float(lon) if lon else None, kind, rev)


class FeatureEngine:
    """Builds read/write command plans and computes features from their results."""

    def __init__(self, settlement_cutoff_hours=(0, 6, 12, 18)):
        self.cutoffs = sorted(int(h) for h in settlement_cutoff_hours)

    # ------------------------------------------------------------------ keys
    @staticmethod
    def keys(tx: dict) -> dict:
        p, q = tx["payer_id"], tx["payee_id"]
        d = tx.get("device_id") or "unknown-device"
        ip = tx.get("ip") or "unknown-ip"
        f = tx.get("payer_fsp") or "unknown-fsp"
        return {
            "ph": f"ph:{p}",
            "zq": f"zt:q:{q}",
            "zd": f"zt:d:{d}",
            "in_q": f"in:{q}",
            "out_q": f"out:{q}",
            "out_p": f"out:{p}",
            "dacc": f"dacc:{d}",
            "pdev": f"pdev:{p}",
            "iacc": f"iacc:{ip}",
            "inf_p": f"inf:{p}",
            "inf_q": f"inf:{q}",
            "cnt_p": f"cnt:{p}",
            "cnt_q": f"cnt:{q}",
            "g_p": f"g:{_node('a', p)}",
            "g_q": f"g:{_node('a', q)}",
            "g_d": f"g:{_node('d', d)}",
            "fspn": f"fspn:{f}",
            "fspr": f"fspr:{f}",
            "dev": d,
            "ip": ip,
        }

    # ------------------------------------------------------------------ read plan
    def read_plan(self, tx: dict) -> list[tuple]:
        k = self.keys(tx)
        ts = int(tx["ts"])
        b = ts // FSP_BUCKET_MS
        buckets = [str(b - i) for i in range(FSP_BASELINE_BUCKETS + 1)]
        return [
            ("lrange", k["ph"], 0, HIST_LEN - 1),  # 0
            ("zcount", k["zq"], ts - HOUR + 1, ts),  # 1
            ("zcount", k["zq"], ts - DAY + 1, ts),  # 2
            ("zcount", k["zd"], ts - HOUR + 1, ts),  # 3
            ("zcount", k["zd"], ts - DAY + 1, ts),  # 4
            ("scard", k["in_q"]),  # 5
            ("scard", k["out_q"]),  # 6
            ("scard", k["out_p"]),  # 7
            ("sismember", k["out_p"], tx["payee_id"]),  # 8
            ("scard", k["dacc"]),  # 9
            ("sismember", k["pdev"], k["dev"]),  # 10
            ("scard", k["pdev"]),  # 11
            ("scard", k["iacc"]),  # 12
            ("sismember", k["iacc"], tx["payer_id"]),  # 13
            ("zrangebyscore", k["inf_p"], ts - DAY + 1, ts),  # 14
            ("hgetall", k["cnt_p"]),  # 15
            ("hgetall", k["cnt_q"]),  # 16
            ("hgetall", k["g_p"]),  # 17
            ("hgetall", k["g_q"]),  # 18
            ("hgetall", k["g_d"]),  # 19
            ("hmget", k["fspn"], buckets),  # 20
            ("hmget", k["fspr"], buckets),  # 21
        ]

    # ------------------------------------------------------------------ write plan
    def write_plan(self, tx: dict) -> list[tuple]:
        k = self.keys(tx)
        ts = int(tx["ts"])
        tid = tx["transfer_id"]
        amt = float(tx["amount"])
        p, q = tx["payer_id"], tx["payee_id"]
        b = ts // FSP_BUCKET_MS
        old = [str(b - FSP_BASELINE_BUCKETS - i) for i in range(1, 5)]
        plan = [
            ("lpush", k["ph"], encode_hist(tx)),
            ("ltrim", k["ph"], 0, HIST_LEN - 1),
            ("zadd", k["zq"], {tid: ts}),
            ("zremrangebyscore", k["zq"], "-inf", ts - DAY - HOUR),
            ("zadd", k["zd"], {tid: ts}),
            ("zremrangebyscore", k["zd"], "-inf", ts - DAY - HOUR),
            ("sadd", k["in_q"], p),
            ("sadd", k["out_p"], q),
            ("sadd", k["dacc"], p),
            ("sadd", k["pdev"], k["dev"]),
            ("sadd", k["iacc"], p),
            ("zadd", k["inf_q"], {f"{tid}:{ts}:{amt:.2f}": ts}),
            ("zremrangebyscore", k["inf_q"], "-inf", ts - DAY - HOUR),
            ("hsetnx", k["cnt_p"], "fs", ts),
            ("hsetnx", k["cnt_q"], "fs", ts),
            ("hincrby", k["cnt_q"], "nin", 1),
            ("hincrby", k["cnt_p"], "nout", 1),
            ("hincrby", k["fspn"], str(b), 1),
            ("hdel", k["fspn"], *old),
        ]
        if tx.get("scenario") in ("REFUND", "REVERSAL"):
            plan.append(("hincrby", k["cnt_p"], "rev", 1))
        if is_round_high_value(amt):
            plan.append(("hincrby", k["fspr"], str(b), 1))
            plan.append(("hdel", k["fspr"], *old))
        return plan

    @staticmethod
    def report_plan(account_id: str) -> list[tuple]:
        """A fraud report (registry feed or operator-confirmed fraud) against a beneficiary account."""
        return [("hincrby", f"cnt:{account_id}", "rep", 1)]

    # ------------------------------------------------------------------ compute
    def compute(self, tx: dict, r: list) -> FeatureVector:
        ts = int(tx["ts"])
        amt = float(tx["amount"])
        hist = []
        for s in r[0] or []:
            try:
                h = decode_hist(s)
            except ValueError:
                continue
            if h[0] <= ts:
                hist.append(h)
        hist.sort(key=lambda h: h[0], reverse=True)  # most recent first; robust to out-of-order arrival

        h30 = [h for h in hist if ts - h[0] <= 30 * DAY]
        amts30 = [h[1] for h in h30]
        n30 = len(amts30)
        if n30:
            mean = sum(amts30) / n30
            mx = max(amts30)
            ratio_mean = amt / max(mean, 1.0)
            ratio_max = amt / max(mx, 1.0)
        else:
            mean = mx = NAN
            ratio_mean = ratio_max = NAN
        if n30 >= 3:
            var = sum((a - mean) ** 2 for a in amts30) / (n30 - 1)
            zscore = (amt - mean) / (math.sqrt(var) + 1.0)
        else:
            zscore = NAN
        rl = round_level(amt)
        near_thr = 1.0 if any(lo <= amt < hi for lo, hi in STRUCTURING_BANDS) else 0.0

        p_1m = sum(1 for h in hist if ts - h[0] < MIN)
        p_1h = sum(1 for h in hist if ts - h[0] < HOUR)
        p_24h = sum(1 for h in hist if ts - h[0] < DAY)
        p_amt_1h = sum(h[1] for h in hist if ts - h[0] < HOUR)

        cnt_p = r[15] or {}
        cnt_q = r[16] or {}
        new_payee = 0.0 if r[8] else 1.0
        payee_fs = cnt_q.get("fs")
        payer_fs = cnt_p.get("fs")
        payee_age_h = max(0.0, (ts - int(payee_fs)) / HOUR) if payee_fs is not None else 0.0
        payer_age_h = max(0.0, (ts - int(payer_fs)) / HOUR) if payer_fs is not None else 0.0
        rep_q = _f(cnt_q.get("rep"))
        rep_p = _f(cnt_p.get("rep"))
        payee_rev_rate = _f(cnt_q.get("rev")) / (_f(cnt_q.get("nin")) + 1.0)
        payer_rev_rate = _f(cnt_p.get("rev")) / (_f(cnt_p.get("nin")) + _f(cnt_p.get("nout")) + 1.0)
        young_payee = 1.0 if payee_fs is None or payee_age_h < 72 else 0.0
        crs = 1.0 - math.exp(-rep_q) * (1.0 - min(1.0, 5.0 * payee_rev_rate)) * (1.0 - 0.5 * young_payee)

        # time features (UTC event time; production would use the switch's local timezone)
        hour = (ts % DAY) / HOUR
        dow = ((ts // DAY) + 3) % 7  # 1970-01-01 was a Thursday -> Monday = 0
        is_weekend = 1.0 if dow >= 5 else 0.0
        if hist:
            angs = [2 * math.pi * ((h[0] % DAY) / DAY) for h in hist]
            sx, cx = sum(math.sin(a) for a in angs), sum(math.cos(a) for a in angs)
            mean_hour = (math.atan2(sx, cx) / (2 * math.pi) * 24) % 24
            d = abs(hour - mean_hour)
            hour_dev = min(d, 24 - d)
            same = sum(1 for h in hist if (((h[0] // DAY) + 3) % 7 >= 5) == (dow >= 5))
            dow_dev = 1.0 - same / len(hist)
            last = hist[0]
            secs_since = max(0.0, (ts - last[0]) / 1000)
            secs_since_log = math.log1p(secs_since)
        else:
            hour_dev = dow_dev = secs_since_log = NAN
            last = None
            secs_since = NAN

        dev_accounts = float(r[9]) + (0.0 if r[10] else 1.0)
        ip_accounts = float(r[12]) + (0.0 if r[13] else 1.0)
        new_device = 0.0 if r[10] else 1.0
        payer_dev_count = float(r[11]) + new_device
        bound = tx.get("device_bound_at")
        bind_age = math.log1p(max(0.0, (ts - int(bound)) / HOUR)) if bound not in (None, "", 0) else NAN

        lat, lon = tx.get("lat"), tx.get("lon")
        geo_d = geo_v = NAN
        if last is not None and last[4] is not None and lat is not None and lon is not None:
            geo_d = haversine_km(last[4], last[5], float(lat), float(lon))
            hours = max((ts - last[0]) / HOUR, 1 / 60)
            geo_v = min(geo_d / hours, 2000.0)

        inflows = []
        for m in r[14] or []:
            try:
                inflows.append(float(m.rsplit(":", 1)[1]))
            except (IndexError, ValueError):
                continue
        in_24h = sum(inflows)
        out_24h = sum(h[1] for h in hist if ts - h[0] < DAY) + amt
        pass_through = min(out_24h / in_24h, 5.0) if in_24h > 0 else 0.0
        mins_since_inflow = NAN
        if r[14]:
            mins_since_inflow = math.log1p(max(0.0, (ts - self._last_inflow_ts(r[14], ts)) / MIN))

        mins_to_cutoff = self.minutes_to_cutoff(ts)

        def zs(vals):
            cur = _f(vals[0])
            base = [_f(v) for v in vals[1:]]
            mu = sum(base) / len(base)
            sd = math.sqrt(sum((x - mu) ** 2 for x in base) / len(base))
            return (cur + 1.0 - mu) / (sd + 1.0)

        fsp_vol_z = zs(r[20])
        fsp_rhv_z = zs(r[21]) if is_round_high_value(amt) else 0.0

        payee_is_m = 1.0 if tx.get("payee_type") == "MERCHANT" else 0.0
        payer_is_b = 1.0 if tx.get("payer_type") in ("BUSINESS", "MERCHANT") else 0.0

        tab = [
            math.log1p(amt),
            zscore,
            ratio_mean,
            ratio_max,
            float(n30),
            1.0 if rl >= 2 else 0.0,
            float(rl),
            near_thr,
            float(p_1m),
            float(p_1h),
            float(p_24h),
            math.log1p(p_amt_1h),
            float(r[1]),
            float(r[2]),
            float(r[3]),
            float(r[4]),
            new_payee,
            float(r[7]),
            float(r[5]),
            float(r[6]),
            math.log1p(payee_age_h),
            math.log1p(payer_age_h),
            rep_q,
            rep_p,
            crs,
            hour,
            1.0 if hour < 5 else 0.0,
            hour_dev,
            is_weekend,
            dow_dev,
            secs_since_log,
            dev_accounts,
            ip_accounts,
            new_device,
            payer_dev_count,
            bind_age,
            geo_d,
            geo_v,
            payee_rev_rate,
            payer_rev_rate,
            1.0 if tx.get("scenario") == "REVERSAL" else 0.0,
            1.0 if tx.get("scenario") == "REFUND" else 0.0,
            math.log1p(in_24h),
            pass_through,
            mins_since_inflow,
            mins_to_cutoff,
            1.0 if mins_to_cutoff <= 15 else 0.0,
            fsp_rhv_z,
            fsp_vol_z,
            float(CHANNELS.get(tx.get("channel") or "", 4)),
            payee_is_m,
            payer_is_b,
            1.0 if tx.get("payer_fsp") != tx.get("payee_fsp") else 0.0,
        ]

        gp, gq, gd = r[17] or {}, r[18] or {}, r[19] or {}
        graph = [
            _f(gp.get("rs")),
            _f(gq.get("rs")),
            _f(gd.get("rs")),
            _f(gp.get("fnr")),
            _f(gq.get("fnr")),
            max(_f(gp.get("sz")), _f(gq.get("sz")), _f(gd.get("sz"))),
            1.0 if (_f(gp.get("cyc")) > 0 or _f(gq.get("cyc")) > 0) else 0.0,
        ]

        seq = self._sequence(tx, hist)
        context = {
            "amount": amt,
            "payer_mean_30d": mean,
            "payer_max_30d": mx,
            "n_hist_30d": n30,
            "device_accounts": dev_accounts,
            "ip_accounts": ip_accounts,
            "geo_distance_km": geo_d,
            "secs_since_last": secs_since,
            "mins_to_cutoff": mins_to_cutoff,
            "in_24h": in_24h,
            "out_24h": out_24h,
            "payee_age_h": payee_age_h if payee_fs is not None else None,
            "payee_reports": rep_q,
            "payer_ring": {k: gp.get(k) for k in ("rs", "rid", "rt", "sz", "cyc")} if gp else None,
            "payee_ring": {k: gq.get(k) for k in ("rs", "rid", "rt", "sz", "cyc")} if gq else None,
            "device_ring": {k: gd.get(k) for k in ("rs", "rid", "rt", "sz")} if gd else None,
            "graph_ts": max(_f(gp.get("ts")), _f(gq.get("ts")), _f(gd.get("ts"))) or None,
        }
        return FeatureVector(tab, graph, seq, context)

    @staticmethod
    def _last_inflow_ts(members: list[str], ts: int) -> int:
        # inflow members are "<transfer_id>:<event_ts>:<amount>" and ZRANGEBYSCORE returns them in score order
        try:
            return int(members[-1].split(":")[-2])
        except (IndexError, ValueError):
            return ts

    def minutes_to_cutoff(self, ts: int) -> float:
        hour = (ts % DAY) / HOUR
        best = 24.0
        for c in self.cutoffs:
            d = (c - hour) % 24
            best = min(best, d)
        return best * 60.0

    def _sequence(self, tx: dict, hist: list) -> np.ndarray:
        steps = list(reversed(hist[: SEQ_LEN - 1]))  # chronological
        cur = (
            int(tx["ts"]),
            float(tx["amount"]),
            tx["payee_id"],
            tx.get("device_id") or "",
            tx.get("lat"),
            tx.get("lon"),
            "M" if tx.get("payee_type") == "MERCHANT" else "C",
            "-",
        )
        steps.append(cur)
        amts = sorted(s[1] for s in steps)
        median = amts[len(amts) // 2]
        out = np.zeros((SEQ_LEN, SEQ_DIM), dtype=np.float32)
        offset = SEQ_LEN - len(steps)
        seen: set[str] = set()
        prev = None
        for i, s in enumerate(steps):
            hr = (s[0] % DAY) / HOUR
            gap = math.log1p(max(0.0, (s[0] - prev[0]) / MIN)) / 10 if prev else 0.0
            jump = 0.0
            if prev and prev[4] is not None and s[4] is not None:
                jump = math.log1p(haversine_km(prev[4], prev[5], float(s[4]), float(s[5]))) / 8
            row = out[offset + i]
            row[0] = math.log1p(s[1]) / 12
            row[1] = max(-3.0, min(3.0, math.log((s[1] + 1) / (median + 1))))
            row[2] = math.sin(2 * math.pi * hr / 24)
            row[3] = math.cos(2 * math.pi * hr / 24)
            row[4] = gap
            row[5] = 0.0 if s[2] in seen else 1.0
            row[6] = 1.0 if prev and prev[3] != s[3] else 0.0
            row[7] = jump
            row[8] = 1.0 if round_level(s[1]) >= 2 else 0.0
            row[9] = 1.0 if s[6] == "M" else 0.0
            row[10] = 1.0 if i == len(steps) - 1 else 0.0
            row[11] = 1.0
            seen.add(s[2])
            prev = s
        return out
