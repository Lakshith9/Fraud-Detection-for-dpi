"""Entity universe for the synthetic payment network: FSPs, customers, merchants, businesses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
MIN_MS = 60_000

# (name, lat, lon, population weight)
CITIES = [
    ("Mumbai", 19.076, 72.8777, 12.4),
    ("Delhi", 28.7041, 77.1025, 11.0),
    ("Bengaluru", 12.9716, 77.5946, 8.4),
    ("Hyderabad", 17.385, 78.4867, 6.8),
    ("Ahmedabad", 23.0225, 72.5714, 5.6),
    ("Chennai", 13.0827, 80.2707, 7.0),
    ("Kolkata", 22.5726, 88.3639, 4.5),
    ("Pune", 18.5204, 73.8567, 3.1),
    ("Jaipur", 26.9124, 75.7873, 3.0),
    ("Lucknow", 26.8467, 80.9462, 2.8),
    ("Kanpur", 26.4499, 80.3319, 2.7),
    ("Nagpur", 21.1458, 79.0882, 2.4),
    ("Indore", 22.7196, 75.8577, 2.0),
    ("Bhopal", 23.2599, 77.4126, 1.8),
    ("Patna", 25.5941, 85.1376, 1.7),
    ("Vadodara", 22.3072, 73.1812, 1.7),
    ("Ludhiana", 30.901, 75.8573, 1.6),
    ("Agra", 27.1767, 78.0081, 1.6),
    ("Nashik", 19.9975, 73.7898, 1.5),
    ("Coimbatore", 11.0168, 76.9558, 1.6),
    ("Kochi", 9.9312, 76.2673, 1.2),
    ("Visakhapatnam", 17.6868, 83.2185, 1.7),
    ("Guwahati", 26.1445, 91.7362, 1.0),
    ("Chandigarh", 30.7333, 76.7794, 1.0),
    ("Thiruvananthapuram", 8.5241, 76.9366, 0.9),
    ("Bhubaneswar", 20.2961, 85.8245, 0.9),
    ("Mysuru", 12.2958, 76.6394, 0.9),
    ("Ranchi", 23.3441, 85.3096, 1.1),
    ("Surat", 21.1702, 72.8311, 4.5),
    ("Varanasi", 25.3176, 82.9739, 1.2),
]

# name: (popularity weight, amount multiplier, lognormal sigma, refund rate, round-amount prob)
MERCHANT_CATEGORIES = {
    "grocery": (0.24, 1.0, 0.6, 0.002, 0.05),
    "food": (0.18, 0.6, 0.5, 0.004, 0.05),
    "fuel": (0.10, 2.0, 0.4, 0.001, 0.35),
    "pharmacy": (0.06, 0.9, 0.6, 0.003, 0.05),
    "utilities": (0.08, 3.0, 0.5, 0.002, 0.40),
    "ecommerce": (0.12, 2.2, 0.8, 0.030, 0.05),
    "electronics": (0.05, 8.0, 0.8, 0.015, 0.15),
    "travel": (0.05, 6.0, 0.7, 0.020, 0.10),
    "education": (0.04, 10.0, 0.6, 0.005, 0.60),
    "gaming": (0.04, 1.5, 0.9, 0.010, 0.30),
    "telecom": (0.04, 1.2, 0.3, 0.002, 0.50),
}
CATEGORY_NAMES = list(MERCHANT_CATEGORIES)


@dataclass
class Accounts:
    """Column-oriented account table. Index = account row; ids are strings."""

    ids: list[str]
    kind: np.ndarray  # 0 consumer, 1 merchant, 2 business
    fsp: np.ndarray  # int fsp index
    alias: list[str]
    city: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    created_ts: np.ndarray  # epoch ms
    device: np.ndarray  # primary device index
    device_bound_ts: np.ndarray
    home_ip: np.ndarray  # ip index
    category: np.ndarray  # merchant category index or -1

    def __len__(self) -> int:
        return len(self.ids)


class Universe:
    """Holds all entities and allocators for new (fraud-created) accounts, devices and IPs."""

    def __init__(self, cfg, rng: np.random.Generator, start_ms: int):
        self.cfg = cfg
        self.rng = rng
        self.start_ms = start_ms
        self.end_ms = start_ms + cfg.days * DAY_MS
        self.fsp_ids = [f"dfsp-{i + 1:02d}" for i in range(cfg.n_fsps)]
        w = 1.0 / np.arange(1, cfg.n_fsps + 1) ** 0.9
        self.fsp_weights = w / w.sum()
        # "lax" FSPs with weaker onboarding host a larger share of mule accounts
        self.lax_fsps = [min(6, cfg.n_fsps - 1), min(10, cfg.n_fsps - 1)]
        cw = np.array([c[3] for c in CITIES])
        self.city_weights = cw / cw.sum()
        self.city_lat = np.array([c[1] for c in CITIES])
        self.city_lon = np.array([c[2] for c in CITIES])
        self.n_devices = 0
        self.n_ips = 0
        self._cols: dict[str, list] = {k: [] for k in Accounts.__dataclass_fields__}
        self._counter = {"cust": 0, "merch": 0, "biz": 0}

    # ----------------------------------------------------------------- allocators
    def new_devices(self, n: int) -> np.ndarray:
        out = np.arange(self.n_devices, self.n_devices + n)
        self.n_devices += n
        return out

    def new_ips(self, n: int) -> np.ndarray:
        out = np.arange(self.n_ips, self.n_ips + n)
        self.n_ips += n
        return out

    def add_accounts(
        self,
        n: int,
        kind: int,
        fsp=None,
        city=None,
        created_ts=None,
        device=None,
        device_bound_ts=None,
        home_ip=None,
        category=None,
    ) -> np.ndarray:
        rng = self.rng
        first = len(self._cols["ids"])
        prefix = {0: "cust", 1: "merch", 2: "biz"}[kind]
        ids = []
        aliases = []
        for _ in range(n):
            self._counter[prefix] += 1
            c = self._counter[prefix]
            ids.append(f"{prefix}-{c:06d}")
            if kind == 0:
                aliases.append("+91" + str(rng.integers(6_000_000_000, 9_999_999_999)))
            else:
                aliases.append(f"{prefix}{c:06d}@upi")
        fsp = rng.choice(len(self.fsp_ids), size=n, p=self.fsp_weights) if fsp is None else np.broadcast_to(fsp, n)
        city = rng.choice(len(CITIES), size=n, p=self.city_weights) if city is None else np.broadcast_to(city, n)
        lat = self.city_lat[city] + rng.normal(0, 0.03, n)
        lon = self.city_lon[city] + rng.normal(0, 0.03, n)
        if created_ts is None:
            created_ts = self.start_ms - rng.integers(30, 2000, n) * DAY_MS
        created_ts = np.broadcast_to(created_ts, n)
        device = self.new_devices(n) if device is None else np.broadcast_to(device, n)
        if device_bound_ts is None:
            device_bound_ts = np.minimum(
                created_ts + rng.integers(0, 400, n) * DAY_MS, self.start_ms - rng.integers(1, 30, n) * DAY_MS
            )
            device_bound_ts = np.maximum(device_bound_ts, created_ts)
        device_bound_ts = np.broadcast_to(device_bound_ts, n)
        home_ip = self.new_ips(n) if home_ip is None else np.broadcast_to(home_ip, n)
        category = np.full(n, -1) if category is None else np.broadcast_to(category, n)
        cols = self._cols
        cols["ids"].extend(ids)
        cols["alias"].extend(aliases)
        cols["kind"].extend([kind] * n)
        for k, v in (
            ("fsp", fsp),
            ("city", city),
            ("lat", lat),
            ("lon", lon),
            ("created_ts", created_ts),
            ("device", device),
            ("device_bound_ts", device_bound_ts),
            ("home_ip", home_ip),
            ("category", category),
        ):
            cols[k].extend(np.asarray(v).tolist())
        return np.arange(first, first + n)

    def freeze(self) -> Accounts:
        c = self._cols
        return Accounts(
            ids=list(c["ids"]),
            alias=list(c["alias"]),
            kind=np.array(c["kind"], dtype=np.int8),
            fsp=np.array(c["fsp"], dtype=np.int16),
            city=np.array(c["city"], dtype=np.int16),
            lat=np.array(c["lat"]),
            lon=np.array(c["lon"]),
            created_ts=np.array(c["created_ts"], dtype=np.int64),
            device=np.array(c["device"], dtype=np.int64),
            device_bound_ts=np.array(c["device_bound_ts"], dtype=np.int64),
            home_ip=np.array(c["home_ip"], dtype=np.int64),
            category=np.array(c["category"], dtype=np.int16),
        )

    @property
    def n_accounts(self) -> int:
        return len(self._cols["ids"])

    def col(self, name: str, idx):
        return np.asarray(self._cols[name])[idx] if not isinstance(idx, (int, np.integer)) else self._cols[name][idx]
