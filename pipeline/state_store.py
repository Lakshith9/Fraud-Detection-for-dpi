"""Online feature-state stores.

Both stores execute the *same* command plans produced by ``pipeline.features.FeatureEngine``:

* ``RedisStore``     - production / serving: one pipelined round-trip per plan.
* ``InMemoryStore``  - offline training replay: a faithful in-process implementation of the small Redis
                       command subset we use (same argument conventions, same return types as redis-py with
                       ``decode_responses=True``).

Because feature computation only ever sees the *results* of a plan, train/serve parity reduces to these two
executors agreeing, which ``tests/unit/test_feature_parity.py`` verifies against a real Redis.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

Plan = list[tuple]  # [(command, *args), ...]


class InMemoryStore:
    """In-process implementation of the Redis subset used by the feature engine and graph worker."""

    def __init__(self):
        self.lists: dict[str, list[str]] = defaultdict(list)  # index 0 = head (LPUSH side)
        self.sets: dict[str, set[str]] = defaultdict(set)
        self.zsets: dict[str, tuple[list[tuple[float, str]], dict[str, float]]] = {}
        self.hashes: dict[str, dict[str, str]] = defaultdict(dict)
        self.strings: dict[str, str] = {}

    # ------------------------------------------------------------- plan execution
    def execute(self, plan: Iterable[tuple]) -> list[Any]:
        out = []
        for cmd, *args in plan:
            out.append(getattr(self, cmd)(*args))
        return out

    # ------------------------------------------------------------- lists
    def lrange(self, key, start, stop):
        lst = self.lists.get(key)
        if not lst:
            return []
        stop = len(lst) - 1 if stop == -1 else stop
        return lst[start : stop + 1]

    def lpush(self, key, *values):
        lst = self.lists[key]
        for v in values:
            lst.insert(0, v)
        return len(lst)

    def ltrim(self, key, start, stop):
        lst = self.lists.get(key)
        if lst is not None:
            del lst[stop + 1 :]
            del lst[:start]
        return True

    # ------------------------------------------------------------- sets
    def sadd(self, key, *members):
        s = self.sets[key]
        before = len(s)
        s.update(members)
        return len(s) - before

    def scard(self, key):
        s = self.sets.get(key)
        return len(s) if s else 0

    def sismember(self, key, member):
        s = self.sets.get(key)
        return bool(s) and member in s

    def smembers(self, key):
        return set(self.sets.get(key, ()))

    # ------------------------------------------------------------- sorted sets
    def _z(self, key):
        z = self.zsets.get(key)
        if z is None:
            z = ([], {})
            self.zsets[key] = z
        return z

    def zadd(self, key, mapping: dict):
        items, index = self._z(key)
        added = 0
        for member, score in mapping.items():
            score = float(score)
            old = index.get(member)
            if old is not None:
                i = bisect.bisect_left(items, (old, member))
                del items[i]
            else:
                added += 1
            index[member] = score
            entry = (score, member)
            if not items or entry >= items[-1]:
                items.append(entry)
            else:
                bisect.insort(items, entry)
        return added

    def zcount(self, key, lo, hi):
        z = self.zsets.get(key)
        if not z:
            return 0
        items = z[0]
        i = bisect.bisect_left(items, (float(lo), ""))
        j = bisect.bisect_right(items, (float(hi), "\U0010ffff"))
        return max(0, j - i)

    def zrangebyscore(self, key, lo, hi):
        z = self.zsets.get(key)
        if not z:
            return []
        items = z[0]
        i = bisect.bisect_left(items, (float(lo), ""))
        j = bisect.bisect_right(items, (float(hi), "\U0010ffff"))
        return [m for _, m in items[i:j]]

    def zremrangebyscore(self, key, lo, hi):
        z = self.zsets.get(key)
        if not z:
            return 0
        items, index = z
        lo = float("-inf") if lo == "-inf" else float(lo)
        i = bisect.bisect_left(items, (lo, ""))
        j = bisect.bisect_right(items, (float(hi), "\U0010ffff"))
        for _, m in items[i:j]:
            index.pop(m, None)
        del items[i:j]
        return j - i

    # ------------------------------------------------------------- hashes
    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hmget(self, key, fields):
        h = self.hashes.get(key, {})
        return [h.get(str(f)) for f in fields]

    def hincrby(self, key, field, amount=1):
        h = self.hashes[key]
        v = int(h.get(str(field), "0")) + int(amount)
        h[str(field)] = str(v)
        return v

    def hsetnx(self, key, field, value):
        h = self.hashes[key]
        if str(field) in h:
            return False
        h[str(field)] = str(value)
        return True

    def hset(self, key, field=None, value=None, mapping=None):
        h = self.hashes[key]
        n = 0
        if field is not None:
            n += str(field) not in h
            h[str(field)] = str(value)
        for f, v in (mapping or {}).items():
            n += str(f) not in h
            h[str(f)] = str(v)
        return n

    def hdel(self, key, *fields):
        h = self.hashes.get(key)
        if not h:
            return 0
        n = 0
        for f in fields:
            if h.pop(str(f), None) is not None:
                n += 1
        return n

    # ------------------------------------------------------------- strings / keys
    def get(self, key):
        return self.strings.get(key)

    def set(self, key, value, ex=None, nx=False):
        if nx and key in self.strings:
            return None
        self.strings[key] = str(value)
        return True

    def delete(self, *keys):
        n = 0
        for k in keys:
            for d in (self.lists, self.sets, self.zsets, self.hashes, self.strings):
                if k in d:
                    del d[k]
                    n += 1
        return n

    def expire(self, key, seconds):  # TTLs are wall-clock housekeeping only; no effect on features
        return True

    # ------------------------------------------------------------- bulk export (bootstrap into Redis)
    def dump_to_redis(self, r, batch: int = 5000) -> int:
        """Copy the entire in-memory state into a Redis instance (used to warm the online store)."""
        pipe = r.pipeline(transaction=False)
        n = 0

        def flush():
            nonlocal pipe
            pipe.execute()
            pipe = r.pipeline(transaction=False)

        for k, lst in self.lists.items():
            if lst:
                pipe.delete(k)
                pipe.rpush(k, *lst)
                n += 1
                if n % batch == 0:
                    flush()
        for k, s in self.sets.items():
            if s:
                pipe.delete(k)
                pipe.sadd(k, *s)
                n += 1
                if n % batch == 0:
                    flush()
        for k, (items, _) in self.zsets.items():
            if items:
                pipe.delete(k)
                pipe.zadd(k, {m: sc for sc, m in items})
                n += 1
                if n % batch == 0:
                    flush()
        for k, h in self.hashes.items():
            if h:
                pipe.delete(k)
                pipe.hset(k, mapping=h)
                n += 1
                if n % batch == 0:
                    flush()
        for k, v in self.strings.items():
            pipe.set(k, v)
            n += 1
            if n % batch == 0:
                flush()
        pipe.execute()
        return n


class RedisStore:
    """Executes command plans against Redis in a single non-transactional pipeline round-trip."""

    def __init__(self, client):
        self.r = client

    def execute(self, plan: Iterable[tuple]) -> list[Any]:
        pipe = self.r.pipeline(transaction=False)
        for cmd, *args in plan:
            getattr(pipe, cmd)(*args)
        return pipe.execute()

    def execute_many(self, plans: list[Plan]) -> list[list[Any]]:
        """Run several plans in ONE round-trip, returning per-plan results."""
        pipe = self.r.pipeline(transaction=False)
        sizes = []
        for plan in plans:
            sizes.append(len(plan))
            for cmd, *args in plan:
                getattr(pipe, cmd)(*args)
        flat = pipe.execute()
        out, i = [], 0
        for s in sizes:
            out.append(flat[i : i + s])
            i += s
        return out
