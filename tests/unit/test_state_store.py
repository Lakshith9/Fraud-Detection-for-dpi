from pipeline.state_store import InMemoryStore


def test_lists_push_trim_range():
    s = InMemoryStore()
    for i in range(5):
        s.lpush("k", str(i))
    s.ltrim("k", 0, 2)
    assert s.lrange("k", 0, -1) == ["4", "3", "2"]
    assert s.lrange("missing", 0, 10) == []


def test_zset_inclusive_ranges_and_member_update():
    s = InMemoryStore()
    s.zadd("z", {"a": 10, "b": 20, "c": 30})
    assert s.zcount("z", 10, 20) == 2
    assert s.zrangebyscore("z", 15, 30) == ["b", "c"]
    s.zadd("z", {"a": 25})  # re-score existing member
    assert s.zrangebyscore("z", 0, 100) == ["b", "a", "c"]
    assert s.zremrangebyscore("z", "-inf", 20) == 1
    assert s.zcount("z", 0, 100) == 2


def test_sets_and_hashes():
    s = InMemoryStore()
    assert s.sadd("s", "x", "y") == 2
    assert s.sadd("s", "x") == 0
    assert s.scard("s") == 2 and s.sismember("s", "x") and not s.sismember("s", "q")
    assert s.hincrby("h", "n", 2) == 2
    assert s.hsetnx("h", "fs", 5) is True and s.hsetnx("h", "fs", 7) is False
    assert s.hgetall("h") == {"n": "2", "fs": "5"}
    assert s.hmget("h", ["n", "zz"]) == ["2", None]
    s.hdel("h", "n")
    assert s.hgetall("h") == {"fs": "5"}


def test_execute_plan_and_dump(test_redis):
    s = InMemoryStore()
    s.execute([("lpush", "l", "a"), ("sadd", "s", "m"), ("zadd", "z", {"t": 1}), ("hset", "h", "f", "v", None)])
    s.set("str", "1")
    s.dump_to_redis(test_redis)
    assert test_redis.lrange("l", 0, -1) == ["a"]
    assert test_redis.smembers("s") == {"m"}
    assert test_redis.zrangebyscore("z", 0, 2) == ["t"]
    assert test_redis.hgetall("h") == {"f": "v"}
    assert test_redis.get("str") == "1"
