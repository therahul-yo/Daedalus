"""PrefixCacheStore tests with real mlx-lm cache objects.

KVCache = trimmable (plain transformer layers).
ArraysCache = non-trimmable (Qwen3.5 / hybrid GDN linear layers) — the case
that forces exact-prefix-only matching.
"""

import threading
import time

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from daedalus.cache.store import (
    PrefixCacheStore,
    _Entry,
    _tokens_digest,
    cache_nbytes,
)


def kv_cache_with(n_tokens: int) -> KVCache:
    c = KVCache()
    keys = mx.arange(n_tokens, dtype=mx.float16).reshape(1, 1, n_tokens, 1)
    keys = mx.broadcast_to(keys, (1, 2, n_tokens, 4))
    c.update_and_fetch(keys, keys)
    return c


def hybrid_cache_with(n_tokens: int):
    """[ArraysCache, KVCache] like a hybrid model layer list."""
    a = ArraysCache(size=2)
    a[0] = mx.ones((1, 4, 8))
    a[1] = mx.full((1, 3, 3), float(n_tokens))
    return [a, kv_cache_with(n_tokens)]


def gemma4_style_cache_with(n_tokens: int, window: int = 64):
    """[KVCache, RotatingKVCache] like Gemma 4's full+sliding layer mix."""
    rot = RotatingKVCache(max_size=window, keep=0)
    step = 16
    for i in range(0, n_tokens, step):
        n = min(step, n_tokens - i)
        keys = mx.ones((1, 2, n, 4), dtype=mx.float16)
        rot.update_and_fetch(keys, keys)
    return [kv_cache_with(n_tokens), rot]


def make_store(tmp_path, **kw):
    kw.setdefault("max_ram_bytes", 10 * 1024**2)
    kw.setdefault("min_persist_tokens", 4)
    return PrefixCacheStore("test-model", cache_dir=tmp_path, **kw)


def test_miss_on_empty_store(tmp_path):
    store = make_store(tmp_path)
    assert store.fetch([1, 2, 3]) is None
    assert store.stats()["misses"] == 1


def test_exclusive_store_lock_prevents_second_owner(tmp_path):
    store = PrefixCacheStore("test-model", cache_dir=tmp_path, exclusive=True)
    with pytest.raises(RuntimeError, match="already owned"):
        PrefixCacheStore("test-model", cache_dir=tmp_path, exclusive=True)
    store.close()
    reopened = PrefixCacheStore("test-model", cache_dir=tmp_path, exclusive=True)
    reopened.close()


def test_deferred_persist_keeps_prefix_available_immediately(tmp_path):
    store = make_store(tmp_path)
    tokens = list(range(20))
    store.put(tokens, [kv_cache_with(20)], persist=False)
    assert store.fetch(tokens + [99]) is not None
    store.persist(tokens)
    reloaded = make_store(tmp_path)
    assert reloaded.fetch(tokens + [99]) is not None
    assert store.stats()["copy_seconds"] >= 0


def test_resident_entry_reports_wall_clock_age(tmp_path):
    store = make_store(tmp_path)
    store.put(list(range(20)), [kv_cache_with(20)], persist=False)
    entry = store.list_entries()[0]
    assert 0 <= entry["age_seconds"] < 5


def test_exact_prefix_hit_trimmable(tmp_path):
    store = make_store(tmp_path)
    prefix = list(range(100))
    store.put(prefix, [kv_cache_with(100)])
    hit = store.fetch(prefix + [777, 888])
    assert hit is not None
    assert hit.matched_tokens == 100
    assert hit.cache[0].offset == 100


def test_superset_entry_is_trimmed_for_trimmable_cache(tmp_path):
    store = make_store(tmp_path)
    stored = list(range(100))
    store.put(stored, [kv_cache_with(100)])
    # Request shares only the first 60 tokens.
    request = list(range(60)) + [900] * 40
    hit = store.fetch(request)
    assert hit is not None
    assert hit.matched_tokens == 60
    assert hit.cache[0].offset == 60  # trimmed 40 off


def test_superset_entry_unusable_for_hybrid_but_prefix_fallback_works(tmp_path):
    store = make_store(tmp_path)
    store.put(list(range(100)), hybrid_cache_with(100))  # superset, non-trimmable
    store.put(list(range(40)), hybrid_cache_with(40))    # strict prefix
    request = list(range(60)) + [900] * 40
    hit = store.fetch(request)
    assert hit is not None
    assert hit.matched_tokens == 40  # fell back to the strict-prefix entry
    assert hit.cache[1].offset == 40


def test_hybrid_superset_with_no_prefix_fallback_misses(tmp_path):
    store = make_store(tmp_path)
    store.put(list(range(100)), hybrid_cache_with(100))
    hit = store.fetch(list(range(60)) + [900] * 40)
    assert hit is None


def test_fetch_returns_deep_copy(tmp_path):
    store = make_store(tmp_path)
    prefix = list(range(50))
    store.put(prefix, [kv_cache_with(50)])
    hit1 = store.fetch(prefix + [1, 2])
    hit1.cache[0].update_and_fetch(
        mx.zeros((1, 2, 10, 4), dtype=mx.float16), mx.zeros((1, 2, 10, 4), dtype=mx.float16)
    )
    hit2 = store.fetch(prefix + [3, 4])
    assert hit2.cache[0].offset == 50  # unaffected by hit1's mutation


def test_identical_request_matches_all_but_last_token(tmp_path):
    store = make_store(tmp_path)
    tokens = list(range(100))
    store.put(tokens[:99], [kv_cache_with(99)])  # end-of-prefill snapshot
    hit = store.fetch(tokens)
    assert hit is not None
    assert hit.matched_tokens == 99  # leaves exactly the final token


def test_disk_persistence_across_restart(tmp_path):
    store = make_store(tmp_path)
    prefix = list(range(2000))
    store.put(prefix, hybrid_cache_with(2000))

    store2 = make_store(tmp_path)  # fresh instance = server restart
    hit = store2.fetch(prefix + [5])
    assert hit is not None
    assert hit.source == "disk"
    assert hit.matched_tokens == 2000
    assert hit.cache[1].offset == 2000
    # ArraysCache state survived the round-trip too.
    assert hit.cache[0][0] is not None


def test_corrupt_disk_entry_is_skipped_not_fatal(tmp_path):
    store = make_store(tmp_path)
    prefix = list(range(500))
    store.put(prefix, [kv_cache_with(500)])
    for f in (store.dir).glob("*.safetensors"):
        f.write_bytes(b"garbage")

    store2 = make_store(tmp_path)
    assert store2.fetch(prefix + [1]) is None  # dropped, no crash


def test_checkpoint_enables_resume(tmp_path):
    store = make_store(tmp_path)
    tokens = list(range(3000))
    # Simulate mid-prefill checkpoint at 2048 tokens.
    store.checkpoint(tokens, 2048, [kv_cache_with(2048)])
    hit = store.fetch(tokens)
    assert hit is not None
    assert hit.matched_tokens == 2048  # resume point, not restart


def test_wrong_model_key_entries_ignored(tmp_path):
    store_a = PrefixCacheStore(
        "model-a", cache_dir=tmp_path, min_persist_tokens=4
    )
    store_a.put(list(range(100)), [kv_cache_with(100)])
    store_b = PrefixCacheStore(
        "model-b", cache_dir=tmp_path, min_persist_tokens=4
    )
    assert store_b.fetch(list(range(100)) + [1]) is None


def test_gemma4_wrapped_window_degrades_to_exact_prefix(tmp_path):
    """Once Gemma 4's sliding window wraps, its cache is non-trimmable —
    superset entries become unusable but strict-prefix reuse still works."""
    store = make_store(tmp_path)
    long_cache = gemma4_style_cache_with(200, window=64)  # wrapped: 200 > 64
    assert not long_cache[1].is_trimmable()
    store.put(list(range(200)), long_cache)

    # Superset entry cannot be trimmed to the 100-token shared prefix -> miss.
    assert store.fetch(list(range(100)) + [999] * 20) is None

    # But a strict-prefix entry is reusable as-is.
    store.put(list(range(100)), gemma4_style_cache_with(100, window=64))
    hit = store.fetch(list(range(100)) + [999] * 20)
    assert hit is not None
    assert hit.matched_tokens == 100


def test_gemma4_unwrapped_window_still_trims(tmp_path):
    """Below the sliding window size the rotating cache is still trimmable."""
    store = make_store(tmp_path)
    cache = gemma4_style_cache_with(40, window=64)  # 40 < 64: not wrapped
    assert cache[1].is_trimmable()
    store.put(list(range(40)), cache)
    hit = store.fetch(list(range(30)) + [999] * 10)
    assert hit is not None
    assert hit.matched_tokens == 30


def test_gemma4_cache_survives_disk_round_trip(tmp_path):
    store = make_store(tmp_path)
    tokens = list(range(2000))
    store.put(tokens, gemma4_style_cache_with(2000, window=64))
    store2 = make_store(tmp_path)
    hit = store2.fetch(tokens + [5])
    assert hit is not None
    assert hit.source == "disk"
    assert hit.matched_tokens == 2000


def test_ram_eviction_keeps_disk_entries_usable(tmp_path):
    store = make_store(tmp_path, max_ram_bytes=1)  # force immediate eviction
    prefix = list(range(1000))
    store.put(prefix, [kv_cache_with(1000)])
    hit = store.fetch(prefix + [1])
    assert hit is not None  # served from disk even though RAM evicted
    assert hit.matched_tokens == 1000


# --------------------------------------------------------------------------
# FIX 1 — persist/evict disk races
# --------------------------------------------------------------------------

def test_repersist_keeps_live_inode_for_open_reader(tmp_path):
    """Re-persisting an already-on-disk key must not unlink the live file a
    concurrent reader is materializing from: the atomic replace keeps the
    reader on the old inode (POSIX)."""
    store = make_store(tmp_path)
    tokens = list(range(2000))
    store.put(tokens, [kv_cache_with(2000)])
    path = store._entries[_tokens_digest(tokens)].path
    assert path is not None and path.exists()

    with path.open("rb") as reader:
        head = reader.read(64)
        # Recurring head-snapshot re-persist of the same key.
        store.put(tokens, [kv_cache_with(2000)])
        rest = reader.read()  # old inode still fully readable
    assert head and (head + rest)
    assert path.exists()
    with store._lock:
        for p in store._disk_usage:
            assert p.exists()


def test_concurrent_persist_and_evict_keeps_index_consistent(tmp_path):
    """Concurrent puts (which persist + evict) must never leave _disk_usage or
    entry.path pointing at a deleted file."""
    store = make_store(tmp_path, max_disk_bytes=30 * 1024)  # tiny → heavy evict
    errors = []
    barrier = threading.Barrier(2)

    def worker(base):
        try:
            barrier.wait()
            for i in range(25):
                toks = list(range(base + i, base + i + 300))
                store.put(toks, [kv_cache_with(300)])
                store.fetch(toks + [7])
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(b,)) for b in (0, 100_000)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    with store._lock:
        for path in store._disk_usage:
            assert path.exists(), f"_disk_usage references missing {path}"
        for key, entry in store._entries.items():
            if entry.path is not None:
                assert entry.path.exists(), f"entry {key} path missing"


# --------------------------------------------------------------------------
# FIX 3 — TTL prune re-runs opportunistically from put()
# --------------------------------------------------------------------------

def _inject_stale_disk_entry(store, tokens, age_days):
    key = _tokens_digest(tokens)
    data = store.dir / f"{key}.safetensors"
    data.write_bytes(b"\x00" * 512)
    old = time.time() - age_days * 86400
    entry = _Entry(tokens=tokens, cache=None, nbytes=0,
                   last_used=time.monotonic(), path=data,
                   created=old, last_used_at=old)
    with store._lock:
        store._entries[key] = entry
        store._index(key, tokens)
        store._disk_usage[data] = (512, old)
    return key


def test_ttl_reprune_triggered_by_put_after_interval(tmp_path):
    store = make_store(tmp_path)
    store.cache_ttl_days = 1
    stale = _inject_stale_disk_entry(store, list(range(4)), age_days=7)
    # Pretend the last prune was long ago so the interval has elapsed.
    store._last_ttl_prune = time.monotonic() - store._ttl_prune_interval - 1

    store.put(list(range(100, 110)), [kv_cache_with(10)], persist=False)

    assert stale not in store._entries  # opportunistic re-prune removed it


def test_ttl_reprune_skipped_within_interval(tmp_path):
    store = make_store(tmp_path)
    store.cache_ttl_days = 1
    stale = _inject_stale_disk_entry(store, list(range(4)), age_days=7)
    store._last_ttl_prune = time.monotonic()  # just pruned

    store.put(list(range(100, 110)), [kv_cache_with(10)], persist=False)

    assert stale in store._entries  # interval not elapsed → not re-pruned


# --------------------------------------------------------------------------
# FIX 4 — fetch must survive a concurrent trim_ram
# --------------------------------------------------------------------------

def test_fetch_hit_survives_concurrent_trim_ram(tmp_path):
    """A RAM-only, unpinned entry is the silent-miss case: trim_ram would null
    its cache and pop it. A fetch that already captured it must still return a
    hit. Block inside _materialize to force the interleaving deterministically.
    """
    store = make_store(tmp_path)
    tokens = list(range(50))
    key = _tokens_digest(tokens)
    entry = _Entry(
        tokens=tokens,
        cache=[kv_cache_with(50)],
        nbytes=cache_nbytes([kv_cache_with(50)]),
        last_used=time.monotonic(),
        path=None,          # RAM-only → trim_ram would pop it entirely
        pinned=False,
        last_used_at=time.time(),
    )
    with store._lock:
        store._entries[key] = entry
        store._index(key, tokens)

    entered = threading.Event()
    proceed = threading.Event()
    orig_materialize = store._materialize

    def blocking_materialize(e):
        entered.set()
        proceed.wait(5)
        return orig_materialize(e)

    store._materialize = blocking_materialize

    result = {}

    def do_fetch():
        result["hit"] = store.fetch(tokens + [999])

    t = threading.Thread(target=do_fetch)
    t.start()
    assert entered.wait(5)      # fetch has pinned the entry, now inside materialize
    store.trim_ram(0)           # attempt to evict everything mid-fetch
    proceed.set()
    t.join(timeout=5)

    assert result["hit"] is not None      # no silent miss
    assert result["hit"].matched_tokens == 50
    assert key in store._entries          # pinned entry was not popped
