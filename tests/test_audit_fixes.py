"""Regression tests for the 2026-07-11 audit fixes (P0 correctness + P1 perf).

Each test pins a specific audit finding by number (see plan file).
"""

import asyncio
import json
import threading
import time

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient
from mlx_lm.models.cache import KVCache

from daedalus.cache.store import PrefixCacheStore
from daedalus.server import create_app

from test_server import FakeEngine, FakeResponse, FakeStore


def kv_cache_with(n_tokens: int) -> KVCache:
    c = KVCache()
    keys = mx.arange(n_tokens, dtype=mx.float16).reshape(1, 1, n_tokens, 1)
    keys = mx.broadcast_to(keys, (1, 2, n_tokens, 4))
    c.update_and_fetch(keys, keys)
    return c


def make_store(tmp_path, **kw):
    kw.setdefault("max_ram_bytes", 10 * 1024**2)
    kw.setdefault("min_persist_tokens", 4)
    return PrefixCacheStore("test-model", cache_dir=tmp_path, **kw)


# ---------------------------------------------------------------- finding 1


def test_deferred_snapshot_pinned_against_ram_eviction(tmp_path):
    """put(persist=False) under RAM pressure must NOT lose the snapshot."""
    store = make_store(tmp_path, max_ram_bytes=1)  # everything over budget
    tokens = list(range(500))
    store.put(tokens, [kv_cache_with(500)], persist=False)
    # Entry survived eviction because it is pinned...
    assert store.fetch(tokens + [1]) is not None
    # ...and persist() lands it on disk, after which it is unpinned.
    store.persist(tokens)
    files = list(store.dir.glob("*.safetensors"))
    assert len(files) == 1
    # Now unpinned: RAM eviction may drop the resident copy, but the disk
    # entry keeps serving.
    store._evict_ram()
    hit = store.fetch(tokens + [1])
    assert hit is not None


def test_unpinned_after_persist_allows_eviction(tmp_path):
    store = make_store(tmp_path, max_ram_bytes=1)
    tokens = list(range(300))
    store.put(tokens, [kv_cache_with(300)], persist=False)
    store.persist(tokens)
    key_entry = list(store._entries.values())[0]
    assert key_entry.pinned is False


# ---------------------------------------------------------------- finding 5


def test_dangling_sidecar_cleaned_on_load(tmp_path):
    store = make_store(tmp_path)
    orphan = store.dir / "deadbeef.safetensors.json"
    orphan.write_text(json.dumps({"version": 1, "model_key": "test-model", "tokens": [1, 2, 3]}))
    store2 = make_store(tmp_path)
    assert not orphan.exists()
    assert store2.stats()["entries"] == 0


# ---------------------------------------------------------------- finding 7


def test_disk_usage_tracked_incrementally(tmp_path):
    store = make_store(tmp_path)
    tokens = list(range(600))
    store.put(tokens, [kv_cache_with(600)])
    assert sum(s for s, _ in store._disk_usage.values()) > 0
    # A fresh store rebuilds the same accounting from disk.
    store2 = make_store(tmp_path)
    assert store2._disk_usage.keys() == store._disk_usage.keys()


def test_disk_eviction_still_enforces_budget(tmp_path):
    store = make_store(tmp_path, max_disk_bytes=1)  # force eviction of all
    tokens = list(range(700))
    store.put(tokens, [kv_cache_with(700)])
    assert list(store.dir.glob("*.safetensors")) == []
    assert store._disk_usage == {}
    # RAM copy remains usable even though the disk file was evicted.
    assert store.fetch(tokens + [1]) is not None


# ---------------------------------------------------------------- finding 3


def _get_chat_handler(app):
    route = next(r for r in app.routes if getattr(r, "path", "") == "/v1/chat/completions")
    return route.endpoint


class SlowAbortableEngine(FakeEngine):
    """Yields one chunk, then spins until should_abort fires."""

    def generate(self, tokens, *, should_abort=None, checkpoint_cb=None, **kw):
        self.generate_calls.append({"tokens": len(tokens)})
        yield FakeResponse("first", 1, None)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if should_abort and should_abort():
                return
            time.sleep(0.02)
        raise AssertionError("abort never observed")


class FakeDisconnectingRequest:
    """Duck-typed starlette Request: reports disconnected after first poll."""

    def __init__(self, body: dict):
        self._body = body
        self.headers = {}
        self.client = None
        self.polls = 0

    async def json(self):
        return self._body

    async def is_disconnected(self):
        self.polls += 1
        return True


def test_non_streaming_disconnect_aborts_engine():
    engine, store = SlowAbortableEngine(), FakeStore()
    app = create_app(engine, store, model_id="test-model")
    handler = _get_chat_handler(app)
    request = FakeDisconnectingRequest(
        {"messages": [{"role": "user", "content": "hi"}]}
    )
    response = asyncio.run(handler(request, authorization=None))
    assert response.status_code == 499
    # Slot released despite the abort.
    assert app.state.daedalus.admitted_requests == 0


# ---------------------------------------------------------------- finding 4


def test_shutdown_waits_for_inflight_then_closes_store():
    closed = []

    class ClosableStore(FakeStore):
        def close(self, timeout: float = 10.0):
            closed.append(time.monotonic())

    engine, store = FakeEngine(), ClosableStore()
    app = create_app(
        engine, store, model_id="test-model", shutdown_drain_seconds=0.5
    )
    state = app.state.daedalus
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        # Simulate a straggler: it should delay close until drain or deadline.
        state.try_admit()
        released = []

        def release_later():
            time.sleep(0.15)
            state.release()
            released.append(time.monotonic())

        threading.Thread(target=release_later, daemon=True).start()
    # Exiting the client context runs lifespan shutdown.
    assert closed, "store.close() never ran on shutdown"
    assert released and closed[0] >= released[0]


# ---------------------------------------------------------------- finding 8/9


def test_head_snapshot_persist_is_deferred():
    """Head snapshots use persist=False in-band; disk write happens post-done."""

    class RecordingStore(FakeStore):
        def __init__(self):
            super().__init__()
            self.persist_calls = []
            self.put_persist_flags = []

        def put(self, tokens, cache, persist=True, async_persist=False):
            self.put_persist_flags.append(persist)
            super().put(tokens, cache, persist=persist)

        def persist(self, tokens):
            self.persist_calls.append(len(tokens))

    class HeadEngine(FakeEngine):
        def generate(self, tokens, *, checkpoint_cb=None, **kw):
            self.generate_calls.append({"tokens": len(tokens)})
            if checkpoint_cb:
                # Simulate the head-boundary chunk landing, then end of prefill.
                checkpoint_cb(len(tokens) // 2, ["cache"])
                checkpoint_cb(len(tokens) - 1, ["cache"])
            yield FakeResponse("out", 1, "stop")

    engine, store = HeadEngine(), RecordingStore()
    app = create_app(engine, store, model_id="test-model")
    client = TestClient(app)
    # Long system message → head boundary probe resolves mid-prompt.
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "s" * 400},
                {"role": "user", "content": "hi"},
            ]
        },
    )
    assert r.status_code == 200
    # Every in-band put was deferred (persist=False)...
    assert store.put_persist_flags and all(f is False for f in store.put_persist_flags)
    # ...and the deferred persists landed after generation.
    assert len(store.persist_calls) >= 1
