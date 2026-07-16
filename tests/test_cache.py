"""Cache tests: TTL eviction, migration, prune counts, lock contention."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from daedalus.cache.store import (
    FORMAT_VERSION,
    _Entry,
    _tokens_digest,
    PrefixCacheStore,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _stub_safetensors(path: Path, size: int = 512) -> None:
    """Write a minimal safetensors stub that passes stat()."""
    header = json.dumps({}).encode()
    path.write_bytes(
        len(header).to_bytes(8, "little") + header + b"\x00" * max(0, size - 8 - len(header))
    )


def _add_disk_entry(store: PrefixCacheStore, tokens: list[int],
                    *, age_days: float = 0, pinned: bool = False,
                    size: int = 512) -> str:
    """Inject a disk‑backed entry directly into *store*, bypassing MLX."""
    key = _tokens_digest(tokens)
    data_path = store.dir / f"{key}.safetensors"
    sidecar_path = store.dir / f"{key}.safetensors.json"

    _stub_safetensors(data_path, size=size)
    sidecar_path.write_text(json.dumps({
        "version": FORMAT_VERSION,
        "model_key": store.model_key,
        "tokens": tokens,
        "created": time.time() - age_days * 86400,
    }))

    entry = _Entry(
        tokens=tokens,
        cache=None,
        nbytes=0,
        last_used=time.time() - age_days * 86400,
        path=data_path,
        pinned=pinned,
        created=time.time() - age_days * 86400,
    )
    with store._lock:
        store._entries[key] = entry
        store._index(key, tokens)
        store._disk_usage[data_path] = (data_path.stat().st_size, time.time())
    return key


@pytest.fixture
def tmp_store(tmp_path: Path) -> PrefixCacheStore:
    """A store backed by a temporary directory (no lock contention)."""
    return PrefixCacheStore(
        model_key="test-model",
        cache_dir=tmp_path,
        max_ram_bytes=1024**3,
        max_disk_bytes=1024**2,
        exclusive=False,
    )


# ---------------------------------------------------------------------------
# 1. TTL eviction respects pins + _disk_usage accounting
# ---------------------------------------------------------------------------

def test_ttl_preserves_pinned_entries(tmp_store: PrefixCacheStore):
    """TTL prune evicts unpinned old entries but keeps pinned ones."""
    store = tmp_store
    store.cache_ttl_days = 1

    key_a = _add_disk_entry(store, [1, 2, 3], age_days=7, pinned=True)
    key_b = _add_disk_entry(store, [4, 5, 6], age_days=7, pinned=False)

    entries_before = len(store._entries)
    removed = store._prune_disk_by_ttl()

    # Pinned entry A survives.
    assert key_a in store._entries
    assert store._entries[key_a].path is not None
    # Unpinned entry B is gone.
    assert key_b not in store._entries
    assert removed == 1

    # _disk_usage consistent: only A's path remains.
    surviving = {store._entries[key_a].path}
    assert set(store._disk_usage.keys()) == surviving

    assert len(store._entries) == entries_before - removed


def test_ttl_accounting_consistent_after_prune(tmp_store: PrefixCacheStore):
    """After TTL prune, _disk_usage no longer references removed files."""
    store = tmp_store
    store.cache_ttl_days = 7

    fresh_key = _add_disk_entry(store, [10, 20], age_days=1)
    stale_key = _add_disk_entry(store, [30, 40], age_days=14)

    store._prune_disk_by_ttl()

    assert fresh_key in store._entries
    assert stale_key not in store._entries

    # The stale file's path must not be in _disk_usage.
    with store._lock:
        for path in store._disk_usage:
            assert path.exists(), f"_disk_usage references missing file {path}"


# ---------------------------------------------------------------------------
# 2. Migration indexes entries in the same pass
# ---------------------------------------------------------------------------

def test_migration_reindexes_in_same_pass(tmp_path: Path):
    """A v1 sidecar is migrated to v2 and indexed during _load_disk_index."""
    tokens = [100, 200, 300]
    key = _tokens_digest(tokens)

    model_dir = tmp_path / "test-model"
    model_dir.mkdir(parents=True)
    data_path = model_dir / f"{key}.safetensors"
    sidecar_path = model_dir / f"{key}.safetensors.json"

    _stub_safetensors(data_path)
    sidecar_path.write_text(json.dumps({
        "version": 1,  # v1 — no model_key, no created, no n_tokens
        "tokens": tokens,
    }))

    # Construct a store — FORMAT_VERSION=2 so _load_disk_index will
    # detect the v1 sidecar, call _migrate_entry, re-read, and index.
    store = PrefixCacheStore(
        model_key="test-model",
        cache_dir=tmp_path,
        max_disk_bytes=1024**2,
        exclusive=False,
    )

    assert key in store._entries, "migrated entry must be in _entries"
    entry = store._entries[key]
    assert entry.tokens == tokens
    assert entry.path == data_path

    # Sidecar on disk upgraded to v2.
    migrated = json.loads(sidecar_path.read_text())
    assert migrated["version"] == FORMAT_VERSION
    assert migrated["model_key"] == "test-model"
    assert migrated.get("created", 0) > 0

    # _disk_usage references the file.
    assert data_path in store._disk_usage

    # The entry is reachable via the trie (key in _entries already proved this,
    # but verify the trie was rebuilt correctly by checking candidate matching).
    candidates = store._candidate_keys([100, 200, 300, 400])
    assert key in candidates, "entry must be in candidate set after migration"

    # stats() counts the migrated entry.
    stats = store.stats()
    assert stats["entries"] >= 1


def test_migration_does_not_lose_on_reload(tmp_path: Path):
    """A migrated entry survives a second store initialization (re-load)."""
    tokens = [111, 222]
    key = _tokens_digest(tokens)

    model_dir = tmp_path / "test-model"
    model_dir.mkdir(parents=True)
    data_path = model_dir / f"{key}.safetensors"
    sidecar_path = model_dir / f"{key}.safetensors.json"

    _stub_safetensors(data_path)
    sidecar_path.write_text(json.dumps({"version": 1, "tokens": tokens}))

    # First init — migration.
    PrefixCacheStore("test-model", cache_dir=tmp_path,
                     max_disk_bytes=1024**2, exclusive=False)

    # Second init — re-load the now-upgraded v2 sidecar.
    store2 = PrefixCacheStore("test-model", cache_dir=tmp_path,
                              max_disk_bytes=1024**2, exclusive=False)
    assert key in store2._entries
    assert store2._entries[key].tokens == tokens


# ---------------------------------------------------------------------------
# 3. Prune returns real counts
# ---------------------------------------------------------------------------

def test_prune_returns_accurate_count(tmp_store: PrefixCacheStore):
    """prune_by_ttl returns the number of evicted entries matching reality."""
    store = tmp_store

    keys = [_add_disk_entry(store, [i, i + 1, i + 2], age_days=10)
            for i in range(2)]
    keys.append(_add_disk_entry(store, [5, 6, 7], age_days=0))  # fresh

    removed = store.prune_by_ttl(ttl_days=5)
    assert removed == 2
    assert keys[2] in store._entries  # fresh survives
    assert keys[0] not in store._entries
    assert keys[1] not in store._entries


def test_prune_zero_when_nothing_expired(tmp_store: PrefixCacheStore):
    """prune_by_ttl returns 0 when all entries are fresh."""
    store = tmp_store
    _add_disk_entry(store, [9, 8], age_days=0)
    _add_disk_entry(store, [7, 6], age_days=0)
    assert store.prune_by_ttl(ttl_days=365) == 0


# ---------------------------------------------------------------------------
# 4. Lock contention
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Torn / temp files left by a crash are never indexed (FIX 2)
# ---------------------------------------------------------------------------

def test_torn_tmp_files_are_cleaned_and_not_indexed(tmp_path: Path):
    """A crash mid-persist leaves ``*.tmp.*`` remnants carrying the real
    token-digest key. _load_disk_index must delete them and index only the
    committed entry, keeping _disk_usage accounting honest."""
    model_dir = tmp_path / "test-model"
    model_dir.mkdir(parents=True)

    tokens = [1, 2, 3, 4]
    key = _tokens_digest(tokens)

    # A committed (real) entry.
    _stub_safetensors(model_dir / f"{key}.safetensors")
    (model_dir / f"{key}.safetensors.json").write_text(json.dumps({
        "version": FORMAT_VERSION,
        "model_key": "test-model",
        "tokens": tokens,
        "created": time.time(),
    }))

    # Crash remnants: a torn tmp data file, an old-style tmp sidecar (matches
    # the *.safetensors.json glob), and a new-style tmp sidecar.
    (model_dir / f"{key}.tmp.111.safetensors").write_bytes(b"torn-partial")
    (model_dir / f"{key}.tmp.111.safetensors.json").write_text("{}")
    (model_dir / f"{key}.tmp.222.sidecar.json").write_text("{}")

    store = PrefixCacheStore("test-model", cache_dir=tmp_path,
                             max_disk_bytes=1024**2, exclusive=False)

    # Exactly one entry — the torn tmp was not mis-indexed under the real key.
    assert key in store._entries
    assert len(store._entries) == 1

    # All temp remnants deleted by the janitor.
    assert list(model_dir.glob("*.tmp.*")) == []

    # _disk_usage references only the committed file, and it exists.
    with store._lock:
        for path in store._disk_usage:
            assert ".tmp." not in path.name
            assert path.exists()
        assert set(store._disk_usage) == {model_dir / f"{key}.safetensors"}


def test_prune_fails_fast_when_server_holds_lock(tmp_path: Path):
    """The CLI helper returns None (with error message) when server holds
    the flock on the cache directory."""
    import fcntl

    model_dir = tmp_path / "test-model"
    model_dir.mkdir(parents=True)
    lock_path = model_dir / ".daedalus.lock"

    lock_fd = lock_path.open("a+")
    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    try:
        from daedalus.cache.cli import _open_store
        result = _open_store("test-model", tmp_path)
        assert result is None, "expected None when lock is held"
    finally:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()
