"""Persistent prefix cache: RAM LRU + disk tier.

The single highest-leverage mechanism for the MacBook Air big-prompt problem:
an agent's 10-40k-token system prompt is prefilled once, persisted, and every
later request — including after a server restart — resumes from cached KV
instead of re-burning the GPU.

Design notes (lineage: mlx-lm LRUPromptCache, vllm-mlx MemoryAwarePrefixCache,
Rapid-MLX runtime/cache.py, baseRT prefix-cache C API):

- Entries are keyed by their full token sequence. Matching returns the entry
  that is the longest usable prefix of the request; a superset entry (longer
  than the request) is trimmed back to the shared prefix when the model's
  cache type supports trimming (hybrid-attention models don't — they degrade
  to pure-prefix matches automatically).
- Fetch returns a deep copy; the live generation mutates its copy and the
  stored entry stays valid.
- Disk entries are mlx-lm ``save_prompt_cache`` safetensors files plus a JSON
  sidecar (format version, token ids, model key). Writes are atomic
  (tmp + rename); unreadable entries are skipped, not fatal.
- Mid-prefill checkpoints reuse the same store: a checkpoint is just a normal
  entry for ``tokens[:done]``, so resume-after-timeout is a plain fetch.
- Cache format version migration: v1 → v2 adds token count metadata, model key
  validation, and TTL-based disk eviction support.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    load_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)

# Structured logging
try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    structlog = None
    HAS_STRUCTLOG = False

# OpenTelemetry
try:
    from opentelemetry import trace
    HAS_OTEL = True
except ImportError:
    trace = None
    HAS_OTEL = False

logger = structlog.get_logger(__name__) if HAS_STRUCTLOG else logging.getLogger(__name__)
tracer = trace.get_tracer("daedalus.cache") if HAS_OTEL and trace else None

# Cache format version. Increment when changing the on-disk format.
# v1: Original format with basic metadata
# v2: Added token count in metadata, model key validation, TTL support
FORMAT_VERSION = 2


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "daedalus" / "prefix"


def _sanitize_model_key(model_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "--", model_key)


def _tokens_digest(tokens: List[int]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(tokens).encode())
    return h.hexdigest()[:24]


def cache_nbytes(prompt_cache: List[Any]) -> int:
    total = 0
    for layer in prompt_cache:
        for _, arr in tree_flatten(layer.state):
            if isinstance(arr, mx.array):
                total += arr.nbytes
    return total


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class CacheHit:
    cache: List[Any]
    matched_tokens: int
    source: str  # "ram" | "disk"


@dataclass
class _Entry:
    tokens: List[int]
    cache: Optional[List[Any]]  # None => disk-only
    nbytes: int
    last_used: float
    path: Optional[Path] = None  # set when persisted


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"]
    keys: set[str]

    def __init__(self) -> None:
        self.children = {}
        self.keys = set()


class PrefixCacheStore:
    def __init__(
        self,
        model_key: str,
        cache_dir: Optional[Path] = None,
        max_ram_bytes: Optional[int] = None,
        max_disk_bytes: int = 10 * 1024**3,
        min_persist_tokens: int = 1024,
        exclusive: bool = False,
        cache_ttl_days: Optional[int] = None,
    ) -> None:
        self.model_key = model_key
        self.dir = (cache_dir or _default_cache_dir()) / _sanitize_model_key(model_key)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._process_lock = None
        if exclusive:
            import fcntl

            self._process_lock = (self.dir / ".daedalus.lock").open("a+")
            try:
                fcntl.flock(self._process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._process_lock.close()
                raise RuntimeError(f"cache directory is already owned: {self.dir}")
        if max_ram_bytes is None:
            try:
                import subprocess

                total = int(
                    subprocess.run(
                        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
                    ).stdout.strip()
                )
            except Exception:
                total = 16 * 1024**3
            max_ram_bytes = int(total * 0.20)
        self.max_ram_bytes = max_ram_bytes
        self.max_disk_bytes = max_disk_bytes
        self.min_persist_tokens = min_persist_tokens
        self.cache_ttl_days = cache_ttl_days
        self._entries: dict[str, _Entry] = {}
        self._trie = _TrieNode()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.copy_seconds = 0.0
        self.lookup_seconds = 0.0
        self.load_seconds = 0.0
        self._load_disk_index()
        self._prune_disk_by_ttl()

    def _index(self, key: str, tokens: List[int]) -> None:
        node = self._trie
        for token in tokens:
            node = node.children.setdefault(token, _TrieNode())
        node.keys.add(key)

    def _rebuild_index(self) -> None:
        """Drop stale trie paths after eviction, corruption, or cache clear."""
        self._trie = _TrieNode()
        for key, entry in self._entries.items():
            self._index(key, entry.tokens)

    def _candidate_keys(self, tokens: List[int]) -> set[str]:
        """Only inspect entries sharing a token prefix with the request."""
        node = self._trie
        keys: set[str] = set()
        for token in tokens[:-1]:
            child = node.children.get(token)
            if child is None:
                return keys if node is self._trie else keys | self._descendant_keys(node)
            node = child
            keys.update(node.keys)  # stored prefix of request
        # Longer entries can be trimmed for regular KV caches. Descendants of
        # the full request prefix are the only possible such candidates.
        return keys | self._descendant_keys(node)

    @staticmethod
    def _descendant_keys(node: _TrieNode) -> set[str]:
        keys: set[str] = set()
        stack = [node]
        while stack:
            current = stack.pop()
            keys.update(current.keys)
            stack.extend(current.children.values())
        return keys

    # ---------------------------------------------------------------- fetch

    def fetch(self, tokens: List[int]) -> Optional[CacheHit]:
        """Best usable cached prefix for ``tokens`` (deep-copied), or None.

        The returned ``matched_tokens`` is always < len(tokens) so at least
        one token remains for the engine to process (decode needs a final
        input token). Superset/overlong entries are trimmed when possible.
        """
        # Start span for cache fetch
        span = tracer.start_span("cache.fetch") if tracer else None
        if span:
            span.set_attribute("token_count", len(tokens))
            span.set_attribute("model_key", self.model_key)
        
        lookup_start = time.perf_counter()
        with self._lock:
            best_key, best_usable = None, 0
            for key in self._candidate_keys(tokens):
                entry = self._entries.get(key)
                if entry is None:
                    continue
                lcp = _common_prefix_len(entry.tokens, tokens)
                # Usable portion: the shared prefix, capped so one token remains.
                usable = min(lcp, len(tokens) - 1)
                if usable <= 0:
                    continue
                if usable < len(entry.tokens):
                    # Entry extends beyond the shared prefix: only usable if
                    # we can trim it back. Decided after materialization —
                    # conservatively require trim support via a probe below.
                    pass
                if usable > best_usable:
                    best_key, best_usable = key, usable
            if best_key is None:
                self.misses += 1
                if span:
                    span.set_attribute("hit", False)
                    span.end()
                logger.info("cache_fetch_miss", token_count=len(tokens), model_key=self.model_key)
                return None
            entry = self._entries[best_key]
        with self._lock:
            self.lookup_seconds += time.perf_counter() - lookup_start

        load_start = time.perf_counter()
        cache_obj, source = self._materialize(entry)
        with self._lock:
            self.load_seconds += time.perf_counter() - load_start
        if cache_obj is None:
            self.misses += 1
            if span:
                span.set_attribute("hit", False)
                span.end()
            logger.info("cache_fetch_materialize_failed", token_count=len(tokens), model_key=self.model_key)
            return None

        overhang = len(entry.tokens) - best_usable
        if overhang > 0:
            if not can_trim_prompt_cache(cache_obj):
                # Non-trimmable (hybrid/rotating) cache: only a pure-prefix
                # entry is usable. Fall back to a strict-prefix search.
                if span:
                    span.end()
                return self._fetch_strict_prefix(tokens, exclude=best_key)
            trim_prompt_cache(cache_obj, overhang)

        with self._lock:
            entry.last_used = time.monotonic()
            self.hits += 1
        
        if span:
            span.set_attribute("hit", True)
            span.set_attribute("matched_tokens", best_usable)
            span.set_attribute("source", source)
            span.set_attribute("overhang_tokens", overhang)
            span.end()
        
        logger.info(
            "cache_fetch_hit",
            token_count=len(tokens),
            matched_tokens=best_usable,
            source=source,
            overhang_tokens=overhang,
            model_key=self.model_key,
        )
        return CacheHit(cache=cache_obj, matched_tokens=best_usable, source=source)

    def _fetch_strict_prefix(
        self, tokens: List[int], exclude: str
    ) -> Optional[CacheHit]:
        with self._lock:
            best_key, best_len = None, 0
            for key in self._candidate_keys(tokens):
                entry = self._entries.get(key)
                if entry is None:
                    continue
                if key == exclude:
                    continue
                lcp = _common_prefix_len(entry.tokens, tokens)
                if lcp == len(entry.tokens) and lcp <= len(tokens) - 1 and lcp > best_len:
                    best_key, best_len = key, lcp
            if best_key is None:
                self.misses += 1
                return None
            entry = self._entries[best_key]
        cache_obj, source = self._materialize(entry)
        if cache_obj is None:
            self.misses += 1
            return None
        with self._lock:
            entry.last_used = time.monotonic()
            self.hits += 1
        return CacheHit(cache=cache_obj, matched_tokens=best_len, source=source)

    def _materialize(self, entry: _Entry) -> tuple[Optional[List[Any]], str]:
        """Deep copy of the entry's cache, loading from disk if needed."""
        if entry.cache is not None:
            return self._copy_cache(entry.cache), "ram"
        try:
            loaded = load_prompt_cache(str(entry.path))
        except Exception as exc:  # corrupt/stale file: drop the entry
            logger.warning("dropping unreadable cache entry %s: %s", entry.path, exc)
            with self._lock:
                self._entries.pop(_tokens_digest(entry.tokens), None)
                self._rebuild_index()
            if entry.path:
                entry.path.unlink(missing_ok=True)
                Path(str(entry.path) + ".json").unlink(missing_ok=True)
            return None, "disk"
        entry.cache = loaded
        entry.nbytes = cache_nbytes(loaded)
        self._evict_ram()
        return self._copy_cache(loaded), "disk"

    def _copy_cache(self, prompt_cache: List[Any]) -> List[Any]:
        """Time the required isolated copy used by mutable generation caches."""
        start = time.perf_counter()
        copied = copy.deepcopy(prompt_cache)
        with self._lock:
            self.copy_seconds += time.perf_counter() - start
        return copied

    # ------------------------------------------------------------------ put

    def put(
        self, tokens: List[int], prompt_cache: List[Any], persist: bool = True,
        async_persist: bool = False,
    ) -> None:
        """Store (a deep copy of) ``prompt_cache`` for ``tokens``."""
        if len(tokens) == 0:
            return
        
        # Start span for cache put
        span = tracer.start_span("cache.put") if tracer else None
        if span:
            span.set_attribute("token_count", len(tokens))
            span.set_attribute("model_key", self.model_key)
            span.set_attribute("persist", persist)
        
        key = _tokens_digest(tokens)
        stored = self._copy_cache(prompt_cache)
        entry = _Entry(
            tokens=list(tokens),
            cache=stored,
            nbytes=cache_nbytes(stored),
            last_used=time.monotonic(),
        )
        with self._lock:
            old = self._entries.get(key)
            if old is not None and old.path is not None:
                entry.path = old.path
            self._entries[key] = entry
            self._index(key, entry.tokens)
        
        logger.info(
            "cache_put",
            token_count=len(tokens),
            size_bytes=entry.nbytes,
            model_key=self.model_key,
            persist=persist,
        )
        
        if span:
            span.set_attribute("size_bytes", entry.nbytes)
            span.end()
        
        if persist and len(tokens) >= self.min_persist_tokens:
            # MLX cache serialization is GPU-stream-bound, so it cannot run
            # on a generic writer thread. ``async_persist`` is retained for
            # compatibility; callers should use ``persist()`` after first
            # token to defer write latency safely on the engine thread.
            if not async_persist:
                self._persist(key, entry)
        self._evict_ram()
        self._evict_disk()

    def persist(self, tokens: List[int]) -> None:
        """Synchronously persist an already-stored immutable snapshot."""
        key = _tokens_digest(tokens)
        
        span = tracer.start_span("cache.persist") if tracer else None
        if span:
            span.set_attribute("token_count", len(tokens))
            span.set_attribute("model_key", self.model_key)
        
        with self._lock:
            entry = self._entries.get(key)
        if entry is not None and entry.cache is not None and len(tokens) >= self.min_persist_tokens:
            self._persist(key, entry)
            self._evict_disk()
            logger.info("cache_persist", token_count=len(tokens), model_key=self.model_key)
        
        if span:
            span.end()

    def close(self, timeout: float = 10.0) -> None:
        """Release the optional single-process cache ownership lock."""
        if self._process_lock is not None:
            import fcntl

            fcntl.flock(self._process_lock.fileno(), fcntl.LOCK_UN)
            self._process_lock.close()
            self._process_lock = None

    def checkpoint(self, tokens: List[int], done: int, prompt_cache: List[Any]) -> None:
        """Persist the live cache state for ``tokens[:done]`` (no deep copy in
        RAM — written straight to disk so a retry can resume)."""
        if done < self.min_persist_tokens:
            return
        
        span = tracer.start_span("cache.checkpoint") if tracer else None
        if span:
            span.set_attribute("token_count", done)
            span.set_attribute("total_tokens", len(tokens))
            span.set_attribute("model_key", self.model_key)
        
        prefix = list(tokens[:done])
        key = _tokens_digest(prefix)
        entry = _Entry(
            tokens=prefix,
            cache=None,  # disk-only; avoids doubling RAM during prefill
            nbytes=0,
            last_used=time.monotonic(),
        )
        path = self._write_entry_files(key, prefix, prompt_cache)
        if path is None:
            if span:
                span.set_attribute("failed", True)
                span.end()
            return
        entry.path = path
        with self._lock:
            existing = self._entries.get(key)
            if existing is None or existing.cache is None:
                self._entries[key] = entry
                self._index(key, prefix)
        self._evict_disk()
        
        logger.info("cache_checkpoint", token_count=done, total_tokens=len(tokens), model_key=self.model_key)
        
        if span:
            span.end()

    # ---------------------------------------------------------- persistence

    def _persist(self, key: str, entry: _Entry) -> None:
        span = tracer.start_span("cache._persist") if tracer else None
        if span:
            span.set_attribute("key", key)
            span.set_attribute("model_key", self.model_key)
        
        path = self._write_entry_files(key, entry.tokens, entry.cache)
        if path is not None:
            entry.path = path
        
        if span:
            span.set_attribute("success", path is not None)
            span.end()

    def _write_entry_files(
        self, key: str, tokens: List[int], prompt_cache: List[Any]
    ) -> Optional[Path]:
        final = self.dir / f"{key}.safetensors"
        # NB: mx.save_safetensors appends ".safetensors" unless the name
        # already ends with it — the tmp name must keep the suffix.
        tmp = self.dir / f"{key}.tmp.{time.monotonic_ns()}.safetensors"
        try:
            save_prompt_cache(
                str(tmp),
                prompt_cache,
                metadata={
                    "daedalus_format": str(FORMAT_VERSION),
                    "model_key": self.model_key,
                    "n_tokens": str(len(tokens)),
                },
            )
            sidecar_tmp = Path(str(tmp) + ".json")
            sidecar_tmp.write_text(
                json.dumps(
                    {
                        "version": FORMAT_VERSION,
                        "model_key": self.model_key,
                        "tokens": tokens,
                        "created": time.time(),
                    }
                )
            )
            tmp.rename(final)
            sidecar_tmp.rename(Path(str(final) + ".json"))
            
            logger.debug("cache_write_entry", key=key, token_count=len(tokens), model_key=self.model_key)
            
            return final
        except Exception as exc:
            logger.warning("failed to persist cache entry %s: %s", key, exc)
            tmp.unlink(missing_ok=True)
            Path(str(tmp) + ".json").unlink(missing_ok=True)
            return None

    def _load_disk_index(self) -> None:
        for sidecar in self.dir.glob("*.safetensors.json"):
            try:
                meta = json.loads(sidecar.read_text())
                # Handle migration from v1 to v2
                if meta.get("version", 1) < FORMAT_VERSION:
                    self._migrate_entry(meta, sidecar)
                    continue
                if meta.get("version") != FORMAT_VERSION:
                    continue
                if meta.get("model_key") != self.model_key:
                    continue
                tokens = meta["tokens"]
                data_path = Path(str(sidecar)[: -len(".json")])
                if not data_path.exists():
                    continue
                key = _tokens_digest(tokens)
                self._entries[key] = _Entry(
                    tokens=tokens,
                    cache=None,
                    nbytes=0,
                    last_used=data_path.stat().st_mtime,
                    path=data_path,
                )
                self._index(key, tokens)
            except Exception as exc:
                logger.warning("skipping bad cache sidecar %s: %s", sidecar, exc)

    def _migrate_entry(self, meta: dict, sidecar: Path) -> None:
        """Migrate a v1 entry to v2 format."""
        try:
            old_version = meta.get("version", 1)
            logger.info("Migrating cache entry from v%s to v%s: %s", old_version, FORMAT_VERSION, sidecar)
            
            # Load the cache to get token count
            data_path = Path(str(sidecar)[: -len(".json")])
            if not data_path.exists():
                logger.warning("Cache data file missing for %s, skipping migration", sidecar)
                return
            
            # v1 didn't have n_tokens in metadata, so we load the cache to count tokens
            try:
                prompt_cache = load_prompt_cache(str(data_path))
                n_tokens = sum(layer.state[0].shape[1] for layer in prompt_cache if hasattr(layer, 'state') and layer.state)
                # Fallback: use token list length from meta
                if n_tokens == 0 and "tokens" in meta:
                    n_tokens = len(meta["tokens"])
            except Exception:
                n_tokens = len(meta.get("tokens", []))
            
            # Update metadata with v2 fields
            meta["version"] = FORMAT_VERSION
            meta["model_key"] = meta.get("model_key", self.model_key)
            meta["n_tokens"] = n_tokens
            meta["created"] = meta.get("created", sidecar.stat().st_mtime)
            meta["ttl_days"] = self.cache_ttl_days
            
            # Write updated sidecar
            sidecar.write_text(json.dumps(meta))
            logger.info("Successfully migrated cache entry %s to v%d", sidecar, FORMAT_VERSION)
        except Exception as exc:
            logger.warning("Failed to migrate cache entry %s: %s", sidecar, exc)
            # Remove corrupted entry
            sidecar.unlink(missing_ok=True)
            Path(str(sidecar)[:-5]).unlink(missing_ok=True)

    def _prune_disk_by_ttl(self) -> None:
        """Remove disk entries older than cache_ttl_days."""
        if self.cache_ttl_days is None:
            return
        cutoff = time.time() - (self.cache_ttl_days * 86400)
        for sidecar in self.dir.glob("*.safetensors.json"):
            try:
                meta = json.loads(sidecar.read_text())
                created = meta.get("created", sidecar.stat().st_mtime)
                if created < cutoff:
                    data_path = Path(str(sidecar)[: -len(".json")])
                    logger.info("TTL eviction: removing cache entry older than %d days: %s", 
                               self.cache_ttl_days, sidecar)
                    data_path.unlink(missing_ok=True)
                    sidecar.unlink(missing_ok=True)
                    # Remove from index if present
                    tokens = meta.get("tokens", [])
                    key = _tokens_digest(tokens)
                    with self._lock:
                        self._entries.pop(key, None)
            except Exception as exc:
                logger.warning("Error during TTL pruning of %s: %s", sidecar, exc)
        with self._lock:
            self._rebuild_index()

    # ------------------------------------------------------------- eviction

    def _evict_ram(self) -> None:
        with self._lock:
            resident = [e for e in self._entries.values() if e.cache is not None]
            total = sum(e.nbytes for e in resident)
            if total <= self.max_ram_bytes:
                return
            resident.sort(key=lambda e: e.last_used)
            for entry in resident:
                if total <= self.max_ram_bytes:
                    break
                # Drop from RAM; keep the entry if it lives on disk.
                total -= entry.nbytes
                entry.cache = None
                entry.nbytes = 0
                if entry.path is None:
                    self._entries.pop(_tokens_digest(entry.tokens), None)
            self._rebuild_index()

    def _evict_disk(self) -> None:
        files = sorted(
            self.dir.glob("*.safetensors"), key=lambda p: p.stat().st_mtime
        )
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= self.max_disk_bytes:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            Path(str(path) + ".json").unlink(missing_ok=True)
            total -= size
            for key, entry in list(self._entries.items()):
                if entry.path == path:
                    if entry.cache is None:
                        self._entries.pop(key, None)
                    else:
                        entry.path = None
        with self._lock:
            self._rebuild_index()

    # ---------------------------------------------------------------- stats

    def stats(self) -> dict:
        with self._lock:
            resident = [e for e in self._entries.values() if e.cache is not None]
            return {
                "entries": len(self._entries),
                "resident_entries": len(resident),
                "resident_bytes": sum(e.nbytes for e in resident),
                "hits": self.hits,
                "misses": self.misses,
                "copy_seconds": self.copy_seconds,
                "lookup_seconds": self.lookup_seconds,
                "load_seconds": self.load_seconds,
            }

    def clear(self) -> int:
        """Remove all cached prefixes for this model and return their count."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._rebuild_index()
            self.hits = 0
            self.misses = 0
        for path in self.dir.glob("*.safetensors"):
            path.unlink(missing_ok=True)
            Path(str(path) + ".json").unlink(missing_ok=True)
        return count

    def trim_ram(self, target_bytes: int = 0) -> int:
        """Drop least-recently-used RAM copies until at or below target."""
        with self._lock:
            resident = [e for e in self._entries.values() if e.cache is not None]
            before = sum(e.nbytes for e in resident)
            resident.sort(key=lambda e: e.last_used)
            total = before
            for entry in resident:
                if total <= target_bytes:
                    break
                total -= entry.nbytes
                entry.cache = None
                entry.nbytes = 0
                if entry.path is None:
                    self._entries.pop(_tokens_digest(entry.tokens), None)
            self._rebuild_index()
            return before - total

    # ----------------------------------------------------------- CLI helpers

    def list_entries(self, model_filter: Optional[str] = None) -> List[dict]:
        """List all cache entries with metadata for CLI display."""
        entries = []
        with self._lock:
            for key, entry in self._entries.items():
                if model_filter and self.model_key != model_filter:
                    continue
                entry_info = {
                    "key": key,
                    "model_key": self.model_key,
                    "token_count": len(entry.tokens),
                    "size_bytes": entry.nbytes if entry.cache is not None else (
                        entry.path.stat().st_size if entry.path and entry.path.exists() else 0
                    ),
                    "in_ram": entry.cache is not None,
                    "on_disk": entry.path is not None and entry.path.exists(),
                    "last_used": entry.last_used,
                    "age_seconds": time.time() - entry.last_used,
                    "hits": getattr(entry, 'hits', 0),
                }
                entries.append(entry_info)
        # Sort by last used (most recent first)
        entries.sort(key=lambda e: e["last_used"], reverse=True)
        return entries

    def inspect_entry(self, key: str) -> Optional[dict]:
        """Get detailed info about a specific cache entry."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            
            info = {
                "key": key,
                "model_key": self.model_key,
                "tokens": entry.tokens,
                "token_count": len(entry.tokens),
                "size_bytes": entry.nbytes if entry.cache is not None else (
                    entry.path.stat().st_size if entry.path and entry.path.exists() else 0
                ),
                "in_ram": entry.cache is not None,
                "on_disk": entry.path is not None and entry.path.exists(),
                "path": str(entry.path) if entry.path else None,
                "last_used": entry.last_used,
                "created": None,
                "hits": getattr(entry, 'hits', 0),
            }
            
            # Try to get creation time from sidecar
            if entry.path and entry.path.exists():
                sidecar = Path(str(entry.path) + ".json")
                if sidecar.exists():
                    try:
                        meta = json.loads(sidecar.read_text())
                        info["created"] = meta.get("created")
                        info["version"] = meta.get("version")
                    except Exception:
                        pass
            
            return info

    def prune_by_ttl(self, ttl_days: Optional[int] = None, model_filter: Optional[str] = None) -> int:
        """Prune disk entries older than TTL days. Returns count of removed entries."""
        ttl = ttl_days if ttl_days is not None else self.cache_ttl_days
        if ttl is None:
            return 0
        
        cutoff = time.time() - (ttl * 86400)
        removed = 0
        
        for sidecar in self.dir.glob("*.safetensors.json"):
            try:
                meta = json.loads(sidecar.read_text())
                if model_filter and meta.get("model_key") != model_filter:
                    continue
                created = meta.get("created", sidecar.stat().st_mtime)
                if created < cutoff:
                    data_path = Path(str(sidecar)[:-5])  # remove .json
                    data_path.unlink(missing_ok=True)
                    sidecar.unlink(missing_ok=True)
                    removed += 1
                    # Remove from index
                    tokens = meta.get("tokens", [])
                    key = _tokens_digest(tokens)
                    with self._lock:
                        self._entries.pop(key, None)
            except Exception as exc:
                logger.warning("Error pruning cache entry %s: %s", sidecar, exc)
        
        with self._lock:
            self._rebuild_index()
        
        return removed

    def warm_from_history(
        self,
        source: str,
        model: str,
        engine,
        limit: Optional[int] = None,
    ) -> int:
        """Warm cache from chat history exports."""
        import json
        from pathlib import Path
        
        if source == "openwebui":
            # OpenWebUI exports: {"chats": [{"messages": [...]}, ...]}
            history_path = Path.home() / ".local" / "share" / "open-webui" / "chats.json"
            if not history_path.exists():
                history_path = Path.home() / "Library" / "Application Support" / "OpenWebUI" / "chats.json"
            if not history_path.exists():
                raise FileNotFoundError(f"OpenWebUI chat history not found at {history_path}")
            data = json.loads(history_path.read_text())
            conversations = data.get("chats", [])
            prompts = []
            for chat in conversations:
                messages = chat.get("messages", [])
                if messages:
                    prompts.append({"messages": messages})
                    
        elif source == "opencode":
            # OpenCode exports: [{"messages": [...]}]
            history_path = Path.home() / ".config" / "opencode" / "history.json"
            if not history_path.exists():
                raise FileNotFoundError(f"OpenCode history not found at {history_path}")
            data = json.loads(history_path.read_text())
            prompts = data if isinstance(data, list) else data.get("conversations", [])
            
        elif source == "hermes":
            # Hermes exports: ~/.hermes/conversations/*.json
            history_dir = Path.home() / ".hermes" / "conversations"
            if not history_dir.exists():
                raise FileNotFoundError(f"Hermes conversations not found at {history_dir}")
            prompts = []
            for conv_file in sorted(history_dir.glob("*.json")):
                data = json.loads(conv_file.read_text())
                if "messages" in data:
                    prompts.append({"messages": data["messages"]})
                    
        else:
            raise ValueError(f"Unknown history source: {source}. Use openwebui, opencode, or hermes")
        
        if limit:
            prompts = prompts[:limit]
        
        warmed = 0
        for i, item in enumerate(prompts):
            messages = item.get("messages", [])
            if not messages:
                continue
            tokens = engine.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            if self.fetch(tokens) is not None:
                print(f"[{i}] already cached ({len(tokens)} tokens)")
                continue
            cache = engine.make_cache()
            report = engine.paced_prefill(tokens, cache)
            self.put(tokens[:report.computed_tokens], cache)
            print(f"[{i}] cached {report.computed_tokens} tokens")
            warmed += 1
            
        return warmed
