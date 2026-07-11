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

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "airlift" / "prefix"


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


class PrefixCacheStore:
    def __init__(
        self,
        model_key: str,
        cache_dir: Optional[Path] = None,
        max_ram_bytes: Optional[int] = None,
        max_disk_bytes: int = 10 * 1024**3,
        min_persist_tokens: int = 1024,
    ) -> None:
        self.model_key = model_key
        self.dir = (cache_dir or _default_cache_dir()) / _sanitize_model_key(model_key)
        self.dir.mkdir(parents=True, exist_ok=True)
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
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self._load_disk_index()

    # ---------------------------------------------------------------- fetch

    def fetch(self, tokens: List[int]) -> Optional[CacheHit]:
        """Best usable cached prefix for ``tokens`` (deep-copied), or None.

        The returned ``matched_tokens`` is always < len(tokens) so at least
        one token remains for the engine to process (decode needs a final
        input token). Superset/overlong entries are trimmed when possible.
        """
        with self._lock:
            best_key, best_usable = None, 0
            for key, entry in self._entries.items():
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
                return None
            entry = self._entries[best_key]

        cache_obj, source = self._materialize(entry)
        if cache_obj is None:
            self.misses += 1
            return None

        overhang = len(entry.tokens) - best_usable
        if overhang > 0:
            if not can_trim_prompt_cache(cache_obj):
                # Non-trimmable (hybrid/rotating) cache: only a pure-prefix
                # entry is usable. Fall back to a strict-prefix search.
                return self._fetch_strict_prefix(tokens, exclude=best_key)
            trim_prompt_cache(cache_obj, overhang)

        with self._lock:
            entry.last_used = time.monotonic()
            self.hits += 1
        return CacheHit(cache=cache_obj, matched_tokens=best_usable, source=source)

    def _fetch_strict_prefix(
        self, tokens: List[int], exclude: str
    ) -> Optional[CacheHit]:
        with self._lock:
            best_key, best_len = None, 0
            for key, entry in self._entries.items():
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
            return copy.deepcopy(entry.cache), "ram"
        try:
            loaded = load_prompt_cache(str(entry.path))
        except Exception as exc:  # corrupt/stale file: drop the entry
            logger.warning("dropping unreadable cache entry %s: %s", entry.path, exc)
            with self._lock:
                self._entries.pop(_tokens_digest(entry.tokens), None)
            if entry.path:
                entry.path.unlink(missing_ok=True)
                Path(str(entry.path) + ".json").unlink(missing_ok=True)
            return None, "disk"
        entry.cache = loaded
        entry.nbytes = cache_nbytes(loaded)
        self._evict_ram()
        return copy.deepcopy(loaded), "disk"

    # ------------------------------------------------------------------ put

    def put(
        self, tokens: List[int], prompt_cache: List[Any], persist: bool = True
    ) -> None:
        """Store (a deep copy of) ``prompt_cache`` for ``tokens``."""
        if len(tokens) == 0:
            return
        key = _tokens_digest(tokens)
        stored = copy.deepcopy(prompt_cache)
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
        if persist and len(tokens) >= self.min_persist_tokens:
            self._persist(key, entry)
        self._evict_ram()
        self._evict_disk()

    def checkpoint(self, tokens: List[int], done: int, prompt_cache: List[Any]) -> None:
        """Persist the live cache state for ``tokens[:done]`` (no deep copy in
        RAM — written straight to disk so a retry can resume)."""
        if done < self.min_persist_tokens:
            return
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
            return
        entry.path = path
        with self._lock:
            existing = self._entries.get(key)
            if existing is None or existing.cache is None:
                self._entries[key] = entry
        self._evict_disk()

    # ---------------------------------------------------------- persistence

    def _persist(self, key: str, entry: _Entry) -> None:
        path = self._write_entry_files(key, entry.tokens, entry.cache)
        if path is not None:
            entry.path = path

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
                    "airlift_format": str(FORMAT_VERSION),
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
            except Exception as exc:
                logger.warning("skipping bad cache sidecar %s: %s", sidecar, exc)

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
            }
