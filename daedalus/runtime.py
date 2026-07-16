"""Runtime identity and resource helpers used to keep persisted state safe."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path


def _checkpoint_content_digest(*candidates: str | None) -> str | None:
    """Short digest of a local checkpoint's weight identity, or None.

    Hashes ``config.json`` plus ``model.safetensors.index.json`` (or, for a
    single-file checkpoint, the first 64KB of ``model.safetensors`` — its
    tensor header). ``candidates`` are probed in order (resolved model dir,
    then the tokenizer path, which is usually the local snapshot dir); the
    first that resolves to a readable checkpoint wins. Returns None for a
    remote-only or unreadable path so the caller can fall back safely.
    """
    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = Path(candidate)
            if not path.exists():
                continue
            base = path if path.is_dir() else path.parent
            config = base / "config.json"
            if not config.exists():
                continue
            h = hashlib.sha256()
            h.update(config.read_bytes())
            index = base / "model.safetensors.index.json"
            if index.exists():
                h.update(index.read_bytes())
            else:
                single = base / "model.safetensors"
                if single.exists():
                    with single.open("rb") as f:
                        h.update(f.read(65536))
            return h.hexdigest()[:16]
        except Exception:
            continue
    return None


def cache_identity(model: str, *, kv_bits: int | None, kv_group_size: int = 64,
                   tokenizer_id: str | None = None, model_revision: str | None = None,
                   draft_model: str | None = None) -> str:
    """Return a cache namespace that cannot cross incompatible runtimes.

    The result becomes a directory name, so it must stay short: macOS caps
    filenames at 255 bytes, and tokenizer_id is often a full local snapshot
    path. Format: ``<model-tail>--<16-hex digest of all fields>``.

    The digest folds in a short content hash of the local checkpoint's
    ``config.json``/index so that re-quantizing or replacing the weights at the
    same path no longer silently reuses stale KV snapshots. Remote-only paths
    hash as ``nohash`` (current behaviour). NOTE: adding this field
    intentionally invalidates every existing cache directory once.
    """
    try:
        import mlx.core as _mx
        import mlx_lm as _mlx_lm

        mlx_version = getattr(_mx, "__version__", "unknown")
        mlx_lm_version = getattr(_mlx_lm, "__version__", "unknown")
    except Exception:
        mlx_version = mlx_lm_version = "unknown"
    content = _checkpoint_content_digest(model, tokenizer_id) or "nohash"
    fields = {
        "model": model,
        "model_revision": model_revision or "unresolved",
        "content": content,
        "tokenizer": tokenizer_id or model,
        # A draft model adds a second cache layout; never share its snapshots
        # with target-only inference or a different draft model.
        "draft_model": draft_model or "none",
        "kv_bits": kv_bits or 16,
        "kv_group": kv_group_size,
        "mlx": mlx_version,
        "mlx_lm": mlx_lm_version,
        "machine": platform.machine(),
    }
    canonical = "|".join(f"{name}={value}" for name, value in fields.items())
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    # Keep the model name visible for humans browsing ~/.cache/daedalus.
    tail = model.rsplit("/", 1)[-1][:80]
    return f"{tail}--{digest}"
