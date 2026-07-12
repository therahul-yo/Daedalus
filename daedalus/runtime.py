"""Runtime identity and resource helpers used to keep persisted state safe."""

from __future__ import annotations

import hashlib
import platform


def cache_identity(model: str, *, kv_bits: int | None, kv_group_size: int = 64,
                   tokenizer_id: str | None = None, model_revision: str | None = None,
                   draft_model: str | None = None) -> str:
    """Return a cache namespace that cannot cross incompatible runtimes.

    The result becomes a directory name, so it must stay short: macOS caps
    filenames at 255 bytes, and tokenizer_id is often a full local snapshot
    path. Format: ``<model-tail>--<16-hex digest of all fields>``.
    """
    try:
        import mlx.core as _mx
        import mlx_lm as _mlx_lm

        mlx_version = getattr(_mx, "__version__", "unknown")
        mlx_lm_version = getattr(_mlx_lm, "__version__", "unknown")
    except Exception:
        mlx_version = mlx_lm_version = "unknown"
    fields = {
        "model": model,
        "model_revision": model_revision or "unresolved",
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
