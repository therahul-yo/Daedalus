"""Runtime identity and resource helpers used to keep persisted state safe."""

from __future__ import annotations

import platform


def cache_identity(model: str, *, kv_bits: int | None, kv_group_size: int = 64,
                   tokenizer_id: str | None = None, model_revision: str | None = None,
                   draft_model: str | None = None) -> str:
    """Return a cache namespace that cannot cross incompatible runtimes."""
    try:
        import mlx
        import mlx_lm

        mlx_version = mlx.__version__
        mlx_lm_version = mlx_lm.__version__
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
    return "|".join(f"{name}={value}" for name, value in fields.items())
