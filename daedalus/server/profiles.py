"""Memory profiles, swap-admission math, and model-config discovery.

Pure logic with no FastAPI dependency: model memory profiles derived from the
checkpoint on disk, the admission arithmetic that keeps a single model resident
on a 16GB M4 Air, and best-effort context-window / KV-cache estimation across
the config spellings common to MLX models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

# Quantized KV (the server's k8v8 cache) packs each K/V element into ~1 byte
# but carries per-group scale/zero-point overhead; reserve 20% above the ideal
# packed size.  Single source of truth for both the reactive KV estimator
# (``estimate_kv_cache_bytes``) and the swap-admission profile derivation
# (``derive_model_profile``).
KV_QUANT_OVERHEAD = 1.2


# ──────────────────────────────────────────────────────────────────────────
# Multi-model: memory profiles + admission (16GB M4 Air, swap-only)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Memory profile for a single model.

    Sizes are derived from the checkpoint on disk (exact, zero maintenance);
    see ``derive_model_profile``. The built-ins below are fallbacks / known
    hybrid-architecture overrides (where KV only grows on the GQA layers).
    """
    model_id: str
    weights_gb: float
    kv_gb_per_8k: float
    kv_gb_per_32k: float
    context_growing_layers: int = 0
    total_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0

    def kv_gb(self, context_tokens: int) -> float:
        """KV cache size at k8v8 for a given context length.

        Linearly interpolate between the 8k and 32k anchors, extrapolating on
        the same slope beyond 32768 (KV grows linearly in context length).
        """
        if context_tokens <= 8192:
            return self.kv_gb_per_8k * (context_tokens / 8192)
        slope = (self.kv_gb_per_32k - self.kv_gb_per_8k) / (32768 - 8192)
        return self.kv_gb_per_8k + slope * (context_tokens - 8192)

    def total_gb(self, context_tokens: int) -> float:
        """Total RAM footprint: weights + KV + 0.5GB Metal cache high-water."""
        return self.weights_gb + self.kv_gb(context_tokens) + 0.5


# 16GB M4 Air: 3.5 (macOS/system) + 0.8 (process) + 1.0 (safety) = 5.3 reserved
# Usable ceiling for (weights + KV) of the active model: 10.7 GB.
MODEL_MEMORY_CEILING_GB = 10.7
# Extra headroom kept free during a swap so both engines are never resident.
SWAP_SAFETY_GB = 1.0

# Known profiles (overrides for hybrid archs where only some layers grow).
MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen3.5-9b": ModelProfile(
        model_id="qwen3.5-9b", weights_gb=5.2, kv_gb_per_8k=0.65, kv_gb_per_32k=2.6,
        context_growing_layers=8, total_layers=32, hidden_size=3584, num_attention_heads=28,
    ),
    "qwen-7b": ModelProfile(
        model_id="qwen-7b", weights_gb=4.7, kv_gb_per_8k=0.58, kv_gb_per_32k=2.3,
        context_growing_layers=28, total_layers=28, hidden_size=3584, num_attention_heads=28,
    ),
    "qwen-3b": ModelProfile(
        model_id="qwen-3b", weights_gb=1.9, kv_gb_per_8k=0.24, kv_gb_per_32k=0.94,
        context_growing_layers=24, total_layers=24, hidden_size=2048, num_attention_heads=16,
    ),
}


def derive_model_profile(model_id: str, model_path: str) -> ModelProfile:
    """Derive a model's memory profile from the checkpoint on disk.

    Exact weights come from the safetensors index header; KV is estimated
    from ``num_hidden_layers`` (conservative for hybrid models — overcounts
    the constant-state layers, which is the safe direction). Built-in hybrid
    profiles (e.g. qwen3.5-9b) take precedence and are exact.
    """
    if model_id in MODEL_PROFILES:
        return MODEL_PROFILES[model_id]
    try:
        cfg = Path(model_path) / "config.json"
        if cfg.exists():
            config = json.loads(cfg.read_text())
            n_layers = int(config.get("num_hidden_layers", 0))
            hidden = int(config.get("hidden_size", 0))
            n_heads = max(int(config.get("num_attention_heads", 1)), 1)
            # GQA: KV grows on num_key_value_heads, not the full attention-head
            # count.  Absent the field, fall back to num_attention_heads (the
            # conservative, err-high direction — overcounts on GQA models).
            n_kv_heads = max(int(config.get("num_key_value_heads", n_heads)), 1)
            head_dim = int(config.get("head_dim", 0)) or (hidden // n_heads if n_heads else 0)
            import glob
            total = 0
            for sf in glob.glob(str(Path(model_path) / "*.safetensors")):
                with open(sf, "rb") as f:
                    n = int.from_bytes(f.read(8), "little")
                    header = json.loads(f.read(n))
                for k, v in header.items():
                    if k == "__metadata__":
                        continue
                    total += v["data_offsets"][1] - v["data_offsets"][0]
            weights_gb = total / 1e9
            # KV bytes at 8k = 2 (K and V) * layers * kv_heads * head_dim *
            # tokens * bytes_per_element (k8v8, see KV_QUANT_OVERHEAD). The old
            # formula dropped the kv-head count and bytes-per-element and so
            # undercounted ~10x, defeating the swap-admission guard.
            kv_per_8k = (2 * n_layers * n_kv_heads * head_dim * 8192 * KV_QUANT_OVERHEAD) / 1e9
            return ModelProfile(
                model_id, weights_gb, kv_per_8k, kv_per_8k * 4,
                n_layers, n_layers, hidden, n_heads,
            )
    except Exception:
        pass
    return ModelProfile(model_id, 5.0, 0.6, 2.4)


def model_fits(
    candidate: ModelProfile,
    active: Optional[ModelProfile],
    max_prompt_tokens: int,
) -> "tuple[bool, float, float]":
    """Whether ``candidate`` can be admitted given the ``active`` model.

    Returns (fits, available_gb, required_gb).
    """
    required = candidate.total_gb(max_prompt_tokens)
    if active is not None:
        available = MODEL_MEMORY_CEILING_GB - active.total_gb(max_prompt_tokens)
    else:
        available = MODEL_MEMORY_CEILING_GB
    # Need SWAP_SAFETY_GB free so both engines are never resident at once.
    return available >= required + SWAP_SAFETY_GB, available, required


def _model_config_owners(model: Any) -> List[Any]:
    """Config-bearing objects for a model, outermost first.

    mlx-lm models expose ``model.args`` (a ModelArgs dataclass), not
    ``.config`` — reading only ``.config`` made every downstream helper a
    silent no-op on real models. Vision-wrapped checkpoints (Qwen3.5) nest
    the text fields another level down in ``text_config``, which may be a
    dict or a dataclass depending on where in mlx-lm it was materialized.
    """
    owners: List[Any] = []
    for attr in ("config", "args"):
        cfg = getattr(model, attr, None)
        if cfg is None:
            continue
        owners.append(cfg)
        text = cfg.get("text_config") if isinstance(cfg, dict) else getattr(cfg, "text_config", None)
        if text is not None:
            owners.append(text)
    return owners


def _cfg_get(owner: Any, *names: str) -> Optional[int]:
    for name in names:
        value = owner.get(name) if isinstance(owner, dict) else getattr(owner, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def model_context_limit(model: Any) -> Optional[int]:
    """Best-effort context-window discovery across common MLX model configs."""
    for owner in _model_config_owners(model):
        value = _cfg_get(owner, "max_position_embeddings", "max_seq_len", "max_sequence_length")
        if value is not None:
            return value
    return None


def estimate_kv_cache_bytes(model: Any, tokens: int, kv_bits: Optional[int]) -> Optional[int]:
    """Conservatively estimate one sequence's target-model KV-cache footprint.

    Returning ``None`` means the architecture does not expose enough standard
    config fields; callers retain the existing reactive memory guard instead
    of making up an unsafe number.
    """
    if tokens < 1:
        return None
    for config in _model_config_owners(model):
        def get(*names: str) -> Optional[int]:
            return _cfg_get(config, *names)

        layers = get("num_hidden_layers", "n_layer", "num_layers")
        heads = get("num_key_value_heads", "num_attention_heads", "n_head")
        head_dim = get("head_dim")
        if head_dim is None:
            hidden = get("hidden_size", "n_embd", "dim")
            attention_heads = get("num_attention_heads", "n_head")
            if hidden is not None and attention_heads is not None and hidden % attention_heads == 0:
                head_dim = hidden // attention_heads
        if layers is None or heads is None or head_dim is None:
            continue
        # Hybrid architectures grow per-token KV only in their full-attention
        # layers; the recurrent (Gated-DeltaNet/SSM) layers hold small
        # constant state. Counting every layer would over-reserve ~4x on
        # Qwen3.5 and spuriously reject long prompts. Two config spellings:
        # an explicit per-layer type list (HF) or an interval (mlx-lm).
        layer_types = (
            config.get("layer_types") if isinstance(config, dict)
            else getattr(config, "layer_types", None)
        )
        if isinstance(layer_types, (list, tuple)) and layer_types:
            kv_layers = sum(
                1 for t in layer_types if isinstance(t, str) and "full" in t
            )
            if kv_layers == 0:
                return None  # pure-recurrent: constant state, nothing to reserve
            layers = kv_layers
        else:
            interval = get("full_attention_interval")
            if interval is not None and interval > 1:
                layers = max(1, layers // interval)
        # keys + values.  Quantized KV has scale/zero-point overhead, so
        # reserve 20% above the ideal packed size rather than relying on an
        # optimistic bit count.  Unquantized MLX cache entries are float16.
        bytes_per_value = 2.0 if kv_bits is None else (kv_bits / 8.0) * KV_QUANT_OVERHEAD
        return int(layers * 2 * heads * head_dim * tokens * bytes_per_value)
    return None
