"""Model-config discovery for context limits and KV admission estimates.

Review finding on the original commits: mlx-lm models expose ``args`` (a
ModelArgs dataclass), never ``.config``, and vision-wrapped checkpoints
(Qwen3.5) nest the text fields another level down in ``text_config`` — so
both ``model_context_limit`` and ``estimate_kv_cache_bytes`` were silent
no-ops on every real model, exercised only by fake-model tests. These tests
pin the real shapes, plus hybrid layer accounting (only full-attention
layers grow per-token KV).
"""

from types import SimpleNamespace

from daedalus.server import estimate_kv_cache_bytes, model_context_limit


def qwen35_like():
    # Mirrors mlx-lm's Qwen3.5: outer ModelArgs with a text_config dict.
    return SimpleNamespace(
        args=SimpleNamespace(
            model_type="qwen3_5",
            text_config={
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
                "full_attention_interval": 4,
                "max_position_embeddings": 131072,
            },
        )
    )


def test_context_limit_found_via_args_text_config():
    assert model_context_limit(qwen35_like()) == 131072


def test_kv_estimate_counts_only_full_attention_layers():
    estimate = estimate_kv_cache_bytes(qwen35_like(), tokens=8192, kv_bits=8)
    # 8 full-attention layers (32 // full_attention_interval=4), not 32:
    # keys+values, 8 KV heads, head_dim 4096/32=128, 1.2x quant overhead.
    expected = int(8 * 2 * 8 * 128 * 8192 * (8 / 8.0) * 1.2)
    assert estimate == expected


def test_kv_estimate_honors_layer_types_list():
    model = SimpleNamespace(
        config={
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "layer_types": ["linear_attention"] * 24 + ["full_attention"] * 8,
        }
    )
    assert estimate_kv_cache_bytes(model, tokens=1000, kv_bits=None) == (
        8 * 2 * 8 * 128 * 1000 * 2
    )


def test_kv_estimate_pure_recurrent_returns_none():
    model = SimpleNamespace(
        config={
            "num_hidden_layers": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "layer_types": ["linear_attention"] * 24,
        }
    )
    assert estimate_kv_cache_bytes(model, tokens=1000, kv_bits=8) is None


def test_no_config_at_all_returns_none():
    class Bare:
        pass

    assert model_context_limit(Bare()) is None
    assert estimate_kv_cache_bytes(Bare(), tokens=1000, kv_bits=8) is None
