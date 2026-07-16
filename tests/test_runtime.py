from daedalus.runtime import cache_identity


def test_cache_identity_changes_for_kv_layout():
    assert cache_identity("model", kv_bits=8) != cache_identity("model", kv_bits=4)


def test_cache_identity_includes_model():
    assert cache_identity("model-a", kv_bits=8) != cache_identity("model-b", kv_bits=8)


def test_cache_identity_includes_pinned_revision():
    assert cache_identity("model", kv_bits=8, model_revision="one") != cache_identity(
        "model", kv_bits=8, model_revision="two"
    )


def test_cache_identity_includes_draft_model():
    assert cache_identity("model", kv_bits=8) != cache_identity(
        "model", kv_bits=8, draft_model="draft-model"
    )


def test_cache_identity_is_filesystem_safe_length():
    """serve was broken on macOS: the identity embedded a full local
    snapshot path and blew the 255-byte filename limit (E2E regression)."""
    key = cache_identity(
        "mlx-community/Qwen3.5-9B-MLX-4bit",
        kv_bits=8,
        tokenizer_id="/Users/x/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/" + "a" * 64,
    )
    assert len(key) <= 120
    assert "/" not in key


def test_cache_identity_changes_with_checkpoint_content(tmp_path):
    """Re-quantizing weights at the same path must not reuse stale snapshots:
    the identity folds in a digest of the checkpoint's config.json."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"quantization": {"bits": 4}}')
    id_v1 = cache_identity(str(ckpt), kv_bits=8)

    # Same path, different weights (re-quant to 8-bit).
    (ckpt / "config.json").write_text('{"quantization": {"bits": 8}}')
    id_v2 = cache_identity(str(ckpt), kv_bits=8)

    assert id_v1 != id_v2


def test_cache_identity_folds_in_safetensors_index(tmp_path):
    """A multi-shard index change (added/renamed shard) invalidates too."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text('{"model_type": "qwen3"}')
    (ckpt / "model.safetensors.index.json").write_text('{"weight_map": {"a": "s1"}}')
    id_a = cache_identity(str(ckpt), kv_bits=8)
    (ckpt / "model.safetensors.index.json").write_text('{"weight_map": {"a": "s2"}}')
    id_b = cache_identity(str(ckpt), kv_bits=8)
    assert id_a != id_b


def test_cache_identity_resolves_via_tokenizer_snapshot_dir(tmp_path):
    """When ``model`` is a bare HF id, the local snapshot is found through the
    tokenizer path (its usual value at serve time)."""
    snap = tmp_path / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text('{"bits": 4}')
    id_1 = cache_identity("mlx-community/model", kv_bits=8, tokenizer_id=str(snap))
    (snap / "config.json").write_text('{"bits": 8}')
    id_2 = cache_identity("mlx-community/model", kv_bits=8, tokenizer_id=str(snap))
    assert id_1 != id_2


def test_cache_identity_nohash_fallback_is_stable(tmp_path):
    """A remote-only / unreadable path falls back to 'nohash' without error and
    stays deterministic across calls."""
    a = cache_identity("mlx-community/does-not-exist-locally", kv_bits=8)
    b = cache_identity("mlx-community/does-not-exist-locally", kv_bits=8)
    assert a == b
    # A directory with no config.json is treated as remote-only.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cache_identity(str(empty), kv_bits=8) == cache_identity(str(empty), kv_bits=8)
