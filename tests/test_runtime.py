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
