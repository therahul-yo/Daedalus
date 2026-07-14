"""Multi-model hot-swap tests (16GB M4 Air, swap-only).

No real weights: a ``FakeEngine`` per model carries a distinct script so the
response proves which engine actually served the request after a swap.
"""

from daedalus.server import (
    create_app,
    model_fits,
    MODEL_PROFILES,
    MODEL_MEMORY_CEILING_GB,
)
from daedalus.server import ModelProfile
from test_server import FakeEngine, FakeStore

# Give the test model ids deliberately small profiles so two fit under the
# 16GB ceiling (the fallback profile would otherwise over-count KV at 64K ctx
# and reject every swap — which is the *correct* production behaviour).
for _mid in ("default", "model-a", "model-b"):
    MODEL_PROFILES[_mid] = ModelProfile(_mid, weights_gb=1.0, kv_gb_per_8k=0.1, kv_gb_per_32k=0.4)


# A store whose fetch is keyed so cache hits prove per-model isolation.
def make_spec(script):
    class ScriptedEngine(FakeEngine):
        def __init__(self):
            super().__init__(script=script)

    return ScriptedEngine(), FakeStore(), f"/models/{script}"


def make_app():
    default_eng, default_store, _ = make_spec("default-model-text")
    a_eng, a_store, a_path = make_spec("model-a-text")
    b_eng, b_store, b_path = make_spec("model-b-text")
    specs = {
        "default": (default_eng, default_store),
        "model-a": (a_eng, a_store),
        "model-b": (b_eng, b_store),
    }
    app = create_app(
        default_eng, default_store, model_id="default",
        model_paths={"model-a": a_path, "model-b": b_path},
        model_loader=specs.__getitem__,
    )
    return app, default_eng, a_eng, b_eng


def test_unknown_model_returns_404():
    app, *_ = make_app()
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"


def test_swap_to_registered_model_serves_with_that_engine():
    app, default_eng, a_eng, b_eng = make_app()
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    # Default model first.
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert "default-model-text" in r.json()["choices"][0]["message"]["content"]
    # Swap to model-a by naming it.
    r = client.post(
        "/v1/chat/completions",
        json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    assert "model-a-text" in r.json()["choices"][0]["message"]["content"]
    # Subsequent default (no model field) request now serves model-a (active).
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert "model-a-text" in r.json()["choices"][0]["message"]["content"]


def test_swap_candidates_are_not_loaded_at_startup():
    default_eng, default_store, _ = make_spec("default-model-text")
    a_eng, a_store, a_path = make_spec("model-a-text")
    loaded = []

    def loader(model_id):
        loaded.append(model_id)
        assert model_id == "model-a"
        return a_eng, a_store

    app = create_app(
        default_eng,
        default_store,
        model_id="default",
        model_paths={"model-a": a_path},
        model_loader=loader,
    )
    assert loaded == []
    ok, message = app.state.daedalus.swap_model("model-a")
    assert ok, message
    assert loaded == ["model-a"]


def test_swap_cooldown_rejects_second_swap():
    app, default_eng, a_eng, b_eng = make_app()
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    # First swap succeeds.
    r = client.post(
        "/v1/chat/completions",
        json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    # Immediate swap to model-b hits cooldown -> 409.
    r = client.post(
        "/v1/chat/completions",
        json={"model": "model-b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 409
    assert r.json()["error"]["type"] == "model_swap_conflict"


def test_swap_timeout_does_not_repoint_or_release_an_unheld_gate(monkeypatch):
    app, default_eng, a_eng, _ = make_app()
    state = app.state.daedalus
    monkeypatch.setattr(state.lock, "acquire_for_swap", lambda timeout=5.0: False)
    ok, message = state.swap_model("model-a")
    assert not ok
    assert message == "engine busy, retry"
    assert state.engine is default_eng
    assert state.model_id == "default"
    assert a_eng.generate_calls == []


def test_swap_waits_for_post_engine_work_before_teardown():
    app, *_ = make_app()
    state = app.state.daedalus
    state.start_engine_task()
    assert not state.wait_for_engine_drain(timeout=0.0)
    state.finish_engine_task()
    assert state.wait_for_engine_drain(timeout=0.0)


def test_cache_isolation_across_models():
    """Each model keeps its own store; a prefix cached for one is not seen by another."""
    app, default_eng, a_eng, b_eng = make_app()
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    msgs = [{"role": "user", "content": "cached prefix please"}]
    # Prime default model cache.
    client.post("/v1/chat/completions", json={"messages": msgs})
    # Swap to model-a; its store must be independent (no cross-model hit).
    r = client.post("/v1/chat/completions", json={"model": "model-a", "messages": msgs})
    assert r.status_code == 200
    # model-a's engine must NOT have seen the prompt as already-cached.
    assert a_eng.generate_calls
    assert a_eng.generate_calls[0]["already_cached"] == 0


def test_admission_rejects_unfit_model():
    """A model that exceeds the memory ceiling (given active) is rejected at 409."""
    # Use derive_model_profile on a fake big id by injecting into MODEL_PROFILES.
    MODEL_PROFILES["big-fake"] = type(MODEL_PROFILES["qwen-7b"])(
        model_id="big-fake", weights_gb=12.0, kv_gb_per_8k=1.0, kv_gb_per_32k=4.0
    )
    app, default_eng, a_eng, b_eng = make_app()
    # Register big-fake lazily; admission must reject it before loading.
    big_eng, big_store, _ = make_spec("big-text")
    previous_loader = app.state.daedalus.model_loader
    app.state.daedalus.model_loader = (
        lambda model_id: (big_eng, big_store)
        if model_id == "big-fake"
        else previous_loader(model_id)
    )
    app.state.daedalus.served_models.add("big-fake")
    app.state.daedalus.model_paths["big-fake"] = "/models/big"
    client = __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "big-fake", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 409
    assert r.json()["error"]["type"] == "model_swap_conflict"
    # Clean up registry mutation.
    MODEL_PROFILES.pop("big-fake", None)


def test_model_fits_math():
    small = MODEL_PROFILES["qwen-3b"]
    big = MODEL_PROFILES["qwen-14b"] if "qwen-14b" in MODEL_PROFILES else type(
        MODEL_PROFILES["qwen-7b"]
    )(model_id="qwen-14b", weights_gb=9.5, kv_gb_per_8k=1.2, kv_gb_per_32k=4.8)
    # With no active model, small fits within the ceiling.
    fits, avail, req = model_fits(small, None, 8192)
    assert fits
    assert avail == MODEL_MEMORY_CEILING_GB
    # With a 7B active, a second 14B does not fit (insufficient free headroom).
    active = MODEL_PROFILES["qwen-7b"]
    fits, avail, req = model_fits(big, active, 8192)
    assert not fits
