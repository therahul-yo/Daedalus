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
from test_server import FakeEngine, FakeGovernor, FakeStore

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


def _client(app):
    return __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app)


def test_invalid_request_does_not_trigger_a_model_swap():
    app, *_ = make_app()
    loaded = []
    original_loader = app.state.daedalus.model_loader

    def loader(model_id):
        loaded.append(model_id)
        return original_loader(model_id)

    app.state.daedalus.model_loader = loader
    client = _client(app)
    # Empty messages fail structural validation (400) BEFORE any swap.
    response = client.post("/v1/chat/completions", json={"model": "model-a", "messages": []})
    assert response.status_code == 400
    assert loaded == []


def test_models_lists_registered_candidates_and_marks_the_resident_one():
    app, *_ = make_app()
    client = _client(app)
    models = {item["id"]: item for item in client.get("/v1/models").json()["data"]}
    assert set(models) == {"default", "model-a", "model-b"}
    assert models["default"]["resident"] is True
    assert models["model-a"]["resident"] is False


def test_swap_does_not_stop_a_service_owned_monitor():
    from daedalus.engine import Engine

    class Monitor:
        def __init__(self):
            self.stopped = 0

        def stop(self):
            self.stopped += 1

    _, default_store, _ = make_spec("default-model-text")
    target_eng, target_store, target_path = make_spec("model-a-text")
    monitor = Monitor()
    # owns_monitor defaults False: the service owns the shared monitor, so a
    # swap that tears down this engine must not stop it.
    service_engine = Engine(None, None, FakeGovernor(), monitor=monitor)
    app = create_app(
        service_engine,
        default_store,
        model_id="default",
        model_paths={"model-a": target_path},
        model_loader=lambda _: (target_eng, target_store),
    )
    ok, message = app.state.daedalus.swap_model("model-a")
    assert ok, message
    assert monitor.stopped == 0


def test_swap_shuts_down_the_old_engine():
    calls = []

    class TrackingEngine(FakeEngine):
        def __init__(self, script):
            super().__init__(script=script)

        def close(self):
            calls.append("close")

        def shutdown(self):
            calls.append("shutdown")

    default_eng = TrackingEngine("default-model-text")
    _, default_store, _ = make_spec("x")
    a_eng, a_store, a_path = make_spec("model-a-text")
    app = create_app(
        default_eng,
        default_store,
        model_id="default",
        model_paths={"model-a": a_path},
        model_loader=lambda _: (a_eng, a_store),
    )
    ok, message = app.state.daedalus.swap_model("model-a")
    assert ok, message
    assert calls == ["close", "shutdown"]


def test_runtime_snapshot_is_coherent_after_swap():
    app, default_eng, a_eng, _ = make_app()
    state = app.state.daedalus
    eng, store, model_id, _, _ = state.runtime_snapshot()
    assert eng is default_eng and model_id == "default"
    ok, message = state.swap_model("model-a")
    assert ok, message
    eng, store, model_id, _, _ = state.runtime_snapshot()
    assert eng is a_eng and model_id == "model-a"
    assert store is state.store


def test_swap_refused_when_shutting_down():
    app, *_ = make_app()
    state = app.state.daedalus
    with state.admission_lock:
        state.accepting = False
    ok, message = state.swap_model("model-a")
    assert not ok
    assert message == "server is shutting down"


def test_served_model_id_reflects_swap_target_non_streaming():
    app, *_ = make_app()
    client = _client(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["model"] == "model-a"


def test_served_model_id_reflects_swap_target_streaming():
    app, *_ = make_app()
    client = _client(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        assert r.status_code == 200
        raw = "".join(r.iter_text())
    import json as _json
    lines = [line for line in raw.split("\n") if line.startswith("data: ") and "[DONE]" not in line]
    chunks = [_json.loads(line[len("data: "):]) for line in lines]
    assert all(c["model"] == "model-a" for c in chunks)
