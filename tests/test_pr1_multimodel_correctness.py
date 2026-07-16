"""Correctness fixes for the multi-model swap path (PR fix/multimodel-correctness).

No real weights: ``FakeEngine``/``FakeStore`` from ``test_server`` stand in.
``MODEL_PROFILES`` is patched with pytest ``monkeypatch`` (never bare global
mutation) so profile sizes stay contained to each test.
"""

import json
import struct

import pytest
from fastapi.testclient import TestClient

from daedalus.server import (
    ModelProfile,
    MODEL_PROFILES,
    _Generation,
    build_prompt_tokens,
    create_app,
    derive_model_profile,
    model_fits,
)
from test_server import FakeEngine, FakeStore, FakeTokenizer

SMALL = dict(weights_gb=1.0, kv_gb_per_8k=0.1, kv_gb_per_32k=0.4)


@pytest.fixture
def small_profiles(monkeypatch):
    for mid in ("default", "model-a", "model-b"):
        monkeypatch.setitem(MODEL_PROFILES, mid, ModelProfile(mid, **SMALL))


class OffsetTokenizer(FakeTokenizer):
    """A tokenizer that appends ``pad`` sentinel ids so two models disagree."""

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def apply_chat_template(self, messages, add_generation_prompt=True, tools=None):
        base = super().apply_chat_template(messages, add_generation_prompt, tools)
        return base + [1] * self.pad


class ModelStub:
    def __init__(self, ctx):
        self.config = {"max_position_embeddings": ctx}


def _make_app(loader_specs, *, default_engine=None, default_store=None, ctx=None):
    default_engine = default_engine or FakeEngine(script="default-text")
    default_store = default_store or FakeStore()
    if ctx is not None:
        default_engine.model = ModelStub(ctx)
    app = create_app(
        default_engine, default_store, model_id="default",
        model_paths={mid: f"/models/{mid}" for mid in loader_specs},
        model_loader=lambda mid: loader_specs[mid],
    )
    return app, default_engine, default_store


# ── FIX 5: /v1/models lists every served model ───────────────────────────────
def test_v1_models_lists_all_served_resident_first(small_profiles):
    app, *_ = _make_app({"model-a": (FakeEngine(), FakeStore()),
                         "model-b": (FakeEngine(), FakeStore())})
    data = TestClient(app).get("/v1/models").json()
    ids = [m["id"] for m in data["data"]]
    assert ids == ["default", "model-a", "model-b"]
    assert data["data"][0]["resident"] is True
    assert all(m["owned_by"] == "daedalus" for m in data["data"])
    assert [m["resident"] for m in data["data"]] == [True, False, False]


# ── FIX 3: context_limit tracks the resident model after a swap ──────────────
def test_context_limit_recomputed_after_swap(small_profiles):
    a_eng = FakeEngine()
    a_eng.model = ModelStub(2048)
    app, default_eng, _ = _make_app({"model-a": (a_eng, FakeStore())}, ctx=8192)
    state = app.state.daedalus
    assert state.context_limit == 8192
    ok, msg = state.swap_model("model-a")
    assert ok, msg
    assert state.context_limit == 2048


def test_context_limit_override_survives_swap(small_profiles):
    a_eng = FakeEngine()
    a_eng.model = ModelStub(2048)
    default_eng = FakeEngine()
    default_eng.model = ModelStub(8192)
    app = create_app(
        default_eng, FakeStore(), model_id="default",
        model_paths={"model-a": "/models/model-a"},
        model_loader=lambda mid: (a_eng, FakeStore()),
        model_context_tokens=999,
    )
    state = app.state.daedalus
    assert state.context_limit == 999
    ok, msg = state.swap_model("model-a")
    assert ok, msg
    assert state.context_limit == 999


# ── FIX 4: cross-tokenizer race → re-tokenize against the resident model ─────
def test_engine_retokenizes_when_swap_epoch_changed(small_profiles):
    """A swap between admission and the engine slot must rebuild the tokens
    with the NEW tokenizer, not run stale ids on the new model."""
    a_eng = FakeEngine()
    a_eng.tokenizer = OffsetTokenizer(pad=5)
    app, default_eng, _ = _make_app({"model-a": (a_eng, FakeStore())})
    default_eng.tokenizer = OffsetTokenizer(pad=0)
    state = app.state.daedalus
    messages = [{"role": "user", "content": "hi"}]
    captured_epoch = state.swap_epoch
    old_tokens = build_prompt_tokens(state, messages, None)
    # Simulate the swap landing after admission built old_tokens.
    ok, msg = state.swap_model("model-a")
    assert ok, msg
    assert state.swap_epoch != captured_epoch
    gen = _Generation(
        state=state, tokens=old_tokens, max_tokens=8, temperature=0.7, top_p=1.0,
        messages=messages, captured_epoch=captured_epoch,
    )
    list(gen._run_engine())
    # The engine that actually ran is model-a, and it saw the re-tokenized
    # (5-longer) prompt, not the stale default ids.
    assert a_eng.generate_calls
    assert a_eng.generate_calls[0]["tokens"] == len(old_tokens) + 5


def test_retokenize_that_no_longer_fits_raises_swap_conflict(small_profiles):
    a_eng = FakeEngine()
    a_eng.tokenizer = OffsetTokenizer(pad=5)
    app, default_eng, _ = _make_app({"model-a": (a_eng, FakeStore())})
    state = app.state.daedalus
    messages = [{"role": "user", "content": "hi"}]
    captured_epoch = state.swap_epoch
    old_tokens = build_prompt_tokens(state, messages, None)
    ok, msg = state.swap_model("model-a")
    assert ok, msg
    state.max_prompt_tokens = 1  # re-tokenized prompt now overflows
    gen = _Generation(
        state=state, tokens=old_tokens, max_tokens=8, temperature=0.7, top_p=1.0,
        messages=messages, captured_epoch=captured_epoch,
    )
    with pytest.raises(_Generation._SwapConflict):
        list(gen._run_engine())
    assert not a_eng.generate_calls  # engine never ran on a bad prompt
    # Regression: the _SwapConflict must not leak the FIFO engine lock, or the
    # next request deadlocks forever.  A second waiter must be able to acquire.
    assert state.lock.acquire(timeout=1.0)
    state.lock.release()


# ── FIX 6 / FIX 7a: degraded mode + no raw exception leak ────────────────────
def test_double_load_failure_enters_degraded_mode(small_profiles):
    def loader(mid):
        raise RuntimeError("secret path /Users/rahul/model")

    app = create_app(
        FakeEngine(), FakeStore(), model_id="default",
        model_paths={"model-a": "/models/model-a"}, model_loader=loader,
    )
    state = app.state.daedalus
    ok, msg = state.swap_model("model-a")
    assert not ok
    assert state.degraded is True
    assert state.engine is None
    # FIX 7a: the class name, never the raw exception text, reaches the caller.
    assert "RuntimeError" in msg
    assert "secret path" not in msg
    client = TestClient(app)
    assert client.get("/health").status_code == 503
    assert client.get("/readyz").status_code == 503
    r = client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    assert r.json()["error"]["message"] == "no model resident"


def test_single_load_failure_restores_old_model_without_leaking_text(small_profiles):
    restored = FakeEngine(script="restored-default")

    def loader(mid):
        if mid == "model-a":
            raise ValueError("private /Users path")
        return restored, FakeStore()

    app = create_app(
        FakeEngine(), FakeStore(), model_id="default",
        model_paths={"model-a": "/models/model-a"}, model_loader=loader,
    )
    state = app.state.daedalus
    ok, msg = state.swap_model("model-a")
    assert not ok
    assert state.degraded is False
    assert state.engine is restored  # prior model restored
    assert "ValueError" in msg
    assert "private" not in msg


# ── FIX 1: an invalid request must not trigger a swap or burn the cooldown ────
def test_invalid_request_does_not_swap_or_burn_cooldown(small_profiles):
    app, default_eng, _ = _make_app({"model-a": (FakeEngine(script="a-text"), FakeStore())})
    state = app.state.daedalus
    client = TestClient(app)
    # Structurally invalid (empty messages) but names a swap target.
    r = client.post("/v1/chat/completions",
                    json={"model": "model-a", "messages": []})
    assert r.status_code == 400
    assert state.model_id == "default"  # no swap happened
    assert state.swap_epoch == 0
    # Cooldown was not burned: a valid swap now succeeds immediately.
    r = client.post("/v1/chat/completions",
                    json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert "a-text" in r.json()["choices"][0]["message"]["content"]


# ── FIX 9: explicit JSON null means "use the default" ────────────────────────
@pytest.mark.parametrize("body", [
    {"max_tokens": None},
    {"max_completion_tokens": None},
    {"temperature": None},
    {"top_p": None},
    {"frequency_penalty": None},
    {"presence_penalty": None},
    {"stream": None},
])
def test_explicit_null_optionals_use_defaults(body):
    app = create_app(FakeEngine(), FakeStore(), model_id="m")
    payload = {"messages": [{"role": "user", "content": "hi"}], **body}
    r = TestClient(app).post("/v1/chat/completions", json=payload)
    assert r.status_code == 200, r.text


# ── FIX 8: deferred pinned snapshots never leak ──────────────────────────────
class PinStore(FakeStore):
    def __init__(self):
        super().__init__()
        self.pinned = set()

    def put(self, tokens, cache, persist=True, async_persist=False):
        super().put(tokens, cache, persist, async_persist)
        if not persist:
            self.pinned.add(tuple(tokens))

    def persist(self, tokens):
        self.pinned.discard(tuple(tokens))


def _run_gen(state, tokens):
    gen = _Generation(
        state=state, tokens=tokens, max_tokens=8, temperature=0.7, top_p=1.0,
        messages=[{"role": "user", "content": "hi"}], captured_epoch=state.swap_epoch,
    )
    return gen


def test_no_pinned_entries_after_successful_request():
    store = PinStore()
    app = create_app(FakeEngine(), store, model_id="m")
    state = app.state.daedalus
    gen = _run_gen(state, list(range(300)))
    list(gen._run_engine())
    assert store.pinned == set()


def test_no_pinned_entries_after_generation_exception():
    class ExplodingEngine(FakeEngine):
        def generate(self, tokens, *, max_tokens, temperature, top_p, prompt_cache,
                     already_cached, snap_points=None, checkpoint_cb=None,
                     should_abort=None, progress_cb=None):
            if checkpoint_cb:
                checkpoint_cb(len(tokens) - 1, prompt_cache)  # creates a pin
            raise RuntimeError("boom during decode")
            yield  # pragma: no cover — makes this a generator

    store = PinStore()
    app = create_app(ExplodingEngine(), store, model_id="m")
    state = app.state.daedalus
    gen = _run_gen(state, list(range(300)))
    with pytest.raises(RuntimeError):
        list(gen._run_engine())
    assert store.pinned == set()


# ── FIX 10: KV estimator correctness ─────────────────────────────────────────
def _write_fake_checkpoint(tmp_path, *, layers, hidden, heads, kv_heads, head_dim, weight_bytes):
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
    }))
    header = {"weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, weight_bytes]}}
    blob = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)


def test_derived_7b_profile_same_order_as_builtin(tmp_path):
    _write_fake_checkpoint(
        tmp_path, layers=28, hidden=3584, heads=28, kv_heads=4, head_dim=128,
        weight_bytes=4_700_000_000,
    )
    derived = derive_model_profile("fake-7b", str(tmp_path))
    builtin = MODEL_PROFILES["qwen-7b"]
    # Weights are exact from the header.
    assert derived.weights_gb == pytest.approx(4.7, abs=0.1)
    # The OLD formula (single head dim, no kv-head count, no bytes/elem)
    # undercounted ~10x: derived KV must be the same order of magnitude as the
    # built-in, not a fraction of it.
    old_broken = (2 * 28 * (3584 // 28) * 8192) / 1e9
    assert derived.kv_gb_per_8k > old_broken * 3
    assert 0.3 < derived.kv_gb_per_8k / builtin.kv_gb_per_8k < 3


def test_derived_7b_no_longer_admitted_at_huge_context(tmp_path):
    _write_fake_checkpoint(
        tmp_path, layers=28, hidden=3584, heads=28, kv_heads=4, head_dim=128,
        weight_bytes=4_700_000_000,
    )
    derived = derive_model_profile("fake-7b", str(tmp_path))
    # At a very long prompt the realistic KV pushes total over the 16GB
    # single-resident ceiling — the old undercount would have wrongly admitted.
    fits, _available, _required = model_fits(derived, None, 200_000)
    assert fits is False


def test_kv_gb_interpolates_beyond_32k():
    p = ModelProfile("x", weights_gb=1.0, kv_gb_per_8k=0.5, kv_gb_per_32k=2.0)
    assert p.kv_gb(8192) == pytest.approx(0.5)
    assert p.kv_gb(32768) == pytest.approx(2.0)
    # Extrapolates on the same slope past 32k (was frozen at the 8k value).
    slope = (2.0 - 0.5) / (32768 - 8192)
    assert p.kv_gb(65536) == pytest.approx(0.5 + slope * (65536 - 8192))
