"""Engine prefill-loop tests with a fake MLX model (no weights needed)."""

import mlx.core as mx
import pytest

from daedalus.engine import Engine, EngineConfig, PrefillAborted
from daedalus.governor import GovernorConfig, LevelPolicy, ThermalGovernor
from daedalus.sensors import ThermalLevel, ThermalMonitor


class FakeLayerCache:
    """Minimal duck-type of an mlx-lm KVCache for the prefill loop."""

    def __init__(self):
        self.offset = 0
        self.state = mx.zeros((1,))


class FakeModel:
    """Records chunk sizes; returns dummy logits. Each call burns fake time."""

    def __init__(self, clock=None, burn_per_chunk=0.5):
        self.chunk_sizes = []
        self.clock = clock
        self.burn_per_chunk = burn_per_chunk

    def __call__(self, inputs, cache=None):
        n = inputs.shape[1]
        self.chunk_sizes.append(n)
        if self.clock is not None:
            self.clock.advance(self.burn_per_chunk)
        for c in cache:
            c.offset += n
        return mx.zeros((1, n, 8))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_engine(level=ThermalLevel.NOMINAL, policies=None, **gov_cfg):
    state = {"level": level}
    monitor = ThermalMonitor(reader=lambda: state["level"], poll_interval=1000)
    cfg = GovernorConfig(**gov_cfg)
    if policies:
        cfg.policies = policies
    clock = FakeClock()
    governor = ThermalGovernor(monitor, cfg, clock=clock)
    # Burn time advances the clock so duty-cycle math sees realistic values.
    model = FakeModel(clock=clock, burn_per_chunk=0.5)
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        clock.advance(s)

    engine = Engine(
        model,
        tokenizer=None,
        governor=governor,
        config=EngineConfig(kv_bits=None),  # fake caches can't quantize
        clock=clock,
        sleep=fake_sleep,
    )
    return engine, model, sleeps, state, monitor


def cache_for(model):
    return [FakeLayerCache(), FakeLayerCache()]


def test_prefill_leaves_last_token_and_covers_rest():
    engine, model, _, _, _ = make_engine()
    tokens = list(range(5000))
    report = engine.paced_prefill(tokens, cache_for(model))
    assert report.computed_tokens == 4999  # all but last
    assert sum(model.chunk_sizes) == 4999
    assert report.chunks == len(model.chunk_sizes)
    assert model.chunk_sizes[0] == 2048  # nominal chunk size


def test_prefill_nominal_never_sleeps():
    engine, model, sleeps, _, _ = make_engine()
    engine.paced_prefill(list(range(6000)), cache_for(model))
    assert sleeps == []


def test_prefill_heavy_duty_cycles():
    engine, model, sleeps, _, _ = make_engine(level=ThermalLevel.HEAVY)
    report = engine.paced_prefill(list(range(8000)), cache_for(model))
    # Heavy: 512-token chunks, duty 0.25 -> idle 3x burn between chunks.
    assert all(size <= 512 for size in model.chunk_sizes)
    assert report.idle_seconds > 0
    assert sleeps  # slept at least once between chunks
    # 0.5s burn, duty 0.25 -> 1.5s idle per gap (within poll-step rounding).
    assert abs(sum(sleeps[: len(sleeps) // max(1, report.chunks - 1)]) - 1.5) < 0.2


def test_small_job_not_paced_even_when_heavy():
    """Pacing a few seconds of burn saves no heat but triples turn latency —
    the exact slowdown seen in live pi usage (569-tok prefill @ 39 tok/s)."""
    engine, model, sleeps, _, _ = make_engine(level=ThermalLevel.HEAVY)
    report = engine.paced_prefill(list(range(2000)), cache_for(model))
    assert sleeps == []
    assert report.idle_seconds == 0
    # Chunk size still respects the thermal level.
    assert all(size <= 512 for size in model.chunk_sizes)


def test_small_residual_after_cache_hit_not_paced():
    engine, model, sleeps, _, _ = make_engine(level=ThermalLevel.HEAVY)
    tokens = list(range(12000))
    engine.paced_prefill(tokens, cache_for(model), already_cached=11000)
    assert sleeps == []  # only 999 fresh tokens: interactive, don't pace


def test_snap_points_clip_chunks():
    engine, model, _, _, _ = make_engine()
    tokens = list(range(6000))
    checkpoints = []
    engine.paced_prefill(
        tokens,
        cache_for(model),
        snap_points=[2129],
        checkpoint_cb=lambda done, cache: checkpoints.append(done),
    )
    # A chunk boundary lands exactly on the snap point...
    assert 2129 in checkpoints
    # ...and every token is still covered exactly once.
    assert sum(model.chunk_sizes) == 5999


def test_snap_point_inside_cached_region_ignored():
    engine, model, _, _, _ = make_engine()
    tokens = list(range(6000))
    checkpoints = []
    engine.paced_prefill(
        tokens,
        cache_for(model),
        already_cached=3000,
        snap_points=[2129],  # already inside the cached prefix
        checkpoint_cb=lambda done, cache: checkpoints.append(done),
    )
    assert 2129 not in checkpoints
    assert sum(model.chunk_sizes) == 2999


def test_prefill_no_sleep_after_final_chunk():
    engine, model, sleeps, _, _ = make_engine(level=ThermalLevel.HEAVY)
    engine.paced_prefill(list(range(513)), cache_for(model))  # exactly one chunk
    assert sleeps == []


def test_progress_callback_sequence():
    engine, model, _, _, _ = make_engine()
    seen = []
    engine.paced_prefill(
        list(range(5000)),
        cache_for(model),
        progress_cb=lambda done, total: seen.append((done, total)),
    )
    assert seen[0] == (0, 5000)
    assert seen[-1] == (4999, 5000)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_abort_mid_prefill_raises():
    engine, model, _, _, _ = make_engine()
    calls = {"n": 0}

    def abort_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(PrefillAborted):
        engine.paced_prefill(
            list(range(50000)), cache_for(model), should_abort=abort_after_two
        )
    assert sum(model.chunk_sizes) < 49999  # stopped early


def test_resume_from_already_cached():
    engine, model, _, _, _ = make_engine()
    tokens = list(range(5000))
    report = engine.paced_prefill(tokens, cache_for(model), already_cached=3000)
    assert report.computed_tokens == 4999
    assert sum(model.chunk_sizes) == 1999  # only the residual was computed


def test_checkpoint_callback_fires_per_chunk():
    engine, model, _, _, _ = make_engine()
    checkpoints = []
    engine.paced_prefill(
        list(range(5000)),
        cache_for(model),
        checkpoint_cb=lambda done, cache: checkpoints.append(done),
    )
    assert checkpoints == [2048, 4096, 4999]


def test_governor_shrinks_chunks_when_pressure_rises_mid_prefill():
    engine, model, _, state, monitor = make_engine()
    tokens = list(range(9000))

    def heat_up(done, total):
        if done >= 4000 and state["level"] == ThermalLevel.NOMINAL:
            state["level"] = ThermalLevel.HEAVY
            monitor.refresh()

    engine.paced_prefill(tokens, cache_for(model), progress_cb=heat_up)
    assert 2048 in model.chunk_sizes  # started nominal
    assert model.chunk_sizes[-1] <= 512  # ended paced
