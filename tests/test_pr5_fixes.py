"""Regression tests for the review fixes applied to the rebased audit-fixes
branch (PR #5): FifoLock abort fairness, stop sequences, penalty validation,
and OpenAI-spec usage-chunk gating.
"""

import json
import threading
import time

from fastapi.testclient import TestClient

from daedalus.scheduler import FifoLock
from daedalus.server import create_app

from test_server import FakeEngine, FakeStore


# ------------------------------------------------------------ FifoLock aborts


def test_aborted_waiters_never_hand_lock_to_second_holder():
    """Queued aborts must not advance the FIFO cursor past a live holder.

    Review finding: each aborting waiter bumped ``_serving`` as if a release
    had happened, so two aborts behind a long generation handed the lock to a
    third waiter while the original holder still ran.
    """
    lock = FifoLock()
    events = []
    a_started = threading.Event()
    a_release = threading.Event()
    results = {}

    def holder():
        assert lock.acquire()
        events.append("A-acquired")
        a_started.set()
        a_release.wait(timeout=10)
        events.append("A-releasing")
        lock.release()

    def waiter(name, abort_event):
        results[name] = lock.acquire(abort_event, timeout=0.05)
        if results[name]:
            events.append(f"{name}-acquired")
            lock.release()

    t_a = threading.Thread(target=holder)
    t_a.start()
    assert a_started.wait(timeout=5)

    b_abort, c_abort, d_abort = (threading.Event() for _ in range(3))
    threads = []
    for name, abort_event in (("B", b_abort), ("C", c_abort), ("D", d_abort)):
        t = threading.Thread(target=waiter, args=(name, abort_event))
        t.start()
        threads.append(t)
        time.sleep(0.1)  # deterministic ticket order: B=1, C=2, D=3

    # B and D abandon the queue while A still holds; C stays.
    b_abort.set()
    d_abort.set()
    threads[0].join(timeout=5)
    threads[2].join(timeout=5)
    assert results["B"] is False and results["D"] is False

    time.sleep(0.3)
    assert "C-acquired" not in events, "C acquired while A still held the lock"

    a_release.set()
    t_a.join(timeout=5)
    threads[1].join(timeout=5)
    assert results["C"] is True
    assert events.index("A-releasing") < events.index("C-acquired")


def test_abort_at_own_turn_passes_lock_on():
    lock = FifoLock()
    aborted = threading.Event()
    aborted.set()
    assert lock.acquire(aborted) is False
    # The abandoned turn must not wedge the queue for the next caller.
    assert lock.acquire() is True
    lock.release()


def test_queued_count_ignores_abandoned_tickets():
    lock = FifoLock()
    assert lock.acquire()
    aborted = threading.Event()
    aborted.set()
    t = threading.Thread(target=lock.acquire, args=(aborted, 0.05))
    t.start()
    t.join(timeout=5)
    assert lock.queued == 0
    lock.release()


# ------------------------------------------------------------- stop sequences


def test_stop_sequence_truncates_stream():
    engine, store = FakeEngine(), FakeStore()
    app = create_app(engine, store, model_id="test-model")
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stop": "world",
        },
    )
    assert r.status_code == 200
    chunks = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c.get("choices")
    )
    assert "world" not in content
    finishes = [
        c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")
    ]
    assert "stop" in finishes


def test_stop_matcher_streams_long_output_without_retaining_history():
    """Only a stop-length suffix is held, so long responses stay linear."""
    engine = FakeEngine(script=["a"] * 200 + ["<END>"])
    app = create_app(engine, FakeStore(), model_id="test-model")
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stop": "<END>",
        },
    )
    chunks = [
        json.loads(line[6:]) for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks if chunk.get("choices")
    )
    assert content == "a" * 200


# ---------------------------------------------------------- penalty handling


def test_out_of_range_penalty_rejected():
    engine, store = FakeEngine(), FakeStore()
    app = create_app(engine, store, model_id="test-model")
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 3.0,
        },
    )
    assert r.status_code == 400


def test_penalties_only_build_processors_when_set():
    class KwargsRecordingEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.kwargs_seen = []

        def generate(self, tokens, **kw):
            self.kwargs_seen.append(dict(kw))
            # FakeEngine mirrors the real engine's keyword-only signature and
            # has no logits_processors parameter.
            kw.pop("logits_processors", None)
            yield from super().generate(tokens, **kw)

    engine, store = KwargsRecordingEngine(), FakeStore()
    app = create_app(engine, store, model_id="test-model")
    client = TestClient(app)

    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert engine.kwargs_seen[0].get("logits_processors") is None

    client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 0.5,
        },
    )
    assert engine.kwargs_seen[1].get("logits_processors")


# --------------------------------------------------------- usage chunk gating


def test_usage_chunk_only_when_requested():
    engine, store = FakeEngine(), FakeStore()
    app = create_app(engine, store, model_id="test-model")
    client = TestClient(app)
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": True}

    # Legacy default: final chunk carries finish_reason AND usage; choices
    # are never empty (OpenCode and friends index choices[0] unguarded).
    r = client.post("/v1/chat/completions", json=body)
    chunks = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert all(c.get("choices") for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["prompt_tokens"] > 0

    # OpenAI spec: stream_options.include_usage=true moves usage into a
    # trailing chunk with an empty choices array.
    r = client.post(
        "/v1/chat/completions",
        json={**body, "stream_options": {"include_usage": True}},
    )
    chunks = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-2].get("usage") is None
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["prompt_tokens"] > 0
