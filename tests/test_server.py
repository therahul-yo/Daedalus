"""Server tests with a fake engine — no model weights needed."""

import json

import pytest
from fastapi.testclient import TestClient

from airlift.server import create_app


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True):
        # Deterministic: token ids derived from the serialized messages.
        text = json.dumps(messages)
        return [ord(c) % 1000 for c in text]


class FakeResponse:
    def __init__(self, text, generation_tokens, finish_reason=None):
        self.text = text
        self.generation_tokens = generation_tokens
        self.finish_reason = finish_reason


class FakeGovernor:
    class _Level:
        name = "NOMINAL"

    effective_level = _Level()


class FakeEngine:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.governor = FakeGovernor()
        self.generate_calls = []

    def make_cache(self):
        return ["fresh-cache"]

    def generate(
        self,
        tokens,
        *,
        max_tokens,
        temperature,
        top_p,
        prompt_cache,
        already_cached,
        checkpoint_cb=None,
        should_abort=None,
        progress_cb=None,
    ):
        self.generate_calls.append(
            {"tokens": len(tokens), "already_cached": already_cached}
        )
        if checkpoint_cb:
            checkpoint_cb(len(tokens) - 1, prompt_cache)  # end-of-prefill snapshot
        words = ["Hello", " world", "!"]
        for i, w in enumerate(words):
            yield FakeResponse(
                w, i + 1, "stop" if i == len(words) - 1 else None
            )


class FakeStore:
    def __init__(self):
        self.entries = {}
        self.put_calls = []

    def fetch(self, tokens):
        key = tuple(tokens[:-1])
        if key in self.entries:
            from airlift.cache.store import CacheHit

            return CacheHit(
                cache=["cached"], matched_tokens=len(tokens) - 1, source="ram"
            )
        return None

    def put(self, tokens, cache, persist=True):
        self.put_calls.append(len(tokens))
        self.entries[tuple(tokens)] = cache

    def checkpoint(self, tokens, done, cache):
        pass

    def stats(self):
        return {"entries": len(self.entries)}


@pytest.fixture
def client_and_fakes():
    engine, store = FakeEngine(), FakeStore()
    app = create_app(engine, store, model_id="test-model")
    return TestClient(app), engine, store


def test_health(client_and_fakes):
    client, _, _ = client_and_fakes
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model"] == "test-model"
    assert r.json()["thermal"] == "NOMINAL"


def test_models(client_and_fakes):
    client, _, _ = client_and_fakes
    data = client.get("/v1/models").json()
    assert data["data"][0]["id"] == "test-model"


def test_non_streaming_completion(client_and_fakes):
    client, engine, store = client_and_fakes
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hello world!"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 3
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    # End-of-prefill snapshot stored for next turn.
    assert store.put_calls


def test_streaming_completion_format(client_and_fakes):
    client, _, _ = client_and_fakes
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        raw = "".join(r.iter_text())

    lines = [l for l in raw.split("\n") if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[len("data: ") :]) for l in lines[:-1]]
    # First chunk: role delta.
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    # Reassembled content.
    content = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks
    )
    assert content == "Hello world!"
    # Final chunk has finish_reason and usage.
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    # OpenCode compat: no chunk may carry an empty tool_calls array.
    for c in chunks:
        assert c["choices"][0]["delta"].get("tool_calls") != []


def test_second_identical_request_hits_cache(client_and_fakes):
    client, engine, store = client_and_fakes
    payload = {"messages": [{"role": "user", "content": "same prompt"}]}
    client.post("/v1/chat/completions", json=payload)
    r = client.post("/v1/chat/completions", json=payload)
    assert r.json()["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert engine.generate_calls[1]["already_cached"] > 0


def test_missing_messages_400(client_and_fakes):
    client, _, _ = client_and_fakes
    r = client.post("/v1/chat/completions", json={})
    assert r.status_code == 400


def test_cache_stats_endpoint(client_and_fakes):
    client, _, _ = client_and_fakes
    assert "entries" in client.get("/v1/cache/stats").json()
