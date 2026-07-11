"""Server tests with a fake engine — no model weights needed."""

import json

import pytest
from fastapi.testclient import TestClient

from daedalus.server import create_app


class FakeTokenizer:
    has_tool_calling = True
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"
    # Simulates whether the chat template left the prompt inside <think>.
    prompt_tail = ""

    def __init__(self):
        self.template_calls = 0

    @staticmethod
    def tool_parser(text, tools=None):
        return json.loads(text)

    def decode(self, tokens):
        return self.prompt_tail

    def apply_chat_template(self, messages, add_generation_prompt=True, tools=None):
        self.template_calls += 1
        # Deterministic: token ids derived from the serialized messages.
        text = json.dumps(messages) + (json.dumps(tools) if tools else "")
        return [ord(c) % 1000 for c in text]


class FakeResponse:
    def __init__(self, text, generation_tokens, finish_reason=None):
        self.text = text
        self.generation_tokens = generation_tokens
        self.generation_tps = 20.0
        self.finish_reason = finish_reason


class FakeGovernor:
    class _Level:
        name = "NOMINAL"

    effective_level = _Level()


class FakeEngine:
    def __init__(self, script=None):
        self.tokenizer = FakeTokenizer()
        self.governor = FakeGovernor()
        self.generate_calls = []
        # script: list of text segments to emit instead of the default.
        self.script = script

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
        snap_points=None,
        checkpoint_cb=None,
        should_abort=None,
        progress_cb=None,
    ):
        self.generate_calls.append(
            {"tokens": len(tokens), "already_cached": already_cached}
        )
        if checkpoint_cb:
            checkpoint_cb(len(tokens) - 1, prompt_cache)  # end-of-prefill snapshot
        words = self.script or ["Hello", " world", "!"]
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
            from daedalus.cache.store import CacheHit

            return CacheHit(
                cache=["cached"], matched_tokens=len(tokens) - 1, source="ram"
            )
        return None

    def put(self, tokens, cache, persist=True, async_persist=False):
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


def test_prompt_tokenization_is_memoized(client_and_fakes):
    client, engine, _ = client_and_fakes
    payload = {"messages": [{"role": "user", "content": "same prompt"}]}
    client.post("/v1/chat/completions", json=payload)
    client.post("/v1/chat/completions", json=payload)
    # The request and shared-head probe are independently memoized; the second
    # stateless resend should not invoke the tokenizer again.
    assert engine.tokenizer.template_calls == 2


def test_missing_messages_400(client_and_fakes):
    client, _, _ = client_and_fakes
    r = client.post("/v1/chat/completions", json={})
    assert r.status_code == 400


def test_cache_stats_endpoint(client_and_fakes):
    client, _, _ = client_and_fakes
    assert "entries" in client.get("/v1/cache/stats").json()


def test_readiness_and_metrics(client_and_fakes):
    client, _, _ = client_and_fakes
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "daedalus_queue_limit 8" in metrics.text
    assert "daedalus_cache_entries" in metrics.text


def test_request_validation(client_and_fakes):
    client, _, _ = client_and_fakes
    assert client.post("/v1/chat/completions", content="not-json").status_code == 400
    assert client.post("/v1/chat/completions", json={"messages": "nope"}).status_code == 400
    assert client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "top_p": 0}
    ).status_code == 400


def test_api_key_protects_v1_endpoints():
    engine, store = FakeEngine(), FakeStore()
    client = TestClient(create_app(engine, store, model_id="test-model", api_key="secret"))
    assert client.get("/v1/models").status_code == 401
    assert client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    ).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_clear_cache_requires_idle_and_authorization():
    class ClearableStore(FakeStore):
        def clear(self):
            count = len(self.entries)
            self.entries.clear()
            return count

    engine, store = FakeEngine(), ClearableStore()
    client = TestClient(create_app(engine, store, model_id="test-model", api_key="secret"))
    store.entries[(1, 2)] = ["cached"]
    response = client.delete("/v1/cache", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json() == {"removed_entries": 1}


def test_bounded_admission_rejects_overloaded_server():
    engine, store = FakeEngine(), FakeStore()
    app = create_app(engine, store, model_id="test-model", max_pending_requests=1)
    app.state.daedalus.admitted_requests = 1
    client = TestClient(app)
    assert client.get("/readyz").status_code == 503
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"


def test_release_slot_is_idempotent():
    """A slot is released exactly once even when multiple guaranteed-release
    paths (worker finally + StreamingResponse background task) both fire."""
    from daedalus.server import _Generation

    engine, store = FakeEngine(), FakeStore()
    app = create_app(engine, store, model_id="test-model", max_pending_requests=4)
    state = app.state.daedalus
    assert state.try_admit() and state.try_admit()  # two admitted
    gen = _Generation(
        state=state, tokens=[1, 2, 3], max_tokens=8, temperature=0.0, top_p=1.0
    )
    gen.release_slot()
    gen.release_slot()  # double-fire must not release the second request's slot
    assert state.admitted_requests == 1


def test_admission_slot_returns_to_zero_after_requests(client_and_fakes):
    client, _, _ = client_and_fakes
    state = client.app.state.daedalus
    client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert state.admitted_requests == 0
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as r:
        "".join(r.iter_text())
    assert state.admitted_requests == 0  # background task + worker: still exactly zero


def test_memory_guard_evicts_cache_then_rejects_when_still_over_limit():
    class MemoryEngine(FakeEngine):
        def active_memory_bytes(self):
            return 100

    class MemoryStore(FakeStore):
        def __init__(self):
            super().__init__()
            self.trimmed = False

        def trim_ram(self, target):
            self.trimmed = True
            return 0

    engine, store = MemoryEngine(), MemoryStore()
    client = TestClient(create_app(engine, store, model_id="test-model", max_active_memory_bytes=50))
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 503
    assert store.trimmed


TOOLS = [
    {
        "type": "function",
        "function": {"name": "read_file", "parameters": {"type": "object"}},
    }
]

TOOL_SCRIPT = [
    "Let me check. ",
    '<tool_call>{"name": "read_file", ',
    '"arguments": {"path": "/etc/hosts"}}</tool_call>',
]


def make_tool_client(script):
    engine, store = FakeEngine(script=script), FakeStore()
    app = create_app(engine, store, model_id="test-model")
    return TestClient(app), engine, store


def test_non_streaming_tool_call():
    client, _, _ = make_tool_client(TOOL_SCRIPT)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS},
    )
    body = r.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "/etc/hosts"}
    assert calls[0]["id"].startswith("call_")
    assert "index" not in calls[0]  # response format carries no index
    assert choice["message"]["content"] == "Let me check. "


def test_streaming_tool_call_deltas():
    client, _, _ = make_tool_client(TOOL_SCRIPT)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": TOOLS,
            "stream": True,
        },
    ) as r:
        raw = "".join(r.iter_text())
    lines = [l for l in raw.split("\n") if l.startswith("data: ")]
    chunks = [json.loads(l[6:]) for l in lines[:-1]]

    tool_chunks = [
        c for c in chunks if "tool_calls" in c["choices"][0]["delta"]
    ]
    assert len(tool_chunks) == 1
    tc = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tc["index"] == 0
    assert tc["function"]["name"] == "read_file"
    # OpenCode compat: no chunk anywhere may carry empty tool_calls.
    for c in chunks:
        assert c["choices"][0]["delta"].get("tool_calls") != []
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_none_disables_tools():
    client, engine, _ = make_tool_client(TOOL_SCRIPT)
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "tools": TOOLS,
            "tool_choice": "none",
        },
    )
    # Tools stripped: the tool-call markup streams through as plain content.
    assert r.json()["choices"][0]["message"].get("tool_calls") is None


def test_no_tools_means_no_filter_interference():
    client, _, _ = make_tool_client(["plain <tool text", " passes"])
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.json()["choices"][0]["message"]["content"] == "plain <tool text passes"


def make_thinking_client(script):
    """Engine whose template opened a <think> block (Qwen3.5 behavior)."""
    engine, store = FakeEngine(script=script), FakeStore()
    engine.tokenizer.prompt_tail = "<|im_start|>assistant\n<think>"
    app = create_app(engine, store, model_id="test-model")
    return TestClient(app), engine, store


THINK_SCRIPT = [
    "The user greets me. ",
    "I should reply.</think>",
    "\n\nYo! How can I help?",
]


def test_reasoning_separated_non_streaming():
    client, _, _ = make_thinking_client(THINK_SCRIPT)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "yo"}]},
    )
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "Yo! How can I help?"
    assert msg["reasoning_content"] == "The user greets me. I should reply."
    assert "</think>" not in msg["content"]


def test_reasoning_separated_streaming():
    client, _, _ = make_thinking_client(THINK_SCRIPT)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "yo"}], "stream": True},
    ) as r:
        raw = "".join(r.iter_text())
    lines = [
        l for l in raw.split("\n") if l.startswith("data: ") and l != "data: [DONE]"
    ]
    chunks = [json.loads(l[6:]) for l in lines]
    reasoning = "".join(
        c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks
    )
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert reasoning == "The user greets me. I should reply."
    assert content == "Yo! How can I help?"
    assert "</think>" not in content


def test_thinking_then_tool_call():
    client, _, _ = make_thinking_client(
        [
            "Need the file.</think>",
            '<tool_call>{"name": "read_file", "arguments": {"path": "/x"}}</tool_call>',
        ]
    )
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "read x"}], "tools": TOOLS},
    )
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read_file"
    assert choice["message"]["reasoning_content"] == "Need the file."


def test_no_think_model_untouched():
    client, _, _ = make_tool_client(["Direct answer, no thinking."])
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    msg = r.json()["choices"][0]["message"]
    assert msg["content"] == "Direct answer, no thinking."
    assert "reasoning_content" not in msg
