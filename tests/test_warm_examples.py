"""The shipped warm prompt pack must satisfy the exact schema `daedalus warm`
accepts, so a copy-paste from the docs cannot be silently rejected."""

import json
from pathlib import Path

from daedalus.cli import _valid_prompt_pack

PACK = Path(__file__).parent.parent / "examples" / "warm-prompts.json"


def test_example_pack_passes_the_real_warm_validator():
    pack = json.loads(PACK.read_text())
    assert _valid_prompt_pack(pack)  # same predicate cmd_warm gates on
    assert 2 <= len(pack) <= 3
    for item in pack:
        assert item["messages"]
        for msg in item["messages"]:
            assert msg["role"] and msg["content"]


def test_validator_rejects_wrong_shapes():
    assert not _valid_prompt_pack({"messages": []})  # object, not an array
    assert not _valid_prompt_pack([{"prompt": "hi"}])  # missing messages
    assert not _valid_prompt_pack([{"messages": "hi"}])  # messages not a list
    assert not _valid_prompt_pack(["nope"])
    assert _valid_prompt_pack([])  # an empty pack is structurally valid


def test_cmd_warm_releases_resources_even_when_prefill_fails(tmp_path, monkeypatch):
    """The warm loop must close the store and tear the engine + monitor down on
    every exit path, so a crashing prefill can't leak the flock or the monitor
    thread."""
    import types

    import daedalus.cli as cli

    events = []

    class FakeMonitor:
        def start(self):
            return self

        def stop(self):
            events.append("monitor.stop")

    class FakeTokenizer:
        name_or_path = "fake"

        def apply_chat_template(self, messages, add_generation_prompt=True):
            return [1, 2, 3]

    class FakeEngine:
        tokenizer = FakeTokenizer()

        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def make_cache(self):
            return []

        def paced_prefill(self, *a, **kw):
            raise RuntimeError("boom")

        def close(self):
            events.append("engine.close")

        def shutdown(self):
            events.append("engine.shutdown")

    class FakeStore:
        def __init__(self, *a, **kw):
            pass

        def fetch(self, tokens):
            return None

        def close(self):
            events.append("store.close")

    monkeypatch.setattr(cli, "_valid_prompt_pack", lambda p: True)
    monkeypatch.setattr("daedalus.sensors.ThermalMonitor", FakeMonitor)
    monkeypatch.setattr("daedalus.engine.Engine", FakeEngine)
    monkeypatch.setattr("daedalus.engine.EngineConfig", lambda **kw: None)
    monkeypatch.setattr("daedalus.cache.store.PrefixCacheStore", FakeStore)
    monkeypatch.setattr("daedalus.runtime.cache_identity", lambda *a, **kw: "key")

    prompts = tmp_path / "prompts.json"
    prompts.write_text('[{"messages": [{"role": "user", "content": "hi"}]}]')
    args = types.SimpleNamespace(
        prompts=str(prompts), model="fake", kv_bits=0, model_revision=None,
    )
    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        cli.cmd_warm(args)
    assert events == ["store.close", "engine.close", "engine.shutdown", "monitor.stop"]
