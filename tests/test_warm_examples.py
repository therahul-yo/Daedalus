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
