"""normalize_messages: OpenAI wire quirks -> template-safe messages.

The failure cases here are real: pi sends role "developer" (crashed Qwen3.5's
template with "Unexpected message role."), and OpenAI-format clients send
tool-call arguments as JSON strings (crashed it with "Can only get item pairs
from a mapping.").
"""

from daedalus.server import normalize_messages


def test_developer_role_becomes_system():
    out = normalize_messages(
        [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "yo"}]
    )
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "be brief"
    assert out[1]["role"] == "user"


def test_unknown_role_becomes_user():
    out = normalize_messages([{"role": "wizard", "content": "abracadabra"}])
    assert out[0]["role"] == "user"


def test_content_parts_flattened():
    out = normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "image_url", "image_url": {"url": "x"}},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
    )
    assert out[0]["content"] == "hello world"


def test_null_content_becomes_empty_string():
    out = normalize_messages([{"role": "assistant", "content": None}])
    assert out[0]["content"] == ""


def test_tool_call_string_arguments_parsed_to_dict():
    out = normalize_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"x": 1}'},
                    }
                ],
            }
        ]
    )
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"x": 1}


def test_tool_call_invalid_json_arguments_wrapped():
    out = normalize_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "f", "arguments": "not json"}}
                ],
            }
        ]
    )
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"_raw": "not json"}


def test_tool_call_dict_arguments_untouched_and_input_not_mutated():
    original = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "f", "arguments": {"x": 2}}}],
    }
    out = normalize_messages([original])
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"x": 2}
    # deep-copied: mutating output must not touch caller's dict
    out[0]["tool_calls"][0]["function"]["arguments"]["x"] = 99
    assert original["tool_calls"][0]["function"]["arguments"]["x"] == 2


def test_standard_messages_pass_through():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    out = normalize_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
