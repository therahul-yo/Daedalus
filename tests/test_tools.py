import json

from daedalus.tools import PassthroughFilter, ToolCallStreamFilter, make_stream_filter


def qwen_parser(text, tools=None):
    return json.loads(text)


def make_filter():
    return ToolCallStreamFilter("<tool_call>", "</tool_call>", qwen_parser)


def feed_all(f, segments):
    content, calls = [], []
    for seg in segments:
        c, k = f.feed(seg)
        content.append(c)
        calls.extend(k)
    c, k = f.finalize()
    content.append(c)
    calls.extend(k)
    return "".join(content), calls


def test_plain_content_passes_through():
    content, calls = feed_all(make_filter(), ["Hello", " world"])
    assert content == "Hello world"
    assert calls == []


def test_single_tool_call():
    payload = '{"name": "read_file", "arguments": {"path": "/x"}}'
    content, calls = feed_all(
        make_filter(), [f"<tool_call>{payload}</tool_call>"]
    )
    assert content == ""
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert json.loads(calls[0].arguments) == {"path": "/x"}
    assert calls[0].id.startswith("call_")


def test_marker_split_across_segments():
    payload = '{"name": "ls", "arguments": {}}'
    segments = ["I'll list files. <tool", "_call>", payload[:10], payload[10:], "</tool_c", "all>"]
    content, calls = feed_all(make_filter(), segments)
    assert content == "I'll list files. "
    assert len(calls) == 1
    assert calls[0].name == "ls"


def test_potential_marker_prefix_is_held_then_released():
    f = make_filter()
    c1, _ = f.feed("text ends with <tool")
    assert c1 == "text ends with "  # "<tool" held back
    c2, _ = f.feed(" but was not a call")
    assert c2 == "<tool but was not a call"


def test_multiple_sequential_calls():
    p1 = '{"name": "a", "arguments": {}}'
    p2 = '{"name": "b", "arguments": {"x": 1}}'
    content, calls = feed_all(
        make_filter(),
        [f"<tool_call>{p1}</tool_call><tool_call>{p2}</tool_call>"],
    )
    assert [c.name for c in calls] == ["a", "b"]


def test_unterminated_call_parsed_at_finalize():
    payload = '{"name": "run", "arguments": {"cmd": "ls"}}'
    content, calls = feed_all(make_filter(), [f"<tool_call>{payload}"])
    assert len(calls) == 1
    assert calls[0].name == "run"


def test_unparseable_call_surfaces_as_content():
    content, calls = feed_all(make_filter(), ["<tool_call>not json at all"])
    assert calls == []
    assert "not json" in content


def test_parser_returning_list():
    def list_parser(text, tools=None):
        return [
            {"name": "x", "arguments": {}},
            {"name": "y", "arguments": {"k": "v"}},
        ]

    f = ToolCallStreamFilter("<tc>", "</tc>", list_parser)
    _, calls = f.feed("<tc>whatever</tc>")
    assert [c.name for c in calls] == ["x", "y"]


def test_openai_format_has_index_and_id():
    payload = '{"name": "f", "arguments": {}}'
    _, calls = feed_all(make_filter(), [f"<tool_call>{payload}</tool_call>"])
    wire = calls[0].as_openai(3)
    assert wire["index"] == 3
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "f"
    assert isinstance(wire["function"]["arguments"], str)


def test_make_stream_filter_passthrough_without_tools():
    class Tok:
        has_tool_calling = True
        tool_call_start = "<tool_call>"
        tool_call_end = "</tool_call>"
        tool_parser = staticmethod(qwen_parser)

    assert isinstance(make_stream_filter(Tok(), None), PassthroughFilter)
    assert isinstance(
        make_stream_filter(Tok(), [{"type": "function"}]), ToolCallStreamFilter
    )

    class NoToolTok:
        has_tool_calling = False
        tool_call_start = None

    assert isinstance(
        make_stream_filter(NoToolTok(), [{"type": "function"}]), PassthroughFilter
    )
