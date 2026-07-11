"""ThinkStreamFilter: split streamed model output into reasoning vs content.

Real failure this guards: Qwen3.5's template opens a <think> block in the
generation prompt, so raw streaming leaked the chain of thought and a stray
</think> into pi's visible reply.
"""

from daedalus.reasoning import NoThinkFilter, ThinkStreamFilter


def feed_all(f, segments):
    reasoning, content = [], []
    for seg in segments:
        r, c = f.feed(seg)
        reasoning.append(r)
        content.append(c)
    r, c = f.finalize()
    reasoning.append(r)
    content.append(c)
    return "".join(reasoning), "".join(content)


def test_prompt_opened_think_block():
    f = ThinkStreamFilter(initially_thinking=True)
    r, c = feed_all(
        f, ["The user wants a greeting.", "</think>", "\n\nHello there!"]
    )
    assert r == "The user wants a greeting."
    assert c == "Hello there!"


def test_end_marker_split_across_segments():
    f = ThinkStreamFilter(initially_thinking=True)
    r, c = feed_all(f, ["thinking...</th", "ink>answer"])
    assert r == "thinking..."
    assert c == "answer"


def test_no_think_content_passes_through():
    f = ThinkStreamFilter(initially_thinking=False)
    r, c = feed_all(f, ["Just a plain ", "answer with < signs > in it"])
    assert r == ""
    assert c == "Just a plain answer with < signs > in it"


def test_model_opens_own_think_block():
    f = ThinkStreamFilter(initially_thinking=False)
    r, c = feed_all(f, ["<think>hmm</think>", "the answer"])
    assert r == "hmm"
    assert c == "the answer"


def test_unclosed_think_is_all_reasoning():
    f = ThinkStreamFilter(initially_thinking=True)
    r, c = feed_all(f, ["model hit EOS while still thinking"])
    assert r == "model hit EOS while still thinking"
    assert c == ""


def test_code_snippet_with_lt_not_swallowed():
    f = ThinkStreamFilter(initially_thinking=False)
    r, c = feed_all(f, ["if a <t", "hreshold: return"])
    assert c == "if a <threshold: return"
    assert r == ""


def test_nothink_filter_passthrough():
    f = NoThinkFilter()
    assert f.feed("abc") == ("", "abc")
    assert f.finalize() == ("", "")
