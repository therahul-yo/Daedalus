"""Real-model compatibility verification.

Usage: python bench/model_verify.py <model_id>

Checks, in order:
1. mlx-lm loads the checkpoint (Qwen3.5/Gemma4 are multimodal — text backbone
   must load with vision weights stripped).
2. Cache classes + trimmability (drives which prefix-cache paths apply).
3. Tool-calling support detected on the tokenizer (parser, markers).
4. Short governed generation produces sane text.
5. Tool-call round trip: tools in template -> model emits call -> filter parses.
"""

import json
import sys

from daedalus.engine import Engine, EngineConfig
from daedalus.sensors import ThermalMonitor
from daedalus.tools import make_stream_filter

MODEL = sys.argv[1]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def main():
    print(f"[1/5] loading {MODEL} ...")
    monitor = ThermalMonitor().start()
    engine = Engine.from_pretrained(
        MODEL, monitor=monitor, config=EngineConfig(kv_bits=8)
    )
    print("      loaded OK")

    print("[2/5] cache classes:")
    from mlx_lm.models.cache import can_trim_prompt_cache

    cache = engine.make_cache()
    kinds = {}
    for c in cache:
        kinds[type(c).__name__] = kinds.get(type(c).__name__, 0) + 1
    trimmable = can_trim_prompt_cache(cache)
    print(f"      {kinds} | trimmable={trimmable}")
    print(
        "      -> cache strategy:",
        "LCP+trim" if trimmable else "exact-prefix only (hybrid)",
    )

    print("[3/5] tool support on tokenizer:")
    tok = engine.tokenizer
    has = getattr(tok, "has_tool_calling", False)
    print(
        f"      has_tool_calling={has}"
        + (f" start={tok.tool_call_start!r} end={tok.tool_call_end!r}" if has else "")
    )

    print("[4/5] short generation:")
    tokens = tok.apply_chat_template(
        [{"role": "user", "content": "Reply with exactly: VERIFY OK"}],
        add_generation_prompt=True,
    )
    text = ""
    for resp in engine.generate(tokens, max_tokens=64, temperature=0.0):
        text += resp.text
    print(f"      {text.strip()[:200]!r}")

    print("[5/5] tool-call round trip:")
    if not has:
        print("      SKIP (no tool parser for this model)")
    else:
        tokens = tok.apply_chat_template(
            [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
            tools=TOOLS,
            add_generation_prompt=True,
        )
        filt = make_stream_filter(tok, TOOLS)
        calls, content = [], []
        for resp in engine.generate(tokens, max_tokens=256, temperature=0.0):
            c, k = filt.feed(resp.text)
            content.append(c)
            calls.extend(k)
        c, k = filt.finalize()
        content.append(c)
        calls.extend(k)
        if calls:
            print(f"      parsed: {calls[0].name}({calls[0].arguments})")
            args = json.loads(calls[0].arguments)
            assert "paris" in json.dumps(args).lower(), args
            print("      TOOL CALL OK")
        else:
            print(f"      NO CALL PARSED. raw content: {''.join(content)[:300]!r}")
            sys.exit(1)

    monitor.stop()
    print("VERIFY PASSED")


if __name__ == "__main__":
    main()
