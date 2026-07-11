"""End-to-end smoke test on a real model: paced prefill + decode.

Usage: python bench/smoke.py [model_id] [prompt_tokens]
"""

import sys
import time

from airlift.engine import Engine
from airlift.governor import ThermalGovernor
from airlift.sensors import ThermalLevel, ThermalMonitor

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-0.6B-4bit"
PROMPT_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000


def main():
    print(f"loading {MODEL} ...")
    t0 = time.perf_counter()
    monitor = ThermalMonitor().start()
    engine = Engine.from_pretrained(MODEL, monitor=monitor)
    print(f"loaded in {time.perf_counter() - t0:.1f}s | thermal={monitor.level.name}")

    # Build a long-ish prompt: repeated context + a question, chat-templated.
    filler = "The quick brown fox jumps over the lazy dog. " * (PROMPT_TOKENS // 10)
    messages = [
        {"role": "system", "content": "You are a concise assistant. " + filler},
        {"role": "user", "content": "Reply with exactly: SMOKE OK"},
    ]
    tokens = engine.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
    print(f"prompt: {len(tokens)} tokens")

    progress_events = []

    def progress(done, total):
        progress_events.append((time.perf_counter(), done, total))

    t1 = time.perf_counter()
    text = ""
    first_token_at = None
    last = None
    for resp in engine.generate(
        tokens, max_tokens=40, temperature=0.0, progress_cb=progress
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        text += resp.text
        last = resp

    ttft = (first_token_at or time.perf_counter()) - t1
    print(f"\n--- output ---\n{text}\n--------------")
    print(f"TTFT: {ttft:.2f}s | prefill events: {len(progress_events)}")
    if last is not None:
        print(
            f"decode: {last.generation_tokens} tokens @ {last.generation_tps:.1f} tok/s"
            f" | peak mem: {last.peak_memory:.2f} GB"
        )
    print(f"thermal after: {monitor.level.name}")
    monitor.stop()
    assert len(progress_events) >= 2, "expected chunked prefill progress"
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
