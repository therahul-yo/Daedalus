"""End-to-end smoke test on a real model: paced prefill + decode.

Usage: python bench/smoke.py [model_id] [prompt_tokens] [--out result.json]
"""

import argparse
import json
import platform
import time
from pathlib import Path

from daedalus.engine import Engine
from daedalus.sensors import ThermalMonitor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("prompt_tokens", nargs="?", type=int, default=4000)
    parser.add_argument("--out", help="write a fingerprinted benchmark artifact")
    args = parser.parse_args()
    print(f"loading {args.model} ...")
    t0 = time.perf_counter()
    monitor = ThermalMonitor().start()
    engine = Engine.from_pretrained(args.model, monitor=monitor)
    print(f"loaded in {time.perf_counter() - t0:.1f}s | thermal={monitor.level.name}")

    # Build a long-ish prompt: repeated context + a question, chat-templated.
    filler = "The quick brown fox jumps over the lazy dog. " * (args.prompt_tokens // 10)
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
    if args.out:
        prefill_wall = first_token_at - t1 if first_token_at else 0.0
        artifact = {
            "schema_version": 1,
            "config": {
                "model": args.model,
                "governor": "balanced",
                "kv_bits": engine.config.kv_bits,
                "prompt_tokens": len(tokens),
                "max_tokens": 40,
                "machine": platform.machine(),
                "macos": platform.mac_ver()[0],
                "cache_mode": "cold",
            },
            "software": {
                "mlx": getattr(__import__("mlx.core", fromlist=["__version__"]), "__version__", "unknown"),
                "mlx_lm": getattr(__import__("mlx_lm"), "__version__", "unknown"),
            },
            "metrics": {
                "ttft_s": round(ttft, 3),
                "prefill_tps_wall": round((len(tokens) - 1) / prefill_wall, 1)
                if prefill_wall else 0.0,
                "generation_tps": round(last.generation_tps, 2) if last else 0.0,
            },
        }
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2) + "\n")
        print(f"wrote {output}")
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
