"""End-to-end smoke test on a real model: paced prefill + decode.

Usage: python bench/smoke.py [model_id] [prompt_tokens]

Also writes a fingerprinted JSON result (config/software/metrics, the schema
bench/regression.py reads) to $SMOKE_OUT (default bench/results/smoke.json) so
the CI regression gate has something real to compare against.
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from daedalus.engine import Engine
from daedalus.sensors import ThermalMonitor

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-0.6B-4bit"
PROMPT_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
MAX_TOKENS = 40
OUT = os.environ.get("SMOKE_OUT", "bench/results/smoke.json")


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
        tokens, max_tokens=MAX_TOKENS, temperature=0.0, progress_cb=progress
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

    # Prefill wall time is the span from generation start to the final chunk
    # progress event; tokens follow bench.py's len(tokens) - 1 convention.
    prefill_wall = (progress_events[-1][0] - t1) if progress_events else 0.0
    prefill_tokens = len(tokens) - 1
    result = {
        "schema_version": 2,
        "config": {
            "model": MODEL,
            # Smoke runs the default engine (no governor A/B); these values are
            # fixed so two smoke runs on the same runner fingerprint-match.
            "governor": "default",
            "kv_bits": None,
            "prompt_tokens": len(tokens),
            "max_tokens": MAX_TOKENS,
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "cache_mode": "cold",
            "hw": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "software": {
            "mlx": getattr(
                __import__("mlx.core", fromlist=["__version__"]), "__version__", "unknown"
            ),
            "mlx_lm": getattr(__import__("mlx_lm"), "__version__", "unknown"),
        },
        "metrics": {
            "ttft_s": round(ttft, 3),
            "prefill_tps_wall": round(prefill_tokens / prefill_wall, 1)
            if prefill_wall
            else None,
            "generation_tokens": last.generation_tokens if last else 0,
            "generation_tps": round(last.generation_tps, 2) if last else 0.0,
            "peak_memory_gb": round(last.peak_memory, 2) if last else 0.0,
            "thermal_end": monitor.level.name,
        },
    }
    monitor.stop()

    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")

    assert len(progress_events) >= 2, "expected chunked prefill progress"
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
