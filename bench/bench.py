"""Benchmark: TTFT / throughput / thermal trace, governor on vs off.

Usage:
  python bench/bench.py --model mlx-community/Qwen3-0.6B-4bit \\
      --prompt-tokens 8000 --max-tokens 64 --governor on --out bench/results/run.json

Emits one JSON document with per-chunk timings and a thermal-pressure trace,
so runs are comparable across settings and machines.
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

from daedalus.engine import Engine, EngineConfig
from daedalus.governor import GovernorConfig, LevelPolicy, ThermalGovernor
from daedalus.sensors import ThermalLevel, ThermalMonitor


def full_speed_policies():
    """Governor-off baseline: mlx-lm-equivalent behavior at every level."""
    return {level: LevelPolicy(chunk_tokens=2048, duty=1.0) for level in ThermalLevel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--prompt-tokens", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--governor", choices=["on", "off"], default="on")
    ap.add_argument("--kv-bits", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    monitor = ThermalMonitor(poll_interval=1.0).start()
    thermal_trace = []
    monitor.on_change(
        lambda old, new: thermal_trace.append(
            {"t": time.time(), "from": old.name, "to": new.name}
        )
    )

    cfg = GovernorConfig()
    if args.governor == "off":
        cfg.policies = full_speed_policies()
        cfg.step_down_seconds = 0.0

    governor = ThermalGovernor(monitor, cfg)
    print(f"loading {args.model} ...")
    engine = Engine.from_pretrained(
        args.model,
        governor=governor,
        config=EngineConfig(kv_bits=args.kv_bits or None),
    )

    filler = "The quick brown fox jumps over the lazy dog. " * (
        args.prompt_tokens // 10
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant. " + filler},
        {"role": "user", "content": "Summarize the above in one sentence."},
    ]
    tokens = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    chunk_events = []
    start = time.perf_counter()

    def progress(done, total):
        chunk_events.append(
            {
                "t": time.perf_counter() - start,
                "done": done,
                "total": total,
                "thermal": monitor.level.name,
            }
        )

    ttft = None
    n_gen = 0
    gen_tps = 0.0
    peak_mem = 0.0
    for resp in engine.generate(
        tokens,
        max_tokens=args.max_tokens,
        temperature=0.0,
        progress_cb=progress,
    ):
        if ttft is None:
            ttft = time.perf_counter() - start
        n_gen = resp.generation_tokens
        gen_tps = resp.generation_tps
        peak_mem = resp.peak_memory

    wall = time.perf_counter() - start
    prefill_tokens = len(tokens) - 1
    prefill_wall = chunk_events[-1]["t"] if chunk_events else 0.0

    result = {
        "config": {
            "model": args.model,
            "governor": args.governor,
            "kv_bits": args.kv_bits,
            "prompt_tokens": len(tokens),
            "max_tokens": args.max_tokens,
            "machine": platform.machine(),
            "hw": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "metrics": {
            "ttft_s": round(ttft or -1, 3),
            "wall_s": round(wall, 3),
            "prefill_wall_s": round(prefill_wall, 3),
            "prefill_tps_wall": round(prefill_tokens / prefill_wall, 1)
            if prefill_wall
            else None,
            "generation_tokens": n_gen,
            "generation_tps": round(gen_tps, 2),
            "peak_memory_gb": round(peak_mem, 2),
            "thermal_start": chunk_events[0]["thermal"] if chunk_events else None,
            "thermal_end": monitor.level.name,
        },
        "thermal_trace": thermal_trace,
        "chunks": chunk_events,
    }
    monitor.stop()

    print(json.dumps(result["metrics"], indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
