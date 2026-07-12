#!/usr/bin/env python3
"""Benchmark: sustained decode TPS, TTFT, peak memory, thermal state.

Usage:
  python bench/throughput_bench.py --model mlx-community/Qwen3-0.6B-4bit \\
      --prompt-tokens 100 --max-tokens 500 --kv-bits 8 \\
      --stream-interval 1 --out bench/results/throughput.json

Measures sustained decode throughput (TPS), time-to-first-token (TTFT),
peak memory usage, and thermal trace. Runs with a full-speed governor
(no thermal pacing) for maximum sustained throughput measurement.
"""

import argparse
import json
import platform
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

from daedalus.engine import Engine, EngineConfig
from daedalus.governor import GovernorConfig, LevelPolicy, ThermalGovernor
from daedalus.sensors import ThermalLevel, ThermalMonitor
from mlx_lm.generate import GenerationResponse


def full_speed_policies() -> Dict[ThermalLevel, LevelPolicy]:
    """Full-speed governor policies: full speed at all thermal levels."""
    return {
        level: LevelPolicy(chunk_tokens=2048, duty=1.0)
        for level in ThermalLevel
    }


def build_filler_prompt(tokenizer, target_tokens: int) -> str:
    """Build a filler prompt targeting approximately target_tokens."""
    # Approximate token ratio: ~4 chars per token for English
    # Use a simple repeating pattern to hit token count
    words = "The quick brown fox jumps over the lazy dog. " * (target_tokens // 8 + 1)
    # Truncate to roughly the right token count
    tokens = tokenizer.encode(words)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens)


def get_machine_info() -> Dict[str, str]:
    """Get machine hardware info."""
    hw = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "machine": platform.machine(),
        "hw": hw,
        "os": platform.platform(),
        "python": platform.python_version(),
        "mlx_version": mx.__version__,
    }


def build_filler_tokens(tokenizer, target_tokens: int) -> list:
    """Build a token list targeting exactly target_tokens."""
    # Use a repetitive pattern that tokenizes predictably
    base_text = "The quick brown fox jumps over the lazy dog. "
    tokens = tokenizer.encode(base_text)
    # Repeat to reach target
    repeat = max(1, target_tokens // len(tokens) + 1)
    full_tokens = (tokens * repeat)[:target_tokens]
    return full_tokens


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark sustained decode throughput, TTFT, memory, thermal"
    )
    ap.add_argument("--model", required=True, help="Model path or HF repo ID")
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=100,
        help="Number of tokens in the prompt (default: 100)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Max tokens to generate (default: 500)",
    )
    ap.add_argument(
        "--kv-bits",
        type=int,
        default=8,
        help="KV cache quantization bits (default: 8)",
    )
    ap.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Progress callback interval in tokens (default: 1)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output JSON file path (optional)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)",
    )
    args = ap.parse_args()

    mx.random.seed(args.seed)

    # Setup thermal monitor with full-speed governor (no thermal pacing)
    monitor = ThermalMonitor(poll_interval=1.0).start()
    thermal_trace: List[Dict[str, Any]] = []
    monitor.on_change(
        lambda old, new: thermal_trace.append(
            {"t": time.time(), "from": old.name, "to": new.name}
        )
    )

    # Full-speed governor: no thermal pacing at any level
    gov_cfg = GovernorConfig()
    gov_cfg.policies = full_speed_policies()
    gov_cfg.step_down_seconds = 0.0

    governor = ThermalGovernor(monitor, gov_cfg)

    print(f"Loading model: {args.model} ...")
    print(f"Initial thermal level: {monitor.level.name}")

    engine = Engine.from_pretrained(
        args.model,
        governor=governor,
        config=EngineConfig(kv_bits=args.kv_bits),
    )

    # Build filler prompt tokens
    print(f"Building prompt with {args.prompt_tokens} tokens...")
    prompt_tokens = build_filler_tokens(engine.tokenizer, args.prompt_tokens)
    print(f"Built prompt with {len(prompt_tokens)} tokens")

    # Track metrics
    chunk_events: List[Dict[str, Any]] = []
    ttft: Optional[float] = None
    generation_tokens = 0
    generation_tps = 0.0
    peak_memory = 0.0
    start_time = time.perf_counter()
    prefill_start = start_time
    first_token_time: Optional[float] = None

    def progress_callback(done: int, total: int):
        nonlocal ttft, first_token_time, generation_tokens, generation_tps, peak_memory
        now = time.perf_counter()
        thermal_level = monitor.level.name

        if ttft is None and done > 0:
            # First token generated (prefill complete + first decode token)
            ttft = now - start_time
            first_token_time = now

        chunk_events.append(
            {
                "t": now - start_time,
                "done": done,
                "total": total,
                "thermal": thermal_level,
            }
        )

    try:
        # Generate tokens
        for resp in engine.generate(
            prompt_tokens,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=1.0,
            progress_cb=progress_callback,
        ):
            generation_tokens = resp.generation_tokens
            generation_tps = resp.generation_tps
            if resp.peak_memory > peak_memory:
                peak_memory = resp.peak_memory
    except Exception as e:
        print(f"Generation error: {e}")
        raise
    finally:
        monitor.stop()

    wall_time = time.perf_counter() - start_time
    prefill_time = (first_token_time - start_time) if first_token_time else 0.0

    # Calculate metrics
    prefill_tokens = len(prompt_tokens) - 1  # minus last token consumed by decode
    prefill_tps_wall = prefill_tokens / prefill_time if prefill_time > 0 else 0.0

    result = {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": time.time(),
        "config": {
            "model": args.model,
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "kv_bits": args.kv_bits,
            "stream_interval": args.stream_interval,
            "seed": args.seed,
            "temperature": args.temperature,
            "governor": "full-speed",
            **get_machine_info(),
        },
        "metrics": {
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "wall_s": round(wall_time, 4),
            "prefill_wall_s": round(prefill_time, 4),
            "prefill_tokens": prefill_tokens,
            "prefill_tps_wall": round(prefill_tps_wall, 2),
            "generation_tokens": generation_tokens,
            "generation_tps": round(generation_tps, 2),
            "peak_memory_gb": round(peak_memory, 3),
            "thermal_start": chunk_events[0]["thermal"] if chunk_events else monitor.level.name,
            "thermal_end": monitor.level.name,
        },
        "thermal_trace": thermal_trace,
        "chunks": chunk_events,
    }

    # Print summary
    print(json.dumps(result["metrics"], indent=2))

    # Save output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()