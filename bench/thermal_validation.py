"""M4 validation: sustained big-prompt load, governor on vs off.

Runs N consecutive DISTINCT large prefills (cache can't help) and records:
- wall time per round, prefill tok/s
- thermal pressure trace (1s resolution)
- governor pacing decisions (chunk sizes, idle time)

Usage:
  python bench/thermal_validation.py --model <id> --prompt-tokens 24000 \\
      --rounds 4 --governor on --out bench/results/thermal_on.json

Run one arm, let the machine cool (thermal back to Nominal), run the other.
"""

import argparse
import json
import time
from pathlib import Path

from daedalus.engine import Engine, EngineConfig
from daedalus.governor import GovernorConfig, LevelPolicy, ThermalGovernor
from daedalus.sensors import ThermalLevel, ThermalMonitor

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey"
).split()


def distinct_prompt(engine, round_idx: int, target_tokens: int):
    """A unique prompt per round so the prefix cache cannot shortcut it."""
    filler = " ".join(
        WORDS[(round_idx * 7 + i) % len(WORDS)] + str(round_idx)
        for i in range(target_tokens)
    )
    messages = [
        {"role": "system", "content": f"Session {round_idx}. Context: {filler}"},
        {"role": "user", "content": "Say OK."},
    ]
    tokens = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return tokens[: target_tokens + 50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=24000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--governor", choices=["on", "off"], default="on")
    ap.add_argument("--kv-bits", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    monitor = ThermalMonitor(poll_interval=1.0).start()
    trace = []
    stop_trace = False

    import threading

    def tracer():
        while not stop_trace:
            trace.append({"t": time.time(), "level": monitor.level.name})
            time.sleep(1.0)

    tracer_thread = threading.Thread(target=tracer, daemon=True)

    cfg = GovernorConfig()
    if args.governor == "off":
        cfg.policies = {
            lvl: LevelPolicy(chunk_tokens=2048, duty=1.0) for lvl in ThermalLevel
        }

    governor = ThermalGovernor(monitor, cfg)
    print(f"loading {args.model} ... (thermal={monitor.level.name})")
    engine = Engine.from_pretrained(
        args.model, governor=governor, config=EngineConfig(kv_bits=args.kv_bits or None)
    )

    start_level = monitor.level
    if start_level > ThermalLevel.NOMINAL:
        print(f"WARNING: starting warm ({start_level.name}) — results will be skewed")

    tracer_thread.start()
    rounds = []
    t_all = time.perf_counter()
    for r in range(args.rounds):
        tokens = distinct_prompt(engine, r, args.prompt_tokens)
        cache = engine.make_cache()
        t0 = time.perf_counter()
        report = engine.paced_prefill(tokens, cache)
        wall = time.perf_counter() - t0
        del cache
        import mlx.core as mx

        mx.clear_cache()
        row = {
            "round": r,
            "tokens": report.computed_tokens,
            "wall_s": round(wall, 2),
            "burn_s": round(report.burn_seconds, 2),
            "idle_s": round(report.idle_seconds, 2),
            "tps_wall": round(report.computed_tokens / wall, 1),
            "tps_burn": round(report.computed_tokens / max(report.burn_seconds, 1e-9), 1),
            "chunks": report.chunks,
            "max_level": ThermalLevel(report.max_level).name,
            "level_after": monitor.level.name,
        }
        rounds.append(row)
        print(json.dumps(row))

    total_wall = time.perf_counter() - t_all
    stop_trace = True
    time.sleep(1.2)
    monitor.stop()

    levels_seen = sorted({p["level"] for p in trace})
    result = {
        "config": vars(args),
        "start_level": start_level.name,
        "total_wall_s": round(total_wall, 1),
        "rounds": rounds,
        "levels_seen": levels_seen,
        "trace": trace,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\ntotal={total_wall:.0f}s levels_seen={levels_seen} -> {out}")


if __name__ == "__main__":
    main()
