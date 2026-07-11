"""airlift CLI: serve, doctor, warm."""

from __future__ import annotations

import argparse
import logging
import sys


def cmd_serve(args) -> int:
    import uvicorn

    from airlift.cache.store import PrefixCacheStore
    from airlift.engine import Engine, EngineConfig
    from airlift.governor import GovernorConfig, ThermalGovernor
    from airlift.sensors import ThermalMonitor
    from airlift.server import create_app

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    monitor = ThermalMonitor().start()
    governor = ThermalGovernor(
        monitor, GovernorConfig(max_duty=args.max_duty)
    )
    print(f"loading {args.model} ...", flush=True)
    engine = Engine.from_pretrained(
        args.model,
        governor=governor,
        config=EngineConfig(kv_bits=args.kv_bits or None),
    )
    store = PrefixCacheStore(args.model)
    app = create_app(engine, store, model_id=args.model)
    print(
        f"airlift serving {args.model} on http://{args.host}:{args.port}/v1"
        f" | thermal={monitor.level.name}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_doctor(args) -> int:
    import platform
    import subprocess

    from airlift.sensors import ThermalMonitor, make_pressure_reader

    hw = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
        capture_output=True,
        text=True,
    ).stdout.split()
    print(f"machine: {' '.join(hw[:-1])} | RAM: {int(hw[-1]) / 1024**3:.0f} GB")
    print(f"macOS: {platform.mac_ver()[0]}")
    try:
        level = make_pressure_reader()()
        print(f"thermal pressure (no sudo): {level.name} — OK")
    except Exception as exc:
        print(f"thermal pressure: FAILED ({exc})")
        return 1
    try:
        import mlx.core as mx

        print(f"mlx: {mx.__version__} | metal: {mx.metal.is_available()}")
        import mlx_lm

        print(f"mlx-lm: {mlx_lm.__version__}")
    except Exception as exc:
        print(f"mlx: FAILED ({exc})")
        return 1
    print("doctor: all good")
    return 0


def cmd_warm(args) -> int:
    """Pre-prefill a prompt file into the persistent cache while cool."""
    import json as _json
    from pathlib import Path

    from airlift.cache.store import PrefixCacheStore
    from airlift.engine import Engine, EngineConfig
    from airlift.governor import ThermalGovernor
    from airlift.sensors import ThermalMonitor

    monitor = ThermalMonitor().start()
    engine = Engine.from_pretrained(
        args.model, monitor=monitor, config=EngineConfig(kv_bits=args.kv_bits or None)
    )
    store = PrefixCacheStore(args.model)

    raw = Path(args.prompts).read_text()
    prompts = _json.loads(raw)  # [{"messages": [...]}, ...]
    for i, item in enumerate(prompts):
        tokens = engine.tokenizer.apply_chat_template(
            item["messages"], add_generation_prompt=True
        )
        if store.fetch(tokens) is not None:
            print(f"[{i}] already cached ({len(tokens)} tokens)")
            continue
        cache = engine.make_cache()
        report = engine.paced_prefill(
            tokens,
            cache,
            progress_cb=lambda done, total: print(
                f"\r[{i}] prefill {done}/{total}", end="", flush=True
            ),
        )
        store.put(tokens[: report.computed_tokens], cache)
        print(f"\n[{i}] cached {report.computed_tokens} tokens")
    print("warm done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="airlift")
    sub = ap.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("--model", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--kv-bits", type=int, default=8)
    serve.add_argument(
        "--max-duty",
        type=float,
        default=1.0,
        help="global GPU duty ceiling (0-1); e.g. 0.5 for quiet mode",
    )
    serve.set_defaults(fn=cmd_serve)

    doctor = sub.add_parser("doctor", help="check thermal sensor + mlx setup")
    doctor.set_defaults(fn=cmd_doctor)

    warm = sub.add_parser("warm", help="pre-prefill prompts into the cache")
    warm.add_argument("--model", required=True)
    warm.add_argument("--prompts", required=True, help="JSON file of {messages}")
    warm.add_argument("--kv-bits", type=int, default=8)
    warm.set_defaults(fn=cmd_warm)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
