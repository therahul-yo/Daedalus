"""daedalus CLI: serve, doctor, warm."""

from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_serve(args) -> int:
    import time

    import uvicorn

    from daedalus import __version__
    from daedalus.cache.store import PrefixCacheStore
    from daedalus.engine import Engine, EngineConfig
    from daedalus.governor import PROFILES, GovernorConfig, ThermalGovernor
    from daedalus.sensors import ThermalMonitor
    from daedalus.server import create_app

    _setup_logging(args.log_level)
    log = logging.getLogger("daedalus")

    monitor = ThermalMonitor().start()
    monitor.on_change(
        lambda old, new: log.log(
            logging.WARNING if int(new) >= 2 else logging.INFO,
            "thermal %s → %s",
            old.name,
            new.name,
        )
    )
    governor = ThermalGovernor(
        monitor,
        GovernorConfig(
            policies=dict(PROFILES[args.profile]), max_duty=args.max_duty
        ),
    )

    log.info("loading %s ...", args.model)
    t0 = time.monotonic()
    engine = Engine.from_pretrained(
        args.model,
        governor=governor,
        config=EngineConfig(kv_bits=args.kv_bits or None),
    )
    load_s = time.monotonic() - t0

    import mlx.core as mx

    store = PrefixCacheStore(args.model)
    cache_stats = store.stats()
    app = create_app(engine, store, model_id=args.model)

    bar = "─" * 62
    for line in (
        bar,
        f"  daedalus v{__version__} — wings that don't melt",
        bar,
        f"  model    {args.model}",
        f"  loaded   {load_s:.1f}s · {mx.get_active_memory() / 1e9:.2f} GB weights"
        f" · kv cache {args.kv_bits or 16}-bit",
        f"  cache    {store.dir} ({cache_stats['entries']} entries)",
        f"  thermal  {monitor.level.name} · profile {args.profile}"
        + (f" · max-duty {args.max_duty}" if args.max_duty < 1.0 else ""),
        f"  api      http://{args.host}:{args.port}/v1  (OpenAI-compatible)",
        bar,
    ):
        print(line, flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_doctor(args) -> int:
    import platform
    import subprocess

    from daedalus.sensors import ThermalMonitor, make_pressure_reader

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

    from daedalus.cache.store import PrefixCacheStore
    from daedalus.engine import Engine, EngineConfig
    from daedalus.governor import ThermalGovernor
    from daedalus.sensors import ThermalMonitor

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
    ap = argparse.ArgumentParser(prog="daedalus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("--model", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--kv-bits", type=int, default=8)
    serve.add_argument(
        "--profile",
        choices=["performance", "balanced", "cool"],
        default="balanced",
        help="thermal pacing profile: performance = full speed until real "
        "throttling (HEAVY); balanced = ease off at MODERATE; cool = quiet/lap",
    )
    serve.add_argument(
        "--max-duty",
        type=float,
        default=1.0,
        help="global GPU duty ceiling (0-1); e.g. 0.5 for quiet mode",
    )
    serve.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="request/thermal log verbosity (default: info)",
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
