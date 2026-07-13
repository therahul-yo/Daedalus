"""daedalus CLI: serve, doctor, warm."""

from __future__ import annotations

import argparse
import logging
import sys


def _setup_logging(level: str, json_output: bool = False) -> None:
    if json_output:
        from daedalus.observability import setup_logging

        setup_logging(level=level, json_output=True)
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s │ %(levelname)-5s │ %(message)s",
            datefmt="%H:%M:%S",
        )
    for noisy in ("httpx", "urllib3", "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_serve(args) -> int:
    import os
    import time
    from pathlib import Path

    import uvicorn

    from daedalus import __version__
    from daedalus.cache.store import PrefixCacheStore
    from daedalus.engine import Engine, EngineConfig
    from daedalus.governor import PROFILES, GovernorConfig, ThermalGovernor
    from daedalus.sensors import ThermalMonitor
    from daedalus.server import create_app
    from daedalus.runtime import cache_identity

    _setup_logging(args.log_level, json_output=getattr(args, "log_json", False))
    log = logging.getLogger("daedalus")

    from daedalus.observability import maybe_init_otel

    if maybe_init_otel():
        log.info("OpenTelemetry tracing enabled (OTLP export)")

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
            policies=dict(PROFILES[args.profile]), max_duty=args.max_duty,
            anticipate_rising=args.anticipate_rising,
        ),
    )

    log.info("loading %s ...", args.model)
    t0 = time.monotonic()
    try:
        engine = Engine.from_pretrained(
            args.model,
            monitor=monitor,
            governor=governor,
            config=EngineConfig(
                kv_bits=args.kv_bits or None,
                prefill_chunk_tokens=args.prefill_chunk_tokens,
                clear_metal_cache_between_chunks=args.clear_metal_cache_between_chunks,
                num_draft_tokens=args.num_draft_tokens,
            ),
            draft_model_path=args.draft_model,
        )
    except Exception as exc:
        monitor.stop()
        print(f"error: could not load model {args.model!r}: {exc}", flush=True)
        print(
            "hint: check the model id for typos (huggingface.co/mlx-community), "
            "network access for a first download, and free disk space.",
            flush=True,
        )
        return 1
    load_s = time.monotonic() - t0

    import mlx.core as mx

    cache_key = cache_identity(
        args.model, kv_bits=args.kv_bits or None,
        tokenizer_id=getattr(engine.tokenizer, "name_or_path", args.model),
        model_revision=args.model_revision,
        draft_model=args.draft_model,
    )
    api_key = args.api_key or (os.environ.get(args.api_key_env) if args.api_key_env else None)
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text().strip()
    if args.host not in {"127.0.0.1", "::1", "localhost"} and not api_key:
        raise ValueError("API key source did not provide a key for non-local binding")
    store = PrefixCacheStore(
        cache_key,
        max_ram_bytes=args.cache_ram_mb * 1024**2 if args.cache_ram_mb else None,
        max_disk_bytes=args.cache_disk_gb * 1024**3,
        exclusive=True,
    )
    cache_stats = store.stats()
    app = create_app(
        engine, store, model_id=args.model, max_pending_requests=args.max_pending_requests,
        api_key=api_key,
        max_active_memory_bytes=args.max_active_memory_gb * 1024**3 if args.max_active_memory_gb else None,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=args.max_completion_tokens,
        requests_per_minute=args.requests_per_minute,
        max_request_bytes=args.max_request_bytes,
        audit_log_path=args.audit_log,
        cors_origins=args.cors_origins,
        global_rps=args.global_rps,
    )

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
        f"  prefill  {args.prefill_chunk_tokens or 'profile'}-token nominal chunks"
        + (" · clearing Metal per chunk" if args.clear_metal_cache_between_chunks else ""),
        f"  api      http://{args.host}:{args.port}/v1  (OpenAI-compatible)",
        f"  queue    max {args.max_pending_requests} requests"
        + (" · API key enabled" if api_key else ""),
        bar,
    ):
        print(line, flush=True)

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        monitor.stop()
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
    from daedalus.runtime import cache_identity
    from daedalus.governor import ThermalGovernor
    from daedalus.sensors import ThermalMonitor

    # Validate the prompts file BEFORE the multi-minute model load.
    try:
        raw = Path(args.prompts).read_text()
        prompts = _json.loads(raw)  # [{"messages": [...]}, ...]
        if not isinstance(prompts, list) or not all(
            isinstance(p, dict) and isinstance(p.get("messages"), list) for p in prompts
        ):
            raise ValueError('expected a JSON array of {"messages": [...]} objects')
    except Exception as exc:
        print(f"error: invalid prompts file {args.prompts!r}: {exc}", flush=True)
        return 1

    monitor = ThermalMonitor().start()
    engine = Engine.from_pretrained(
        args.model, monitor=monitor, config=EngineConfig(kv_bits=args.kv_bits or None)
    )
    store = PrefixCacheStore(cache_identity(
        args.model, kv_bits=args.kv_bits or None,
        tokenizer_id=getattr(engine.tokenizer, "name_or_path", args.model),
        model_revision=args.model_revision,
    ), exclusive=True)
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


def cmd_tune(args) -> int:
    """Measure nominal prefill chunk candidates on the current machine."""
    import json
    import statistics
    import time
    from pathlib import Path

    import mlx.core as mx

    from daedalus.engine import Engine, EngineConfig
    from daedalus.governor import GovernorConfig, LevelPolicy, ThermalGovernor
    from daedalus.sensors import ThermalLevel, ThermalMonitor

    candidates = [int(value) for value in args.candidates.split(",") if value.strip()]
    if not candidates or any(value < 128 for value in candidates):
        raise ValueError("--candidates must be comma-separated integers of at least 128")
    monitor = ThermalMonitor().start()
    # No idle gaps while benchmarking nominal chunk throughput. Thermal state
    # is still recorded and callers should start from NOMINAL.
    policies = {level: LevelPolicy(chunk_tokens=2048, duty=1.0) for level in ThermalLevel}
    governor = ThermalGovernor(monitor, GovernorConfig(policies=policies))
    print(f"loading {args.model} (start thermal={monitor.level.name}) ...")
    engine = Engine.from_pretrained(args.model, governor=governor, config=EngineConfig(kv_bits=args.kv_bits or None))
    filler = "Daedalus measures prefill throughput on Apple Silicon. " * max(1, args.prompt_tokens // 8)
    tokens = engine.tokenizer.apply_chat_template(
        [{"role": "system", "content": filler}, {"role": "user", "content": "Reply OK."}],
        add_generation_prompt=True,
    )[: args.prompt_tokens + 64]
    samples = {chunk: [] for chunk in candidates}
    order = [candidates[(i + r) % len(candidates)] for r in range(args.repeats) for i in range(len(candidates))]
    try:
        for chunk in order:
            runner = Engine(engine.model, engine.tokenizer, governor, EngineConfig(
                kv_bits=args.kv_bits or None, prefill_chunk_tokens=chunk,
                clear_metal_cache_between_chunks=args.clear_metal_cache_between_chunks,
            ))
            cache = runner.make_cache()
            started = time.perf_counter()
            report = runner.paced_prefill(tokens, cache)
            elapsed = time.perf_counter() - started
            tps = report.computed_tokens / elapsed
            samples[chunk].append(tps)
            print(f"chunk={chunk:5d}  {tps:7.1f} tok/s  thermal={monitor.level.name}")
            del cache
            mx.clear_cache()
        medians = {chunk: round(statistics.median(values), 2) for chunk, values in samples.items()}
        best = max(medians, key=medians.get)
        result = {"model": args.model, "prompt_tokens": len(tokens), "samples_tps": samples,
                  "median_tps": medians, "recommended_prefill_chunk_tokens": best,
                  "final_thermal": monitor.level.name}
        print(json.dumps(result, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        return 0
    finally:
        monitor.stop()


def main() -> int:
    ap = argparse.ArgumentParser(prog="daedalus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("--model", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument(
        "--api-key", help="require this Bearer token for /v1 endpoints; required for non-local host"
    )
    serve.add_argument("--api-key-env", help="environment variable holding the API key")
    serve.add_argument("--api-key-file", help="file containing the API key (preferred for services)")
    serve.add_argument("--model-revision", help="immutable model revision included in the cache namespace")
    serve.add_argument("--draft-model", help="optional compatible draft model for speculative decoding")
    serve.add_argument("--num-draft-tokens", type=int, default=0,
                       help="tokens proposed per speculative decoding step (requires --draft-model)")
    serve.add_argument(
        "--max-pending-requests", type=int, default=8,
        help="maximum active or queued requests (default: 8)",
    )
    serve.add_argument(
        "--cache-ram-mb", type=int,
        help="RAM budget for prefix cache in MiB (default: 20%% of system RAM)",
    )
    serve.add_argument(
        "--cache-disk-gb", type=int, default=10,
        help="disk budget for prefix cache in GiB (default: 10)",
    )
    serve.add_argument(
        "--max-active-memory-gb", type=float,
        help="reject new work after cache eviction if MLX active memory exceeds this limit",
    )
    serve.add_argument("--max-prompt-tokens", type=int, default=65536)
    serve.add_argument("--max-completion-tokens", type=int, default=4096)
    serve.add_argument("--requests-per-minute", type=int, default=0,
                       help="per-client LAN limit; 0 disables rate limiting")
    serve.add_argument("--max-request-bytes", type=int, default=2 * 1024 * 1024)
    serve.add_argument(
        "--audit-log",
        help="path to a structured (NDJSON) audit log for auth failures, rate-limit events, and cache-admin operations",
    )
    serve.add_argument(
        "--log-json", action="store_true",
        help="emit structured JSON logs (uses structlog when installed)",
    )
    serve.add_argument("--kv-bits", type=int, default=8)
    serve.add_argument(
        "--prefill-chunk-tokens", type=int,
        help="measured nominal prefill chunk size; thermal policies still override when hot",
    )
    serve.add_argument(
        "--clear-metal-cache-between-chunks", action="store_true",
        help="free Metal allocations after each prefill chunk (slower but useful under tight memory)",
    )
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
        "--anticipate-rising",
        action="store_true",
        help="pace one thermal level ahead while pressure is rising — the "
        "macOS signal lags the heat ramp on a fanless chassis",
    )
    serve.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="request/thermal log verbosity (default: info)",
    )
    serve.add_argument(
        "--cors-origin", action="append", dest="cors_origins", default=None,
        help="permitted CORS origin; repeat for multiple origins",
    )
    serve.add_argument(
        "--global-rps", type=float, default=0.0,
        help="global rate limit across all clients in requests per second (0 = disabled)",
    )
    serve.set_defaults(fn=cmd_serve)

    doctor = sub.add_parser("doctor", help="check thermal sensor + mlx setup")
    doctor.set_defaults(fn=cmd_doctor)

    warm = sub.add_parser("warm", help="pre-prefill prompts into the cache")
    warm.add_argument("--model", required=True)
    warm.add_argument("--prompts", required=True, help="JSON file of {messages}")
    warm.add_argument("--kv-bits", type=int, default=8)
    warm.add_argument("--model-revision", help="immutable model revision included in the cache namespace")
    warm.set_defaults(fn=cmd_warm)

    tune = sub.add_parser("tune", help="benchmark prefill chunks and recommend the fastest nominal size")
    tune.add_argument("--model", required=True)
    tune.add_argument("--candidates", default="1024,2048,4096")
    tune.add_argument("--prompt-tokens", type=int, default=8192)
    tune.add_argument("--repeats", type=int, default=2)
    tune.add_argument("--kv-bits", type=int, default=8)
    tune.add_argument("--clear-metal-cache-between-chunks", action="store_true")
    tune.add_argument("--out", help="write JSON result to this path")
    tune.set_defaults(fn=cmd_tune)

    from daedalus.cache.cli import add_cache_parser
    add_cache_parser(sub)

    args = ap.parse_args()
    if args.cmd == "serve":
        if args.max_pending_requests < 1:
            ap.error("--max-pending-requests must be at least 1")
        if args.cache_ram_mb is not None and args.cache_ram_mb < 1:
            ap.error("--cache-ram-mb must be positive")
        if args.cache_disk_gb < 1:
            ap.error("--cache-disk-gb must be positive")
        if args.host not in {"127.0.0.1", "::1", "localhost"} and not args.api_key:
            if not args.api_key_env and not args.api_key_file:
                ap.error("an API key source is required when binding outside localhost")
        if sum(bool(value) for value in (args.api_key, args.api_key_env, args.api_key_file)) > 1:
            ap.error("use only one of --api-key, --api-key-env, or --api-key-file")
        if args.prefill_chunk_tokens is not None and args.prefill_chunk_tokens < 128:
            ap.error("--prefill-chunk-tokens must be at least 128")
        if args.max_active_memory_gb is not None and args.max_active_memory_gb <= 0:
            ap.error("--max-active-memory-gb must be positive")
        if args.max_prompt_tokens < 1 or args.max_completion_tokens < 1:
            ap.error("token limits must be positive")
        if args.requests_per_minute < 0:
            ap.error("--requests-per-minute cannot be negative")
        if args.max_request_bytes < 1:
            ap.error("--max-request-bytes must be positive")
        if args.num_draft_tokens < 0:
            ap.error("--num-draft-tokens cannot be negative")
        if args.num_draft_tokens and not args.draft_model:
            ap.error("--num-draft-tokens requires --draft-model")
        if args.draft_model and not args.num_draft_tokens:
            # A draft model with zero draft tokens silently loads a second
            # multi-GB model that never speculates.
            ap.error("--draft-model requires --num-draft-tokens (e.g. 3)")
    if args.cmd == "tune" and (args.prompt_tokens < 512 or args.repeats < 1):
        ap.error("--prompt-tokens must be at least 512 and --repeats at least 1")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
