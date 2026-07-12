"""daedalus CLI: serve, doctor, warm, config, inspect-cache, benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _setup_logging(level: str) -> None:
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

    # Load all models
    models = []
    engines = {}
    stores = {}
    total_memory_gb = 0.0
    import mlx.core as mx

    for model_path in args.model:
        log.info("loading %s ...", model_path)
        t0 = time.monotonic()
        engine = Engine.from_pretrained(
            model_path,
            governor=governor,
            config=EngineConfig(
                kv_bits=args.kv_bits or None,
                prefill_chunk_tokens=args.prefill_chunk_tokens,
                clear_metal_cache_between_chunks=args.clear_metal_cache_between_chunks,
                num_draft_tokens=args.num_draft_tokens,
            ),
            draft_model_path=args.draft_model,
        )
        load_s = time.monotonic() - t0

        cache_key = cache_identity(
            model_path,
            kv_bits=args.kv_bits or None,
            tokenizer_id=getattr(engine.tokenizer, "name_or_path", model_path),
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
            cache_ttl_days=args.cache_ttl_days,
            exclusive=True,
        )
        cache_stats = store.stats()
        models.append({
            "id": model_path,
            "engine": engine,
            "store": store,
            "load_time": load_s,
            "memory_gb": mx.get_active_memory() / 1e9,
            "kv_bits": args.kv_bits or 16,
            "cache_entries": cache_stats['entries'],
            "cache_dir": str(store.dir),
        })
        engines[model_path] = engine
        stores[model_path] = store
        total_memory_gb += mx.get_active_memory() / 1e9

    # Log startup banner
    bar = "─" * 62
    lines = [
        bar,
        f"  daedalus v{__version__} — wings that don't melt",
        bar,
    ]
    for m in models:
        lines.append(f"  model    {m['id']}")
        lines.append(f"  loaded   {m['load_time']:.1f}s · {m['memory_gb']:.2f} GB weights · kv cache {m['kv_bits']}-bit")
        lines.append(f"  cache    {m['cache_dir']} ({m['cache_entries']} entries)")
    lines.append(f"  total    {total_memory_gb:.2f} GB weights")
    lines.append(f"  thermal  {monitor.level.name} · profile {args.profile}" + (f" · max-duty {args.max_duty}" if args.max_duty < 1.0 else ""))
    lines.append(f"  prefill  {args.prefill_chunk_tokens or 'profile'}-token nominal chunks" + (" · clearing Metal per chunk" if args.clear_metal_cache_between_chunks else ""))
    lines.append(f"  api      http://{args.host}:{args.port}/v1  (OpenAI-compatible)")
    lines.append(f"  queue    max {args.max_pending_requests} requests" + (" · API key enabled" if api_key else ""))
    lines.append(bar)
    for line in lines:
        print(line, flush=True)

    # Multi-model server state
    app = create_app(
        engines, stores, args.model, max_pending_requests=args.max_pending_requests,
        api_key=api_key,
        max_active_memory_bytes=args.max_active_memory_gb * 1024**3 if args.max_active_memory_gb else None,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=args.max_completion_tokens,
        requests_per_minute=args.requests_per_minute,
        max_request_bytes=args.max_request_bytes,
        global_rps=args.global_rps,
        global_burst=args.global_burst,
        shutdown_timeout=args.shutdown_timeout,
        cors_origins=args.cors_origins.split(",") if args.cors_origins else [],
        cors_allow_credentials=args.cors_allow_credentials,
        audit_log_path=args.audit_log_path,
        governor=governor,
        monitor=monitor,
        stream_interval=args.stream_interval,
    )

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

    monitor = ThermalMonitor().start()
    engine = Engine.from_pretrained(
        args.model, monitor=monitor, config=EngineConfig(kv_bits=args.kv_bits or None)
    )
    store = PrefixCacheStore(cache_identity(
        args.model, kv_bits=args.kv_bits or None,
        tokenizer_id=getattr(engine.tokenizer, "name_or_path", args.model),
        model_revision=args.model_revision,
    ), exclusive=True)

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


def cmd_config_show(args) -> int:
    """Show resolved configuration (CLI + env + defaults merged)."""
    import json
    import os
    import platform
    import subprocess

    # Build config dict from args (which already has defaults applied by argparse)
    config = {
        "model": args.model,
        "kv_bits": args.kv_bits,
        "profile": args.profile,
        "max_duty": args.max_duty,
        "prefill_chunk_tokens": args.prefill_chunk_tokens,
        "clear_metal_cache_between_chunks": args.clear_metal_cache_between_chunks,
        "cache_ram_mb": args.cache_ram_mb,
        "cache_disk_gb": args.cache_disk_gb,
        "cache_ttl_days": args.cache_ttl_days,
        "max_active_memory_gb": args.max_active_memory_gb,
        "max_pending_requests": args.max_pending_requests,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_completion_tokens": args.max_completion_tokens,
        "requests_per_minute": args.requests_per_minute,
        "max_request_bytes": args.max_request_bytes,
        "global_rps": args.global_rps,
        "global_burst": args.global_burst,
        "shutdown_timeout": args.shutdown_timeout,
        "cors_origins": args.cors_origins,
        "cors_allow_credentials": args.cors_allow_credentials,
        "audit_log_path": args.audit_log_path,
        "log_level": args.log_level,
        "host": args.host,
        "port": args.port,
        "api_key": args.api_key,
        "api_key_env": args.api_key_env,
        "api_key_file": args.api_key_file,
        "model_revision": args.model_revision,
        "draft_model": args.draft_model,
        "num_draft_tokens": args.num_draft_tokens,
    }

    # Add environment variable overrides
    env_overrides = {}
    for key, value in os.environ.items():
        if key.startswith("DAEDALUS_"):
            env_key = key[8:].lower()
            env_overrides[env_key] = value
    if env_overrides:
        config["env_overrides"] = env_overrides

    # Add system info
    hw = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
        capture_output=True,
        text=True,
    ).stdout.split()
    config["system"] = {
        "cpu": " ".join(hw[:-1]) if hw else "unknown",
        "ram_gb": int(hw[-1]) / 1024**3 if hw else 0,
        "macos": platform.mac_ver()[0],
        "platform": platform.machine(),
    }

    # Try to import MLX info
    try:
        import mlx.core as mx
        config["mlx"] = {
            "version": mx.__version__,
            "metal_available": mx.metal.is_available(),
        }
    except Exception as exc:
        config["mlx"] = {"error": str(exc)}

    # Cache identity if model provided
    if args.model:
        from daedalus.runtime import cache_identity
        from daedalus.engine import Engine, EngineConfig
        engine = Engine.from_pretrained(args.model, config=EngineConfig(kv_bits=args.kv_bits or None))
        config["cache_identity"] = cache_identity(
            args.model,
            kv_bits=args.kv_bits or None,
            tokenizer_id=getattr(engine.tokenizer, "name_or_path", args.model),
            model_revision=args.model_revision,
            draft_model=args.draft_model,
        )
        del engine

    print(json.dumps(config, indent=2, default=str))
    return 0


def cmd_config_doctor(args) -> int:
    """Validate configuration, warn on conflicts."""
    import json
    import os
    import platform
    import subprocess
    import warnings

    issues = []
    warnings_list = []

    # Check model
    if not args.model:
        issues.append("No model specified (--model required)")
    else:
        try:
            from daedalus.engine import Engine, EngineConfig
            engine = Engine.from_pretrained(args.model, config=EngineConfig(kv_bits=args.kv_bits or None))
            del engine
        except Exception as exc:
            issues.append(f"Model '{args.model}' failed to load: {exc}")

    # Check KV bits
    if args.kv_bits not in (4, 8, 16):
        warnings_list.append(f"KV bits {args.kv_bits} may not be supported; recommended: 4, 8, or 16")

    # Check thermal profile
    if args.profile not in ("performance", "balanced", "cool"):
        issues.append(f"Invalid thermal profile: {args.profile}")

    # Check max duty
    if not (0 < args.max_duty <= 1.0):
        issues.append(f"max-duty must be in (0, 1], got {args.max_duty}")

    # Check prefill chunk tokens
    if args.prefill_chunk_tokens is not None and args.prefill_chunk_tokens < 128:
        warnings_list.append(f"prefill-chunk-tokens {args.prefill_chunk_tokens} < 128 may be inefficient")

    # Check cache RAM
    if args.cache_ram_mb is not None:
        if args.cache_ram_mb < 1:
            issues.append("cache-ram-mb must be positive")
        else:
            # Check against system RAM
            hw = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            ).stdout.strip()
            if hw:
                total_ram_gb = int(hw) / 1024**3
                if args.cache_ram_mb > total_ram_gb * 1024 * 0.8:
                    warnings_list.append(f"cache-ram-mb {args.cache_ram_mb} > 80% of system RAM ({total_ram_gb:.0f}GB)")

    # Check cache disk
    if args.cache_disk_gb < 1:
        issues.append("cache-disk-gb must be positive")

    # Check cache TTL
    if args.cache_ttl_days is not None and args.cache_ttl_days < 1:
        issues.append("cache-ttl-days must be positive")

    # Check max active memory
    if args.max_active_memory_gb is not None and args.max_active_memory_gb <= 0:
        issues.append("max-active-memory-gb must be positive")

    # Check host binding + API key
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        has_key = bool(args.api_key or args.api_key_env or args.api_key_file)
        if not has_key:
            issues.append("API key required when binding outside localhost (--api-key, --api-key-env, or --api-key-file)")

    # Check API key sources
    key_sources = sum(bool(v) for v in (args.api_key, args.api_key_env, args.api_key_file))
    if key_sources > 1:
        issues.append("Use only one of --api-key, --api-key-env, or --api-key-file")

    # Check requests per minute
    if args.requests_per_minute < 0:
        issues.append("requests-per-minute cannot be negative")

    # Check max request bytes
    if args.max_request_bytes < 1:
        issues.append("max-request-bytes must be positive")

    # Check draft model
    if args.num_draft_tokens > 0 and not args.draft_model:
        issues.append("--num-draft-tokens requires --draft-model")

    # Check global RPS
    if args.global_rps < 0:
        issues.append("global-rps cannot be negative")
    if args.global_burst < 0:
        issues.append("global-burst cannot be negative")

    # Check shutdown timeout
    if args.shutdown_timeout <= 0:
        issues.append("shutdown-timeout must be positive")

    # Check CORS
    if args.cors_origins:
        origins = [o.strip() for o in args.cors_origins.split(",") if o.strip()]
        for origin in origins:
            if not (origin.startswith("http://") or origin.startswith("https://") or origin == "*"):
                warnings_list.append(f"CORS origin '{origin}' should start with http:// or https:// (or be *)")

    # Check thermal sensor
    try:
        from daedalus.sensors import make_pressure_reader
        level = make_pressure_reader()()
        if level.value >= 2:
            warnings_list.append(f"System thermal pressure is {level.name} - consider 'cool' profile")
    except Exception as exc:
        warnings_list.append(f"Could not read thermal pressure: {exc}")

    # Print results
    print("=== Daedalus Configuration Doctor ===")
    print(f"Model: {args.model}")
    print(f"KV bits: {args.kv_bits}")
    print(f"Thermal profile: {args.profile} (max duty: {args.max_duty})")
    print()

    if issues:
        print("❌ ERRORS:")
        for issue in issues:
            print(f"  - {issue}")
        print()
    else:
        print("✅ No configuration errors found.")
        print()

    if warnings_list:
        print("⚠️  WARNINGS:")
        for warn in warnings_list:
            print(f"  - {warn}")
        print()
    else:
        print("✅ No configuration warnings.")
        print()

    # Print system info
    hw = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
        capture_output=True,
        text=True,
    ).stdout.split()
    print(f"System: {' '.join(hw[:-1])} | RAM: {int(hw[-1]) / 1024**3:.0f} GB")
    print(f"macOS: {platform.mac_ver()[0]}")

    try:
        import mlx.core as mx
        print(f"MLX: {mx.__version__} | Metal: {mx.metal.is_available()}")
    except Exception as exc:
        print(f"MLX: FAILED ({exc})")

    return 1 if issues else 0


def cmd_inspect_cache(args) -> int:
    """Rich cache inspection for a model."""
    import logging
    import time
    from pathlib import Path

    logging.basicConfig(level=logging.WARNING)

    from daedalus.cache.store import PrefixCacheStore, _default_cache_dir, _sanitize_model_key
    from daedalus.runtime import cache_identity
    from daedalus.engine import Engine, EngineConfig

    cache_dir = Path(args.cache_dir) if args.cache_dir else _default_cache_dir()

    # Build cache identity
    engine = Engine.from_pretrained(args.model, config=EngineConfig(kv_bits=args.kv_bits or None))
    cache_key = cache_identity(
        args.model,
        kv_bits=args.kv_bits or None,
        tokenizer_id=getattr(engine.tokenizer, "name_or_path", args.model),
        model_revision=args.model_revision,
        draft_model=args.draft_model,
    )
    del engine

    store = PrefixCacheStore(
        cache_key,
        cache_dir=cache_dir,
        exclusive=False,
    )

    print(f"Cache inspection for model: {args.model}")
    print(f"Cache key: {cache_key}")
    print(f"Cache directory: {store.dir}")
    print()

    # Get store stats
    stats = store.stats()
    print(f"=== Summary ===")
    print(f"Total entries: {stats['entries']}")
    print(f"Resident in RAM: {stats['resident_entries']}")
    print(f"Resident size: {stats['resident_bytes'] / 1024**2:.1f} MB")
    print(f"Hits: {stats['hits']}")
    print(f"Misses: {stats['misses']}")
    hit_rate = stats['hits'] / (stats['hits'] + stats['misses']) * 100 if (stats['hits'] + stats['misses']) > 0 else 0
    print(f"Hit rate: {hit_rate:.1f}%")
    print(f"Lookup time: {stats['lookup_seconds']:.3f}s")
    print(f"Load time: {stats['load_seconds']:.3f}s")
    print(f"Copy time: {stats['copy_seconds']:.3f}s")
    print()

    # List entries
    entries = store.list_entries()
    if not entries:
        print("No cache entries found.")
        return 0

    # Sort by last used (most recent first)
    entries.sort(key=lambda e: e["last_used"], reverse=True)

    print(f"=== Entries ({len(entries)}) ===")
    print(f"{'KEY':<26} {'TOKENS':>8} {'SIZE':>10} {'RAM':>4} {'DISK':>5} {'AGE':>8} {'HITS':>5}")
    print("-" * 80)

    total_tokens = 0
    total_size = 0
    for entry in entries:
        key = entry["key"][:24]
        tokens = entry["token_count"]
        size_bytes = entry["size_bytes"]
        in_ram = "yes" if entry["in_ram"] else "no"
        on_disk = "yes" if entry["on_disk"] else "no"
        age_seconds = entry["age_seconds"]
        hits = entry["hits"]

        # Format age
        if age_seconds < 60:
            age_str = f"{age_seconds:.0f}s"
        elif age_seconds < 3600:
            age_str = f"{age_seconds / 60:.1f}m"
        elif age_seconds < 86400:
            age_str = f"{age_seconds / 3600:.1f}h"
        else:
            age_str = f"{age_seconds / 86400:.1f}d"

        print(f"{key:<26} {tokens:>8} {size_bytes / 1024**2:>9.1f}MB {in_ram:>4} {on_disk:>5} {age_str:>8} {hits:>5}")
        total_tokens += tokens
        total_size += size_bytes

    print("-" * 80)
    print(f"Total: {len(entries)} entries, {total_tokens:,} tokens, {total_size / 1024**2:.1f} MB")
    print()

    # Detailed view if requested
    if args.detailed:
        print("=== Detailed Entry Info ===")
        for entry in entries:
            print(f"\n--- {entry['key'][:24]} ---")
            print(f"  Model: {entry['model_key']}")
            print(f"  Tokens: {entry['token_count']:,}")
            print(f"  Size: {entry['size_bytes'] / 1024**2:.1f} MB")
            print(f"  In RAM: {'Yes' if entry['in_ram'] else 'No'}")
            print(f"  On Disk: {'Yes' if entry['on_disk'] else 'No'}")
            print(f"  Last used: {time.ctime(entry['last_used'])} ({entry['age_seconds']:.0f}s ago)")
            print(f"  Hits: {entry['hits']}")

            if args.show_tokens:
                # Load the entry to get tokens
                detailed = store.inspect_entry(entry['key'])
                if detailed and detailed.get('tokens'):
                    tokens = detailed['tokens']
                    print(f"  First 20 tokens: {tokens[:20]}")
                    if len(tokens) > 40:
                        print(f"  ... ({len(tokens) - 40} tokens omitted) ...")
                    print(f"  Last 20 tokens: {tokens[-20:]}")

    return 0


def cmd_benchmark(args) -> int:
    """Run benchmark and optional regression gate."""
    import json
    import platform
    import subprocess
    import time
    from pathlib import Path

    from daedalus.engine import Engine, EngineConfig
    from daedalus.governor import GovernorConfig, LevelPolicy, ThermalGovernor
    from daedalus.sensors import ThermalLevel, ThermalMonitor

    def full_speed_policies():
        return {level: LevelPolicy(chunk_tokens=2048, duty=1.0) for level in ThermalLevel}

    monitor = ThermalMonitor(poll_interval=1.0).start()
    thermal_trace = []
    monitor.on_change(
        lambda old, new: thermal_trace.append(
            {"t": time.time(), "from": old.name, "to": new.name}
        )
    )

    cfg = GovernorConfig()
    if args.profile != "balanced":
        from daedalus.governor import PROFILES
        cfg.policies = dict(PROFILES[args.profile])
    cfg.max_duty = args.max_duty

    if args.governor == "off":
        cfg.policies = full_speed_policies()
        cfg.step_down_seconds = 0.0

    governor = ThermalGovernor(monitor, cfg)
    print(f"loading {args.model} ...")
    engine = Engine.from_pretrained(
        args.model,
        governor=governor,
        config=EngineConfig(
            kv_bits=args.kv_bits or None,
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            clear_metal_cache_between_chunks=args.clear_metal_cache_between_chunks,
        ),
    )

    filler = "The quick brown fox jumps over the lazy dog. " * (args.prompt_tokens // 10)
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
    try:
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
    finally:
        monitor.stop()

    wall = time.perf_counter() - start
    prefill_tokens = len(tokens) - 1
    prefill_wall = chunk_events[-1]["t"] if chunk_events else 0.0

    result = {
        "config": {
            "model": args.model,
            "governor": args.governor,
            "profile": args.profile,
            "max_duty": args.max_duty,
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

    print(json.dumps(result["metrics"], indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out}")

    # Regression gate
    if args.baseline and args.candidate:
        baseline = json.loads(Path(args.baseline).read_text())["metrics"]
        candidate = json.loads(Path(args.candidate).read_text())["metrics"]
        return run_regression_gate(baseline, candidate, args.max_ttft_regression, args.max_throughput_regression)

    return 0


def run_regression_gate(baseline: dict, candidate: dict, max_ttft_regression: float, max_throughput_regression: float) -> int:
    """Run regression gate comparison."""
    import sys

    failures = []

    if candidate["ttft_s"] > baseline["ttft_s"] * (1 + max_ttft_regression):
        failures.append(
            f"TTFT regressed: {baseline['ttft_s']}s -> {candidate['ttft_s']}s "
            f"(threshold: {max_ttft_regression * 100:.0f}%)"
        )

    if baseline.get("prefill_tps_wall") and candidate.get("prefill_tps_wall"):
        if candidate["prefill_tps_wall"] < baseline["prefill_tps_wall"] * (1 - max_throughput_regression):
            failures.append(
                f"Prefill throughput regressed: "
                f"{baseline['prefill_tps_wall']} -> {candidate['prefill_tps_wall']} tok/s "
                f"(threshold: {max_throughput_regression * 100:.0f}%)"
            )

    if failures:
        print("Benchmark regression detected:\n- " + "\n- ".join(failures), file=sys.stderr)
        return 1

    print("Benchmark gate passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="daedalus")
    sub = ap.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    serve.add_argument("--model", required=True, action="append", help="model to serve (can be specified multiple times for multi-model serving)")
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
        "--cache-ttl-days", type=int,
        help="evict disk cache entries older than N days (default: no TTL eviction)",
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
        "--global-rps",
        type=float,
        default=0.0,
        help="global token bucket rate limit (total RPS across all clients); 0 disables",
    )
    serve.add_argument(
        "--global-burst",
        type=int,
        default=0,
        help="burst allowance for global rate limiter (default: 2x global_rps)",
    )
    serve.add_argument(
        "--shutdown-timeout",
        type=float,
        default=30.0,
        help="graceful shutdown drain timeout in seconds (default: 30)",
    )
    serve.add_argument(
        "--cors-origins",
        type=str,
        help="comma-separated list of allowed CORS origins (e.g., https://app.example.com,https://dev.example.com)",
    )
    serve.add_argument(
        "--cors-allow-credentials",
        action="store_true",
        help="allow credentials (cookies, auth headers) in CORS requests",
    )
    serve.add_argument(
        "--audit-log-path",
        type=str,
        help="path to audit log file (JSON lines); stderr if omitted",
    )
    serve.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="tokens per SSE yield; batch tokens to reduce Python overhead (default: 1, vllm-mlx uses 16)",
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

    # Cache subcommand group
    from daedalus.cache.cli import add_cache_parser
    add_cache_parser(sub)

    # Config subcommand group
    config = sub.add_parser("config", help="configuration commands")
    config_sub = config.add_subparsers(dest="config_cmd", required=True)

    config_show = config_sub.add_parser("show", help="show resolved configuration (CLI + env + defaults merged)")
    config_show.add_argument("--model", help="model to show config for (affects cache identity)")
    config_show.add_argument("--kv-bits", type=int, default=8, help="KV cache quantization bits")
    config_show.add_argument("--profile", choices=["performance", "balanced", "cool"], default="balanced", help="thermal profile")
    config_show.add_argument("--max-duty", type=float, default=1.0, help="max duty cycle (0-1)")
    config_show.add_argument("--prefill-chunk-tokens", type=int, help="prefill chunk size")
    config_show.add_argument("--clear-metal-cache-between-chunks", action="store_true")
    config_show.add_argument("--cache-ram-mb", type=int, help="cache RAM budget in MiB")
    config_show.add_argument("--cache-disk-gb", type=int, default=10, help="cache disk budget in GiB")
    config_show.add_argument("--cache-ttl-days", type=int, help="cache TTL in days")
    config_show.add_argument("--max-active-memory-gb", type=float, help="max active memory in GB")
    config_show.add_argument("--max-pending-requests", type=int, default=8)
    config_show.add_argument("--max-prompt-tokens", type=int, default=65536)
    config_show.add_argument("--max-completion-tokens", type=int, default=4096)
    config_show.add_argument("--requests-per-minute", type=int, default=0)
    config_show.add_argument("--max-request-bytes", type=int, default=2097152)
    config_show.add_argument("--global-rps", type=float, default=0.0)
    config_show.add_argument("--global-burst", type=int, default=0)
    config_show.add_argument("--shutdown-timeout", type=float, default=30.0)
    config_show.add_argument("--cors-origins", type=str, help="comma-separated CORS origins")
    config_show.add_argument("--cors-allow-credentials", action="store_true")
    config_show.add_argument("--audit-log-path", type=str)
    config_show.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info")
    config_show.add_argument("--host", default="127.0.0.1")
    config_show.add_argument("--port", type=int, default=8080)
    config_show.add_argument("--api-key", help="API key")
    config_show.add_argument("--api-key-env", help="API key env var")
    config_show.add_argument("--api-key-file", help="API key file")
    config_show.add_argument("--model-revision", help="model revision")
    config_show.add_argument("--draft-model", help="draft model")
    config_show.add_argument("--num-draft-tokens", type=int, default=0)
    config_show.set_defaults(fn=cmd_config_show)

    config_doctor = config_sub.add_parser("doctor", help="validate configuration, warn on conflicts")
    config_doctor.add_argument("--model", required=True, help="model to validate config for")
    config_doctor.add_argument("--kv-bits", type=int, default=8)
    config_doctor.add_argument("--profile", choices=["performance", "balanced", "cool"], default="balanced")
    config_doctor.add_argument("--max-duty", type=float, default=1.0)
    config_doctor.add_argument("--prefill-chunk-tokens", type=int)
    config_doctor.add_argument("--clear-metal-cache-between-chunks", action="store_true")
    config_doctor.add_argument("--cache-ram-mb", type=int)
    config_doctor.add_argument("--cache-disk-gb", type=int, default=10)
    config_doctor.add_argument("--cache-ttl-days", type=int)
    config_doctor.add_argument("--max-active-memory-gb", type=float)
    config_doctor.add_argument("--max-pending-requests", type=int, default=8)
    config_doctor.add_argument("--max-prompt-tokens", type=int, default=65536)
    config_doctor.add_argument("--max-completion-tokens", type=int, default=4096)
    config_doctor.add_argument("--requests-per-minute", type=int, default=0)
    config_doctor.add_argument("--max-request-bytes", type=int, default=2097152)
    config_doctor.add_argument("--global-rps", type=float, default=0.0)
    config_doctor.add_argument("--global-burst", type=int, default=0)
    config_doctor.add_argument("--shutdown-timeout", type=float, default=30.0)
    config_doctor.add_argument("--cors-origins", type=str)
    config_doctor.add_argument("--cors-allow-credentials", action="store_true")
    config_doctor.add_argument("--audit-log-path", type=str)
    config_doctor.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info")
    config_doctor.add_argument("--host", default="127.0.0.1")
    config_doctor.add_argument("--port", type=int, default=8080)
    config_doctor.add_argument("--api-key", help="API key")
    config_doctor.add_argument("--api-key-env", help="API key env var")
    config_doctor.add_argument("--api-key-file", help="API key file")
    config_doctor.add_argument("--model-revision", help="model revision")
    config_doctor.add_argument("--draft-model", help="draft model")
    config_doctor.add_argument("--num-draft-tokens", type=int, default=0)
    config_doctor.set_defaults(fn=cmd_config_doctor)

    # Inspect-cache command
    inspect_cache = sub.add_parser("inspect-cache", help="rich cache inspection for a model")
    inspect_cache.add_argument("--model", required=True, help="model to inspect cache for")
    inspect_cache.add_argument("--kv-bits", type=int, default=8)
    inspect_cache.add_argument("--model-revision", help="model revision")
    inspect_cache.add_argument("--draft-model", help="draft model")
    inspect_cache.add_argument("--cache-dir", type=str, help="cache directory (default: ~/.cache/daedalus/prefix)")
    inspect_cache.add_argument("--show-tokens", action="store_true", help="show token IDs in output")
    inspect_cache.add_argument("--detailed", action="store_true", help="show detailed per-entry info")
    inspect_cache.set_defaults(fn=cmd_inspect_cache)

    # Benchmark command
    benchmark = sub.add_parser("benchmark", help="run benchmark and optional regression gate")
    benchmark.add_argument("--model", required=True, help="model to benchmark")
    benchmark.add_argument("--baseline", help="baseline JSON file for regression comparison")
    benchmark.add_argument("--candidate", help="candidate JSON file (output of this run) for regression comparison")
    benchmark.add_argument("--prompt-tokens", type=int, default=8000, help="prompt tokens for benchmark")
    benchmark.add_argument("--max-tokens", type=int, default=64, help="max generation tokens")
    benchmark.add_argument("--governor", choices=["on", "off"], default="on", help="thermal governor on/off")
    benchmark.add_argument("--kv-bits", type=int, default=8)
    benchmark.add_argument("--profile", choices=["performance", "balanced", "cool"], default="balanced")
    benchmark.add_argument("--max-duty", type=float, default=1.0)
    benchmark.add_argument("--prefill-chunk-tokens", type=int)
    benchmark.add_argument("--clear-metal-cache-between-chunks", action="store_true")
    benchmark.add_argument("--out", help="output JSON file for benchmark results")
    benchmark.add_argument("--max-ttft-regression", type=float, default=0.10, help="max TTFT regression threshold (default 10%%)")
    benchmark.add_argument("--max-throughput-regression", type=float, default=0.05, help="max throughput regression threshold (default 5%%)")
    benchmark.set_defaults(fn=cmd_benchmark)

    args = ap.parse_args()
    if args.cmd == "serve":
        if args.max_pending_requests < 1:
            ap.error("--max-pending-requests must be at least 1")
        if args.cache_ram_mb is not None and args.cache_ram_mb < 1:
            ap.error("--cache-ram-mb must be positive")
        if args.cache_disk_gb < 1:
            ap.error("--cache-disk-gb must be positive")
        if args.cache_ttl_days is not None and args.cache_ttl_days < 1:
            ap.error("--cache-ttl-days must be positive")
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
        if args.global_rps < 0:
            ap.error("--global-rps cannot be negative")
        if args.global_burst < 0:
            ap.error("--global-burst cannot be negative")
        if args.shutdown_timeout <= 0:
            ap.error("--shutdown-timeout must be positive")
        if args.cors_origins:
            args.cors_origins = [origin.strip() for origin in args.cors_origins.split(",") if origin.strip()]
    if args.cmd == "tune" and (args.prompt_tokens < 512 or args.repeats < 1):
        ap.error("--prompt-tokens must be at least 512 and --repeats at least 1")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())