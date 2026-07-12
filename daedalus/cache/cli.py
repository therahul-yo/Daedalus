"""Cache CLI commands: list, inspect, prune, warm-from-history."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from daedalus.cache.store import PrefixCacheStore, _default_cache_dir, _sanitize_model_key


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _format_age(seconds: float) -> str:
    """Format age in seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


def cmd_cache_list(args: argparse.Namespace) -> int:
    """List all cache entries."""
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else _default_cache_dir()
    
    if args.model:
        # List entries for a specific model
        model_key = _sanitize_model_key(args.model)
        store = PrefixCacheStore(
            model_key=args.model,
            cache_dir=cache_dir,
            exclusive=False,
        )
        entries = store.list_entries(model_filter=args.model)
        print(f"Cache entries for model: {args.model}")
        print(f"Cache directory: {store.dir}")
    else:
        # List entries for all models
        entries = []
        for model_dir in cache_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_key = model_dir.name.replace("--", "/")  # approximate reverse
            store = PrefixCacheStore(
                model_key=model_key,
                cache_dir=cache_dir,
                exclusive=False,
            )
            entries.extend(store.list_entries())
        
        # Sort by last used (most recent first)
        entries.sort(key=lambda e: e["last_used"], reverse=True)
        print(f"Cache entries across all models in: {cache_dir}")
    
    if not entries:
        print("No cache entries found.")
        return 0
    
    # Print table header
    print(f"\n{'KEY':<26} {'MODEL':<30} {'TOKENS':>8} {'SIZE':>10} {'RAM':>4} {'DISK':>5} {'AGE':>8} {'HITS':>5}")
    print("-" * 110)
    
    total_size = 0
    total_tokens = 0
    for entry in entries:
        key = entry["key"][:24]
        model = entry["model_key"][:28]
        tokens = entry["token_count"]
        size = _format_size(entry["size_bytes"])
        in_ram = "yes" if entry["in_ram"] else "no"
        on_disk = "yes" if entry["on_disk"] else "no"
        age = _format_age(entry["age_seconds"])
        hits = entry["hits"]
        
        print(f"{key:<26} {model:<30} {tokens:>8} {size:>10} {in_ram:>4} {on_disk:>5} {age:>8} {hits:>5}")
        total_size += entry["size_bytes"]
        total_tokens += tokens
    
    print("-" * 110)
    print(f"Total: {len(entries)} entries, {total_tokens:,} tokens, {_format_size(total_size)}")
    return 0


def cmd_cache_inspect(args: argparse.Namespace) -> int:
    """Inspect a specific cache entry."""
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else _default_cache_dir()
    
    # Find the entry across all models
    entry = None
    model_key = None
    
    if args.model:
        model_key = _sanitize_model_key(args.model)
        store = PrefixCacheStore(
            model_key=args.model,
            cache_dir=cache_dir,
            exclusive=False,
        )
        entry = store.inspect_entry(args.key)
    else:
        # Search across all models
        for model_dir in cache_dir.iterdir():
            if not model_dir.is_dir():
                continue
            mk = model_dir.name.replace("--", "/")
            store = PrefixCacheStore(
                model_key=mk,
                cache_dir=cache_dir,
                exclusive=False,
            )
            entry = store.inspect_entry(args.key)
            if entry:
                model_key = mk
                break
    
    if not entry:
        print(f"Cache entry not found: {args.key}")
        return 1
    
    print(f"Cache Entry: {entry['key']}")
    print(f"  Model:       {entry['model_key']}")
    print(f"  Tokens:      {entry['token_count']:,}")
    print(f"  Size:        {_format_size(entry['size_bytes'])}")
    print(f"  In RAM:      {'Yes' if entry['in_ram'] else 'No'}")
    print(f"  On Disk:     {'Yes' if entry['on_disk'] else 'No'}")
    if entry['path']:
        print(f"  Disk Path:   {entry['path']}")
    print(f"  Last Used:   {time.ctime(entry['last_used'])} ({_format_age(time.time() - entry['last_used'])} ago)")
    if entry['created']:
        print(f"  Created:     {time.ctime(entry['created'])}")
    if entry.get('version'):
        print(f"  Format Ver:  v{entry['version']}")
    print(f"  Hits:        {entry['hits']}")
    
    # Show first/last tokens
    if args.show_tokens:
        tokens = entry['tokens']
        print(f"\n  First 20 tokens: {tokens[:20]}")
        if len(tokens) > 40:
            print(f"  ... ({len(tokens) - 40} tokens omitted) ...")
        print(f"  Last 20 tokens:  {tokens[-20:]}")
    
    return 0


def cmd_cache_prune(args: argparse.Namespace) -> int:
    """Prune cache entries older than TTL."""
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else _default_cache_dir()
    
    total_removed = 0
    
    if args.model:
        model_key = _sanitize_model_key(args.model)
        store = PrefixCacheStore(
            model_key=args.model,
            cache_dir=cache_dir,
            cache_ttl_days=args.ttl_days,
            exclusive=False,
        )
        removed = store.prune_by_ttl(ttl_days=args.ttl_days, model_filter=args.model)
        print(f"Pruned {removed} entries for model: {args.model}")
        total_removed = removed
    else:
        # Prune across all models
        for model_dir in cache_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_key = model_dir.name.replace("--", "/")
            store = PrefixCacheStore(
                model_key=model_key,
                cache_dir=cache_dir,
                cache_ttl_days=args.ttl_days,
                exclusive=False,
            )
            removed = store.prune_by_ttl(ttl_days=args.ttl_days, model_filter=model_key)
            if removed > 0:
                print(f"Pruned {removed} entries for model: {model_key}")
            total_removed += removed
    
    print(f"Total entries removed: {total_removed}")
    return 0


def cmd_cache_warm_from_history(args: argparse.Namespace) -> int:
    """Warm cache from chat history exports."""
    import logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    from daedalus.engine import Engine, EngineConfig
    from daedalus.runtime import cache_identity
    from daedalus.governor import ThermalGovernor, GovernorConfig, LevelPolicy
    from daedalus.sensors import ThermalMonitor, ThermalLevel
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else _default_cache_dir()
    
    # Set up engine
    monitor = ThermalMonitor().start()
    policies = {level: LevelPolicy(chunk_tokens=2048, duty=1.0) 
                for level in ThermalLevel}
    governor = ThermalGovernor(monitor, GovernorConfig(policies=policies))
    
    engine = Engine.from_pretrained(
        args.model,
        governor=governor,
        config=EngineConfig(kv_bits=args.kv_bits or 8),
    )
    
    cache_key = cache_identity(
        args.model,
        kv_bits=args.kv_bits or 8,
        tokenizer_id=getattr(engine.tokenizer, "name_or_path", args.model),
        model_revision=args.model_revision,
    )
    
    store = PrefixCacheStore(
        cache_key,
        cache_dir=cache_dir,
        exclusive=True,
    )
    
    print(f"Warming cache for model: {args.model}")
    print(f"History source: {args.source}")
    print(f"Cache directory: {store.dir}")
    
    try:
        warmed = store.warm_from_history(
            source=args.source,
            model=args.model,
            engine=engine,
            limit=args.limit,
        )
        print(f"\nWarmed {warmed} new cache entries.")
    finally:
        monitor.stop()
    
    return 0


def add_cache_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add cache subcommand group to the main parser."""
    cache = subparsers.add_parser("cache", help="Cache management commands")
    cache_sub = cache.add_subparsers(dest="cache_cmd", required=True)
    
    # Common arguments
    for sub in [cache_sub]:
        pass  # We'll add common args to each subcommand
    
    # cache list
    list_parser = cache_sub.add_parser("list", help="List cache entries")
    list_parser.add_argument("--model", help="Filter by model key")
    list_parser.add_argument("--cache-dir", help="Cache directory (default: ~/.cache/daedalus/prefix)")
    list_parser.set_defaults(fn=cmd_cache_list)
    
    # cache inspect
    inspect_parser = cache_sub.add_parser("inspect", help="Inspect a cache entry")
    inspect_parser.add_argument("key", help="Cache entry key (24-char prefix)")
    inspect_parser.add_argument("--model", help="Model key (faster lookup)")
    inspect_parser.add_argument("--cache-dir", help="Cache directory")
    inspect_parser.add_argument("--show-tokens", action="store_true", help="Show token IDs")
    inspect_parser.set_defaults(fn=cmd_cache_inspect)
    
    # cache prune
    prune_parser = cache_sub.add_parser("prune", help="Prune stale cache entries")
    prune_parser.add_argument("--ttl-days", type=int, default=30, help="Remove entries older than N days (default: 30)")
    prune_parser.add_argument("--model", help="Filter by model key")
    prune_parser.add_argument("--cache-dir", help="Cache directory")
    prune_parser.set_defaults(fn=cmd_cache_prune)
    
    # cache warm-from-history
    warm_parser = cache_sub.add_parser("warm-from-history", help="Warm cache from chat history exports")
    warm_parser.add_argument("--source", required=True, choices=["openwebui", "opencode", "hermes"],
                           help="History source: openwebui, opencode, or hermes")
    warm_parser.add_argument("--model", required=True, help="Model to warm cache for")
    warm_parser.add_argument("--kv-bits", type=int, default=8, help="KV cache quantization bits")
    warm_parser.add_argument("--model-revision", help="Model revision for cache namespace")
    warm_parser.add_argument("--limit", type=int, help="Limit number of conversations to process")
    warm_parser.add_argument("--cache-dir", help="Cache directory")
    warm_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    warm_parser.set_defaults(fn=cmd_cache_warm_from_history)