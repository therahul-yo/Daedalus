#!/usr/bin/env python3
"""Automated prefill chunk tuning per model."""

import argparse
import json
import time
from pathlib import Path

from daedalus.engine import Engine, EngineConfig
from daedalus.governor import GovernorConfig, LevelPolicy, ThermalGovernor
from daedalus.sensors import ThermalLevel, ThermalMonitor


def tune_prefill_chunk(model: str, chunks: list[int]) -> dict:
    """Test each chunk size from cool state."""
    results = {}
    for chunk in chunks:
        monitor = ThermalMonitor(poll_interval=1.0).start()
        policies = {level: LevelPolicy(chunk_tokens=2048, duty=1.0) for level in ThermalLevel}
        governor = ThermalGovernor(monitor, GovernorConfig(policies=policies))
        
        engine = Engine.from_pretrained(model, governor=governor, 
                                        config=EngineConfig(prefill_chunk_tokens=chunk))
        
        # Cool down wait
        while monitor.level != ThermalLevel.NOMINAL:
            time.sleep(2)
        
        tokens = engine.tokenizer.apply_chat_template([
            {"role": "system", "content": "x" * 5000},
            {"role": "user", "content": "Go"},
        ], add_generation_prompt=True)[:8192]
        
        cache = engine.make_cache()
        start = time.perf_counter()
        report = engine.paced_prefill(tokens, cache)
        elapsed = time.perf_counter() - start
        
        results[chunk] = {
            "tps": report.computed_tokens / elapsed,
            "elapsed": elapsed,
            "chunks": report.chunks,
            "thermal_after": monitor.level.name,
        }
        monitor.stop()
    
    best = max(results, key=lambda k: results[k]["tps"])
    return {"model": model, "results": results, "best_chunk": best}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", default="512,1024,2048,4096,8192")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    
    chunks = [int(x) for x in args.chunks.split(",")]
    result = tune_prefill_chunk(args.model, chunks)
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2))
