# Daedalus Throughput Optimization Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Maximize sustained tokens/second (decode throughput) and overall LLM speed on MacBook Air M4 (16GB) for Daedalus inference server.

**Architecture:** Daedalus is a single-user sequential MLX inference server with thermal-aware prefill, persistent prefix cache, and checkpoint/resume. Current bottleneck: decode phase is memory-bandwidth bound; prefill is compute-bound but thermally throttled.

**Tech Stack:** MLX 0.31+, mlx-lm 0.31+, Metal/Metal Performance Shaders, macOS Darwin thermal pressure API.

---

## Current Baseline (from memory)

| Model | Sustained Decode TPS | Notes |
|-------|---------------------|-------|
| ZAYA1-8B-JANGTQ4 | 30-40 TPS | Best local speed on M4 Air |
| Qwen3.5-4B (vllm-mlx) | ~30 TPS | With `--cache-memory-mb 3072`, `--stream-interval 16` |
| Ornith-1.0-9B | ~15 TPS | Chain-of-thought overhead |
| Daedalus (current) | Unknown | Need measurement |

---

## Optimization Areas

### 1. Decode Phase (Memory-Bandwidth Bound) — Highest Impact
- 8-bit KV quantization (already done) → try 4-bit KV with optimal group size
- Metal kernel optimization: ensure MLX's latest `gemm` kernels are used
- Stream interval tuning (vllm-mlx uses `--stream-interval 16`)
- Reduce Python overhead in decode loop

### 2. Prefill Phase (Compute-Bound, Thermally Constrained)
- Larger tuned chunk size when thermal = NOMINAL
- Profile optimal `--prefill-chunk-tokens` per model
- Disable `clear_metal_cache_between_chunks` (default) to retain allocations

### 3. KV Cache Optimization
- 4-bit KV with group_size=32 or 64 (test vs 8-bit)
- `quantized_kv_start` tuning (default 4096)
- Prefix cache hit rate → reduces prefill entirely

### 4. Server/Streaming Overhead
- Stream interval tuning (vllm-mlx uses 16)
- Reduce per-token Python overhead in `_stream_response`
- Batch SSE writes

### 5. Model Selection
- Use ZAYA1-8B or Qwen3.5-4B for max throughput
- Avoid chain-of-thought models (Ornith) for throughput-critical workloads

---

## Step-by-Step Plan

### Task 1: Add Benchmark Infrastructure

**Objective:** Create reproducible benchmark to measure baseline and track improvements.

**Files:**
- Create: `bench/throughput_bench.py`
- Modify: `bench/__init__.py` (if needed)

**Step 1: Write failing test**

```python
# bench/throughput_bench.py
def test_decode_throughput_measurement():
    """Measure sustained decode TPS for a given model."""
    # This is a benchmark, not a unit test - we just need it runnable
    pass
```

**Step 2: Run to verify it runs**

```bash
cd /tmp/Daedalus-audit && python bench/throughput_bench.py --model mlx-community/Qwen3.5-4B-MLX-4bit --prompt-tokens 100 --max-tokens 500
```

Expected: Outputs TPS, TTFT, memory usage

**Step 3: Implement benchmark script**

```python
#!/usr/bin/env python3
"""Throughput benchmark for Daedalus decode speed.

Measures:
- TTFT (time to first token)
- Sustained decode TPS (tokens/second after first token)
- Peak memory usage
- Thermal state throughout
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


def run_benchmark(model: str, prompt_tokens: int, max_tokens: int, 
                  kv_bits: int = 8, stream_interval: int = 1) -> dict:
    monitor = ThermalMonitor(poll_interval=0.5).start()
    thermal_trace = []
    monitor.on_change(lambda old, new: thermal_trace.append(
        {"t": time.time(), "from": old.name, "to": new.name}
    ))

    # Governor: no thermal pacing for pure throughput measurement
    policies = {level: LevelPolicy(chunk_tokens=2048, duty=1.0) for level in ThermalLevel}
    governor = ThermalGovernor(monitor, GovernorConfig(policies=policies, step_down_seconds=0.0))

    engine = Engine.from_pretrained(
        model,
        governor=governor,
        config=EngineConfig(kv_bits=kv_bits),
    )

    # Build prompt
    filler = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. " + filler},
        {"role": "user", "content": "Continue the story."},
    ]
    tokens = engine.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    chunk_events = []
    start = time.perf_counter()

    def progress(done, total):
        chunk_events.append({
            "t": time.perf_counter() - start,
            "done": done,
            "total": total,
            "thermal": monitor.level.name,
        })

    ttft = None
    n_gen = 0
    gen_tps = 0.0
    peak_mem = 0.0

    for resp in engine.generate(
        tokens,
        max_tokens=max_tokens,
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

    return {
        "config": {
            "model": model,
            "kv_bits": kv_bits,
            "prompt_tokens": len(tokens),
            "max_tokens": max_tokens,
            "stream_interval": stream_interval,
            "machine": platform.machine(),
            "hw": subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            ).stdout.strip(),
        },
        "metrics": {
            "ttft_s": round(ttft or -1, 3),
            "wall_s": round(wall, 3),
            "prefill_wall_s": round(prefill_wall, 3),
            "prefill_tps_wall": round(prefill_tokens / prefill_wall, 1) if prefill_wall else None,
            "generation_tokens": n_gen,
            "generation_tps": round(gen_tps, 2),
            "peak_memory_gb": round(peak_mem, 2),
            "thermal_start": chunk_events[0]["thermal"] if chunk_events else None,
            "thermal_end": monitor.level.name,
        },
        "thermal_trace": thermal_trace,
        "chunks": chunk_events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--prompt-tokens", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--kv-bits", type=int, default=8)
    ap.add_argument("--stream-interval", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    monitor = ThermalMonitor(poll_interval=0.5).start()
    print(f"Starting benchmark: {args.model} | prompt={args.prompt_tokens} | max_tokens={args.max_tokens} | kv_bits={args.kv_bits}")
    
    result = run_benchmark(
        args.model, args.prompt_tokens, args.max_tokens,
        kv_bits=args.kv_bits, stream_interval=args.stream_interval
    )
    
    monitor.stop()
    
    print(json.dumps(result["metrics"], indent=2))
    
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

**Step 4: Run benchmark**

```bash
cd /tmp/Daedalus-audit && python bench/throughput_bench.py --model mlx-community/Qwen3.5-4B-MLX-4bit --prompt-tokens 100 --max-tokens 500 --out bench/results/baseline_qwen4b.json
cd /tmp/Daedalus-audit && python bench/throughput_bench.py --model mlx-community/ZAYA1-8B-MLX-4bit --prompt-tokens 100 --max-tokens 500 --out bench/results/baseline_zaya8b.json
```

Expected: JSON output with TPS, TTFT, memory

**Step 5: Commit**

```bash
git add bench/throughput_bench.py
git commit -m "feat: add throughput benchmark script"
```

---

### Task 2: Tune Stream Interval for Decode Throughput

**Objective:** Find optimal stream interval (like vllm-mlx's `--stream-interval 16`) to reduce Python overhead per token.

**Files:**
- Modify: `daedalus/server.py` (add `--stream-interval` param to `create_app` and `_stream_response`)

**Step 1: Add stream_interval parameter**

```python
# In create_app signature
def create_app(
    engines: Dict[str, Engine],
    stores: Dict[str, PrefixCacheStore],
    model_ids: List[str],
    *,
    ...
    stream_interval: int = 1,  # NEW: tokens per SSE yield
    ...
) -> FastAPI:
```

**Step 2: Pass to _Generation and _stream_response**

```python
# In chat_completions endpoint
gen = _Generation(
    ...
    stream_interval=stream_interval,
    ...
)

# In _Generation.__init__
def __init__(self, ..., stream_interval: int = 1):
    self.stream_interval = stream_interval
    self.tokens_since_yield = 0

# In _run_engine, yield every N tokens instead of every token
def _run_engine(self):
    ...
    for resp in state.engine.generate(...):
        self.tokens_since_yield += resp.generation_tokens
        if self.tokens_since_yield >= self.stream_interval:
            yield {"type": "delta", "text": resp.text}
            self.tokens_since_yield = 0
```

**Step 3: Benchmark stream intervals**

```bash
for interval in 1 4 8 16 32; do
  python bench/throughput_bench.py --model mlx-community/Qwen3.5-4B-MLX-4bit \
    --prompt-tokens 100 --max-tokens 500 --stream-interval $interval \
    --out bench/results/stream_interval_${interval}.json
done
```

Expected: Find sweet spot (vllm-mlx uses 16)

**Step 4: Commit**

```bash
git add daedalus/server.py
git commit -m "feat: add stream_interval parameter for throughput tuning"
```

---

### Task 3: Optimize KV Quantization (4-bit vs 8-bit)

**Objective:** Test 4-bit KV cache with optimal group size for decode throughput.

**Files:**
- Modify: `daedalus/engine.py` (EngineConfig defaults, test 4-bit)
- Modify: `bench/throughput_bench.py` (add kv_bits parameter sweep)

**Step 1: Add KV bit sweep to benchmark**

```bash
for bits in 4 8; do
  for group in 32 64 128; do
    python bench/throughput_bench.py --model mlx-community/Qwen3.5-4B-MLX-4bit \
      --prompt-tokens 100 --max-tokens 500 --kv-bits $bits \
      --out bench/results/kv_${bits}bit_group${group}.json
  done
done
```

**Step 2: Update EngineConfig defaults if 4-bit wins**

```python
# In EngineConfig
kv_bits: Optional[int] = 4  # if 4-bit is faster
kv_group_size: int = 64     # optimal group size from sweep
```

**Step 3: Commit**

```bash
git add daedalus/engine.py bench/throughput_bench.py
git commit -m "feat: optimize KV quantization for decode throughput"
```

---

### Task 4: Tune Prefill Chunk Size per Model

**Objective:** Find optimal `--prefill-chunk-tokens` for each model when thermal = NOMINAL.

**Files:**
- Modify: `daedalus/cli.py` (tune command already exists, extend)
- Create: `bench/prefill_tune.py` (automated per-model tuning)

**Step 1: Create prefill tune script**

```python
#!/usr/bin/env python3
"""Automated prefill chunk tuning per model."""

import json
import subprocess
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", default="512,1024,2048,4096,8192")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    
    chunks = [int(x) for x in args.chunks.split(",")]
    result = tune_prefill_chunk(args.model, chunks)
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2))
```

**Step 2: Run for target models**

```bash
python bench/prefill_tune.py --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --out bench/results/prefill_tune_qwen4b.json
python bench/prefill_tune.py --model mlx-community/ZAYA1-8B-MLX-4bit \
  --out bench/results/prefill_tune_zaya8b.json
```

**Step 3: Apply best chunk sizes to CLI defaults**

```bash
# Update daedalus serve --prefill-chunk-tokens defaults per model
```

**Step 4: Commit**

```bash
git add bench/prefill_tune.py daedalus/cli.py
git commit -m "feat: automated prefill chunk tuning per model"
```

---

### Task 5: Reduce Python Overhead in Decode Loop

**Objective:** Minimize per-token Python overhead in `_stream_response` and `_Generation._run_engine`.

**Files:**
- Modify: `daedalus/server.py` (`_stream_response`, `_Generation._run_engine`)

**Optimizations:**

1. **Batch SSE writes** - accumulate tokens, write every N tokens
2. **Remove per-token logging** - only log at chunk boundaries
3. **Pre-allocate buffers** - avoid list append in hot path
4. **Use local variable references** - avoid attribute lookups in loop

```python
# In _stream_response - batch writes
async def _stream_response(...):
    buffer = []
    tokens_in_buffer = 0
    BATCH_SIZE = 16  # configurable
    
    while True:
        event = await queue.get()
        if event["type"] == "delta":
            buffer.append(event["text"])
            tokens_in_buffer += 1
            if tokens_in_buffer >= BATCH_SIZE:
                yield _sse(_chunk(..., {"content": "".join(buffer)}))
                buffer.clear()
                tokens_in_buffer = 0
        elif event["type"] == "done":
            if buffer:
                yield _sse(_chunk(..., {"content": "".join(buffer)}))
            yield _sse(_chunk(...))
            break
```

**Step 2: Benchmark before/after**

```bash
python bench/throughput_bench.py --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --prompt-tokens 100 --max-tokens 500 --out bench/results/overhead_optimized.json
```

**Step 3: Commit**

```bash
git add daedalus/server.py
git commit -m "perf: reduce Python overhead in decode streaming loop"
```

---

### Task 6: Optimize Metal Cache Management

**Objective:** Tune `metal_cache_high_water_bytes` and `clear_metal_cache_between_chunks` for sustained throughput.

**Files:**
- Modify: `daedalus/engine.py` (EngineConfig defaults)
- Create: `bench/metal_tune.py`

**Step 1: Test Metal cache configurations**

```bash
for highwater in 1073741824 1610612736 2147483648; do  # 1GB, 1.5GB, 2GB
  for clear in true false; do
    python bench/throughput_bench.py --model mlx-community/Qwen3.5-4B-MLX-4bit \
      --prompt-tokens 100 --max-tokens 500 \
      --out bench/results/metal_hw${highwater}_clear${clear}.json
  done
done
```

**Step 2: Update EngineConfig**

```python
# In EngineConfig
metal_cache_high_water_bytes: int = 1_610_612_736  # 1.5GB optimal
clear_metal_cache_between_chunks: bool = False
```

**Step 3: Commit**

```bash
git add daedalus/engine.py
git commit -m "perf: optimize Metal cache high-water mark for sustained throughput"
```

---

### Task 7: Prefix Cache Hit Rate Optimization

**Objective:** Maximize cache hit rate to eliminate prefill entirely for repeated prompts.

**Files:**
- Modify: `daedalus/cache/store.py` (tune LRU eviction)
- Modify: `daedalus/server.py` (token cache size)

**Optimizations:**
1. Increase `token_cache_entries` default (256 → 1024)
2. Tune `PromptTokenCache.max_tokens` (200k → 500k)
3. Tune `SharedHeadIndex.max_entries` (128 → 512)
4. Pre-warm cache at startup for known workloads

**Step 1: Update defaults**

```python
# In create_app
token_cache_entries: int = 1024,  # was 256

# In ServerState
token_cache=PromptTokenCache(token_cache_entries=1024),
head_cache=SharedHeadIndex(max_entries=512),
```

**Step 2: Add cache warm command for production workloads**

```bash
# Already implemented: daedalus cache warm-from-history
# Test with real OpenCode/Hermes history
```

**Step 3: Commit**

```bash
git add daedalus/server.py
git commit -m "perf: increase prefix cache sizes for higher hit rates"
```

---

### Task 8: Model-Specific Throughput Profiles

**Objective:** Create optimal config profiles per model (ZAYA1-8B, Qwen3.5-4B, etc.)

**Files:**
- Create: `daedalus/profiles/` (YAML config files)
- Modify: `daedalus/cli.py` (load profile)

**Step 1: Create profile files**

```yaml
# daedalus/profiles/zaya1-8b.yaml
model: mlx-community/ZAYA1-8B-MLX-4bit
kv_bits: 4
kv_group_size: 64
prefill_chunk_tokens: 8192
stream_interval: 16
metal_cache_high_water_bytes: 2147483648
clear_metal_cache_between_chunks: false
token_cache_entries: 1024
```

```yaml
# daedalus/profiles/qwen3.5-4b.yaml
model: mlx-community/Qwen3.5-4B-MLX-4bit
kv_bits: 8
kv_group_size: 64
prefill_chunk_tokens: 4096
stream_interval: 16
metal_cache_high_water_bytes: 1610612736
clear_metal_cache_between_chunks: false
token_cache_entries: 1024
```

**Step 2: Add --profile flag to CLI**

```bash
daedalus serve --profile zaya1-8b
```

**Step 3: Commit**

```bash
git add daedalus/profiles/ daedalus/cli.py
git commit -m "feat: model-specific throughput profiles"
```

---

### Task 9: Run Full Benchmark Suite & Document Results

**Objective:** Run complete benchmark suite with all optimizations, compare to baseline.

**Files:**
- Create: `bench/run_full_suite.py`

**Step 1: Run full comparison**

```bash
# Baseline (original settings)
python bench/throughput_bench.py --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --prompt-tokens 100 --max-tokens 500 --out bench/results/final_qwen4b.json

# Optimized (all flags)
python bench/throughput_bench.py --model mlx-community/Qwen3.5-4B-MLX-4bit \
  --prompt-tokens 100 --max-tokens 500 \
  --kv-bits 4 --stream-interval 16 \
  --out bench/results/optimized_qwen4b.json
```

**Step 2: Generate comparison report**

```python
# Script to compare baseline vs optimized
import json
baseline = json.load(open("bench/results/baseline_qwen4b.json"))
optimized = json.load(open("bench/results/optimized_qwen4b.json"))

print(f"TPS: {baseline['metrics']['generation_tps']} -> {optimized['metrics']['generation_tps']} ({optimized['metrics']['generation_tps']/baseline['metrics']['generation_tps']:.1f}x)")
print(f"TTFT: {baseline['metrics']['ttft_s']} -> {optimized['metrics']['ttft_s']}")
print(f"Memory: {baseline['metrics']['peak_memory_gb']} -> {optimized['metrics']['peak_memory_gb']}")
```

**Step 3: Document in README**

```markdown
## Throughput Benchmarks (M4 Air 16GB)

| Model | Config | Decode TPS | TTFT | Peak Mem |
|-------|--------|-----------|------|----------|
| Qwen3.5-4B | Baseline | XX | XX | XX |
| Qwen3.5-4B | Optimized | XX | XX | XX |
| ZAYA1-8B | Baseline | XX | XX | XX |
| ZAYA1-8B | Optimized | XX | XX | XX |
```

**Step 4: Commit**

```bash
git add bench/results/*.json README.md
git commit -m "docs: throughput benchmark results with optimizations"
```

---

## Validation Checklist

- [ ] Baseline benchmarks run and produce JSON
- [ ] Stream interval sweep finds optimum (target: 16)
- [ ] KV quantization sweep finds 4-bit or 8-bit winner
- [ ] Prefill chunk tuning per model documented
- [ ] Python overhead reduction measurable (>5% TPS gain)
- [ ] Metal cache config tuned
- [ ] Cache hit rate >90% for repeated prompts
- [ ] Model profiles created and loadable
- [ ] Full benchmark suite passes
- [ ] Results documented in README

---

## Risks & Tradeoffs

| Risk | Mitigation |
|------|------------|
| 4-bit KV quality loss | Test perplexity/quality on held-out prompts |
| Larger stream interval = higher latency | Keep configurable; default to balanced |
| 4-bit KV may not work on all models | Test per-model; fallback to 8-bit |
| Thermal governor may limit sustained TPS | Test with governor OFF for peak measurement |
| MTP/speculative decoding not tested | Memory says MTP slower on 16GB; skip for now |

---

## Open Questions

1. What's the actual baseline TPS for Daedalus on Qwen3.5-4B and ZAYA1-8B?
2. Does 4-bit KV work reliably on ZAYA1-8B (different architecture)?
3. Does `stream_interval=16` break any client compatibility?
4. Should we add `--max-num-seqs` style batching for throughput (currently single-user)?
5. Is `quantized_kv_start=4096` optimal or should we tune per model?