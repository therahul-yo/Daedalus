# daedalus

**A MacBook-Air-first MLX inference engine.** Local coding agents (OpenCode, Hermes Agent,
Pi Code) send 10–40k-token first prompts; on a passively-cooled Air, prefilling that prompt
is a multi-minute full-power GPU burn that thermally throttles the machine and times out the
client. daedalus is a local OpenAI-compatible server built on [mlx-lm](https://github.com/ml-explore/mlx-lm)
that treats the Air's thermal envelope as a first-class scheduling constraint.

## Measured (M4 Air 16GB, Qwen3.5-9B MLX 4-bit)

**Prefix cache** — 8,027-token prompt (OpenCode-scale first hit), real HTTP streaming:

| request | TTFT | cached tokens |
|---|---|---|
| cold (first ever) | 52.2 s | 0 |
| warm (agent's next turn) | **0.24 s** | 8,026 |

Coding agents are stateless — they resend the whole conversation every turn. After the
first prefill, the Air never re-burns the GPU for that prefix, **even across restarts**.

**Thermal governor** — 4 consecutive distinct 20k-token prefills (80,196 tokens total,
cache disabled; the unavoidable cold-prefill worst case), same machine, cooled to
Nominal before each arm:

| | governor OFF | governor ON |
|---|---|---|
| wall clock | 647 s | 1012 s |
| GPU burn time | 647 s | **515 s (−20%)** |
| burn-rate per round (tok/s) | 133 → 119 → 121 → 124 | **152 → 157 → 157 → 157** |
| time at HEAVY (throttled) | **50%** | 21% |
| thermal state after rounds | HEAVY, HEAVY, MODERATE, HEAVY | NOMINAL, MODERATE, MODERATE, NOMINAL |

Unpaced, the Air hits HEAVY pressure inside the first prefill, throttles ~10% immediately,
and keeps degrading. Paced, the silicon runs unthrottled whenever it runs — the same work
takes 20% fewer GPU-seconds (less total heat) at a stable rate that does not decay, in
exchange for longer wall-clock spread across idle gaps that keep the machine usable.
With the prefix cache, this cost is paid once per unique prefix, ever.

## How it works

1. **The best prefill is no prefill** — persistent prefix cache (RAM LRU + disk
   safetensors, atomic writes, corruption-tolerant). Correct for hybrid-attention
   models (Qwen3.5's Gated-DeltaNet, Gemma 4's sliding windows) whose caches cannot
   be trimmed: exact-prefix matching with end-of-prefill snapshots — the reuse that
   stock mlx-lm's server misses for these models.
2. **Never re-do work** — mid-prefill checkpoints every 4k tokens; a client timeout,
   crash, or restart resumes where it stopped instead of restarting a 30k prefill.
3. **Duty-cycle the burn** — a governor reads macOS's 5-level thermal-pressure signal
   (`com.apple.system.thermalpressurelevel`, no sudo — finer than NSProcessInfo, which
   hides the level where throttling actually starts) and paces prefill: full speed at
   Nominal, smaller chunks + idle gaps at Moderate, hold at Heavy. Escalates instantly,
   de-escalates with hysteresis (fanless chassis cool slowly). `--max-duty 0.5` = quiet mode.
4. **Never let the client die** — SSE keepalives with progress/thermal/ETA from the
   first second of prefill; well-formed streamed tool-call deltas (explicit `index`,
   never an empty `tool_calls` array — a known client-hang trigger).
5. **Sized for the machine** — 8-bit quantized KV by default, Metal wired-limit
   management, single-user sequential engine (no batching complexity to fight).

Tool calling rides mlx-lm's auto-detected per-model parsers (Qwen, Gemma 4, …) behind a
streaming marker filter, so any model mlx-lm can parse tools for works here too.

## Install

```bash
# recommended: uv (or pipx) puts `daedalus` on your PATH
uv tool install --editable . --with "transformers>=5.0,<5.11"
# or: pipx install .
```

## Usage

```bash
daedalus doctor                      # verify thermal sensor + mlx setup
daedalus serve --model mlx-community/Qwen3.5-9B-MLX-4bit --port 8484
# point any OpenAI-compatible agent at http://127.0.0.1:8484/v1
daedalus warm --model ... --prompts prompts.json   # pre-prefill while cool
```

`serve` prints a startup banner (model, memory, cache entries, thermal state)
and logs every request: cache hit/miss, prefill progress with tok/s and
thermal level, decode rate, and finish reason. Reasoning models' `<think>`
output is separated into `reasoning_content` so it never leaks into the reply,
and thermal transitions are logged as they happen.

Target models: Qwen3.5-9B MLX 4-bit, Gemma 4 E4B / 12B MLX 4-bit — anything mlx-lm loads.

## Operating Daedalus safely

Daedalus binds to `127.0.0.1` by default. To expose it on a LAN, an API key is
required explicitly:

```bash
daedalus serve --model ... --host 0.0.0.0 --api-key "replace-with-a-long-secret"
```

Use `Authorization: Bearer <key>` with `/v1` endpoints. The server admits at
most eight active-or-queued requests by default (change with
`--max-pending-requests`) and returns a standard 429 response when full rather
than accumulating unbounded worker threads. Cache budgets are configurable with
`--cache-ram-mb` and `--cache-disk-gb`.

Operational endpoints:

- `GET /health` — process/model/thermal liveness
- `GET /readyz` — admission readiness
- `GET /metrics` — Prometheus text metrics for cache, queue, request outcomes,
  and thermal level
- `GET /v1/cache/stats` and `DELETE /v1/cache` — authenticated cache inspection
  and maintenance when the server is idle

## Tune for speed

The fastest safe prefill chunk varies by model, RAM, and Mac generation. Run
the built-in measurement from a cool machine, then pass its recommendation to
`serve`:

```bash
daedalus tune --model mlx-community/Qwen3.5-9B-MLX-4bit --out tune.json
# use the returned recommended_prefill_chunk_tokens value
daedalus serve --model mlx-community/Qwen3.5-9B-MLX-4bit --prefill-chunk-tokens 4096
```

The tuned size applies only while thermal pressure is Nominal. Daedalus still
automatically switches to smaller thermal-profile chunks when the laptop gets
hot. By default it also keeps Metal allocations between prefill chunks for
throughput; use `--clear-metal-cache-between-chunks` only when memory pressure
matters more than speed.

Full-prompt tokenization is memoized for repeated stateless agent requests,
and final cache snapshots are held in RAM immediately then written after the
first streamed output. `/metrics` exposes token-cache hits and KV-copy time so
you can verify whether those optimizations help your workload.

Design lineage: runtime = mlx-lm public primitives (no monkey-patching of internals);
cache design informed by vllm-mlx and Rapid-MLX (Apache-2.0); prefix-cache semantics
inspired by baseRT's public C API. The thermal governor exists in none of them.

## Status

Working vertical slice, 56 tests. In progress: real-model validation on Qwen3.5-9B and
Gemma 4 E4B, sustained thermal A/B on the M4 Air (`bench/thermal_validation.py`).

## License

Apache-2.0
