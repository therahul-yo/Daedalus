```
██                            ██
████                        ████
████████                ████████
  ██████████        ██████████
    ██████████    ██████████
        ████████████████
            ████████
```

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

**vs llama.cpp** — same machine, same 8,000-token agent prompt, Qwen3.5-9B at 4-bit
(MLX 4-bit vs GGUF Q4_K_M), identical protocol via `bench/head_to_head.py
--wait-nominal` (each arm's cold start waits for NOMINAL thermal pressure, so
neither engine inherits the other's heat):

| | cold TTFT | warm TTFT | TTFT after server restart | decode |
|---|---|---|---|---|
| daedalus | 50.8 s | 0.22 s | **0.55 s** | 18.2 tok/s |
| llama-server | 57.3 s | 0.60 s | 78.6 s | 13.8 tok/s |

From an identical cool chassis, daedalus prefills 11% faster while thermally pacing
itself and decodes 32% faster. The decisive column is the restart: llama.cpp's slot
cache dies with the process, so a daemon restart re-pays the full 8k prefill — on a
chassis its own workload just heated, throttled down to 11.6 tok/s mid-prefill —
while daedalus reloads the snapshot from disk in half a second. Every restart,
crash, or reboot re-runs this experiment. Reproduce with
`python bench/head_to_head.py --mlx-model <mlx-id> --gguf <hf-repo:quant> --wait-nominal`.

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
# from a release wheel (recommended) — puts `daedalus` on your PATH
uv tool install https://github.com/therahul-yo/Daedalus/releases/download/v0.2.0/daedalus-0.2.0-py3-none-any.whl
# or from source (editable):
uv tool install --editable . --with "transformers>=5.0,<5.11"
# or: pipx install .
```

Requires an Apple Silicon Mac (MLX). `--audit-log`, `--log-json`, and the
`observability` extra (`pip install 'daedalus[observability]'` for structlog +
OpenTelemetry) are optional.

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

For a service, avoid putting a key in shell history: use
`--api-key-env DAEDALUS_API_KEY` or `--api-key-file /path/to/secret` instead.
Terminate TLS at a local reverse proxy before making the service reachable on a
LAN. See [agent integrations](docs/integrations.md) for compatible-client and
deployment notes.

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

Persistent caches are automatically isolated by model, tokenizer, MLX/MLX-LM
versions, KV layout, and machine architecture. This prevents an upgrade from
loading an incompatible KV snapshot. Pass `--model-revision <immutable-id>`
when a model name may resolve to mutable upstream weights.

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

For release-performance gating, compare two real-model benchmark artifacts:

```bash
python bench/regression.py baseline.json candidate.json
```

The default gate allows at most 10% TTFT and 5% prefill-throughput regression;
adjust the thresholds only with a documented hardware/model reason.

For kernel investigation, capture a real model before changing MLX-LM internals:

```bash
python bench/bench.py --model <model> --prompt-tokens 20000 \
  --metal-capture /tmp/daedalus.gputrace --out run.json
```

Only pursue a custom kernel when the capture identifies one operation as a
material cold-prefill bottleneck. Qwen hybrid models already use MLX-LM Metal
kernels for their recurrent path, so Daedalus keeps stock MLX-LM kernels by
default.

Speculative decoding is available experimentally with a tokenizer-compatible
draft model: `--draft-model <id> --num-draft-tokens 2`. Benchmark memory,
acceptance rate, and sustained thermals first; a draft model can slow a 16GB
Air when its extra unified-memory use outweighs decode savings.

For a persistent Mac service, copy and customize
[`scripts/com.daedalus.server.plist.template`](scripts/com.daedalus.server.plist.template),
then load it with `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.daedalus.server.plist`.
Keep the API secret in a root-readable environment file or secret manager rather
than committing it into the plist.

Tagged releases include `SHA256SUMS` and `requirements.lock.txt` alongside the
wheel and source archive. Verify the checksum before installing an artifact;
the locked manifest records the exact runtime dependency resolution used for
that release.

Design lineage: runtime = mlx-lm public primitives (no monkey-patching of internals);
cache design informed by vllm-mlx and Rapid-MLX (Apache-2.0); prefix-cache semantics
inspired by baseRT's public C API. The thermal governor exists in none of them.

## Status

**v0.2.0 released** — engine, persistent prefix cache, thermal governor, and
OpenAI-compatible server, validated on real hardware (M4 Air 16GB, Qwen3.5-9B-4bit):
171 tests, ruff-gated, thermal A/B benchmarked (`bench/thermal_validation.py`), and
measured head-to-head against llama.cpp (`bench/head_to_head.py`); tool calls and
reasoning separation verified against pi.

v0.2 adds penalties / stop / usage options; request-ID, CORS, and global + per-client
rate limiting; predictive (hybrid-aware) KV admission and context-window guards; the
`daedalus cache` CLI with TTL eviction; the NDJSON audit log, `--log-json`, OpenTelemetry,
and latency histograms; `daedalus status`; and a SHA-pinned release shipping a wheel +
checksums. See [`CHANGELOG.md`](CHANGELOG.md) and the
[v0.2.0 release](https://github.com/therahul-yo/Daedalus/releases/tag/v0.2.0).

**On `main`, toward v0.3** — swap-only multi-model serving: register extra models with
`--swap-model`, and a request for a non-resident model triggers a hot-swap that tears down
the current engine (releasing its Metal memory and cache flock) before loading the target,
so only one model is ever resident on your 16GB. Admission is checked against a derived,
hybrid-aware weights+KV estimate; swaps are rate-limited by a cooldown, and a failed target
load restores the previous model rather than stranding the server. Implemented and
unit-tested (180 tests) — **real-hardware swap validation is still pending**, so it is not
in the v0.2.0 release.

Deliberately deferred: the MTP decode head (the 4-bit checkpoint ships no MTP weights)
and a COW cache-copy redesign (implemented, A/B-measured, no gain — see
`bench/copy_cost.py`). Also next: Gemma 4 real-model run, and a signed standalone binary
(the current build is best-effort — MLX's `@rpath` native libs don't survive PyInstaller yet).

## License

Apache-2.0
