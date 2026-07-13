# Changelog

## v0.2.0 — 2026-07-13

Multi-agent development round: features built by three coding agents,
review-hardened (every PR adversarially verified before merge).

### API
- `frequency_penalty` / `presence_penalty` via mlx-lm logits processors,
  built only when a request sets them.
- `stop` sequences (string or list): rolling cross-chunk match, output
  truncated at the matched string.
- `stream_options: {"include_usage": true}` — OpenAI-spec trailing usage
  chunk with empty `choices`; the default keeps usage on the final content
  chunk so unguarded `choices[0]` clients (OpenCode) never break.

### Server hardening
- `X-Request-ID` extracted/generated and echoed on every response.
- Configurable CORS (`--cors-origin`, repeatable; off by default).
- Global token-bucket rate limiter (`--global-rps`) alongside the per-IP
  window; queued-abort-safe FifoLock (an aborting waiter can no longer
  hand the engine lock to two holders).
- Shutdown drain now tracks in-flight requests on both response paths.
- psutil-based low-memory guard feeding cache trim before admission.

### Cache
- `daedalus cache list | inspect | prune` CLI with exclusive-lock
  fail-fast when a server owns the cache dir.
- TTL-based disk eviction (`--cache-ttl-days`), pin-aware, incremental
  accounting preserved.
- Sidecar format v2 (adds `created`); v1 entries migrate in place on
  first load, indexed in the same pass.

### Observability
- Structured NDJSON audit log (`--audit-log`): auth failures, rate-limit
  hits, queue/memory rejections, cache-admin ops — real client addresses,
  10 MiB rotation.
- `--log-json` structured logging (structlog when installed, stdlib
  fallback); OpenTelemetry OTLP tracing auto-enables via standard
  `OTEL_*` env vars when the optional packages are present.

### Tests
- 144 tests (was 111): FifoLock fairness, stop/penalty/usage regression,
  request-ID/global-limit/drain, cache TTL/migration/prune, audit and
  observability coverage; deflaked thermal-sensor test.

## v0.1.0 — 2026-07-11

First working release, built and validated in one day on the target hardware
(M4 MacBook Air 16GB, Qwen3.5-9B MLX 4-bit).

### Engine
- Thermally-governed chunked prefill: duty-cycles GPU burn off macOS's 5-level
  thermal-pressure signal (no sudo). Profiles: `performance / balanced / cool`.
  Measured: same 80k-token workload with 20% less GPU time, throttled residency
  50% → 21%, stable 157 tok/s prefill vs decaying 133 → 119 unpaced.
- Small jobs (<4k fresh tokens) never pace — interactive turns stay fast.
- Metal cache high-water guard bounds retained allocations (1.5GB default).
- Snap points land chunks exactly on shared-prompt boundaries.

### Prefix cache
- Persistent (RAM LRU + disk safetensors) with exact-prefix semantics correct
  for hybrid-attention models (Qwen3.5 Gated-DeltaNet, wrapped Gemma 4
  windows) where stock servers silently miss. Measured: 8k-token agent prompt
  52.2s cold → 0.24s warm (218×), surviving restarts.
- Trie-indexed lookup, shared-head snapshots for new sessions, mid-prefill
  checkpoints (resumable after timeout), incremental disk accounting,
  crash-safe sidecar-first writes, exclusive flock per cache dir,
  cache-identity namespacing (model/kv/mlx versions/draft).
- Deferred persists: disk writes land after the client's final chunk; pinned
  entries make deferral safe under RAM pressure.

### Server
- OpenAI-compatible `/v1/chat/completions` (SSE + non-streaming) with strict
  tool-call deltas (never empty `tool_calls`), `reasoning_content` separation,
  prefill keepalives with thermal + ETA, request validation, bounded admission
  (FIFO), per-IP rate limiting, API-key auth (constant-time), request-size and
  token limits, `/metrics` (auth-gated when keyed), `/readyz`, `/health`,
  graceful drain on shutdown, disconnect aborts for both response modes.

### CLI & ops
- `daedalus serve | doctor | warm | tune`; startup banner with memory/cache/
  thermal; friendly model-load errors; prompts-file validation before load;
  launchd template using `--api-key-file`; CI + tag-triggered release
  workflows with SHA-pinned actions.
