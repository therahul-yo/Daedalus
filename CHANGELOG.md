# Changelog

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
