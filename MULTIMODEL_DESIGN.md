# Daedalus Multi-Model (16GB M4 Air, swap-only)

## Decision (revised per review)
- **No manager indirection layer.** `ServerState` keeps the single access path
  `state.engine` / `state.store`. A hot-swap repoints those two fields (plus the
  token/head caches) atomically behind the existing `FifoLock`. No 15 call-site
  rewrites, no dual legacy/manager path.
- **Builds on #17's model-404.** `chat_completions` already returns 404 for
  unknown models. The swap branch is added *after* that check: registered-but-
  not-resident models trigger `state.swap_model()`, which returns 409 (over
  budget / cooldown) or performs the swap.
- **Concurrent models don't fit on 16GB** → one resident engine at a time.

## Memory math
- Reserved: 3.5 (macOS) + 0.8 (process) + 1.0 (safety) = 5.3 GB
- Usable ceiling `MODEL_MEMORY_CEILING_GB = 10.7` for weights + KV of the active model.
- `SWAP_SAFETY_GB = 1.0` kept free during a swap so both engines are never resident.
- Admission: `candidate.total_gb(ctx) + SWAP_SAFETY_GB <= 10.7`. The old model
  is released before the target loads, so inactive models never consume RAM.

## Profiles — derived, not hardcoded
- `derive_model_profile(model_id, model_path)`:
  - Known hybrid archs (`qwen3.5-9b`) have exact built-in `MODEL_PROFILES`.
  - Otherwise reads `config.json` (layers/hidden/heads) + sums safetensors file
    sizes from the index header for exact weights. KV estimated from layer count
    (conservative for hybrids — overcounts constant-state layers = safe).
- CLI override is **not** needed: weights are exact, KV is config-derived.
- Context length for the math comes from `--max-prompt-tokens` (server config),
  not a baked-in 8K/32K.

## Swap sequence (serialized by FifoLock)
1. Request names a registered non-resident model.
2. `state.swap_model()`:
   - 404 already handled upstream (unknown model).
   - Reject if within `swap_cooldown_seconds` (default 30s) → 409 "swap cooldown".
   - `FifoLock.acquire_for_swap()` blocks new admits and waits until the engine
     is idle (no holder, queue drained).
   - Drain deferred snapshot persistence, close the old store, tear down the
     old engine, clear Metal memory, then lazy-load the target and repoint
     `state.engine`/`state.store`/`token_cache`/`head_cache`/`model_id`.
   - `FifoLock.release_after_swap()` lets new admits proceed.
   - Emit `audit_logger.model_swap(from, to)`.
3. In-flight requests that already hold the lock finish on the *old* engine; the
   next acquire sees the new engine. Cache namespaces are per-model (separate
   `PrefixCacheStore` per model key), so no cross-model KV reuse.

## API
- `POST /v1/chat/completions {"model": "qwen-7b"}`:
  - `model == resident` → fast path (unchanged).
  - `model` registered, not resident, fits → swap, then serve.
  - `model` registered, not resident, over budget/cooldown → **409** with detail.
  - `model` unknown → **404** (model_not_found, from #17).
- `GET /v1/models` lists registered models and marks the resident one with
  `resident: true`; selecting another id still performs a swap on demand.

## CLI
- `daedalus serve --model qwen-7b --swap-model qwen-3b --swap-model qwen-3.5-9b`
  registers each target path without loading it. The first request for a target
  pays its load time, but only one model is resident on a 16GB machine.

## Tests (tests/test_multimodel.py — all green)
- unknown model → 404
- swap to registered model → served by that engine; becomes resident
- swap cooldown → second immediate swap → 409
- cache isolation: model-a store independent of default
- admission rejects over-budget model → 409
- `model_fits` math: small fits with no active; 14B doesn't fit alongside 7B

## Files touched
- `daedalus/server.py`: `ModelProfile`, `MODEL_PROFILES`, `derive_model_profile`,
  `model_fits`; `ServerState.models/served_models/model_paths/swap_*`,
  `register_model`, `swap_model`; `create_app(model_specs=...)`;
  `chat_completions` swap branch on top of #17's 404.
- `daedalus/scheduler.py`: `FifoLock` gains `acquire_for_swap`/`release_after_swap`
  (swap gate) — FIFO order preserved for normal admits.
- `daedalus/audit.py`: `model_swap` emit helper.
- `daedalus/cli.py`: `--swap-model` (repeatable) → `model_specs`.
- `tests/test_multimodel.py`: 6 tests.
