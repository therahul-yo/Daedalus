# Daedalus Improvements Implementation Plan

## Overview
Implement all 8 improvement categories identified in the audit as independent tasks, each executed by a dedicated subagent with two-stage review.

## Tasks

### Task 1: CI/CD Pipeline (GitHub Actions)
**Files to create/modify:**
- `.github/workflows/ci.yml` — main CI workflow
- `.github/workflows/bench.yml` — scheduled benchmark workflow

**Requirements:**
- macOS runner (Apple Silicon for MLX)
- Install dependencies: `pip install -e ".[dev]"`
- Run full test suite: `python -m pytest tests/ -v`
- Run smoke benchmark: `python bench/smoke.py --model mlx-community/Qwen3-0.6B-4bit`
- Run thermal validation: `python bench/thermal_validation.py --model mlx-community/Qwen3-0.6B-4bit --prompt-tokens 2000 --governor on --out bench/results/thermal_on.json` and `--governor off`
- Run regression gate on benchmark artifacts
- Cache MLX/HuggingFace downloads between runs

### Task 2: Structured Logging + OpenTelemetry
**Files to modify:**
- `daedalus/server.py` — add OTel instrumentation, structured JSON logging
- `daedalus/engine.py` — add spans for prefill/decode phases
- `daedalus/cache/store.py` — add spans for cache fetch/put
- `daedalus/governor.py` — add thermal state change spans
- `pyproject.toml` — add `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `structlog` to optional deps

**Requirements:**
- Structured JSON logs with `structlog` (level, timestamp, request_id, thermal_level, cache_hit, tokens_processed)
- OpenTelemetry spans for: request lifecycle, prefill chunks, cache operations, thermal transitions
- OTLP exporter configurable via env var `OTEL_EXPORTER_OTLP_ENDPOINT`
- Request ID propagation via `X-Request-ID` header

### Task 3: Cache Durability & Migration + CLI
**Files to create/modify:**
- `daedalus/cache/store.py` — add schema version migration, TTL eviction
- `daedalus/cli.py` — add `cache` subcommand group
- `daedalus/cache/cli.py` (new) — cache list, inspect, prune, warm-from-history

**Requirements:**
- Cache format version migration path (v1 → v2+)
- `--cache-ttl-days` parameter for disk eviction
- `daedalus cache list [--model MODEL]` — show entries with size, age, hits
- `daedalus cache inspect <entry>` — show token count, source, timestamps
- `daedalus cache prune [--ttl-days N] [--model MODEL]` — remove stale entries
- `daedalus cache warm-from-history --source openwebui|opencode|hermes --model MODEL` — import prompts from chat history exports

### Task 4: Server Hardening
**Files to modify:**
- `daedalus/server.py` — request ID propagation, graceful shutdown drain, global rate limit, priority queue

**Requirements:**
- Read/propagate `X-Request-ID` header through all log lines and OTel spans
- Graceful shutdown: stop accepting new requests, wait for in-flight (with timeout), then exit
- Global token bucket rate limiter (total RPS across all clients) in addition to per-client
- Priority queue: short prompts (<2k tokens) jump ahead of long prompts
- Configurable CORS origins via `--cors-origins` (comma-separated)

### Task 5: Multi-Model Serving
**Files to modify:**
- `daedalus/cli.py` — accept `--model` multiple times
- `daedalus/server.py` — per-model engine/store/router
- `daedalus/runtime.py` — cache identity per model

**Requirements:**
- `--model` can be specified multiple times: `daedalus serve --model A --model B`
- Each model gets isolated cache namespace (already via cache_identity)
- `/v1/models` returns all loaded models
- Request routing by `model` field in chat completions
- Shared thermal governor (single monitor) across models
- Memory admission control respects sum of active models

### Task 6: Developer Experience Commands
**Files to modify:**
- `daedalus/cli.py` — add `config`, `inspect-cache`, `benchmark` commands

**Requirements:**
- `daedalus config show` — resolved config (CLI + env + defaults merged)
- `daedalus config doctor` — validate config, warn on conflicts
- `daedalus inspect-cache --model MODEL` — same as `cache list` but richer output
- `daedalus benchmark --model MODEL --baseline baseline.json --candidate candidate.json` — run bench and run regression gate

### Task 7: Packaging & Distribution
**Files to create:**
- `packaging/homebrew/daedalus.rb` — Homebrew formula
- `packaging/pyinstaller/daedalus.spec` — PyInstaller spec for standalone binary
- `.github/workflows/release.yml` — build + notarize + upload assets

**Requirements:**
- Homebrew formula installs `daedalus` CLI with all deps
- PyInstaller build produces signed, notarized macOS binary (requires Apple Developer ID)
- GitHub Actions release workflow: on tag push, build binary, notarize, create release with artifacts
- Document installation methods in README: `brew install`, `uv tool install`, binary download

### Task 8: Security Hardening
**Files to modify:**
- `daedalus/server.py` — cookie auth, audit logging, configurable CORS

**Requirements:**
- Cookie-based auth for browser/websocket clients (`Authorization: Bearer` + `Cookie: daedalus_token=`)
- Audit log: structured JSON lines to `daedalus_audit.log` (auth success/fail, cache admin ops, rate limit hits, model load)
- Configurable CORS: `--cors-origins "https://app.example.com,https://dev.example.com"`
- API key rotation: `--api-key-file` watches file for changes (reload on SIGHUP or poll)

---

## Execution Order
Tasks 1-8 can run in parallel (independent file sets). Each task gets:
1. Implementer subagent
2. Spec compliance reviewer
3. Code quality reviewer
4. Mark complete

Final integration review after all 8 tasks done.