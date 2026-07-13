# Master audit plan

Audited against `origin/master` commit `67b3465` on 2026-07-13.  No source
code changes are included in this branch yet.

## Evidence

- Unit and integration suite: `144 passed`.
- Source distribution and wheel build successfully with `uv build`.
- The package build warns that the table form of `project.license` is
  deprecated and the repository does not ship a `LICENSE` file.
- The benchmark and Metal-capture tooling exist, but there is no automated
  real-model test matrix or statistically controlled performance gate.

## Priority 0: request safety and correctness

1. Enforce the request-body limit while reading, not only from
   `Content-Length`.  Chunked requests without that header are currently read
   in full by `request.json()`, so a LAN client can exceed
   `--max-request-bytes`.  Add a bounded streaming JSON-body reader and tests
   for chunked/absent/malformed length cases.
2. Validate the complete OpenAI request shape before admission: every stop
   value must be a non-empty string, tools must be a list of valid function
   definitions, and an explicitly requested `model` must equal the served
   model.  Return a documented 400/404 rather than allowing malformed input
   to fail in the generation thread.
3. Make the cache-admin clear operation atomic with admission.  It currently
   checks `admitted_requests` then clears outside the admission lock, allowing
   a new request to be admitted between the check and clear.  Introduce a
   maintenance gate or hold a dedicated cache-admin lock around both steps.
4. Add a lifecycle owner for the thermal monitor and stop it on application
   shutdown.  This makes in-process embeddings and tests safe, not only the
   short-lived CLI process.

Acceptance: focused negative-path tests plus the full test suite; no request
can cause unbounded body buffering or race a cache clear.

## Priority 1: durable production packaging and operations

1. Ship the Apache-2.0 license text, convert `project.license` to the current
   SPDX form, and assert that wheel/sdist contain the license and documentation.
2. Add a compatibility-tested dependency policy: retain `uv.lock` for
   development, constrain known-compatible MLX/MLX-LM/Transformers ranges for
   installs, and add a scheduled dependency-update verification job.
3. Add static quality gates (Ruff formatting/lint and Pyright) and coverage
   reporting with a non-flaky floor.  Keep macOS ARM as the release test lane;
   use a Linux lane only for pure protocol/unit tests if it proves useful.
4. Make releases traceable: changelog/version consistency check, SBOM,
   artifact checksums, and provenance/attestation.  Code signing/notarization
   remains an externally configured follow-up because it needs Apple
   credentials.
5. Document reverse-proxy deployment precisely: trusted-proxy configuration
   for client IP rate limits, TLS termination, CORS allow-list examples, and
   health/readiness exposure.  Do not trust `X-Forwarded-For` unless a proxy
   allow-list is explicitly configured.

Acceptance: a clean clone produces an installable artifact containing legal
metadata; CI blocks formatting, typing, tests, packaging, and provenance
regressions.

## Priority 2: measured inference performance

1. Extend the benchmark schema with model revision, MLX/MLX-LM versions,
   macOS version, RAM, power state, thermal starting state, cache mode, and
   trial number.  Reject comparisons whose machine/model/config fingerprints
   differ.
2. Add a repeat-run performance runner with cooldown/nominal-thermal gating
   and median/p95 reporting for cold TTFT, RAM-cache TTFT, disk-cache restart
   TTFT, prefill throughput, decode throughput, and peak memory.
3. Add real-model nightly/manual verification for at least one trimmable and
   one hybrid-cache model, including restart, checkpoint resume, tools,
   reasoning, client disconnect, and cache corruption recovery.
4. Add a capacity planner that estimates KV-cache bytes from model config,
   context length, KV quantization, and draft-model overhead before loading or
   admitting a request.  Replace the current reactive active-memory threshold
   with a prediction plus a safety reserve.
5. Make speculative decoding evidence-driven: benchmark candidate draft models
   on the exact hardware; surface acceptance rate, extra unified-memory cost,
   and net tokens/s in metrics.  Keep it unavailable for non-trimmable hybrid
   caches as it is today.

Acceptance: a candidate change cannot be called faster without comparable,
multi-trial artifacts; the server rejects configurations that cannot fit the
machine rather than failing under memory pressure.

## Priority 3: optional capability work

1. Multi-model serving with explicit eviction/admission math.  Do not load
   models opportunistically on a 16GB Air.
2. Model-native MTP/speculative heads only where the selected model and
   MLX-LM expose a supported implementation; benchmark against normal decode.
3. A custom Metal kernel only after a `.gputrace` identifies one material,
   repeatable hot operation not already handled by MLX-LM.  The current Qwen
   hybrid recurrent path already uses MLX-LM Metal kernels, so this is not the
   first optimization to implement.

Acceptance: each feature is opt-in, has a memory budget, retains OpenAI API
compatibility, and improves the measured target workload.

## Suggested delivery order

1. Priority 0 as one small safety PR.
2. Priority 1 packaging/CI as a second PR.
3. Priority 2 benchmark schema and real-model matrix before altering engine
   algorithms.
4. Only then choose one measured engine optimization from Priority 2/3.
