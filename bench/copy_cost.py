"""Quantify the hidden cost of copying MLX KV caches (audit finding 6).

Hypothesis under test: ``copy.deepcopy`` on an mx.array-backed KV cache is
lazy/COW — near-free at copy time — and the real buffer duplication happens
on the *first write* to the copy, i.e. inside TTFT on every cache hit.

VERDICT (2026-07-13, closes the audit's deferred "COW ownership redesign"):
measured on real hardware, the redesign is NOT worth shipping. A loan-out
fetch (hand the caller the stored object, skip the deepcopy) was implemented
and A/B-measured on Qwen3.5-9B with an 8k prompt: warm TTFT 0.25s vs 0.22s
on stock master — no improvement, within noise. Two reasons the synthetic
number here (~66ms at 8k) does not transfer:

1. The end-of-prefill snapshot ``put()`` deepcopies the request's cache back
   into the store, making the request's buffers copy-on-write again — the
   first decode write pays one materialization regardless of what fetch()
   avoided. The last copy before the first decode write is the one that
   bills; removing the fetch-side copy changes nothing.
2. This bench models 8 full-attention layers with real 8k-token KV. On the
   actual hybrid model only 8 of 32 layers are GQA; the 24 Gated-DeltaNet
   layers carry small constant state, so the true per-copy tax is roughly a
   quarter of the synthetic figure (~15ms) — inside run-to-run noise.

Do not re-attempt without first invalidating one of those two facts (e.g. a
snapshot mechanism that does not re-share the request's buffers).

Usage: python bench/copy_cost.py [n_tokens]
No model needed — synthetic KV shaped like Qwen3.5-9B's 8 GQA layers.
"""

import copy
import sys
import time

import mlx.core as mx
from mlx_lm.models.cache import KVCache

N_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
N_LAYERS = 8          # Qwen3.5-9B growing (attention) layers
KV_HEADS, HEAD_DIM = 4, 256


def build_cache(n_tokens: int) -> list:
    layers = []
    for _ in range(N_LAYERS):
        c = KVCache()
        k = mx.random.normal((1, KV_HEADS, n_tokens, HEAD_DIM)).astype(mx.float16)
        v = mx.random.normal((1, KV_HEADS, n_tokens, HEAD_DIM)).astype(mx.float16)
        c.update_and_fetch(k, v)
        layers.append(c)
    mx.eval([c.state for c in layers])
    return layers


def first_write(cache: list) -> None:
    """One decode-step append — the write that materializes a COW copy."""
    for c in cache:
        k = mx.random.normal((1, KV_HEADS, 1, HEAD_DIM)).astype(mx.float16)
        c.update_and_fetch(k, k)
    mx.eval([c.state for c in cache])


def main() -> None:
    nbytes = 2 * N_LAYERS * KV_HEADS * N_TOKENS * HEAD_DIM * 2
    print(f"cache: {N_LAYERS} layers x {N_TOKENS} tok -> {nbytes / 1e6:.0f} MB fp16")

    base = build_cache(N_TOKENS)

    t0 = time.perf_counter()
    copied = copy.deepcopy(base)
    t_copy = time.perf_counter() - t0
    print(f"deepcopy:                 {t_copy * 1000:8.2f} ms")

    t0 = time.perf_counter()
    first_write(copied)
    t_write_copy = time.perf_counter() - t0
    print(f"first write on the copy:  {t_write_copy * 1000:8.2f} ms")

    fresh = build_cache(N_TOKENS)
    t0 = time.perf_counter()
    first_write(fresh)
    t_write_fresh = time.perf_counter() - t0
    print(f"first write, unshared:    {t_write_fresh * 1000:8.2f} ms")

    hidden = t_write_copy - t_write_fresh
    print(
        f"\nhidden COW materialization: {hidden * 1000:.2f} ms "
        f"({t_write_copy / max(t_write_fresh, 1e-9):.1f}x the unshared write)"
    )
    print(
        "verdict:",
        "COW tax CONFIRMED — copy cost lands in TTFT on first decode step"
        if hidden > 5 * max(t_write_fresh, 1e-9) or hidden > 0.010
        else "no significant COW tax on this MLX version",
    )


if __name__ == "__main__":
    main()
