# airlift

**A MacBook-Air-first MLX inference engine.** Local coding agents (OpenCode, Hermes Agent,
Pi Code) send 10–40k-token first prompts; on a passively-cooled Air, prefilling that prompt
is a multi-minute full-power GPU burn that thermally throttles the machine and times out the
client. airlift is a local OpenAI-compatible server built on [mlx-lm](https://github.com/ml-explore/mlx-lm)
that treats the Air's thermal envelope as a first-class scheduling constraint.

## How it solves the big-prompt problem

1. **The best prefill is no prefill** — a persistent prefix cache (RAM + disk, survives
   restarts) means the agent's giant system prompt is prefilled once, ever.
2. **Never re-do work** — mid-prefill checkpoints make long prefills resumable; a client
   timeout or crash never repeats completed work.
3. **Duty-cycle the burn** — a thermal governor reads macOS's 5-level thermal-pressure
   signal (no sudo) and paces prefill chunks: full speed while Nominal, smaller chunks with
   idle gaps at Moderate, hold at Heavy.
4. **Never let the client die** — SSE keepalives stream from the first second of prefill.
5. **Sized for the machine** — quantized KV cache by default; Metal memory limits set from
   the actual RAM budget. Single-user sequential engine: no batching complexity.

Design lineage: runtime = mlx-lm primitives (no monkey-patching); cache design informed by
vllm-mlx and Rapid-MLX (Apache-2.0); prefix-cache API semantics inspired by baseRT's public
C API. The thermal governor is what none of them have.

## Status

Early development. Target/dev hardware: M4 MacBook Air 16GB.

## License

Apache-2.0
