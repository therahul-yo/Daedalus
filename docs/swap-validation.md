# Multi-model swap validation (real 16GB M-series)

A short checklist for confirming hot-swap works on real hardware, where the
constraint that bites is RAM: two 7-9B models cannot be resident at once on a
16GB machine, so a swap must fully release the old model before loading the new.

## Run

Start the server with at least two models registered:

```sh
daedalus serve --model qwen-7b --model qwen-3b --port 8484
```

Then, from another shell, drive the harness against the running server:

```sh
python scripts/validate_swap.py http://127.0.0.1:8484 --target qwen-3b
# add --api-key KEY if the server is key-protected
```

It lists `/v1/models`, chats the resident model, swaps to `--target` (timing
the load and retrying once through the 30s cooldown), records free memory
before/after, swaps back, and prints a PASS/FAIL checklist.

## What to watch

- **`/v1/models`** must list every `--model` id, resident one first, each with
  `resident: true|false`.
- **`/metrics`** (needs the API key when the server is keyed):
  - `active_memory_bytes` should drop to roughly the new model's weight
    footprint right after the swap — evidence the old model was released, not
    stacked on top of the new one.
  - watch that the cache stats reset for the swapped-in model (its store starts
    empty).
- **`/health`** should report `model` == the swapped-in id and stay `200`. A
  `503` with `status: degraded` means a swap failed to load the target *and*
  could not restore the previous model — investigate the server log.
- **Activity Monitor / `psutil`**: free memory before vs. after a swap should
  move by roughly `|weights(new) - weights(old)|`, never by the *sum* of both.

## Expected memory ceiling

On 16GB, keep `weights + KV` of the single resident model under ~10.7 GB
(`MODEL_MEMORY_CEILING_GB`), with ~1 GB (`SWAP_SAFETY_GB`) kept free during the
swap so the two engines are never both resident. A swap that would exceed this
is rejected up front with a `409 model_swap_conflict` naming the required vs.
available GB — that rejection is correct behavior, not a bug.
