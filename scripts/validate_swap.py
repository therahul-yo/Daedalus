#!/usr/bin/env python3
"""Real-hardware multi-model swap validation.

Run against an ALREADY-RUNNING daedalus server (start it separately with two
``--model`` entries). Uses only the stdlib plus psutil so it can run from the
system Python on a 16GB M-series machine.

    python scripts/validate_swap.py http://127.0.0.1:8484 \
        --target qwen-3b [--api-key KEY]

It lists models, chats the resident model, hot-swaps to the target (timing the
load and tolerating the 30s cooldown with a retry), records process-free memory
before/after via psutil, swaps back, and prints a pass/fail checklist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

import psutil


def _post(base: str, path: str, body: dict, api_key: str | None) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="POST")
    req.add_header("content-type", "application/json")
    if api_key:
        req.add_header("authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _get(base: str, path: str, api_key: str | None) -> tuple[int, dict]:
    req = urllib.request.Request(base + path, method="GET")
    if api_key:
        req.add_header("authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _chat(base: str, model: str | None, api_key: str | None) -> tuple[int, dict]:
    body: dict = {"messages": [{"role": "user", "content": "Reply with the single word OK."}],
                  "max_tokens": 8}
    if model is not None:
        body["model"] = model
    return _post(base, "/v1/chat/completions", body, api_key)


def _swap(base: str, target: str, api_key: str | None) -> tuple[bool, float, str]:
    """Request the swap target; retry once through the cooldown (409)."""
    for attempt in range(2):
        started = time.monotonic()
        status, payload = _chat(base, target, api_key)
        elapsed = time.monotonic() - started
        if status == 200:
            return True, elapsed, "swapped"
        message = (payload.get("error") or {}).get("message", str(payload))
        if status == 409 and "cooldown" in message.lower() and attempt == 0:
            print(f"  cooldown hit ({message!r}); waiting 31s and retrying…")
            time.sleep(31)
            continue
        return False, elapsed, f"{status}: {message}"
    return False, 0.0, "swap did not complete"


def _mem_free_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="server base URL, e.g. http://127.0.0.1:8484")
    parser.add_argument("--target", required=True, help="model id to swap to")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    key = args.api_key

    checklist: list[tuple[str, bool]] = []

    status, models = _get(base, "/v1/models", key)
    ids = [m.get("id") for m in models.get("data", [])]
    resident = next((m["id"] for m in models.get("data", []) if m.get("resident")), None)
    checklist.append((f"/v1/models lists >1 model (got {ids})", status == 200 and len(ids) > 1))
    checklist.append((f"target {args.target!r} is served", args.target in ids))

    status, _ = _chat(base, None, key)
    checklist.append((f"resident model {resident!r} answers a chat", status == 200))

    before = _mem_free_gb()
    ok, elapsed, detail = _swap(base, args.target, key)
    after = _mem_free_gb()
    checklist.append((f"swap to {args.target!r} succeeded ({detail}, {elapsed:.1f}s)", ok))
    print(f"  free memory: {before:.2f} GB before → {after:.2f} GB after swap")

    status, health = _get(base, "/health", key)
    checklist.append((f"/health reports {args.target!r} resident",
                      status == 200 and health.get("model") == args.target))

    # Swap back to the original resident (best-effort, tolerates cooldown).
    if resident and resident != args.target:
        ok_back, _, detail_back = _swap(base, resident, key)
        checklist.append((f"swap back to {resident!r} succeeded ({detail_back})", ok_back))

    print("\nChecklist:")
    all_ok = True
    for label, passed in checklist:
        all_ok = all_ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
