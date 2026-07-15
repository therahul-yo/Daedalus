"""Manual real-model verification of the single-resident swap path.

Runs one completion on a registered target model and confirms that the server
reports that target as active afterwards. Intended for an Apple Silicon host;
both models must fit individually in the configured memory budget.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("swap_model")
    parser.add_argument("--port", type=int, default=8876)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "daedalus.cli", "serve",
            "--model", args.model,
            "--swap-model", args.swap_model,
            "--port", str(args.port),
        ],
    )
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                if get_json(f"{base}/health").get("status") == "ok":
                    break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError("server did not become healthy")

        request = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(
                {"model": args.swap_model, "messages": [{"role": "user", "content": "Reply OK."}], "max_tokens": 8}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            completion = json.load(response)
        assert completion["model"] == args.swap_model, completion
        health = get_json(f"{base}/health")
        assert health["model"] == args.swap_model, health
        assert health["thermal"] in {"NOMINAL", "MODERATE", "HEAVY", "TRAPPING", "SLEEPING"}
        print(json.dumps({"swapped_to": args.swap_model, "thermal": health["thermal"]}))
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
