"""E2E: launch the real server, send two identical big-prompt requests,
verify streaming + keepalives + warm-cache TTFT improvement.
"""

import json
import subprocess
import sys
import time
import urllib.request

PORT = 8765
MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-0.6B-4bit"
PROMPT_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000


def wait_for_server(timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/health", timeout=2
            ) as r:
                return json.load(r)
        except Exception:
            time.sleep(1)
    raise TimeoutError("server did not come up")


def timed_request(label):
    filler = "The quick brown fox jumps over the lazy dog. " * (PROMPT_TOKENS // 10)
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "You are concise. " + filler},
                {"role": "user", "content": "Reply with exactly: E2E OK"},
            ],
            "stream": True,
            "max_tokens": 30,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    first_content = None
    keepalives = 0
    text = ""
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith(":"):
                keepalives += 1
                continue
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                if first_content is None:
                    first_content = time.time() - start
                text += delta["content"]
            if chunk.get("usage"):
                usage = chunk["usage"]
    cached = usage["prompt_tokens_details"]["cached_tokens"] if usage else 0
    print(
        f"{label}: TTFT={first_content:.2f}s keepalives={keepalives} "
        f"cached_tokens={cached} prompt_tokens={usage['prompt_tokens']}"
    )
    return first_content, cached


def main():
    proc = subprocess.Popen(
        [".venv/bin/python", "-m", "airlift.cli", "serve", "--model", MODEL,
         "--port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        health = wait_for_server()
        print(f"server up: {health}")
        cold_ttft, cold_cached = timed_request("cold")
        warm_ttft, warm_cached = timed_request("warm")
        assert cold_cached == 0, "first request must be a cache miss"
        assert warm_cached > 0, "second request must hit the prefix cache"
        assert warm_ttft < cold_ttft, "warm TTFT must beat cold TTFT"
        stats = json.load(
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/cache/stats")
        )
        print(f"cache stats: {stats}")
        print("E2E PASSED")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
