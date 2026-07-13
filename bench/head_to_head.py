"""Head-to-head: daedalus vs llama.cpp (llama-server) on the stateless-agent
workload — same prompt, cold TTFT, warm TTFT, and TTFT after a server restart
(the persistent-cache test).

Usage:
    python bench/head_to_head.py \
        [--mlx-model mlx-community/Qwen3-0.6B-4bit] \
        [--gguf ggml-org/Qwen3-0.6B-GGUF:Q4_K_M] \
        [--prompt-tokens 6000]

llama-server must be on PATH (brew install llama.cpp). The GGUF is fetched
from Hugging Face by llama-server itself on first run (-hf).
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request

DAEDALUS_PORT = 8767
LLAMA_PORT = 8768


def wait_health(url: str, timeout: float = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"server at {url} did not come up")


def _spawn(cmd: list, log_name: str, health_url: str) -> subprocess.Popen:
    log = open(f"bench/results/{log_name}.log", "ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{cmd[0]} exited rc={proc.returncode} — see bench/results/{log_name}.log"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2):
                return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError(
        f"{cmd[0]} never became healthy — see bench/results/{log_name}.log"
    )


def start_daedalus(model: str) -> subprocess.Popen:
    return _spawn(
        [sys.executable, "-m", "daedalus.cli", "serve",
         "--model", model, "--port", str(DAEDALUS_PORT)],
        "h2h_daedalus",
        f"http://127.0.0.1:{DAEDALUS_PORT}/health",
    )


def start_llama(gguf: str, ctx: int) -> subprocess.Popen:
    return _spawn(
        ["llama-server", "-hf", gguf, "--port", str(LLAMA_PORT),
         "-c", str(ctx), "--no-webui"],
        "h2h_llama",
        f"http://127.0.0.1:{LLAMA_PORT}/health",
    )


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def timed_request(port: int, filler: str, *, cache_prompt: bool = True):
    """One streamed chat completion; returns (ttft_s, decode_tps, n_completion)."""
    body = {
        "messages": [
            {"role": "system", "content": "You are concise. " + filler},
            {"role": "user", "content": "Reply with exactly: BENCH OK"},
        ],
        "stream": True,
        "max_tokens": 60,
        "temperature": 0,
        # llama-server honors this; daedalus ignores unknown fields.
        "cache_prompt": cache_prompt,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    first = None
    last = None
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.time()
                if first is None:
                    first = now - start
                last = now
                n_tokens += 1
    decode_s = (last - start - first) if (first is not None and last) else 0.0
    tps = (n_tokens - 1) / decode_s if decode_s > 0 and n_tokens > 1 else float("nan")
    return first, tps, n_tokens


def wait_thermal_nominal(timeout: float = 1800) -> None:
    """Block until macOS reports NOMINAL thermal pressure (fair cold starts)."""
    try:
        from daedalus.sensors import make_pressure_reader
        read = make_pressure_reader()
    except Exception:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        level = read()
        if level.name == "NOMINAL":
            return
        print(f"  (thermal {level.name} — cooling before this arm...)", flush=True)
        time.sleep(30)
    print("  (warning: never reached NOMINAL; proceeding warm)", flush=True)


def run_arm(name: str, start_fn, port: int, filler: str, results: dict,
            wait_nominal: bool = False) -> None:
    print(f"\n=== {name} ===", flush=True)
    if wait_nominal:
        wait_thermal_nominal()
    proc = start_fn()
    try:
        cold, tps_c, _ = timed_request(port, filler)
        print(f"  cold TTFT   {cold:8.2f}s   decode {tps_c:6.1f} tok/s", flush=True)
        warm, tps_w, _ = timed_request(port, filler)
        print(f"  warm TTFT   {warm:8.2f}s   decode {tps_w:6.1f} tok/s", flush=True)
    finally:
        stop(proc)
    # Restart: does the prompt cache survive?
    proc = start_fn()
    try:
        restart, tps_r, _ = timed_request(port, filler)
        print(f"  restart TTFT{restart:8.2f}s   decode {tps_r:6.1f} tok/s", flush=True)
    finally:
        stop(proc)
    results[name] = {"cold": cold, "warm": warm, "restart": restart,
                     "decode_tps": tps_w}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-model", default="mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--gguf", default="unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
    ap.add_argument("--prompt-tokens", type=int, default=6000)
    ap.add_argument("--wait-nominal", action="store_true",
                    help="cool to NOMINAL thermal pressure before each arm")
    ap.add_argument("--out", help="write results JSON here")
    args = ap.parse_args()

    if not shutil.which("llama-server"):
        print("error: llama-server not on PATH (brew install llama.cpp)")
        return 1

    # Unique per run so day-old daedalus disk cache can't fake a cold hit,
    # but IDENTICAL across arms and across the restart inside one run.
    nonce = str(time.time_ns())
    filler = f"Run {nonce}. " + (
        "The quick brown fox jumps over the lazy dog. " * (args.prompt_tokens // 10)
    )

    results: dict = {}
    run_arm("daedalus", lambda: start_daedalus(args.mlx_model),
            DAEDALUS_PORT, filler, results, wait_nominal=args.wait_nominal)
    run_arm("llama.cpp", lambda: start_llama(args.gguf, args.prompt_tokens + 4096),
            LLAMA_PORT, filler, results, wait_nominal=args.wait_nominal)

    print("\n=== summary (identical prompt, temp 0, stream) ===")
    print(f"{'':12} {'cold TTFT':>10} {'warm TTFT':>10} {'restart TTFT':>13} {'decode':>10}")
    for name, r in results.items():
        print(f"{name:12} {r['cold']:>9.2f}s {r['warm']:>9.2f}s "
              f"{r['restart']:>12.2f}s {r['decode_tps']:>7.1f} t/s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
