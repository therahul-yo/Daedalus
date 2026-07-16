# Agent integrations

Daedalus exposes the standard OpenAI-compatible base URL:

```text
http://127.0.0.1:8080/v1
```

For any OpenAI-compatible client, select the loaded model ID and set its base
URL to the value above. On a LAN deployment, use HTTPS from a reverse proxy and
provide `Authorization: Bearer <DAEDALUS_API_KEY>`.

The model ID is whatever you passed to `daedalus serve --model`. Ask the running
server for the exact string rather than guessing:

```bash
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

## curl (streaming)

Pass `-N` so curl does not buffer the response and the tokens stream as they are
produced. Drop the `Authorization` header when the server runs unkeyed on
localhost.

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $DAEDALUS_API_KEY" \
  -d '{
    "model": "YOUR_MODEL_ID",
    "stream": true,
    "messages": [
      {"role": "system", "content": "You are a terse coding assistant."},
      {"role": "user", "content": "Write a Python one-liner to reverse a list."}
    ]
  }'
```

While a cold prompt is still prefilling, Daedalus emits SSE **comment** lines
that begin with a colon, e.g.:

```text
: prefill 1024/8192 thermal=nominal eta=4s
```

These reset a client's idle timeout and are invisible to JSON parsers — only the
`data:` lines carry chunks. The stream terminates with `data: [DONE]`. A
non-streaming call (omit `"stream": true`) returns a single JSON body instead.

## OpenCode

Add Daedalus as a custom OpenAI-compatible provider in `opencode.json`. OpenCode
drives it through the AI SDK's `@ai-sdk/openai-compatible` package and passes
`options` straight to that provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "daedalus": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Daedalus (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "YOUR_MODEL_ID": {}
      }
    }
  }
}
```

If the server enforces `--api-key`, add an `apiKey` field alongside `baseURL`
under `options`. Leave streaming enabled: Daedalus sends the SSE keepalives
described above while a cold prompt prefill is in progress.

## Continue

Add a models entry to Continue's `config.yaml`. Use the built-in `openai`
provider and point `apiBase` at Daedalus:

```yaml
models:
  - name: Daedalus
    provider: openai
    model: YOUR_MODEL_ID
    apiBase: http://127.0.0.1:8080/v1
    apiKey: unused   # any non-empty string; use the real key if the server enforces one
    roles:
      - chat
      - edit
```

Continue's schema requires a non-empty `apiKey` even for a local server; supply
the real Bearer token when `--api-key` is set, otherwise any placeholder works.

## Pi, Hermes, and other OpenAI-compatible clients

Use their OpenAI-compatible provider mode with the same base URL and model ID.
Daedalus supports chat-completions streaming, tool-call deltas, and
`reasoning_content`; clients that do not understand reasoning deltas can ignore
that optional field.

## Warming the cache

`daedalus warm` pre-prefills a set of prompts into the persistent prefix cache
while the machine is cool, so the first real request that shares that prefix
(typically a fixed agent system prompt) skips the cold prefill. The prompts file
is a JSON array of `{"messages": [...]}` objects — the same message shape you
send to `/v1/chat/completions`. A ready-to-edit pack ships at
[`examples/warm-prompts.json`](../examples/warm-prompts.json):

```bash
daedalus warm \
  --model YOUR_MODEL_ID \
  --prompts examples/warm-prompts.json
```

The prefix cache is namespaced by model, `--kv-bits`, and `--model-revision`, so
warm and serve must agree on all three or the warmed entries will not be found.
Both default to `--kv-bits 8`; if you override it on `serve`, pass the same value
to `warm`.

### Warming at login

The repository ships a launchd template for the server itself at
[`scripts/com.daedalus.server.plist.template`](../scripts/com.daedalus.server.plist.template)
(copy it, edit the paths, and load it with `launchctl bootstrap`, as described in
the README). To warm at login, adapt a *copy* of that template into a one-shot
agent: replace the `serve ...` arguments in `ProgramArguments` with
`warm --model YOUR_MODEL_ID --prompts /absolute/path/to/warm-prompts.json`, keep
`RunAtLoad`, and drop `KeepAlive` (warm exits when it finishes rather than
staying resident). Point it at an absolute prompts path — launchd does not run
from your project directory.

Before serving from an automated context you can gate on a machine-readable
health check: `daedalus doctor --json` prints `{"ok": ..., "checks": [...]}` to
stdout and exits non-zero if the thermal sensor or MLX is unavailable.

## Production LAN setup

1. Run with `--host 127.0.0.1 --api-key-env DAEDALUS_API_KEY`.
2. Terminate TLS in Caddy, nginx, or another local reverse proxy.
3. Restrict the proxy to the intended LAN/VPN and forward the Authorization
   header unchanged. If rate limits should distinguish downstream clients,
   pass the proxy's peer address with `--trusted-proxy-host`; Daedalus ignores
   `X-Forwarded-For` from every other peer.
4. Consider `--cache-ttl-days 30` for a long-running shared machine. Scrape
   `/metrics`, probe `/health` and `/readyz`, and set a conservative
   `--max-active-memory-gb` limit.
