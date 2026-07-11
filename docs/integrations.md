# Agent integrations

Daedalus exposes the standard OpenAI-compatible base URL:

```text
http://127.0.0.1:8080/v1
```

For any OpenAI-compatible client, select the loaded model ID and set its base
URL to the value above. On a LAN deployment, use HTTPS from a reverse proxy and
provide `Authorization: Bearer <DAEDALUS_API_KEY>`.

## OpenCode

Set its OpenAI-compatible provider base URL to `http://127.0.0.1:8080/v1` and
use the Daedalus model ID. Leave streaming enabled: Daedalus sends SSE
keepalives while a cold prompt prefill is in progress.

## Pi, Hermes, Continue, and VS Code clients

Use their OpenAI-compatible provider mode with the same base URL and model ID.
Daedalus supports chat-completions streaming, tool-call deltas, and
`reasoning_content`; clients that do not understand reasoning deltas can ignore
that optional field.

## Production LAN setup

1. Run with `--host 127.0.0.1 --api-key-env DAEDALUS_API_KEY`.
2. Terminate TLS in Caddy, nginx, or another local reverse proxy.
3. Restrict the proxy to the intended LAN/VPN and forward the Authorization
   header unchanged.
4. Scrape `/metrics`, probe `/health` and `/readyz`, and set a conservative
   `--max-active-memory-gb` limit.
