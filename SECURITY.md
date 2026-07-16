# Security Policy

## Threat model

Daedalus is a single-user, single-machine MLX inference server. By default it
binds to `127.0.0.1`, so nothing is reachable off the host and no authentication
is required.

Exposing it beyond localhost is an explicit, deliberate step. To serve a LAN you
must both:

1. Set an API key — `--api-key-env DAEDALUS_API_KEY` (or `--api-key-file`), and
2. Terminate TLS at a local reverse proxy (Caddy, nginx, …) in front of the
   server. See [docs/integrations.md](docs/integrations.md) for the reference
   LAN deployment, including `--trusted-proxy-host` handling of forwarded peer
   addresses.

Requests to `/v1` then require `Authorization: Bearer <key>`. The server caps
active-or-queued requests (`--max-pending-requests`, default 8) and returns 429
when full rather than growing worker threads without bound.

## Not in scope

The following are explicitly outside the security model — do not rely on
Daedalus to provide them:

- **Multi-tenant isolation.** All callers sharing one server share one process,
  one prefix cache, and one memory budget. The API key is an on/off gate, not a
  per-tenant boundary.
- **Untrusted-model sandboxing.** A model id you load is executed with your
  privileges (weights, tokenizer, and chat template are trusted inputs). Only
  load models you trust; there is no sandbox around model code or assets.

## Reporting a vulnerability

Please report privately through GitHub's **Security Advisories** ("Report a
vulnerability" on the repository's Security tab), not via public issues. Include
the version or commit, your platform and hardware, and a minimal reproduction.
We aim to acknowledge reports within a few days.

## Supported versions

Security fixes target the latest tagged release and the `master` branch. Older
releases are not maintained; upgrade to the latest release to receive fixes.
