"""Small dependency-free Prometheus metrics registry for the local server."""

from __future__ import annotations

import threading
from collections import Counter


class ServerMetrics:
    """Thread-safe counters and gauges exported by ``/metrics``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = Counter()
        self.errors = Counter()
        self.cache_admin = Counter()

    def inc_request(self, outcome: str) -> None:
        with self._lock:
            self.requests[outcome] += 1

    def inc_error(self, kind: str) -> None:
        with self._lock:
            self.errors[kind] += 1

    def inc_cache_admin(self, action: str) -> None:
        with self._lock:
            self.cache_admin[action] += 1

    def render(self, *, active: int, limit: int, cache: dict, thermal: str) -> str:
        with self._lock:
            requests = dict(self.requests)
            errors = dict(self.errors)
            cache_admin = dict(self.cache_admin)
        lines = [
            "# HELP daedalus_requests_total Chat completion requests by outcome.",
            "# TYPE daedalus_requests_total counter",
        ]
        lines += [f'daedalus_requests_total{{outcome="{k}"}} {v}' for k, v in sorted(requests.items())]
        lines += [
            "# HELP daedalus_errors_total Server errors by type.",
            "# TYPE daedalus_errors_total counter",
        ]
        lines += [f'daedalus_errors_total{{type="{k}"}} {v}' for k, v in sorted(errors.items())]
        lines += [
            "# HELP daedalus_queue_requests Number of admitted active or queued requests.",
            "# TYPE daedalus_queue_requests gauge",
            f"daedalus_queue_requests {active}",
            "# HELP daedalus_queue_limit Maximum admitted active or queued requests.",
            "# TYPE daedalus_queue_limit gauge",
            f"daedalus_queue_limit {limit}",
            "# HELP daedalus_cache_entries Persistent prefix-cache entries.",
            "# TYPE daedalus_cache_entries gauge",
            f"daedalus_cache_entries {cache.get('entries', 0)}",
            "# HELP daedalus_cache_hits_total Prefix-cache hits.",
            "# TYPE daedalus_cache_hits_total counter",
            f"daedalus_cache_hits_total {cache.get('hits', 0)}",
            "# HELP daedalus_cache_misses_total Prefix-cache misses.",
            "# TYPE daedalus_cache_misses_total counter",
            f"daedalus_cache_misses_total {cache.get('misses', 0)}",
            "# HELP daedalus_cache_copy_seconds_total Time spent copying mutable KV caches.",
            "# TYPE daedalus_cache_copy_seconds_total counter",
            f"daedalus_cache_copy_seconds_total {cache.get('copy_seconds', 0.0)}",
            "# HELP daedalus_cache_lookup_seconds_total Time spent choosing a prefix-cache candidate.",
            "# TYPE daedalus_cache_lookup_seconds_total counter",
            f"daedalus_cache_lookup_seconds_total {cache.get('lookup_seconds', 0.0)}",
            "# HELP daedalus_cache_load_seconds_total Time spent materializing cache entries.",
            "# TYPE daedalus_cache_load_seconds_total counter",
            f"daedalus_cache_load_seconds_total {cache.get('load_seconds', 0.0)}",
            "# HELP daedalus_cache_lookup_candidates_total Prefix-cache candidates inspected.",
            "# TYPE daedalus_cache_lookup_candidates_total counter",
            f"daedalus_cache_lookup_candidates_total {cache.get('candidate_keys_examined', 0)}",
            "# HELP daedalus_prompt_token_cache_hits_total Chat-template token cache hits.",
            "# TYPE daedalus_prompt_token_cache_hits_total counter",
            f"daedalus_prompt_token_cache_hits_total {cache.get('tokenization', {}).get('hits', 0)}",
            "# HELP daedalus_prompt_token_cache_misses_total Chat-template token cache misses.",
            "# TYPE daedalus_prompt_token_cache_misses_total counter",
            f"daedalus_prompt_token_cache_misses_total {cache.get('tokenization', {}).get('misses', 0)}",
            "# HELP daedalus_shared_head_cache_hits_total Reused system/tool prompt head boundaries.",
            "# TYPE daedalus_shared_head_cache_hits_total counter",
            f"daedalus_shared_head_cache_hits_total {cache.get('shared_head', {}).get('hits', 0)}",
            "# HELP daedalus_thermal_level Current thermal level (0=nominal through 4=sleeping).",
            "# TYPE daedalus_thermal_level gauge",
            f"daedalus_thermal_level{{level=\"{thermal.lower()}\"}} {['NOMINAL', 'MODERATE', 'HEAVY', 'TRAPPING', 'SLEEPING'].index(thermal)}",
        ]
        lines += [f'daedalus_cache_admin_total{{action="{k}"}} {v}' for k, v in sorted(cache_admin.items())]
        return "\n".join(lines) + "\n"
