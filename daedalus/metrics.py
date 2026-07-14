"""Small dependency-free Prometheus metrics registry for the local server."""

from __future__ import annotations

import bisect
import threading
from collections import Counter
from typing import List, Sequence


class Histogram:
    """Fixed-bucket Prometheus histogram (cumulative, with +Inf/_sum/_count)."""

    def __init__(self, buckets: Sequence[float]) -> None:
        self.uppers = sorted(buckets)
        self.counts = [0] * (len(self.uppers) + 1)  # last slot = +Inf
        self.total = 0.0
        self.n = 0

    def observe(self, value: float) -> None:
        self.counts[bisect.bisect_left(self.uppers, value)] += 1
        self.total += value
        self.n += 1

    def render(self, name: str, help_text: str) -> List[str]:
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        cumulative = 0
        for upper, count in zip(self.uppers, self.counts):
            cumulative += count
            lines.append(f'{name}_bucket{{le="{upper}"}} {cumulative}')
        lines.append(f'{name}_bucket{{le="+Inf"}} {self.n}')
        lines.append(f"{name}_sum {self.total}")
        lines.append(f"{name}_count {self.n}")
        return lines


# Prefill on a paced fanless Air legitimately spans 100ms (warm hit) to
# minutes (cold 40k prompt at MODERATE) — buckets must cover both regimes.
TTFT_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0)
QUEUE_WAIT_BUCKETS = (0.01, 0.05, 0.25, 1.0, 5.0, 15.0, 60.0, 180.0)
DECODE_TPS_BUCKETS = (2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0)


class ServerMetrics:
    """Thread-safe counters and gauges exported by ``/metrics``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = Counter()
        self.errors = Counter()
        self.cache_admin = Counter()
        self.ttft = Histogram(TTFT_BUCKETS)
        self.queue_wait = Histogram(QUEUE_WAIT_BUCKETS)
        self.decode_tps = Histogram(DECODE_TPS_BUCKETS)

    def inc_request(self, outcome: str) -> None:
        with self._lock:
            self.requests[outcome] += 1

    def inc_error(self, kind: str) -> None:
        with self._lock:
            self.errors[kind] += 1

    def inc_cache_admin(self, action: str) -> None:
        with self._lock:
            self.cache_admin[action] += 1

    def observe_ttft(self, seconds: float) -> None:
        with self._lock:
            self.ttft.observe(seconds)

    def observe_queue_wait(self, seconds: float) -> None:
        with self._lock:
            self.queue_wait.observe(seconds)

    def observe_decode_tps(self, tps: float) -> None:
        if tps > 0:
            with self._lock:
                self.decode_tps.observe(tps)

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
        with self._lock:
            lines += self.ttft.render(
                "daedalus_ttft_seconds", "Time to first generated token."
            )
            lines += self.queue_wait.render(
                "daedalus_queue_wait_seconds",
                "Time between admission and engine-lock acquisition.",
            )
            lines += self.decode_tps.render(
                "daedalus_decode_tokens_per_second", "Decode throughput per request."
            )
        return "\n".join(lines) + "\n"
