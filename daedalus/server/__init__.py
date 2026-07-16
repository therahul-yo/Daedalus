"""OpenAI-compatible server for daedalus.

Endpoints: /v1/chat/completions (SSE + non-stream), /v1/models, /health,
/v1/cache/stats.

This package is a pure structural split of the original ``daedalus.server``
module; every name importable from ``daedalus.server`` before the split is
re-exported here unchanged. The submodules are:

- ``profiles``   — memory profiles, swap-admission math, config discovery
- ``http_utils`` — request-id middleware, rate limiters, SSE/body helpers
- ``state``      — ``ServerState`` plus the prompt/head caches
- ``prompts``    — message normalization, prompt building, tool validation
- ``generation`` — ``_Generation`` engine worker and SSE streaming pump
- ``app``        — ``create_app`` FastAPI application factory
"""

from __future__ import annotations

import logging

from daedalus.server.app import (
    BackgroundTask as BackgroundTask,
    CORSMiddleware as CORSMiddleware,
    FastAPI as FastAPI,
    Header as Header,
    JSONResponse as JSONResponse,
    PlainTextResponse as PlainTextResponse,
    StreamingResponse as StreamingResponse,
    asynccontextmanager as asynccontextmanager,
    audit_logger as audit_logger,
    create_app as create_app,
    hmac as hmac,
    math as math,
    run_in_threadpool as run_in_threadpool,
)
from daedalus.server.generation import (
    CHECKPOINT_EVERY_TOKENS as CHECKPOINT_EVERY_TOKENS,
    CHECKPOINT_MIN_INTERVAL_S as CHECKPOINT_MIN_INTERVAL_S,
    CHECKPOINT_MIN_JOB_TOKENS as CHECKPOINT_MIN_JOB_TOKENS,
    KEEPALIVE_INTERVAL_S as KEEPALIVE_INTERVAL_S,
    AsyncGenerator as AsyncGenerator,
    PrefillAborted as PrefillAborted,
    ThinkStreamFilter as ThinkStreamFilter,
    _Generation as _Generation,
    _stream_response as _stream_response,
    asyncio as asyncio,
    concurrent as concurrent,
    make_stream_filter as make_stream_filter,
)
from daedalus.server.http_utils import (
    BaseHTTPMiddleware as BaseHTTPMiddleware,
    ClientRateLimiter as ClientRateLimiter,
    GlobalRateLimiter as GlobalRateLimiter,
    Request as Request,
    RequestBodyTooLarge as RequestBodyTooLarge,
    RequestIdMiddleware as RequestIdMiddleware,
    StarletteRequest as StarletteRequest,
    StarletteResponse as StarletteResponse,
    _chunk as _chunk,
    _sse as _sse,
    contextvars as contextvars,
    read_json_body as read_json_body,
    request_client_ip as request_client_ip,
    request_id_var as request_id_var,
    uuid as uuid,
)
from daedalus.server.profiles import (
    KV_QUANT_OVERHEAD as KV_QUANT_OVERHEAD,
    MODEL_MEMORY_CEILING_GB as MODEL_MEMORY_CEILING_GB,
    MODEL_PROFILES as MODEL_PROFILES,
    SWAP_SAFETY_GB as SWAP_SAFETY_GB,
    Any as Any,
    List as List,
    ModelProfile as ModelProfile,
    Optional as Optional,
    Path as Path,
    _cfg_get as _cfg_get,
    _model_config_owners as _model_config_owners,
    dataclass as dataclass,
    derive_model_profile as derive_model_profile,
    estimate_kv_cache_bytes as estimate_kv_cache_bytes,
    json as json,
    model_context_limit as model_context_limit,
    model_fits as model_fits,
)
from daedalus.server.prompts import (
    _TEMPLATE_ROLES as _TEMPLATE_ROLES,
    build_prompt_tokens as build_prompt_tokens,
    normalize_messages as normalize_messages,
    validate_tools as validate_tools,
)
from daedalus.server.state import (
    Callable as Callable,
    Engine as Engine,
    FifoLock as FifoLock,
    OrderedDict as OrderedDict,
    PrefixCacheStore as PrefixCacheStore,
    PromptTokenCache as PromptTokenCache,
    ServerMetrics as ServerMetrics,
    ServerState as ServerState,
    SharedHeadIndex as SharedHeadIndex,
    field as field,
    psutil as psutil,
    threading as threading,
    time as time,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Application factory
    "create_app",
    # Memory profiles / admission / config discovery
    "KV_QUANT_OVERHEAD",
    "ModelProfile",
    "MODEL_MEMORY_CEILING_GB",
    "SWAP_SAFETY_GB",
    "MODEL_PROFILES",
    "derive_model_profile",
    "model_fits",
    "model_context_limit",
    "estimate_kv_cache_bytes",
    # HTTP utilities
    "RequestIdMiddleware",
    "GlobalRateLimiter",
    "ClientRateLimiter",
    "RequestBodyTooLarge",
    "read_json_body",
    "request_client_ip",
    "request_id_var",
    # Runtime state and caches
    "ServerState",
    "PromptTokenCache",
    "SharedHeadIndex",
    # Prompt handling
    "normalize_messages",
    "build_prompt_tokens",
    "validate_tools",
    # Generation constants
    "KEEPALIVE_INTERVAL_S",
    "CHECKPOINT_EVERY_TOKENS",
    "CHECKPOINT_MIN_JOB_TOKENS",
    "CHECKPOINT_MIN_INTERVAL_S",
]
