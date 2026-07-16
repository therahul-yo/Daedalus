"""Prompt normalization, chat-template rendering, and tool-schema validation.

Maps OpenAI wire-format quirks onto what HF chat templates accept, builds
(and caches) the prompt token ids for a request, and validates the OpenAI
``tools`` array before it reaches the engine.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from daedalus.server.state import ServerState

logger = logging.getLogger(__name__)

_TEMPLATE_ROLES = {"system", "user", "assistant", "tool"}


def normalize_messages(messages: List[dict]) -> List[dict]:
    """Map OpenAI wire-format quirks onto what HF chat templates accept.

    - role "developer" (newer OpenAI convention, sent by pi) -> "system";
      other unknown roles -> "user" (templates raise on unknown roles)
    - content parts [{type: "text", text: ...}] -> flattened string
    - assistant tool_calls function.arguments JSON string -> dict (Qwen-style
      templates iterate arguments as a mapping)
    - content: null -> ""
    """
    out = []
    for msg in messages:
        m = dict(msg)
        role = m.get("role")
        if role not in _TEMPLATE_ROLES:
            m["role"] = "system" if role == "developer" else "user"
            if role != "developer":
                logger.warning("unknown message role %r -> user", role)
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        elif content is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                call = json.loads(json.dumps(call))  # deep copy
                fn = call.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {"_raw": args}
                calls.append(call)
            m["tool_calls"] = calls
        out.append(m)
    return out


def build_prompt_tokens(
    state: ServerState, messages: List[dict], tools: Optional[List[dict]] = None
) -> List[int]:
    kwargs = {"add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    normalized = normalize_messages(messages)
    return state.token_cache.get_or_build(
        normalized, tools,
        lambda: state.engine.tokenizer.apply_chat_template(normalized, **kwargs),
    )


def validate_tools(tools: Any) -> Optional[str]:
    """Return a public validation error for malformed OpenAI tool schemas."""
    if tools is None:
        return None
    if not isinstance(tools, list):
        return "tools must be an array"
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            return "each tool must be a function definition"
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"].strip():
            return "each tool function must have a non-empty name"
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            return "tool function parameters must be an object"
    return None
