"""Streaming tool-call extraction.

Wraps mlx-lm's per-model tool parser (auto-detected on the TokenizerWrapper:
``tool_call_start``/``tool_call_end`` markers + ``tool_parser`` function) in
a stream filter that:

- passes normal content through as it streams,
- holds back any text that might be the start of a tool-call marker (markers
  can be split across decode steps),
- buffers marker-delimited regions and parses them into OpenAI-format tool
  calls when the end marker arrives.

Client-compat rules (from OpenCode/pi research): tool-call deltas must carry
an explicit ``index``; a chunk must NEVER contain ``"tool_calls": []``.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: str  # JSON-encoded string (OpenAI wire format)
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")

    def as_openai(self, index: int) -> dict:
        return {
            "index": index,
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


def _normalize_parsed(parsed: Any) -> List[ToolCall]:
    """mlx-lm parsers return {name, arguments} or a list of them."""
    items = parsed if isinstance(parsed, list) else [parsed]
    calls = []
    for item in items:
        args = item.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args)
        calls.append(ToolCall(name=item["name"], arguments=args))
    return calls


def _held_marker_prefix(text: str, marker: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of
    ``marker`` (text that must be held back until more arrives)."""
    max_check = min(len(text), len(marker) - 1)
    for n in range(max_check, 0, -1):
        if text.endswith(marker[:n]):
            return n
    return 0


class ToolCallStreamFilter:
    """Feed streamed text segments; get (safe_content, completed_tool_calls)."""

    def __init__(
        self,
        start_marker: str,
        end_marker: str,
        parser: Callable[[str, Any], Any],
        tools: Optional[List[dict]] = None,
    ) -> None:
        self.start = start_marker
        self.end = end_marker
        self.parser = parser
        self.tools = tools
        self._buf = ""
        self._in_call = False

    def feed(self, segment: str) -> Tuple[str, List[ToolCall]]:
        self._buf += segment
        content_out: List[str] = []
        calls_out: List[ToolCall] = []

        while True:
            if not self._in_call:
                idx = self._buf.find(self.start)
                if idx >= 0:
                    content_out.append(self._buf[:idx])
                    self._buf = self._buf[idx:]
                    self._in_call = True
                    continue
                held = _held_marker_prefix(self._buf, self.start)
                emit_upto = len(self._buf) - held
                content_out.append(self._buf[:emit_upto])
                self._buf = self._buf[emit_upto:]
                break
            else:
                end_idx = self._buf.find(self.end, len(self.start))
                if end_idx < 0:
                    break  # wait for more of the call
                call_end = end_idx + len(self.end)
                raw = self._buf[:call_end]
                self._buf = self._buf[call_end:]
                self._in_call = False
                calls_out.extend(self._parse(raw))
                continue

        return "".join(content_out), calls_out

    def finalize(self) -> Tuple[str, List[ToolCall]]:
        """Flush at end of generation. An unterminated tool-call region is
        parsed on a best-effort basis (models sometimes stop at EOS before
        emitting the end marker)."""
        buf, self._buf = self._buf, ""
        if not buf:
            return "", []
        if self._in_call:
            self._in_call = False
            calls = self._parse(buf)
            if calls:
                return "", calls
            return buf, []  # unparseable: surface as content, don't drop
        return buf, []

    def _parse(self, raw: str) -> List[ToolCall]:
        inner = raw
        if inner.startswith(self.start):
            inner = inner[len(self.start) :]
        if inner.endswith(self.end):
            inner = inner[: -len(self.end)]
        for candidate in (inner, raw):
            try:
                return _normalize_parsed(self.parser(candidate, self.tools))
            except Exception:
                continue
        logger.warning("unparseable tool call (%d chars): %.120r", len(raw), raw)
        return []


class PassthroughFilter:
    """Used when the model/tokenizer has no tool-calling support."""

    def feed(self, segment: str) -> Tuple[str, List[ToolCall]]:
        return segment, []

    def finalize(self) -> Tuple[str, List[ToolCall]]:
        return "", []


def make_stream_filter(tokenizer: Any, tools: Optional[List[dict]]):
    if (
        tools
        and getattr(tokenizer, "has_tool_calling", False)
        and tokenizer.tool_call_start
    ):
        return ToolCallStreamFilter(
            start_marker=tokenizer.tool_call_start,
            end_marker=tokenizer.tool_call_end,
            parser=tokenizer.tool_parser,
            tools=tools,
        )
    return PassthroughFilter()
