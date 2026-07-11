"""Reasoning ("think") stream separation.

Qwen3.5-style chat templates end the generation prompt with an opened
``<think>`` block, so the model's output *starts inside* a reasoning span and
the client never sees the opening tag. Without handling, the raw chain of
thought (and a stray ``</think>``) leaks into the visible reply — exactly what
pi showed in testing.

``ThinkStreamFilter`` splits streamed text into (reasoning, content) pairs.
The server emits reasoning as OpenAI-style ``reasoning_content`` deltas, which
reasoning-aware clients (pi, OpenCode) render dimmed/collapsed, and plain
clients simply ignore.
"""

from __future__ import annotations

from typing import Tuple

from daedalus.tools import _held_marker_prefix


class ThinkStreamFilter:
    """Feed streamed text; get (reasoning, content) splits.

    ``initially_thinking=True`` when the prompt itself opened a think block
    (detected by the server from the templated prompt's tail).
    """

    def __init__(
        self,
        start_marker: str = "<think>",
        end_marker: str = "</think>",
        initially_thinking: bool = False,
    ) -> None:
        self.start = start_marker
        self.end = end_marker
        self._thinking = initially_thinking
        self._buf = ""
        self._content_started = False

    def feed(self, segment: str) -> Tuple[str, str]:
        self._buf += segment
        reasoning_out: list[str] = []
        content_out: list[str] = []

        while True:
            if self._thinking:
                idx = self._buf.find(self.end)
                if idx < 0:
                    held = _held_marker_prefix(self._buf, self.end)
                    emit = len(self._buf) - held
                    reasoning_out.append(self._buf[:emit])
                    self._buf = self._buf[emit:]
                    break
                reasoning_out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(self.end) :]
                self._thinking = False
            else:
                idx = self._buf.find(self.start)
                if idx < 0:
                    held = _held_marker_prefix(self._buf, self.start)
                    emit = len(self._buf) - held
                    chunk = self._buf[:emit]
                    self._buf = self._buf[emit:]
                    if not self._content_started:
                        # Swallow the whitespace the template leaves after
                        # a closed think block before real content starts.
                        chunk = chunk.lstrip("\n")
                        if chunk:
                            self._content_started = True
                    content_out.append(chunk)
                    break
                chunk = self._buf[:idx]
                if not self._content_started:
                    chunk = chunk.lstrip("\n")
                    if chunk:
                        self._content_started = True
                content_out.append(chunk)
                self._buf = self._buf[idx + len(self.start) :]
                self._thinking = True

        return "".join(reasoning_out), "".join(content_out)

    def finalize(self) -> Tuple[str, str]:
        buf, self._buf = self._buf, ""
        if not buf:
            return "", ""
        if self._thinking:
            # Model hit EOS before closing the think block: everything
            # buffered is reasoning.
            return buf, ""
        if not self._content_started:
            buf = buf.lstrip("\n")
        return "", buf


class NoThinkFilter:
    """Model/template without reasoning markers: pass everything through."""

    def feed(self, segment: str) -> Tuple[str, str]:
        return "", segment

    def finalize(self) -> Tuple[str, str]:
        return "", ""
