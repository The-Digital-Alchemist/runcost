"""SSE stream parser for extracting usage from Anthropic streaming responses.

Claude Code always uses streaming. The proxy forwards SSE chunks in real-time
while parsing message_start and message_delta events to extract usage stats
for accurate cost tracking.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StreamUsage:
    """Aggregated usage from an SSE stream."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class SSEParser:
    """Incremental SSE parser that extracts usage from Anthropic streaming events.

    Usage:
        parser = SSEParser()
        for chunk in stream:
            parser.feed(chunk)
            yield chunk  # forward to client
        usage = parser.usage  # aggregated usage after stream ends
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._current_event: Optional[str] = None
        self._current_data: list[str] = []
        self.usage = StreamUsage()

    def feed(self, chunk: bytes) -> None:
        """Feed a raw SSE chunk (bytes) into the parser."""
        try:
            text = chunk.decode("utf-8", errors="replace")
        except Exception:
            return
        self._buffer += text
        self._process_buffer()

    def _process_buffer(self) -> None:
        """Process buffered text, extracting complete SSE events."""
        while "\n\n" in self._buffer:
            event_block, self._buffer = self._buffer.split("\n\n", 1)
            self._parse_event_block(event_block)

    def _parse_event_block(self, block: str) -> None:
        """Parse a single SSE event block."""
        event_type: Optional[str] = None
        data_lines: list[str] = []

        for line in block.split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if not data_lines:
            return

        data_str = "\n".join(data_lines)
        if not data_str or data_str == "[DONE]":
            return

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return

        self._handle_event(event_type, data)

    def _handle_event(self, event_type: Optional[str], data: dict) -> None:
        """Extract usage information from parsed SSE events."""
        msg_type = data.get("type", "")

        # message_start contains input token counts and model
        if msg_type == "message_start" or event_type == "message_start":
            message = data.get("message", {})
            self.usage.model = message.get("model", self.usage.model)
            usage = message.get("usage", {})
            self.usage.input_tokens = usage.get("input_tokens", 0)
            self.usage.cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
            self.usage.cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)

        # message_delta contains output token count (final event)
        elif msg_type == "message_delta" or event_type == "message_delta":
            usage = data.get("usage", {})
            self.usage.output_tokens = usage.get("output_tokens", self.usage.output_tokens)

    def finalize(self) -> StreamUsage:
        """Process any remaining buffer and return final usage."""
        if self._buffer.strip():
            self._parse_event_block(self._buffer)
            self._buffer = ""
        return self.usage


def parse_openai_sse_usage(chunk: bytes, accumulated: dict) -> None:
    """Parse OpenAI SSE chunks to extract usage. OpenAI includes usage in the final chunk.

    Args:
        chunk: Raw SSE chunk bytes.
        accumulated: Dict to accumulate usage into (mutated in place).
    """
    try:
        text = chunk.decode("utf-8", errors="replace")
    except Exception:
        return

    for line in text.split("\n"):
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # OpenAI includes usage in the final chunk when stream_options.include_usage=True
        usage = data.get("usage")
        if usage:
            accumulated["prompt_tokens"] = usage.get("prompt_tokens", 0)
            accumulated["completion_tokens"] = usage.get("completion_tokens", 0)
            accumulated["total_tokens"] = usage.get("total_tokens", 0)
