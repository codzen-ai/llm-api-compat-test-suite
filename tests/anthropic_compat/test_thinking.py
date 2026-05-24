"""Test Anthropic-compatible extended thinking.

A model supports thinking only if it both (a) accepts the ``thinking`` request
parameter without 4xx and (b) actually emits a ``type: "thinking"`` content
block (non-streaming) or ``thinking_delta`` events (streaming). Silently
ignoring the parameter and returning text-only output is a common
pseudo-compatibility failure mode that 200-OK alone does not catch.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from http_client import LoggingHttpClient

# Extended thinking requires max_tokens > budget_tokens. The prompt asks for a
# short answer to a reasoning-heavy question so the model is likely to spend
# tokens in the thinking channel rather than rambling in visible text.
THINKING_BUDGET = 1024
MAX_TOKENS = THINKING_BUDGET + 256
PROMPT = (
    "What is 47 * 53? Think step by step, then give the final number."
)


@pytest.mark.capability("thinking")
class TestThinking:
    """POST /v1/messages with ``thinking: {type: enabled, ...}``."""

    def test_thinking_non_streaming(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Response content must include at least one ``type: thinking`` block."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": MAX_TOKENS,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": THINKING_BUDGET,
                },
                "messages": [{"role": "user", "content": PROMPT}],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        content: Any = body.get("content")
        assert isinstance(content, list) and len(content) > 0

        thinking_blocks = [
            b for b in content
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) > 0, (
            "Response contained no `type: thinking` content block — "
            "the model may have silently ignored the `thinking` parameter. "
            f"Got block types: {[b.get('type') for b in content]}"
        )
        first = thinking_blocks[0]
        thinking_text: Any = first.get("thinking")
        assert isinstance(thinking_text, str) and len(thinking_text) > 0, (
            "`thinking` block missing non-empty `thinking` text field"
        )

    @pytest.mark.capability("streaming")
    def test_thinking_streaming(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Stream must include at least one ``thinking_delta`` event."""
        status, lines = client.request_stream(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": MAX_TOKENS,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": THINKING_BUDGET,
                },
                "messages": [{"role": "user", "content": PROMPT}],
                "stream": True,
            },
        )

        assert status == 200, f"Expected 200, got {status}"

        saw_thinking_delta = False
        for raw_line in lines:
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload:
                continue
            try:
                chunk: Any = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("type") != "content_block_delta":
                continue
            delta: Any = chunk.get("delta")
            if (
                isinstance(delta, dict)
                and delta.get("type") == "thinking_delta"
                and isinstance(delta.get("thinking"), str)
                and delta["thinking"]
            ):
                saw_thinking_delta = True
                break

        assert saw_thinking_delta, (
            "Stream contained no `thinking_delta` event — "
            "the model may have silently ignored the `thinking` parameter"
        )
