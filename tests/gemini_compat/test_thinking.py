"""Test Gemini-compatible thinking (extended reasoning) support.

A model supports thinking only if it both (a) accepts the
``generationConfig.thinkingConfig`` parameter without 4xx and (b) actually
reasons — proved by ``usageMetadata.thoughtsTokenCount > 0`` and (when
``includeThoughts: true``) a part with ``thought: true`` in the response.
Silently dropping ``thinkingConfig`` and returning a plain completion is a
common pseudo-compatibility failure that 200 OK alone does not catch.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest

from http_client import LoggingHttpClient

PROMPT = (
    "What is 47 * 53? Think step by step, then give the final number."
)
# -1 lets the model choose its own budget (dynamic thinking). A concrete cap
# would also work but varies per snapshot; -1 is the most portable signal.
THINKING_BUDGET = -1


@pytest.mark.capability("thinking")
class TestThinking:
    """POST /v1beta/models/{model}:generateContent with ``thinkingConfig``."""

    def test_thinking_non_streaming(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """``usageMetadata.thoughtsTokenCount`` must be > 0, and with
        ``includeThoughts: true`` at least one part must carry ``thought: true``."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": PROMPT}]},
                ],
                "generationConfig": {
                    "thinkingConfig": {
                        "thinkingBudget": THINKING_BUDGET,
                        "includeThoughts": True,
                    },
                },
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        usage: Any = body.get("usageMetadata")
        assert isinstance(usage, dict), "Response missing `usageMetadata`"
        thoughts: Any = usage.get("thoughtsTokenCount")
        assert isinstance(thoughts, int) and thoughts > 0, (
            "`thoughtsTokenCount` is 0 or missing — the model may have "
            f"silently ignored `thinkingConfig`. Got usage: {usage!r}"
        )

        parts: Any = body["candidates"][0]["content"]["parts"]
        assert isinstance(parts, list) and len(parts) > 0
        thought_parts = [
            p for p in parts if isinstance(p, dict) and p.get("thought") is True
        ]
        assert len(thought_parts) > 0, (
            "`includeThoughts: true` set but no part with `thought: true` — "
            f"got parts: {parts!r}"
        )

    @pytest.mark.capability("streaming")
    def test_thinking_streaming(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Streaming must surface thinking via either a ``thought: true`` part
        in some chunk or a final ``thoughtsTokenCount > 0`` in usageMetadata."""
        status, lines = client.request_stream(
            "POST",
            f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": PROMPT}]},
                ],
                "generationConfig": {
                    "thinkingConfig": {
                        "thinkingBudget": THINKING_BUDGET,
                        "includeThoughts": True,
                    },
                },
            },
        )

        assert status == 200, f"Expected 200, got {status}"

        chunks: list[dict[str, Any]] = []
        for raw_line in lines:
            if not raw_line.startswith("data: "):
                continue
            with contextlib.suppress(json.JSONDecodeError):
                chunks.append(json.loads(raw_line[6:]))

        saw_thought_part = False
        thoughts_token_final: int | None = None
        for chunk in chunks:
            candidates: Any = chunk.get("candidates") or []
            if candidates:
                parts: Any = (
                    candidates[0].get("content", {}).get("parts") or []
                )
                for part in parts:
                    if isinstance(part, dict) and part.get("thought") is True:
                        saw_thought_part = True
                        break

            usage: Any = chunk.get("usageMetadata")
            if isinstance(usage, dict):
                tt: Any = usage.get("thoughtsTokenCount")
                if isinstance(tt, int):
                    thoughts_token_final = tt

        assert saw_thought_part or (
            thoughts_token_final is not None and thoughts_token_final > 0
        ), (
            "Stream had no `thought: true` part and no positive "
            "`thoughtsTokenCount` in usageMetadata — the model may have "
            "silently ignored `thinkingConfig`"
        )
