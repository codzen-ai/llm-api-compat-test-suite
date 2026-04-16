"""Test OpenAI-compatible streaming chat completions."""

from __future__ import annotations

import json
from typing import Any

import pytest

from http_client import LoggingHttpClient


def parse_sse_data(lines: list[str]) -> list[dict[str, Any] | str]:
    """Extract data payloads from SSE lines."""
    results: list[dict[str, Any] | str] = []
    for line in lines:
        if line.startswith("data: "):
            payload = line[6:]
            if payload.strip() == "[DONE]":
                results.append("[DONE]")
            else:
                try:
                    results.append(json.loads(payload))
                except json.JSONDecodeError:
                    results.append(payload)
    return results


@pytest.mark.capability("streaming")
class TestChatStreaming:
    """POST /v1/chat/completions with stream=true."""

    def test_basic_stream(self, client: LoggingHttpClient, model: str) -> None:
        """Streaming should return SSE chunks with correct structure."""
        status, lines = client.request_stream(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": True,
            },
        )

        assert status == 200, f"Expected 200, got {status}"

        chunks = parse_sse_data(lines)
        assert len(chunks) > 0, "No SSE data chunks received"

        # Last item should be [DONE]
        assert chunks[-1] == "[DONE]", "Stream should end with [DONE]"

        # Parse actual data chunks (exclude [DONE])
        data_chunks = [c for c in chunks if isinstance(c, dict)]
        assert len(data_chunks) > 0, "No data chunks received"

        # Validate first chunk structure
        first = data_chunks[0]
        assert "id" in first
        assert "object" in first
        assert first["object"] == "chat.completion.chunk"
        assert "choices" in first

    def test_stream_delta_content(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Stream chunks should contain delta with content or role."""
        status, lines = client.request_stream(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": True,
            },
        )

        assert status == 200

        chunks = parse_sse_data(lines)
        data_chunks = [c for c in chunks if isinstance(c, dict)]

        collected_content = ""
        saw_role = False
        saw_finish_reason = False

        for chunk in data_chunks:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            if "role" in delta:
                assert delta["role"] == "assistant"
                saw_role = True
            if "content" in delta:
                collected_content += delta["content"]
            if choice.get("finish_reason") is not None:
                saw_finish_reason = True

        assert saw_role, "No chunk contained role='assistant'"
        assert len(collected_content) > 0, "No content received in stream"
        assert saw_finish_reason, "No chunk contained finish_reason"

    def test_stream_usage(self, client: LoggingHttpClient, model: str) -> None:
        """stream_options include_usage should return usage in final chunk."""
        status, lines = client.request_stream(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hi."}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

        assert status == 200

        chunks = parse_sse_data(lines)
        data_chunks = [c for c in chunks if isinstance(c, dict)]

        # The last data chunk (before [DONE]) should contain usage
        last_data = data_chunks[-1]
        if "usage" in last_data:
            usage = last_data["usage"]
            assert "prompt_tokens" in usage
            assert "completion_tokens" in usage
            assert "total_tokens" in usage
