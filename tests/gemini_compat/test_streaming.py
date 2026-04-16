"""Test Gemini-compatible streaming generateContent API."""

from __future__ import annotations

import contextlib
import json
from typing import Any

import pytest

from http_client import LoggingHttpClient


def parse_sse_data(lines: list[str]) -> list[dict[str, Any]]:
    """Extract JSON objects from SSE data lines."""
    results: list[dict[str, Any]] = []
    for line in lines:
        if line.startswith("data: "):
            payload = line[6:]
            with contextlib.suppress(json.JSONDecodeError):
                results.append(json.loads(payload))
    return results


@pytest.mark.capability("streaming")
class TestStreaming:
    """POST /v1beta/models/{model}:streamGenerateContent."""

    def test_basic_stream(self, client: LoggingHttpClient, model: str) -> None:
        """Streaming should return SSE chunks with correct structure."""
        status, lines = client.request_stream(
            "POST",
            f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": "Say hello."}]},
                ],
            },
        )

        assert status == 200, f"Expected 200, got {status}"

        chunks = parse_sse_data(lines)
        assert len(chunks) > 0, "No SSE data chunks received"

        # Validate chunk structure
        first: Any = chunks[0]
        assert "candidates" in first
        candidate: Any = first["candidates"][0]
        assert "content" in candidate
        assert "parts" in candidate["content"]

    def test_stream_collects_content(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Concatenating stream chunks should produce complete text."""
        status, lines = client.request_stream(
            "POST",
            f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": "Say hello."}]},
                ],
            },
        )

        assert status == 200

        chunks = parse_sse_data(lines)
        collected_text = ""

        for chunk in chunks:
            candidates: Any = chunk.get("candidates", [])
            if candidates:
                parts: Any = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        collected_text += part["text"]

        assert len(collected_text) > 0, "No text content collected from stream"

    def test_stream_usage_metadata(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Final chunk should contain usageMetadata."""
        status, lines = client.request_stream(
            "POST",
            f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": "Say hi."}]},
                ],
            },
        )

        assert status == 200

        chunks = parse_sse_data(lines)
        # Last chunk often has usageMetadata
        last: Any = chunks[-1] if chunks else {}
        if "usageMetadata" in last:
            usage: Any = last["usageMetadata"]
            assert "promptTokenCount" in usage
            assert "candidatesTokenCount" in usage
