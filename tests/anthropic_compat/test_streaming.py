"""Test Anthropic-compatible streaming Messages API."""

from __future__ import annotations

import json
from typing import Any

import pytest

from http_client import LoggingHttpClient


def parse_sse_events(lines: list[str]) -> list[dict[str, Any]]:
    """Parse SSE lines into event dicts with 'event' and 'data' keys."""
    events: list[dict[str, Any]] = []
    current_event: str | None = None

    for line in lines:
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            data_str = line[6:]
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str
            events.append({"event": current_event, "data": data})
            current_event = None

    return events


@pytest.mark.capability("streaming")
class TestStreaming:
    """POST /v1/messages with stream=true."""

    def test_basic_stream(self, client: LoggingHttpClient, model: str) -> None:
        """Streaming should return correct SSE event sequence."""
        status, lines = client.request_stream(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": True,
            },
        )

        assert status == 200, f"Expected 200, got {status}"

        events = parse_sse_events(lines)
        assert len(events) > 0, "No SSE events received"

        event_types: list[Any] = [e["event"] for e in events]

        # Required event sequence
        assert "message_start" in event_types, "Missing 'message_start' event"
        assert "message_stop" in event_types, "Missing 'message_stop' event"

    def test_stream_event_structure(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Each event type should have the correct data structure."""
        status, lines = client.request_stream(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": True,
            },
        )

        assert status == 200

        events = parse_sse_events(lines)

        for event in events:
            if event["event"] == "message_start":
                data: Any = event["data"]
                assert data.get("type") == "message_start"
                assert "message" in data
                msg: Any = data["message"]
                assert "id" in msg
                assert msg.get("role") == "assistant"

            elif event["event"] == "content_block_start":
                data = event["data"]
                assert data.get("type") == "content_block_start"
                assert "content_block" in data

            elif event["event"] == "content_block_delta":
                data = event["data"]
                assert data.get("type") == "content_block_delta"
                assert "delta" in data

            elif event["event"] == "message_delta":
                data = event["data"]
                assert data.get("type") == "message_delta"

    def test_stream_collects_content(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Concatenating content_block_delta events should produce text."""
        status, lines = client.request_stream(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Say hello."}],
                "stream": True,
            },
        )

        assert status == 200

        events = parse_sse_events(lines)
        collected_text = ""

        for event in events:
            if event["event"] == "content_block_delta":
                delta: Any = event["data"].get("delta", {})
                if delta.get("type") == "text_delta":
                    collected_text += delta.get("text", "")

        assert len(collected_text) > 0, "No text content collected from stream"
