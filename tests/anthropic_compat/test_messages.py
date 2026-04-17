"""Test Anthropic-compatible Messages API (non-streaming)."""

from __future__ import annotations

from typing import Any

import pytest

from http_client import LoggingHttpClient


@pytest.mark.capability("chat")
class TestMessages:
    """POST /v1/messages — basic non-streaming requests."""

    def test_simple_message(self, client: LoggingHttpClient, model: str) -> None:
        """A simple user message should return a well-formed response."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        # Top-level required fields
        assert "id" in body, "Response missing 'id'"
        assert body.get("type") == "message"
        assert body.get("role") == "assistant"
        assert "model" in body

        # content blocks
        assert "content" in body
        content: Any = body["content"]
        assert isinstance(content, list) and len(content) > 0

        text_block: Any = content[0]
        assert text_block.get("type") == "text"
        assert isinstance(text_block.get("text"), str)
        assert len(text_block["text"]) > 0

        assert body.get("stop_reason") in ("end_turn", "max_tokens", "stop_sequence")

    def test_system_message(self, client: LoggingHttpClient, model: str) -> None:
        """Top-level system parameter should work."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        content: Any = body["content"]
        assert content[0]["type"] == "text"

    def test_multi_turn(self, client: LoggingHttpClient, model: str) -> None:
        """Multi-turn conversation should be supported."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        content: Any = body["content"]
        text: Any = content[0]["text"]
        assert isinstance(text, str) and len(text) > 0

    def test_usage_field(self, client: LoggingHttpClient, model: str) -> None:
        """Response should contain usage statistics."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert status == 200
        assert isinstance(body, dict)
        assert "usage" in body

        usage: Any = body["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert isinstance(usage["input_tokens"], int)
        assert isinstance(usage["output_tokens"], int)

    def test_max_tokens_respected(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """max_tokens should limit response length."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 5,
                "messages": [
                    {"role": "user", "content": "Write a long story about a cat."},
                ],
            },
        )

        assert status == 200
        assert isinstance(body, dict)
        # With very low max_tokens, stop_reason may be max_tokens
        assert body.get("stop_reason") in ("end_turn", "max_tokens")

    def test_temperature(self, client: LoggingHttpClient, model: str) -> None:
        """temperature parameter should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Say hello."}],
                "temperature": 0.0,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_top_p(self, client: LoggingHttpClient, model: str) -> None:
        """top_p parameter should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Say hello."}],
                "top_p": 0.9,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_top_k(self, client: LoggingHttpClient, model: str) -> None:
        """top_k parameter should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Say hello."}],
                "top_k": 40,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_metadata_user_id(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """metadata.user_id should be accepted for tracking purposes."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Say hello."}],
                "metadata": {"user_id": "test-user-123"},
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_stop_sequences(self, client: LoggingHttpClient, model: str) -> None:
        """stop_sequences parameter should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Count from 1 to 10."}],
                "stop_sequences": ["5"],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
