"""Test OpenAI-compatible chat completions (non-streaming)."""

from __future__ import annotations

import json

import pytest

from http_client import LoggingHttpClient


@pytest.mark.capability("chat")
class TestChatBasic:
    """POST /v1/chat/completions — basic non-streaming requests."""

    def test_simple_message(self, client: LoggingHttpClient, model: str) -> None:
        """A simple user message should return a well-formed response."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        # Top-level required fields
        assert "id" in body, "Response missing 'id'"
        assert "object" in body, "Response missing 'object'"
        assert body["object"] == "chat.completion"
        assert "created" in body, "Response missing 'created'"
        assert "model" in body, "Response missing 'model'"

        # choices
        assert "choices" in body, "Response missing 'choices'"
        choices = body["choices"]
        assert isinstance(choices, list) and len(choices) > 0

        choice = choices[0]
        assert "index" in choice
        assert "message" in choice

        message = choice["message"]
        assert message.get("role") == "assistant"
        assert isinstance(message.get("content"), str)
        assert len(message["content"]) > 0

        assert "finish_reason" in choice

    def test_system_message(self, client: LoggingHttpClient, model: str) -> None:
        """System + user message should work correctly."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello."},
                ],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        assert body["choices"][0]["message"]["role"] == "assistant"

    def test_multi_turn(self, client: LoggingHttpClient, model: str) -> None:
        """Multi-turn conversation should be supported."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        content = body["choices"][0]["message"]["content"]
        assert isinstance(content, str) and len(content) > 0

    def test_usage_field(self, client: LoggingHttpClient, model: str) -> None:
        """Response should contain usage statistics."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert status == 200
        assert isinstance(body, dict)
        assert "usage" in body, "Response missing 'usage'"

        usage = body["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)

    def test_max_tokens(self, client: LoggingHttpClient, model: str) -> None:
        """max_tokens parameter should limit response length."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Write a long story about a cat."},
                ],
                "max_tokens": 10,
            },
        )

        assert status == 200
        assert isinstance(body, dict)
        # With max_tokens=10, completion tokens should be small
        usage = body.get("usage", {})
        if usage:
            assert usage.get("completion_tokens", 0) <= 20  # Allow some slack

    def test_temperature(self, client: LoggingHttpClient, model: str) -> None:
        """temperature parameter should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "temperature": 0.0,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_n_parameter(self, client: LoggingHttpClient, model: str) -> None:
        """n parameter should return multiple choices."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "n": 2,
            },
        )

        assert status == 200
        assert isinstance(body, dict)
        assert len(body["choices"]) == 2
        assert body["choices"][0]["index"] == 0
        assert body["choices"][1]["index"] == 1

    def test_stop_sequence(self, client: LoggingHttpClient, model: str) -> None:
        """stop parameter should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Count 1 to 10."}],
                "stop": ["5"],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_content_as_array(self, client: LoggingHttpClient, model: str) -> None:
        """content as array of content parts should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Say hello."}],
                    },
                ],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        content = body["choices"][0]["message"]["content"]
        assert isinstance(content, str) and len(content) > 0

    def test_message_name_field(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """name field on messages should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "name": "alice",
                        "content": "Say hello.",
                    },
                ],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_max_completion_tokens(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """max_completion_tokens parameter should limit response length."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Write a long story about a cat."},
                ],
                "max_completion_tokens": 10,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        usage = body.get("usage", {})
        if usage:
            assert usage.get("completion_tokens", 0) <= 20

    def test_top_p(self, client: LoggingHttpClient, model: str) -> None:
        """top_p parameter should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "top_p": 0.9,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_frequency_penalty(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """frequency_penalty parameter should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "frequency_penalty": 0.5,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_presence_penalty(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """presence_penalty parameter should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "presence_penalty": 0.5,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_seed(self, client: LoggingHttpClient, model: str) -> None:
        """seed parameter should be accepted for deterministic output."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "seed": 42,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        # system_fingerprint may be returned when seed is used
        assert isinstance(body, dict)

    def test_logprobs(self, client: LoggingHttpClient, model: str) -> None:
        """logprobs and top_logprobs parameters should return log probabilities."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "logprobs": True,
                "top_logprobs": 3,
                "max_tokens": 10,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        choice = body["choices"][0]
        if "logprobs" in choice and choice["logprobs"] is not None:
            logprobs = choice["logprobs"]
            assert "content" in logprobs
            assert isinstance(logprobs["content"], list)
            if len(logprobs["content"]) > 0:
                token_info = logprobs["content"][0]
                assert "token" in token_info
                assert "logprob" in token_info
                if "top_logprobs" in token_info:
                    assert isinstance(token_info["top_logprobs"], list)
                    assert len(token_info["top_logprobs"]) <= 3

    def test_user_field(self, client: LoggingHttpClient, model: str) -> None:
        """user field should be accepted for tracking purposes."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "user": "test-user-123",
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_json_mode(self, client: LoggingHttpClient, model: str) -> None:
        """response_format=json_object should return valid JSON content."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": 'Return a JSON object with key '
                        '"greeting" and value "hello".',
                    },
                ],
                "response_format": {"type": "json_object"},
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
