"""Test OpenAI-compatible function/tool calling."""

from __future__ import annotations

import json

import pytest

from http_client import LoggingHttpClient

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                },
            },
            "required": ["location"],
        },
    },
}


@pytest.mark.capability("tools")
class TestToolCalling:
    """POST /v1/chat/completions with tools parameter."""

    def test_tool_call_response(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Model should return tool_calls when tools are provided."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo?",
                    },
                ],
                "tools": [WEATHER_TOOL],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        choice = body["choices"][0]
        assert choice["finish_reason"] in ("tool_calls", "stop")

        if choice["finish_reason"] == "tool_calls":
            message = choice["message"]
            assert "tool_calls" in message
            tool_calls = message["tool_calls"]
            assert isinstance(tool_calls, list) and len(tool_calls) > 0

            tc = tool_calls[0]
            assert "id" in tc
            assert tc["type"] == "function"
            assert "function" in tc
            assert tc["function"]["name"] == "get_weather"
            args = json.loads(tc["function"]["arguments"])
            assert "location" in args

    def test_tool_result_roundtrip(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Model should handle tool result messages and produce a final answer."""
        # First request: get tool call
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo?",
                    },
                ],
                "tools": [WEATHER_TOOL],
            },
        )

        assert status == 200
        assert isinstance(body, dict)

        choice = body["choices"][0]
        if choice["finish_reason"] != "tool_calls":
            pytest.skip("Model did not produce tool_calls")

        assistant_msg = choice["message"]
        tool_call_id = assistant_msg["tool_calls"][0]["id"]

        # Second request: provide tool result
        status2, body2 = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {"role": "user", "content": "What's the weather in Tokyo?"},
                    assistant_msg,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(
                            {"temperature": "22°C", "condition": "sunny"}
                        ),
                    },
                ],
                "tools": [WEATHER_TOOL],
            },
        )

        assert status2 == 200, f"Expected 200, got {status2}: {body2}"
        assert isinstance(body2, dict)

        final_message = body2["choices"][0]["message"]
        assert final_message["role"] == "assistant"
        assert isinstance(final_message.get("content"), str)

    def test_parallel_tool_calls(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Model should support parallel tool calls when appropriate."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo and London?",
                    },
                ],
                "tools": [WEATHER_TOOL],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        choice = body["choices"][0]
        if choice["finish_reason"] == "tool_calls":
            tool_calls = choice["message"]["tool_calls"]
            # May or may not produce parallel calls — just verify structure
            for tc in tool_calls:
                assert "id" in tc
                assert tc["type"] == "function"
                assert "function" in tc
