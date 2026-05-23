"""Test Anthropic-compatible tool use."""

from __future__ import annotations

import json
from typing import Any

import pytest

from http_client import LoggingHttpClient

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name",
            },
        },
        "required": ["location"],
    },
}


@pytest.mark.capability("tools")
class TestToolUse:
    """POST /v1/messages with tools parameter."""

    def test_tool_use_response(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Model should return tool_use content block."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "What's the weather in Tokyo?"},
                ],
                "tools": [WEATHER_TOOL],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        # Find tool_use content block
        content: Any = body["content"]
        tool_use_blocks: list[Any] = [b for b in content if b.get("type") == "tool_use"]

        if body.get("stop_reason") == "tool_use":
            assert (
                len(tool_use_blocks) > 0
            ), "stop_reason=tool_use but no tool_use block"

            block: Any = tool_use_blocks[0]
            assert "id" in block
            assert block["name"] == "get_weather"
            assert "input" in block
            assert isinstance(block["input"], dict)
            assert "location" in block["input"]

    def test_tool_result_roundtrip(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Model should handle tool_result and produce a final text response."""
        # First request: get tool use
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "What's the weather in Tokyo?"},
                ],
                "tools": [WEATHER_TOOL],
            },
        )

        assert status == 200
        assert isinstance(body, dict)

        if body.get("stop_reason") != "tool_use":
            pytest.skip("Model did not produce tool_use")

        content: Any = body["content"]
        tool_use_block: Any = next(
            b for b in content if b["type"] == "tool_use"
        )

        # Second request: provide tool result
        status2, body2 = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "What's the weather in Tokyo?"},
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_block["id"],
                                "content": json.dumps(
                                    {"temperature": "22°C", "condition": "sunny"}
                                ),
                            },
                        ],
                    },
                ],
                "tools": [WEATHER_TOOL],
            },
        )

        assert status2 == 200, f"Expected 200, got {status2}: {body2}"
        assert isinstance(body2, dict)

        content2: Any = body2["content"]
        text_blocks: list[Any] = [b for b in content2 if b["type"] == "text"]
        assert len(text_blocks) > 0, "No text block in tool result response"
        assert len(text_blocks[0]["text"]) > 0

    @pytest.mark.capability("json_schema")
    def test_structured_output_via_forced_tool(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Anthropic's documented structured-output pattern: declare a tool
        whose input_schema is the desired output shape, and force the model
        to call it via tool_choice. The tool_use.input must conform."""
        city_info_tool = {
            "name": "record_city_info",
            "description": "Record structured information about a city.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "population": {"type": "integer"},
                },
                "required": ["city", "population"],
            },
        }
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 1024,
                "messages": [
                    {"role": "user", "content": "Record data for Tokyo."},
                ],
                "tools": [city_info_tool],
                "tool_choice": {"type": "tool", "name": "record_city_info"},
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        content: Any = body["content"]
        tool_use_blocks: list[Any] = [b for b in content if b.get("type") == "tool_use"]
        assert len(tool_use_blocks) > 0, "Forced tool_choice produced no tool_use block"

        block: Any = tool_use_blocks[0]
        assert block["name"] == "record_city_info"
        payload: Any = block["input"]
        assert isinstance(payload, dict)
        assert set(payload.keys()) >= {"city", "population"}, (
            f"Schema violation: missing required keys in {payload}"
        )
        assert isinstance(payload["city"], str)
        assert isinstance(payload["population"], int)
