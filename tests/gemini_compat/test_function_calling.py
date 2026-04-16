"""Test Gemini-compatible function calling."""

from __future__ import annotations

from typing import Any

import pytest

from http_client import LoggingHttpClient

WEATHER_TOOL = {
    "functionDeclarations": [
        {
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
    ],
}


@pytest.mark.capability("tools")
class TestFunctionCalling:
    """POST /v1beta/models/{model}:generateContent with tools."""

    def test_function_call_response(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Model should return functionCall part when tools are provided."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "What's the weather in Tokyo?"}],
                    },
                ],
                "tools": [WEATHER_TOOL],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        candidate: Any = body["candidates"][0]
        parts: Any = candidate["content"]["parts"]

        # Check if any part has functionCall
        fc_parts: list[Any] = [p for p in parts if "functionCall" in p]
        if fc_parts:
            fc: Any = fc_parts[0]["functionCall"]
            assert fc["name"] == "get_weather"
            assert "args" in fc
            assert "location" in fc["args"]

    def test_function_response_roundtrip(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Model should handle functionResponse and produce final answer."""
        # First request: get function call
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "What's the weather in Tokyo?"}],
                    },
                ],
                "tools": [WEATHER_TOOL],
            },
        )

        assert status == 200
        assert isinstance(body, dict)

        candidate: Any = body["candidates"][0]
        parts: Any = candidate["content"]["parts"]
        fc_parts: list[Any] = [p for p in parts if "functionCall" in p]

        if not fc_parts:
            pytest.skip("Model did not produce functionCall")

        fc: Any = fc_parts[0]["functionCall"]

        # Second request: provide function response
        status2, body2 = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "What's the weather in Tokyo?"}],
                    },
                    {
                        "role": "model",
                        "parts": [{"functionCall": fc}],
                    },
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": fc["name"],
                                    "response": {
                                        "temperature": "22°C",
                                        "condition": "sunny",
                                    },
                                },
                            },
                        ],
                    },
                ],
                "tools": [WEATHER_TOOL],
            },
        )

        assert status2 == 200, f"Expected 200, got {status2}: {body2}"
        assert isinstance(body2, dict)

        text_parts: list[Any] = [
            p for p in body2["candidates"][0]["content"]["parts"] if "text" in p
        ]
        assert len(text_parts) > 0, "No text part in function response roundtrip"
