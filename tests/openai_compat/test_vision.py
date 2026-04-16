"""Test OpenAI-compatible vision (image input)."""

from __future__ import annotations

import pytest

from http_client import LoggingHttpClient

# Small 1x1 red PNG, base64 encoded
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
)


@pytest.mark.capability("vision")
class TestVision:
    """POST /v1/chat/completions with image content."""

    def test_image_url_input(self, client: LoggingHttpClient, model: str) -> None:
        """Model should accept image_url content type."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What do you see in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{TINY_PNG_B64}",
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": 100,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        message = body["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert isinstance(message.get("content"), str)

    def test_image_url_with_detail(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """detail parameter (low/high/auto) should be accepted."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{TINY_PNG_B64}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                "max_tokens": 100,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
