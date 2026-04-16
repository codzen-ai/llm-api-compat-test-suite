"""Test Anthropic-compatible vision (image input)."""

from __future__ import annotations

from typing import Any

import pytest

from http_client import LoggingHttpClient

# Small 1x1 red PNG, base64 encoded
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "nGP4z8BQDwAEgAF/pooBPQAAAABJRU5ErkJggg=="
)


@pytest.mark.capability("vision")
class TestVision:
    """POST /v1/messages with image content blocks."""

    def test_base64_image_input(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Model should accept base64 image content block."""
        status, body = client.request(
            "POST",
            "/v1/messages",
            json_body={
                "model": model,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": TINY_PNG_B64,
                                },
                            },
                            {
                                "type": "text",
                                "text": "What do you see in this image?",
                            },
                        ],
                    },
                ],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        content: Any = body["content"]
        text_blocks: list[Any] = [b for b in content if b["type"] == "text"]
        assert len(text_blocks) > 0
        assert len(text_blocks[0]["text"]) > 0
