"""Test OpenAI-compatible vision (image input)."""

from __future__ import annotations

import pytest

from http_client import LoggingHttpClient

# 64x64 PNG with shapes, base64 encoded (OpenAI vision rejects tiny 1x1 images)
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAABXUlEQVR4nO2a3W7DIBSDbavv/8pn"
    "arOLTVsTDj8Bs31XkRpRGwMtHBgRcEYwRzBHMEcwRzBHMEcw5zGiUZLvPur+s/MYIToKX4sOZtje"
    "yqGpopXDSqOAJgPV0r81gifVMuonMcloVo9XC3E6bS5kVFjv0vG/NIsnWT2a1fG9olCF+qFE0oOW"
    "Ul/hQaupz3qw/yuhBbs/FcL1Mnqinv1Wo/hcRX9+xcXCuvsQ4ozBkxpIWyfA2d1fEsLWCVgg7GqA"
    "a0yAy2mwbwIuCOYI5gjmCOYI5gi7GoiIyqOmAfD9tmbfBFwQNjYQa0wDnu7rt04AC4TA7Y9VWHIe"
    "P2tzw4JyQVECUwYSy4od9kNIhe/dHAKLa02JBG7zwEylLDeEbvDAZJ0vPQcODyNsMK++qdDdd21l"
    "ba27fhXqFQUb1P/5uxI1t1W+PK9yW2XufSH+X3ydjGCOYI5gjmCOZgto5QPMYJlbNJq0MAAAAABJ"
    "RU5ErkJggg=="
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
                "max_completion_tokens": 100,
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
                "max_completion_tokens": 100,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
