"""Test OpenAI-compatible embeddings API."""

from __future__ import annotations

import pytest

from http_client import LoggingHttpClient


@pytest.mark.capability("embeddings")
class TestEmbeddings:
    """POST /v1/embeddings."""

    def test_single_input(self, client: LoggingHttpClient, model: str) -> None:
        """Single text input should return embedding vector."""
        status, body = client.request(
            "POST",
            "/v1/embeddings",
            json_body={
                "model": model,
                "input": "Hello world",
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        assert body.get("object") == "list"
        assert "data" in body
        data = body["data"]
        assert isinstance(data, list) and len(data) > 0

        embedding_obj = data[0]
        assert embedding_obj.get("object") == "embedding"
        assert "embedding" in embedding_obj
        assert isinstance(embedding_obj["embedding"], list)
        assert all(isinstance(v, float) for v in embedding_obj["embedding"])
        assert "index" in embedding_obj

        assert "model" in body
        assert "usage" in body

    def test_batch_input(self, client: LoggingHttpClient, model: str) -> None:
        """Multiple text inputs should return multiple embeddings."""
        status, body = client.request(
            "POST",
            "/v1/embeddings",
            json_body={
                "model": model,
                "input": ["Hello", "World"],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        assert len(body["data"]) == 2
        assert body["data"][0]["index"] == 0
        assert body["data"][1]["index"] == 1
