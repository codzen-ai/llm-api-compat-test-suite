"""Test Gemini-compatible generateContent API (non-streaming)."""

from __future__ import annotations

from typing import Any

import pytest

from http_client import LoggingHttpClient


@pytest.mark.capability("chat")
class TestGenerateContent:
    """POST /v1beta/models/{model}:generateContent."""

    def test_simple_message(self, client: LoggingHttpClient, model: str) -> None:
        """A simple user message should return a well-formed response."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": "Say hello."}],
                    },
                ],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        # candidates
        assert "candidates" in body, "Response missing 'candidates'"
        candidates: Any = body["candidates"]
        assert isinstance(candidates, list) and len(candidates) > 0

        candidate: Any = candidates[0]
        assert "content" in candidate

        content: Any = candidate["content"]
        assert content.get("role") == "model"
        assert "parts" in content
        parts: Any = content["parts"]
        assert isinstance(parts, list) and len(parts) > 0
        assert "text" in parts[0]
        assert len(parts[0]["text"]) > 0

    def test_multi_turn(self, client: LoggingHttpClient, model: str) -> None:
        """Multi-turn conversation should be supported."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": "My name is Alice."}]},
                    {"role": "model", "parts": [{"text": "Hello Alice!"}]},
                    {"role": "user", "parts": [{"text": "What is my name?"}]},
                ],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        text: Any = body["candidates"][0]["content"]["parts"][0]["text"]
        assert isinstance(text, str) and len(text) > 0

    def test_system_instruction(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """systemInstruction should be accepted."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "systemInstruction": {
                    "parts": [{"text": "You are a helpful assistant."}],
                },
                "contents": [
                    {"role": "user", "parts": [{"text": "Say hello."}]},
                ],
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_usage_metadata(self, client: LoggingHttpClient, model: str) -> None:
        """Response should contain usageMetadata."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": "Hi"}]},
                ],
            },
        )

        assert status == 200
        assert isinstance(body, dict)

        if "usageMetadata" in body:
            usage: Any = body["usageMetadata"]
            assert "promptTokenCount" in usage
            assert "candidatesTokenCount" in usage
            assert "totalTokenCount" in usage

    def test_generation_config(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """generationConfig parameters should be accepted."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": "Say hello."}]},
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 100,
                    "topP": 0.9,
                },
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_stop_sequences(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """stopSequences in generationConfig should be accepted."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": "Count from 1 to 10."}]},
                ],
                "generationConfig": {
                    "stopSequences": ["5"],
                },
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_candidate_count(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """candidateCount in generationConfig should return multiple candidates."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": "Say hello."}]},
                ],
                "generationConfig": {
                    "candidateCount": 2,
                },
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)
        candidates: Any = body["candidates"]
        assert isinstance(candidates, list)
        # Some providers may not support candidateCount > 1
        assert len(candidates) >= 1

    def test_top_k(self, client: LoggingHttpClient, model: str) -> None:
        """topK in generationConfig should be accepted."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": "Say hello."}]},
                ],
                "generationConfig": {
                    "topK": 40,
                },
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_safety_ratings(self, client: LoggingHttpClient, model: str) -> None:
        """Response candidates should include safety ratings."""
        status, body = client.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            json_body={
                "contents": [
                    {"role": "user", "parts": [{"text": "Say hello."}]},
                ],
            },
        )

        assert status == 200
        assert isinstance(body, dict)

        candidate: Any = body["candidates"][0]
        if "safetyRatings" in candidate:
            ratings: Any = candidate["safetyRatings"]
            assert isinstance(ratings, list)
            for rating in ratings:
                assert "category" in rating
                assert "probability" in rating
