"""Test OpenAI-compatible reasoning (chain-of-thought) support.

A reasoning model supports thinking only if it both (a) accepts the
``reasoning_effort`` request parameter without 4xx and (b) actually reasons —
i.e. the response reports ``usage.completion_tokens_details.reasoning_tokens
> 0`` (non-streaming) or emits ``delta.reasoning_content`` chunks (streaming).
Silently ignoring ``reasoning_effort`` and returning a regular completion is
a common pseudo-compatibility failure mode that a bare 200 OK does not catch.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from http_client import LoggingHttpClient

# A question that genuinely benefits from reasoning, so the model is likely to
# spend tokens in the reasoning channel even at low effort.
PROMPT = (
    "What is 47 * 53? Think step by step, then give the final number."
)
# gpt-5.x / o-series reject `max_tokens`; reasoning models also need enough
# budget to cover both reasoning and visible output.
MAX_COMPLETION_TOKENS = 2048


@pytest.mark.capability("thinking")
class TestReasoning:
    """POST /v1/chat/completions with ``reasoning_effort``."""

    def test_reasoning_effort_non_streaming(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """``usage.completion_tokens_details.reasoning_tokens`` must be > 0."""
        status, body = client.request(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "reasoning_effort": "low",
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
            },
        )

        assert status == 200, f"Expected 200, got {status}: {body}"
        assert isinstance(body, dict)

        usage: Any = body.get("usage")
        assert isinstance(usage, dict), "Response missing `usage`"
        details: Any = usage.get("completion_tokens_details")
        assert isinstance(details, dict), (
            "Response missing `usage.completion_tokens_details` — "
            "the model may have silently ignored `reasoning_effort`"
        )
        reasoning_tokens: Any = details.get("reasoning_tokens")
        assert isinstance(reasoning_tokens, int) and reasoning_tokens > 0, (
            "`reasoning_tokens` is 0 or missing — the model may have "
            f"silently ignored `reasoning_effort`. Got: {details!r}"
        )

    @pytest.mark.capability("streaming")
    def test_reasoning_effort_streaming(
        self, client: LoggingHttpClient, model: str
    ) -> None:
        """Stream must surface reasoning via either ``delta.reasoning_content``
        chunks or a final ``usage.completion_tokens_details.reasoning_tokens
        > 0`` (whichever the provider exposes — both are valid signals)."""
        status, lines = client.request_stream(
            "POST",
            "/v1/chat/completions",
            json_body={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT}],
                "reasoning_effort": "low",
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

        assert status == 200, f"Expected 200, got {status}"

        saw_reasoning_delta, reasoning_tokens_final = _scan_stream(lines)
        assert saw_reasoning_delta or (
            reasoning_tokens_final is not None and reasoning_tokens_final > 0
        ), (
            "Stream had no `reasoning_content` deltas and no positive "
            "`reasoning_tokens` in final usage — the model may have "
            "silently ignored `reasoning_effort`"
        )


def _scan_stream(lines: list[str]) -> tuple[bool, int | None]:
    """Return (saw_reasoning_delta, reasoning_tokens_in_final_usage)."""
    saw_reasoning_delta = False
    reasoning_tokens_final: int | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk: Any = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue

        if _chunk_has_reasoning_delta(chunk):
            saw_reasoning_delta = True
        rt = _chunk_reasoning_tokens(chunk)
        if rt is not None:
            reasoning_tokens_final = rt
    return saw_reasoning_delta, reasoning_tokens_final


def _chunk_has_reasoning_delta(chunk: dict[str, Any]) -> bool:
    choices: Any = chunk.get("choices") or []
    if not choices:
        return False
    delta: Any = choices[0].get("delta") or {}
    rc: Any = delta.get("reasoning_content")
    return isinstance(rc, str) and bool(rc)


def _chunk_reasoning_tokens(chunk: dict[str, Any]) -> int | None:
    usage: Any = chunk.get("usage")
    if not isinstance(usage, dict):
        return None
    details: Any = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    rt: Any = details.get("reasoning_tokens")
    return rt if isinstance(rt, int) else None
