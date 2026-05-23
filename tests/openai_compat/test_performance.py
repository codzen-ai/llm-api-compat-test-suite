"""OpenAI streaming latency: TTFT and TPOT against the profile's budget.

TTFT is the time from request send to the first SSE chunk carrying a
non-empty ``delta.content`` *or* ``delta.reasoning_content`` — role-only
opener chunks don't count. Reasoning-style models (o-series, glm-5.x,
DeepSeek-R1) often stream their chain-of-thought first and only later (or
never, if truncated) emit visible content; from a user-perceived latency
perspective the first ``reasoning_content`` chunk is still "first token",
so we count it.

TPOT is ``(t_last_output - t_first_output) / (completion_tokens - 1)``,
spanning reasoning and content tokens. ``completion_tokens`` comes from the
final chunk's ``usage`` block, which is why the request sets
``stream_options.include_usage: true``.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest
from _perf_common import MAX_OUTPUT_TOKENS, PROMPT, run_perf_assertion

from http_client import LoggingHttpClient
from model_profile import ResolvedModel


@pytest.mark.capability("performance")
@pytest.mark.capability("streaming")
class TestPerformance:
    """POST /v1/chat/completions — TTFT and TPOT against profile budget."""

    def test_streaming_latency(
        self,
        client: LoggingHttpClient,
        model: str,
        resolved_model: ResolvedModel,
        record_property: Callable[[str, object], None],
    ) -> None:
        run_perf_assertion(
            resolved_model=resolved_model,
            record_property=record_property,
            measure_once=lambda: _measure_once(client, model),
        )


def _measure_once(
    client: LoggingHttpClient, model: str
) -> tuple[float, float]:
    status, timed_lines = client.request_stream_timed(
        "POST",
        "/v1/chat/completions",
        json_body={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
        },
    )
    assert status == 200, f"Expected 200, got {status}"

    first_output_ms: float | None = None
    last_output_ms: float | None = None
    completion_tokens: int | None = None

    for ts_ms, raw_line in timed_lines:
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            has_output = (isinstance(content, str) and content) or (
                isinstance(reasoning, str) and reasoning
            )
            if has_output:
                if first_output_ms is None:
                    first_output_ms = ts_ms
                last_output_ms = ts_ms

        usage = chunk.get("usage")
        if isinstance(usage, dict) and isinstance(
            usage.get("completion_tokens"), int
        ):
            completion_tokens = usage["completion_tokens"]

    assert first_output_ms is not None, (
        "Stream contained no content or reasoning_content delta — "
        "cannot compute TTFT"
    )
    assert last_output_ms is not None
    assert completion_tokens is not None and completion_tokens > 0, (
        "Stream did not report completion_tokens — "
        "stream_options.include_usage: true is required"
    )
    assert completion_tokens > 1, (
        f"Need >1 output token to measure TPOT, got {completion_tokens}"
    )

    ttft = first_output_ms
    tpot = (last_output_ms - first_output_ms) / (completion_tokens - 1)
    return ttft, tpot
