"""Anthropic streaming latency: TTFT and TPOT against the profile's budget.

The Messages stream emits a sequence of events; we measure the timestamp of
the first ``content_block_delta`` whose payload is ``text_delta`` text or
``thinking_delta`` thinking (extended-thinking models like Claude Sonnet
4.5 may stream thinking content before any visible text). Both count as
"first token" for TTFT.

TPOT is ``(t_last_output - t_first_output) / (output_tokens - 1)``, where
``output_tokens`` comes from the final ``message_delta`` event's ``usage``
block (cumulative; the ``message_start`` event also has a ``usage`` field
but it carries the initial small count, not the final total).
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
    """POST /v1/messages with stream=true — TTFT and TPOT vs. budget."""

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


def _has_output_text(chunk: dict[str, object]) -> bool:
    """True if ``chunk`` is a ``content_block_delta`` carrying non-empty
    visible or thinking text."""
    if chunk.get("type") != "content_block_delta":
        return False
    delta = chunk.get("delta")
    if not isinstance(delta, dict):
        return False
    delta_type = delta.get("type")
    if delta_type == "text_delta":
        text = delta.get("text")
    elif delta_type == "thinking_delta":
        text = delta.get("thinking")
    else:
        return False
    return isinstance(text, str) and bool(text)


def _measure_once(
    client: LoggingHttpClient, model: str
) -> tuple[float, float]:
    status, timed_lines = client.request_stream_timed(
        "POST",
        "/v1/messages",
        json_body={
            "model": model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
        },
    )
    assert status == 200, f"Expected 200, got {status}"

    first_output_ms: float | None = None
    last_output_ms: float | None = None
    output_tokens: int | None = None

    for ts_ms, raw_line in timed_lines:
        line = raw_line.strip()
        # Anthropic SSE alternates `event: <name>` and `data: {...}` lines.
        # We only need the data payload to know what arrived.
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload:
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if _has_output_text(chunk):
            if first_output_ms is None:
                first_output_ms = ts_ms
            last_output_ms = ts_ms
        elif chunk.get("type") == "message_delta":
            usage = chunk.get("usage") or {}
            if isinstance(usage.get("output_tokens"), int):
                output_tokens = usage["output_tokens"]

    assert first_output_ms is not None, (
        "Stream contained no text_delta or thinking_delta — cannot compute TTFT"
    )
    assert last_output_ms is not None
    assert output_tokens is not None and output_tokens > 0, (
        "Stream did not report output_tokens in final message_delta"
    )
    assert output_tokens > 1, (
        f"Need >1 output token to measure TPOT, got {output_tokens}"
    )

    ttft = first_output_ms
    tpot = (last_output_ms - first_output_ms) / (output_tokens - 1)
    return ttft, tpot
