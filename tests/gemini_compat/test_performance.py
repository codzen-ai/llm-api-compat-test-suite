"""Gemini streaming latency: TTFT and TPOT against the profile's budget.

The ``streamGenerateContent?alt=sse`` endpoint emits JSON chunks shaped
``{"candidates": [{"content": {"parts": [...]}}], "usageMetadata": {...}}``.
TTFT is the timestamp of the first chunk where ``candidates[0].content.parts``
contains a non-empty ``text``. Thinking-style Gemini models also expose a
``thought`` flag on the part; either way the field that timestamps the
emission is ``text``.

TPOT is ``(t_last_output - t_first_output) / (candidatesTokenCount - 1)``.
Gemini may include ``usageMetadata`` cumulatively in each chunk; we keep
the last value seen, which equals the total at stream end.

Note: there is no Gemini profile in ``model_profiles/gemini/`` yet, so this
test will not be picked up by any real run until one is authored — at that
point ``performance`` must appear in its ``capabilities`` list alongside a
``performance:`` budget block.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable

import pytest
from _perf_common import MAX_OUTPUT_TOKENS, PROMPT, run_perf_assertion

from http_client import LoggingHttpClient
from model_profile import ResolvedModel


@pytest.mark.capability("performance")
@pytest.mark.capability("streaming")
class TestPerformance:
    """POST /v1beta/models/{model}:streamGenerateContent?alt=sse."""

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
        f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
        json_body={
            "contents": [{"role": "user", "parts": [{"text": PROMPT}]}],
            "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS},
        },
    )
    assert status == 200, f"Expected 200, got {status}"

    first_output_ms: float | None = None
    last_output_ms: float | None = None
    output_tokens: int | None = None

    for ts_ms, raw_line in timed_lines:
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload:
            continue
        chunk: object = None
        with contextlib.suppress(json.JSONDecodeError):
            chunk = json.loads(payload)
        if not isinstance(chunk, dict):
            continue

        candidates = chunk.get("candidates") or []
        if candidates:
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            has_text = any(
                isinstance(p, dict)
                and isinstance(p.get("text"), str)
                and p["text"]
                for p in parts
            )
            if has_text:
                if first_output_ms is None:
                    first_output_ms = ts_ms
                last_output_ms = ts_ms

        usage = chunk.get("usageMetadata")
        if isinstance(usage, dict) and isinstance(
            usage.get("candidatesTokenCount"), int
        ):
            output_tokens = usage["candidatesTokenCount"]

    assert first_output_ms is not None, (
        "Stream contained no text part — cannot compute TTFT"
    )
    assert last_output_ms is not None
    assert output_tokens is not None and output_tokens > 0, (
        "Stream did not report usageMetadata.candidatesTokenCount"
    )
    assert output_tokens > 1, (
        f"Need >1 output token to measure TPOT, got {output_tokens}"
    )

    ttft = first_output_ms
    tpot = (last_output_ms - first_output_ms) / (output_tokens - 1)
    return ttft, tpot
