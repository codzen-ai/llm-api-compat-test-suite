"""Streaming latency tests: TTFT (time-to-first-token) and TPOT (time-per-
output-token), measured against the budget declared in the model's profile
(optionally overridden in user config).

TTFT is defined as the time from request send to the first SSE chunk whose
``delta`` carries a non-empty ``content`` *or* ``reasoning_content`` —
role-only opener chunks don't count. Reasoning-style models (e.g. glm-5.x,
o-series, DeepSeek-R1) often stream their chain-of-thought first and only
later (or never, if truncated) emit visible content; from a user-perceived
latency perspective, the first ``reasoning_content`` chunk is still "first
token", so we count it.

TPOT is ``(t_last_output - t_first_output) / (completion_tokens - 1)``,
spanning both reasoning and content tokens. ``completion_tokens`` comes from
the final chunk's ``usage`` block, which is why the request sets
``stream_options.include_usage: true``.

Three samples per measurement, compared against the budget by median to keep
single-shot network/cold-start jitter from flipping the result.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable

import pytest

from http_client import LoggingHttpClient
from model_profile import ResolvedModel

ITERATIONS = 3
PROMPT = (
    "Write a short paragraph of about 50 words introducing the concept of "
    "compound interest in plain language."
)
# Headroom for reasoning models, which can spend hundreds of tokens on
# chain-of-thought before emitting visible content. Too tight a cap (e.g.
# 120) leaves them with no content at all, making the request useless for
# latency measurement.
MAX_COMPLETION_TOKENS = 512


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
        budget = resolved_model.performance_budget
        if budget is None:
            pytest.skip("No performance budget configured for this model")

        ttft_samples: list[float] = []
        tpot_samples: list[float] = []
        for _ in range(ITERATIONS):
            ttft_ms, tpot_ms = _measure_once(client, model)
            ttft_samples.append(ttft_ms)
            tpot_samples.append(tpot_ms)

        median_ttft = statistics.median(ttft_samples)
        median_tpot = statistics.median(tpot_samples)

        ttft_fmt = [f"{s:.0f}" for s in ttft_samples]
        tpot_fmt = [f"{s:.1f}" for s in tpot_samples]

        ttft_pass = "PASS" if median_ttft <= budget.ttft_ms else "FAIL"
        tpot_pass = "PASS" if median_tpot <= budget.tpot_ms else "FAIL"

        sample_headers = " | ".join(
            f"Run {i + 1}" for i in range(ITERATIONS)
        )
        sample_separator = " | ".join(["---"] * ITERATIONS)
        ttft_row = " | ".join(ttft_fmt)
        tpot_row = " | ".join(tpot_fmt)

        # Record details before the assertion so PASS runs also surface the
        # numbers in the summary's "Test Details" section.
        record_property(
            "details",
            (
                f"| Metric | {sample_headers} | Median | Budget | Result |\n"
                f"|---|{sample_separator}|---|---|---|\n"
                f"| TTFT (ms) | {ttft_row} | "
                f"**{median_ttft:.0f}** | {budget.ttft_ms:.0f} | {ttft_pass} |\n"
                f"| TPOT (ms/tok) | {tpot_row} | "
                f"**{median_tpot:.1f}** | {budget.tpot_ms:.1f} | {tpot_pass} |"
            ),
        )

        failures: list[str] = []
        if median_ttft > budget.ttft_ms:
            failures.append(
                f"TTFT median {median_ttft:.0f}ms exceeds budget "
                f"{budget.ttft_ms:.0f}ms (samples ms: {ttft_fmt})"
            )
        if median_tpot > budget.tpot_ms:
            failures.append(
                f"TPOT median {median_tpot:.1f}ms/tok exceeds budget "
                f"{budget.tpot_ms:.1f}ms/tok (samples ms/tok: {tpot_fmt})"
            )
        assert not failures, "\n".join(failures)


def _measure_once(
    client: LoggingHttpClient, model: str
) -> tuple[float, float]:
    """Run one streaming request and return ``(ttft_ms, tpot_ms)``."""
    status, timed_lines = client.request_stream_timed(
        "POST",
        "/v1/chat/completions",
        json_body={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
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
