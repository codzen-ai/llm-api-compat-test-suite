"""Shared orchestration for TTFT/TPOT performance tests across API formats.

Each format's ``test_performance.py`` defines a single-iteration measurement
closure that issues one streaming request and returns ``(ttft_ms, tpot_ms)``;
this module then runs ``ITERATIONS`` samples, records a result table to the
report's "Test Details" section, and asserts both medians against the
profile's :class:`PerformanceBudget`.

Leading underscore in the filename keeps pytest from collecting it as a
test module.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

import pytest

from model_profile import ResolvedModel

# Three samples balance noise reduction against API cost. The median (not
# the mean) shields the budget check from a single cold-start outlier.
ITERATIONS = 3

# Headroom for reasoning models whose chain-of-thought eats hundreds of
# tokens before any visible content is emitted. Too tight a cap leaves
# such models with no output to time, breaking the measurement.
MAX_OUTPUT_TOKENS = 512

PROMPT = (
    "Write a short paragraph of about 50 words introducing the concept of "
    "compound interest in plain language."
)


def run_perf_assertion(
    *,
    resolved_model: ResolvedModel,
    record_property: Callable[[str, object], None],
    measure_once: Callable[[], tuple[float, float]],
) -> None:
    """Drive the per-test loop and emit a result table to the summary.

    ``measure_once`` is the format-specific function that issues one
    streaming request and returns ``(ttft_ms, tpot_ms)``.
    """
    budget = resolved_model.performance_budget
    if budget is None:
        pytest.skip("No performance budget configured for this model")

    ttft_samples: list[float] = []
    tpot_samples: list[float] = []
    for _ in range(ITERATIONS):
        ttft_ms, tpot_ms = measure_once()
        ttft_samples.append(ttft_ms)
        tpot_samples.append(tpot_ms)

    median_ttft = statistics.median(ttft_samples)
    median_tpot = statistics.median(tpot_samples)

    ttft_fmt = [f"{s:.0f}" for s in ttft_samples]
    tpot_fmt = [f"{s:.1f}" for s in tpot_samples]
    ttft_pass = "PASS" if median_ttft <= budget.ttft_ms else "FAIL"
    tpot_pass = "PASS" if median_tpot <= budget.tpot_ms else "FAIL"

    sample_headers = " | ".join(f"Run {i + 1}" for i in range(ITERATIONS))
    sample_separator = " | ".join(["---"] * ITERATIONS)
    ttft_row = " | ".join(ttft_fmt)
    tpot_row = " | ".join(tpot_fmt)
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
