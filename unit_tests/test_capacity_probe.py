"""Unit tests for scripts/capacity_probe.py.

Covers the pure-function helpers (usage extraction, error detail, retry-after
parsing, default classifier) plus an httpx.MockTransport-driven integration
over ``run_load`` that verifies:

  - target_rpm pacing is honored within ~15%
  - Retry-After is observed (all workers pause cooperatively)
  - records are populated with usage tokens + classification
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from capacity_probe import (
    CapacityOutcome,
    MinuteSample,
    ProbeOutcome,
    RequestRecord,
    RequestSpec,
    StageResult,
    _auto_workers_for_rpm,
    _classify_bottleneck,
    _compute_429_rate,
    _compute_throughput,
    _count_classifications,
    _discover_limits_from_headers,
    _extract_error_detail,
    _extract_usage_tokens,
    _make_big_spec_factory,
    _make_decision,
    _make_small_spec,
    _write_capacity_per_minute_csv,
    _write_capacity_summary_md,
    _write_config_json,
    _write_probe_summary_md,
    classify_429,
    classify_default,
    parse_retry_after,
    run_load,
    stage_0_header_sniff,
)

from config import ApiFormat, ModelConfig, ProviderConfig

# ── Pure helpers ─────────────────────────────────────────────────────────────


class TestExtractUsageTokens:
    @pytest.mark.parametrize(("body", "expected"), [
        ({"usage": {"prompt_tokens": 100, "completion_tokens": 50}}, (100, 50)),
        ({"usage": {"prompt_tokens": 0, "completion_tokens": 0}}, (0, 0)),
        ({"usage": {}}, (0, 0)),
        ({}, (0, 0)),
        ({"usage": None}, (0, 0)),
        ({"usage": "not a dict"}, (0, 0)),
        (None, (0, 0)),
        ("error: rate limited", (0, 0)),
        ({"usage": {"prompt_tokens": "100", "completion_tokens": "50"}}, (100, 50)),
        ({"usage": {"prompt_tokens": "bad", "completion_tokens": 50}}, (0, 0)),
    ])
    def test_extraction(
        self,
        body: dict[str, Any] | str | None,
        expected: tuple[int, int],
    ) -> None:
        assert _extract_usage_tokens(body) == expected


class TestExtractErrorDetail:
    def test_string_body_truncates(self) -> None:
        long = "x" * 500
        assert _extract_error_detail(long) == "x" * 200

    def test_openai_error_shape(self) -> None:
        body = {"error": {"message": "rate limited", "code": "429"}}
        assert _extract_error_detail(body) == "rate limited"

    def test_falls_back_to_code(self) -> None:
        body = {"error": {"code": "context_length_exceeded"}}
        assert _extract_error_detail(body) == "context_length_exceeded"

    def test_no_error_key_returns_body_repr(self) -> None:
        body = {"foo": "bar"}
        assert "foo" in _extract_error_detail(body)

    def test_none_returns_empty(self) -> None:
        assert _extract_error_detail(None) == ""


class TestParseRetryAfter:
    def test_lowercase(self) -> None:
        assert parse_retry_after({"retry-after": "60"}) == 60.0

    def test_titlecase(self) -> None:
        assert parse_retry_after({"Retry-After": "30"}) == 30.0

    def test_missing(self) -> None:
        assert parse_retry_after({}) is None

    def test_http_date_unsupported(self) -> None:
        # We deliberately don't support HTTP-date form; returns None
        date_form = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
        assert parse_retry_after(date_form) is None


class TestClassifyDefault:
    @pytest.mark.parametrize(("status", "expected"), [
        (200, "ok"),
        (201, "ok"),
        (299, "ok"),
        (429, "429_unclassified"),
        (500, "5xx"),
        (503, "5xx"),
        (400, "http_400"),
        (404, "http_404"),
    ])
    def test_status_mapping(self, status: int, expected: str) -> None:
        assert classify_default(status, {}, None) == expected


# ── 429 classifier (three tiers) ─────────────────────────────────────────────


class TestClassify429:
    def test_non_429_delegates_to_default(self) -> None:
        assert classify_429(200, {}, None) == "ok"
        assert classify_429(500, {}, None) == "5xx"

    def test_tier1_rate_limit_type_requests(self) -> None:
        assert classify_429(429, {"rate_limit_type": "requests"}, None) == "429_rpm"

    def test_tier1_rate_limit_type_tokens(self) -> None:
        assert classify_429(429, {"rate_limit_type": "tokens"}, None) == "429_tpm"

    def test_tier1_unknown_value_falls_through(self) -> None:
        # Unknown value → tier 2/3 take over; with empty rest → unclassified
        result = classify_429(429, {"rate_limit_type": "weird"}, None)
        assert result == "429_unclassified"

    def test_tier2_openai_remaining_requests_zero(self) -> None:
        h = {"x-ratelimit-remaining-requests": "0",
             "x-ratelimit-remaining-tokens": "1000"}
        assert classify_429(429, h, None) == "429_rpm"

    def test_tier2_openai_remaining_tokens_zero(self) -> None:
        h = {"x-ratelimit-remaining-requests": "100",
             "x-ratelimit-remaining-tokens": "0"}
        assert classify_429(429, h, None) == "429_tpm"

    def test_tier2_litellm_api_key_infix(self) -> None:
        # litellm uses x-ratelimit-api_key-remaining-* — confirmed on mgtv
        h = {"x-ratelimit-api_key-remaining-requests": "0",
             "x-ratelimit-api_key-limit-requests": "60"}
        assert classify_429(429, h, None) == "429_rpm"

    def test_tier3_body_rpm_keyword(self) -> None:
        # litellm body shape (confirmed on mgtv)
        body = {"error": {"message": "Rate limit exceeded for api_key: abc. "
                                      "Limit type: requests. Current limit: 60.",
                          "code": "429"}}
        assert classify_429(429, {}, body) == "429_rpm"

    def test_tier3_body_tpm_keyword(self) -> None:
        body = {"error": {"message": "Rate limit exceeded. Limit type: tokens. "
                                      "Current limit: 100000."}}
        assert classify_429(429, {}, body) == "429_tpm"

    def test_tier3_string_body(self) -> None:
        assert classify_429(429, {}, "tokens per minute exceeded") == "429_tpm"
        assert classify_429(429, {}, "too many requests") == "429_rpm"

    def test_unclassified_when_no_signals(self) -> None:
        assert classify_429(429, {}, None) == "429_unclassified"
        vague = {"error": {"message": "limited"}}
        assert classify_429(429, {}, vague) == "429_unclassified"

    def test_tier_priority_header_beats_body(self) -> None:
        # Header says tokens but body says requests → header (Tier 1) wins
        h = {"rate_limit_type": "tokens"}
        body = {"error": {"message": "Limit type: requests"}}
        assert classify_429(429, h, body) == "429_tpm"

    def test_real_mgtv_429_sample(self) -> None:
        """Verbatim sample captured from mgtv during the design-phase probe.

        Hits Tier 1 (``rate_limit_type: requests``). Pinning this test means
        any regression that drops Tier 1 handling will surface immediately
        rather than silently degrading to Tier 3 fuzzy matching."""
        headers = {
            "retry-after": "60",
            "rate_limit_type": "requests",
            "reset_at": "2026-05-24 21:54:36 UTC",
            "x-litellm-call-id": "ded129c0-7e54-4379-a34e-abe7e87d9595",
            "x-litellm-key-rpm-limit": "60",
        }
        body = {
            "error": {
                "message": (
                    "Rate limit exceeded for api_key: 5822f2c3b. "
                    "Limit type: requests. Current limit: 60, "
                    "Remaining: 0. Limit resets at: 2026-05-24 21:54:36 UTC"
                ),
                "type": "None",
                "param": "None",
                "code": "429",
            },
        }
        assert classify_429(429, headers, body) == "429_rpm"


# ── run_load integration via MockTransport ───────────────────────────────────


def _ok_response(req: httpx.Request) -> httpx.Response:  # noqa: ARG001
    body = {
        "id": "mock-1",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return httpx.Response(200, json=body)


def _make_spec() -> RequestSpec:
    return RequestSpec(prompt_text="hi", input_tokens_target=10, max_output_tokens=5)


class TestRunLoad:
    def test_records_populated_on_success(self) -> None:
        async def go() -> list[RequestRecord]:
            return await run_load(
                url="https://mock/v1/chat/completions",
                headers={"Authorization": "Bearer test"},
                model="m",
                spec_fn=_make_spec,
                target_rpm=600,  # one every 100ms
                duration_sec=0.5,  # ~5 dispatches
                n_workers=4,
                timeout_sec=5.0,
                transport=httpx.MockTransport(_ok_response),
            )

        records = asyncio.run(go())
        assert len(records) >= 3, f"got {len(records)} records, expected >= 3"
        assert all(r.classification == "ok" for r in records)
        assert all(r.input_tokens == 10 and r.output_tokens == 5 for r in records)
        assert all(r.status == 200 for r in records)

    def test_target_rpm_pacing(self) -> None:
        """At 120 RPM (one every 500ms) over 2.0s, expect ~4 dispatches.

        Tolerance is wide (3-6 records) because dispatcher schedules at
        absolute times but the first dispatch happens at t=0 — so over 2s
        you get dispatches at 0.0, 0.5, 1.0, 1.5 → 4 records. Allow ±1 for
        async scheduling jitter on slow CI.
        """
        async def go() -> list[RequestRecord]:
            return await run_load(
                url="https://mock/v1/chat/completions",
                headers={},
                model="m",
                spec_fn=_make_spec,
                target_rpm=120,
                duration_sec=2.0,
                n_workers=2,
                timeout_sec=5.0,
                transport=httpx.MockTransport(_ok_response),
            )

        records = asyncio.run(go())
        # absolute scheduling: dispatches at 0.0, 0.5, 1.0, 1.5 (next would be 2.0
        # but loop exits at deadline) → expect 4
        assert 3 <= len(records) <= 6, f"got {len(records)} records at 120 RPM over 2s"

    def test_retry_after_pauses_all_workers(self) -> None:
        """First request returns 429 with Retry-After=1; subsequent requests
        from all workers must pause until that deadline."""
        call_times: list[float] = []
        first_call_done = False

        def handler(req: httpx.Request) -> httpx.Response:  # noqa: ARG001
            nonlocal first_call_done
            call_times.append(time.monotonic())
            if not first_call_done:
                first_call_done = True
                return httpx.Response(
                    429,
                    headers={"retry-after": "1"},
                    json={"error": {"message": "rate limit"}},
                )
            return _ok_response(req)

        async def go() -> list[RequestRecord]:
            return await run_load(
                url="https://mock/v1/chat/completions",
                headers={},
                model="m",
                spec_fn=_make_spec,
                target_rpm=600,  # fast dispatch — would normally pile up
                duration_sec=2.0,
                n_workers=4,
                timeout_sec=5.0,
                transport=httpx.MockTransport(handler),
            )

        records = asyncio.run(go())
        assert any(r.classification == "429_unclassified" for r in records)
        assert any(r.classification == "ok" for r in records)
        # All ok-responses must have arrived at least ~0.9s after the 429
        # (allow 100ms slack vs the 1s Retry-After)
        first_429_time = next(t for t, r in zip(call_times, records, strict=False)
                               if r.classification == "429_unclassified")
        ok_call_times = [t for t, r in zip(call_times, records, strict=False)
                          if r.classification == "ok"]
        for ok_t in ok_call_times:
            assert ok_t - first_429_time >= 0.9, (
                f"OK response at t={ok_t - first_429_time:.2f}s after 429 — "
                "Retry-After pause not honored"
            )

    def test_classifier_is_invoked(self) -> None:
        calls: list[tuple[int, str]] = []

        def fake_classifier(
            status: int,
            headers: dict[str, str],  # noqa: ARG001
            body: dict[str, Any] | str | None,  # noqa: ARG001
        ) -> str:
            calls.append((status, "called"))
            return "custom"

        async def go() -> list[RequestRecord]:
            return await run_load(
                url="https://mock/v1/chat/completions",
                headers={},
                model="m",
                spec_fn=_make_spec,
                target_rpm=600,
                duration_sec=0.5,
                n_workers=2,
                timeout_sec=5.0,
                classifier=fake_classifier,
                transport=httpx.MockTransport(_ok_response),
            )

        records = asyncio.run(go())
        assert len(calls) == len(records) > 0
        assert all(r.classification == "custom" for r in records)

    def test_on_record_callback_fires(self) -> None:
        observed: list[str] = []

        async def go() -> list[RequestRecord]:
            return await run_load(
                url="https://mock/v1/chat/completions",
                headers={},
                model="m",
                spec_fn=_make_spec,
                target_rpm=600,
                duration_sec=0.5,
                n_workers=2,
                timeout_sec=5.0,
                on_record=lambda r: observed.append(r.classification),
                transport=httpx.MockTransport(_ok_response),
            )

        records = asyncio.run(go())
        assert len(observed) == len(records)

    def test_zero_rpm_raises(self) -> None:
        async def go() -> list[RequestRecord]:
            return await run_load(
                url="https://mock/",
                headers={},
                model="m",
                spec_fn=_make_spec,
                target_rpm=0,
                duration_sec=0.1,
                n_workers=1,
                timeout_sec=5.0,
                transport=httpx.MockTransport(_ok_response),
            )

        with pytest.raises(ValueError, match="target_rpm must be positive"):
            asyncio.run(go())

    def test_network_error_classified(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:  # noqa: ARG001
            msg = "connection reset"
            raise httpx.ConnectError(msg)

        async def go() -> list[RequestRecord]:
            return await run_load(
                url="https://mock/",
                headers={},
                model="m",
                spec_fn=_make_spec,
                target_rpm=600,
                duration_sec=0.3,
                n_workers=2,
                timeout_sec=5.0,
                transport=httpx.MockTransport(handler),
            )

        records = asyncio.run(go())
        assert len(records) > 0
        assert all(r.classification == "network" for r in records)
        assert all(r.status == 0 for r in records)


# ── Probe helpers ────────────────────────────────────────────────────────────


def _rec(
    cls: str = "ok", pt: int = 100, ct: int = 50, ts: float = 0.0,
) -> RequestRecord:
    return RequestRecord(
        ts=ts, elapsed_ms=100.0, status=200 if cls == "ok" else 429,
        input_tokens=pt if cls == "ok" else 0,
        output_tokens=ct if cls == "ok" else 0,
        classification=cls, retry_after_sec=None,
    )


class TestComputeThroughput:
    def test_only_ok_counted(self) -> None:
        # Spread ts across 60s so the OK-window equals one minute exactly:
        # last_finish (59.9 + 0.1) − first_start (0.0) = 60.0
        records = [_rec("ok", pt=10, ct=5, ts=0.0),
                   _rec("ok", pt=20, ct=10, ts=59.9),
                   _rec("429_rpm", ts=30.0)]
        rpm, tpm = _compute_throughput(records, elapsed_sec=60.0)
        # 2 ok in 60s → 2 RPM; tokens = (10+5)+(20+10) = 45 → 45 TPM
        assert rpm == 2.0
        assert tpm == 45.0

    def test_zero_elapsed_returns_zero(self) -> None:
        # Single OK record + zero elapsed → no window is derivable
        assert _compute_throughput([_rec("ok")], elapsed_sec=0.0) == (0.0, 0.0)

    def test_empty_returns_zero(self) -> None:
        assert _compute_throughput([], elapsed_sec=60.0) == (0.0, 0.0)

    def test_partial_minute_scales_up(self) -> None:
        # 5 OK records spanning 30s — window is derived from ts, not elapsed
        records = [_rec("ok", pt=100, ct=50, ts=t)
                   for t in (0.0, 7.5, 15.0, 22.5, 29.9)]
        rpm, tpm = _compute_throughput(records, elapsed_sec=30.0)
        assert rpm == 10.0
        assert tpm == 1500.0

    def test_collapsed_ts_falls_back_to_elapsed(self) -> None:
        # All ts identical (synthetic / clock skew) → derivation collapses,
        # fall back to elapsed_sec so the result is still meaningful.
        records = [_rec("ok", pt=100, ct=50, ts=0.0) for _ in range(5)]
        # Force elapsed_ms to 0 so the derived window is also 0
        for r in records:
            r.elapsed_ms = 0.0
        rpm, tpm = _compute_throughput(records, elapsed_sec=30.0)
        assert rpm == 10.0
        assert tpm == 1500.0


class TestCountClassifications:
    def test_aggregates(self) -> None:
        records = [_rec("ok"), _rec("ok"), _rec("429_rpm"), _rec("network")]
        assert _count_classifications(records) == {"ok": 2, "429_rpm": 1, "network": 1}

    def test_empty(self) -> None:
        assert _count_classifications([]) == {}


class TestDiscoverLimitsFromHeaders:
    def test_litellm_rpm_only(self) -> None:
        # mgtv-confirmed: rpm via x-litellm-key-rpm-limit, no tpm header
        h = {"x-litellm-key-rpm-limit": "60"}
        assert _discover_limits_from_headers(h) == (60, None)

    def test_litellm_both(self) -> None:
        h = {"x-litellm-key-rpm-limit": "60", "x-litellm-key-tpm-limit": "100000"}
        assert _discover_limits_from_headers(h) == (60, 100000)

    def test_openai_style(self) -> None:
        h = {"x-ratelimit-limit-requests": "500",
             "x-ratelimit-limit-tokens": "150000",
             "x-ratelimit-remaining-requests": "499",  # should be skipped
             "x-ratelimit-remaining-tokens": "149000"}
        assert _discover_limits_from_headers(h) == (500, 150000)

    def test_litellm_api_key_infix(self) -> None:
        h = {"x-ratelimit-api_key-limit-requests": "60"}
        assert _discover_limits_from_headers(h) == (60, None)

    def test_none_when_no_headers(self) -> None:
        assert _discover_limits_from_headers({}) == (None, None)

    def test_real_mgtv_baseline_headers(self) -> None:
        """Verbatim baseline headers from mgtv (deepseek-v4-flash, ok response).

        Pin: confirms Stage 0 will short-circuit RPM probe on this provider."""
        h = {
            "content-type": "application/json",
            "x-litellm-call-id": "c355d50e-76bc-4c8a-845e-7f204ea1f961",
            "x-litellm-version": "1.85.0",
            "x-litellm-key-rpm-limit": "60",
            "x-litellm-key-max-budget": "100.0",
            "x-ratelimit-api_key-remaining-requests": "58",
            "x-ratelimit-api_key-limit-requests": "60",
        }
        rpm, tpm = _discover_limits_from_headers(h)
        assert rpm == 60
        assert tpm is None


class TestSpecFactories:
    def test_small_spec_is_tiny(self) -> None:
        s = _make_small_spec()
        assert s.max_output_tokens == 10
        assert len(s.prompt_text) < 100

    def test_big_spec_scales_with_max_context(self) -> None:
        factory = _make_big_spec_factory(max_context=10000)
        s = factory()
        # input ~70% * 10000 = 7000 tokens → ~28000 chars (4 chars/token)
        assert s.input_tokens_target == 7000
        assert s.max_output_tokens == 2500
        assert len(s.prompt_text) > 20000  # ~70% * 10000 * 4 chars

    def test_big_spec_factory_returns_same_text(self) -> None:
        """The text is precomputed once and reused — important for
        reproducibility and to avoid wasting CPU regenerating on each call."""
        factory = _make_big_spec_factory(max_context=1000)
        s1 = factory()
        s2 = factory()
        assert s1.prompt_text is s2.prompt_text


class TestStage0HeaderSniff:
    def test_extracts_litellm_rpm(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:  # noqa: ARG001
            return httpx.Response(
                200,
                headers={"x-litellm-key-rpm-limit": "60"},
                json={"choices": [{"message": {"role": "assistant",
                                                "content": "ok"}}],
                      "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
            )

        rpm, tpm, raw = asyncio.run(stage_0_header_sniff(
            url="https://mock/v1/chat/completions",
            headers={},
            model="m",
            timeout_sec=5.0,
            transport=httpx.MockTransport(handler),
        ))
        assert rpm == 60
        assert tpm is None
        assert "x-litellm-key-rpm-limit" in raw

    def test_network_error_returns_empties(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:  # noqa: ARG001
            msg = "boom"
            raise httpx.ConnectError(msg)

        rpm, tpm, raw = asyncio.run(stage_0_header_sniff(
            url="https://mock/",
            headers={},
            model="m",
            timeout_sec=5.0,
            transport=httpx.MockTransport(handler),
        ))
        assert (rpm, tpm, raw) == (None, None, {})

    def test_no_limit_headers_returns_none_but_keeps_raw(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:  # noqa: ARG001
            return httpx.Response(200, json={"usage": {"prompt_tokens": 5,
                                                        "completion_tokens": 2}})

        rpm, tpm, raw = asyncio.run(stage_0_header_sniff(
            url="https://mock/",
            headers={},
            model="m",
            timeout_sec=5.0,
            transport=httpx.MockTransport(handler),
        ))
        assert rpm is None
        assert tpm is None
        assert raw != {}  # we still keep content-type etc.


# ── Capacity controller pieces ───────────────────────────────────────────────


class TestCompute429Rate:
    def test_no_records(self) -> None:
        assert _compute_429_rate([]) == 0.0

    def test_all_ok(self) -> None:
        assert _compute_429_rate([_rec("ok"), _rec("ok")]) == 0.0

    def test_mixed(self) -> None:
        records = [_rec("ok"), _rec("ok"), _rec("429_rpm"), _rec("429_tpm")]
        assert _compute_429_rate(records) == 0.5

    def test_includes_unclassified(self) -> None:
        records = [_rec("ok"), _rec("429_unclassified")]
        assert _compute_429_rate(records) == 0.5

    def test_network_errors_not_counted_as_429(self) -> None:
        records = [_rec("ok"), _rec("network"), _rec("timeout")]
        assert _compute_429_rate(records) == 0.0


class TestClassifyBottleneck:
    def test_no_429_records(self) -> None:
        assert _classify_bottleneck([_rec("ok"), _rec("ok")]) == "NO_429"

    def test_pure_rpm(self) -> None:
        records = [_rec("429_rpm") for _ in range(10)]
        assert _classify_bottleneck(records) == "RPM"

    def test_pure_tpm(self) -> None:
        records = [_rec("429_tpm") for _ in range(10)]
        assert _classify_bottleneck(records) == "TPM"

    def test_70pct_rpm_is_rpm(self) -> None:
        records = [_rec("429_rpm")] * 7 + [_rec("429_tpm")] * 3
        assert _classify_bottleneck(records) == "RPM"

    def test_50_50_is_mixed(self) -> None:
        records = [_rec("429_rpm")] * 5 + [_rec("429_tpm")] * 5
        assert _classify_bottleneck(records) == "MIXED"

    def test_unclassified_ignored(self) -> None:
        # unclassified is not RPM or TPM → does not count toward either
        records = (
            [_rec("429_unclassified")] * 100
            + [_rec("429_rpm")] * 7
            + [_rec("429_tpm")] * 3
        )
        assert _classify_bottleneck(records) == "RPM"


class TestMakeDecision:
    @pytest.mark.parametrize(("rate", "expected"), [
        (0.0, "ramp_up"),
        (0.005, "ramp_up"),
        (0.01, "hold"),       # exactly at low bound → hold (not <)
        (0.05, "hold"),
        (0.10, "hold"),       # exactly at high bound → hold (not >)
        (0.11, "back_off"),
        (0.50, "back_off"),
    ])
    def test_band_boundaries(self, rate: float, expected: str) -> None:
        assert _make_decision(rate) == expected


class TestAutoWorkers:
    def test_minimum_floor(self) -> None:
        # Very low RPM still gets 10 workers minimum
        assert _auto_workers_for_rpm(1.0) == 10
        assert _auto_workers_for_rpm(60.0) == 10  # 60/12+1=6, floored to 10

    def test_scales_with_rpm(self) -> None:
        # 120 RPM / 12 = 10, +1 = 11 (above floor)
        assert _auto_workers_for_rpm(120.0) == 11
        # 300 RPM / 12 = 25, +1 = 26
        assert _auto_workers_for_rpm(300.0) == 26


# ── Report writers (Step 7) ──────────────────────────────────────────────────


def _make_provider() -> ProviderConfig:
    return ProviderConfig(
        name="testp",
        base_url="https://example.com",
        api_key="x",
        api_format=ApiFormat.OPENAI,
        models=[],
    )


class TestReportWriters:
    def _make_capacity_outcome(self) -> CapacityOutcome:
        m1 = MinuteSample(
            minute_idx=0, target_rpm=10.0, n_workers=5,
            records=[_rec("ok", pt=100, ct=50, ts=1.0),
                     _rec("ok", pt=100, ct=50, ts=2.0),
                     _rec("429_rpm", ts=3.0)],
            rate_429=0.33, decision="hold",
        )
        m2 = MinuteSample(
            minute_idx=1, target_rpm=10.0, n_workers=5,
            records=[_rec("ok", pt=100, ct=50, ts=60.0)],
            rate_429=0.0, decision="hold",
        )
        return CapacityOutcome(
            converged=True, sustained_rpm=2.0, sustained_tpm=300.0,
            sustained_input_tpm=200.0, sustained_output_tpm=100.0,
            bottleneck="RPM", final_target_rpm=10.0,
            minutes=[m1, m2], steady_minutes_collected=2,
        )

    def test_capacity_per_minute_csv(self, tmp_path: Path) -> None:
        outcome = self._make_capacity_outcome()
        _write_capacity_per_minute_csv(tmp_path, outcome)
        rows = (tmp_path / "per_minute.csv").read_text().splitlines()
        assert rows[0].startswith("minute,target_rpm,n_workers")
        assert len(rows) == 3  # header + 2 minutes
        # minute 1 had 2 ok + 1 429_rpm
        assert "1,10.0,5,3,2,1,0,0,0,200,100" in rows[1]

    def test_capacity_summary_md_headline(self, tmp_path: Path) -> None:
        import argparse as _ap
        outcome = self._make_capacity_outcome()
        args = _ap.Namespace(
            mode="capacity", avg_input_tokens=100, avg_output_tokens=50,
            max_input_tokens=None, max_output_tokens=None, seed=42,
            steady_state_minutes=2, start_rpm=100, workers=None,
        )
        provider = _make_provider()
        model_cfg = ModelConfig(name="testm", profile="x")
        _write_capacity_summary_md(
            tmp_path, args, provider, model_cfg, outcome,
            "2026-05-24T10:00:00+00:00", "2026-05-24T10:01:00+00:00", 60.0,
        )
        content = (tmp_path / "summary.md").read_text()
        assert "# Capacity Probe Report (capacity mode)" in content
        assert "Sustained throughput: **2.0 RPM**" in content
        assert "**RPM**" in content  # bottleneck
        assert "## Minute-by-minute trace" in content
        assert "## Business request profile" in content

    def test_config_json_round_trips(self, tmp_path: Path) -> None:
        import argparse as _ap
        args = _ap.Namespace(
            mode="probe", config="config.yaml", provider="p", model="m",
            max_context=32000, seed=42, max_duration_minutes=10,
        )
        provider = _make_provider()
        model_cfg = ModelConfig(name="testm", profile="x")
        _write_config_json(
            tmp_path, args, provider, model_cfg,
            "2026-05-24T10:00:00+00:00", "2026-05-24T10:01:00+00:00",
        )
        loaded = json.loads((tmp_path / "config.json").read_text())
        assert loaded["mode"] == "probe"
        assert loaded["provider"] == "testp"
        assert loaded["base_url"] == "https://example.com"
        assert loaded["model"] == "testm"
        assert loaded["args"]["max_context"] == 32000
        assert "git_commit" in loaded

    def test_probe_summary_md_caveat_when_no_429(self, tmp_path: Path) -> None:
        import argparse as _ap
        # Stage that ran without hitting 429 → caveat should fire
        stage = StageResult(
            stage="Stage 2 (TPM)", elapsed_sec=60.0,
            records=[_rec("ok", pt=2000, ct=500) for _ in range(10)],
            measured_rpm=10.0, measured_tpm=25000.0,
            class_counts={"ok": 10},
        )
        outcome = ProbeOutcome(
            rpm_limit=60.0, tpm_limit=25000.0,
            rpm_source="header", tpm_source="bounded_by_rpm",
            stage_0_headers={"x-litellm-key-rpm-limit": "60"},
            stages=[stage],
        )
        args = _ap.Namespace(
            mode="probe", max_context=4000,
            stage1_workers=200, stage2_workers=200, saturation_rpm=60000.0,
        )
        provider = _make_provider()
        model_cfg = ModelConfig(name="testm", profile="x")
        _write_probe_summary_md(
            tmp_path, args, provider, model_cfg, outcome,
            "2026-05-24T10:00:00+00:00", "2026-05-24T10:01:00+00:00", 60.0,
        )
        content = (tmp_path / "summary.md").read_text()
        assert "lower bound" in content.lower()
        assert "client concurrency" in content.lower()
        assert "x-litellm-key-rpm-limit" in content


def test_json_serializable_record() -> None:
    """RequestRecord must serialize via dataclasses.asdict + json — used by
    config.json output later. Catches accidental non-serializable field
    additions early."""
    from dataclasses import asdict
    r = RequestRecord(
        ts=1.0, elapsed_ms=10.0, status=200, input_tokens=5, output_tokens=3,
        classification="ok", retry_after_sec=None,
    )
    s = json.dumps(asdict(r))
    assert "ok" in s
