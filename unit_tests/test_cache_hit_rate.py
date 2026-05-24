"""Unit tests for scripts/cache_hit_rate.py.

Covers the pure-function branches that aren't exercised by any other test
(overflow classification, sampler reproducibility, lorem generator, usage
extraction) plus an httpx-mocked integration over ``run_test`` that
verifies session-restart semantics.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from typing import Any
from unittest.mock import MagicMock, patch

import cache_hit_rate as chr_module
import httpx
import pytest
from _common import (
    CHARS_PER_TOKEN,
    Sampler,
    fmt_duration,
    make_text_for_tokens,
    select_target,
)
from cache_hit_rate import (
    RunState,
    _extract_assistant_content,
    _extract_usage,
    is_overflow,
    run_test,
)
from rich.console import Console

from config import ApiFormat, ModelConfig, ProviderConfig, SuiteConfig

# ── Pure-function tests ──────────────────────────────────────────────────────


class TestIsOverflow:
    @pytest.mark.parametrize(("status", "body", "expected"), [
        (400, "context_length_exceeded: too many tokens", True),
        (400, {"error": {"message": "maximum context length is 32768"}}, True),
        (413, "input is too long", True),
        (422, {"error": {"code": "context_window_exceeded"}}, True),
        (500, "context_length whatever", False),  # 5xx is never overflow
        (400, "rate limit exceeded", False),
        (400, {"error": {"code": "invalid_request"}}, False),
        (200, "context_length anything", False),  # 200 is success, not overflow
    ])
    def test_classification(
        self, status: int, body: dict[str, Any] | str, expected: bool,
    ) -> None:
        assert is_overflow(status, body) is expected

    def test_none_body_is_not_overflow(self) -> None:
        assert is_overflow(400, None) is False


class TestSampler:
    def test_all_samples_within_bounds(self) -> None:
        rng = random.Random(0)
        s = Sampler(avg=1000, max_value=5000, rng=rng)
        samples = [s() for _ in range(2000)]
        assert all(1 <= x <= 5000 for x in samples)

    def test_same_seed_same_sequence(self) -> None:
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        sa = Sampler(avg=300, max_value=1500, rng=rng_a)
        sb = Sampler(avg=300, max_value=1500, rng=rng_b)
        assert [sa() for _ in range(50)] == [sb() for _ in range(50)]

    def test_empirical_mean_close_to_target(self) -> None:
        rng = random.Random(7)
        s = Sampler(avg=28000, max_value=80000, rng=rng)
        samples = [s() for _ in range(500)]
        mean = sum(samples) / len(samples)
        # Hard clamp on the lognormal-ish distribution biases slightly; ±10%
        # is plenty of headroom for spec validation without flaky tests.
        assert abs(mean - 28000) / 28000 < 0.10


class TestMakeTextForTokens:
    @pytest.mark.parametrize("n", [1, 10, 1000, 28000])
    def test_exact_char_count(self, n: int) -> None:
        assert len(make_text_for_tokens(n)) == n * CHARS_PER_TOKEN

    def test_zero_request_returns_at_least_one_char(self) -> None:
        # 0 tokens is nonsensical; the function floors at 1 *character* so
        # the API never gets an empty user message body.
        assert len(make_text_for_tokens(0)) == 1


class TestExtractUsage:
    def test_full_usage(self) -> None:
        state = RunState()
        parsed = {"usage": {
            "prompt_tokens": 1234,
            "completion_tokens": 56,
            "prompt_tokens_details": {"cached_tokens": 789},
        }}
        pt, ct, comp = _extract_usage(parsed, state)
        assert (pt, ct, comp) == (1234, 789, 56)
        assert state.cached_field_missing_logged is False

    def test_details_missing_logs_warning_once(self) -> None:
        state = RunState()
        parsed = {"usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        pt1, ct1, _ = _extract_usage(parsed, state)
        pt2, ct2, _ = _extract_usage(parsed, state)
        # cached_tokens silently 0 on both
        assert (pt1, ct1) == (100, 0)
        assert (pt2, ct2) == (100, 0)
        # And we only flip the flag once (caller uses it to gate WARN print)
        assert state.cached_field_missing_logged is True

    def test_usage_block_missing(self) -> None:
        state = RunState()
        pt, ct, comp = _extract_usage({}, state)
        assert (pt, ct, comp) == (0, 0, 0)


class TestExtractAssistantContent:
    def test_happy_path(self) -> None:
        parsed = {"choices": [
            {"message": {"role": "assistant", "content": "hello"}}
        ]}
        assert _extract_assistant_content(parsed) == "hello"

    @pytest.mark.parametrize("parsed", [
        {},  # no choices
        {"choices": []},  # empty choices
        {"choices": [{}]},  # no message
        {"choices": [{"message": {}}]},  # no content
        {"choices": [{"message": {"content": None}}]},  # null content
        {"choices": "not a list"},  # malformed
    ])
    def test_missing_fields_return_empty(self, parsed: dict[str, Any]) -> None:
        assert _extract_assistant_content(parsed) == ""


class TestFmtDuration:
    @pytest.mark.parametrize(("seconds", "expected"), [
        (5.0, "5s"),
        (75.0, "1m 15s"),
        (3725.0, "1h 2m 5s"),
        (0.0, "0s"),
    ])
    def test_format(self, seconds: float, expected: str) -> None:
        assert fmt_duration(seconds) == expected


class TestSelectTarget:
    def _suite(self, *providers: ProviderConfig) -> SuiteConfig:
        return SuiteConfig(providers=list(providers))

    def _provider(
        self, name: str, *models: str, api_format: ApiFormat = ApiFormat.OPENAI,
    ) -> ProviderConfig:
        return ProviderConfig(
            name=name, base_url="http://x", api_key="k", api_format=api_format,
            models=[ModelConfig(name=m, profile="gpt-5.4-mini") for m in models],
        )

    def test_single_provider_single_model_no_flags(self) -> None:
        suite = self._suite(self._provider("only", "m1"))
        provider, model = select_target(suite, None, None)
        assert provider.name == "only"
        assert model.name == "m1"

    def test_multiple_providers_requires_flag(self) -> None:
        suite = self._suite(
            self._provider("a", "m1"), self._provider("b", "m2"),
        )
        with pytest.raises(ValueError, match="multiple providers"):
            select_target(suite, None, None)

    def test_provider_not_found_lists_alternatives(self) -> None:
        suite = self._suite(self._provider("a", "m1"))
        with pytest.raises(ValueError, match=r"not found.*'a'"):
            select_target(suite, "missing", None)

    def test_non_openai_format_rejected(self) -> None:
        suite = self._suite(
            self._provider("a", "m1", api_format=ApiFormat.ANTHROPIC),
        )
        with pytest.raises(ValueError, match="only openai"):
            select_target(suite, None, None)

    def test_multiple_models_requires_flag(self) -> None:
        suite = self._suite(self._provider("a", "m1", "m2"))
        with pytest.raises(ValueError, match="multiple models"):
            select_target(suite, None, None)

    def test_model_not_found(self) -> None:
        suite = self._suite(self._provider("a", "m1", "m2"))
        with pytest.raises(ValueError, match=r"model 'missing' not found"):
            select_target(suite, None, "missing")

    def test_empty_providers(self) -> None:
        with pytest.raises(ValueError, match="no providers"):
            select_target(SuiteConfig(providers=[]), None, None)


# ── Integration tests over run_test with mocked httpx ────────────────────────


def _args(**overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = dict(
        config="", provider=None, model=None,
        total_requests=10, interval_sec=0.0,
        initial_avg_tokens=100, initial_max_tokens=200,
        append_avg_tokens=10, append_max_tokens=20,
        output_avg_tokens=5, output_max_tokens=10,
        seed=42, output_dir="unused-in-run_test",
        request_timeout_sec=5.0, max_attempts_multiplier=3.0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="test", base_url="http://mock", api_key="x",
        api_format=ApiFormat.OPENAI, verify_ssl=False,
    )


def _model() -> ModelConfig:
    return ModelConfig(name="mock-model", profile="gpt-5.4-mini")


def _ok(prompt: int = 1000, cached: int = 500, completion: int = 50) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    body = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def _overflow() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 400
    body = {"error": {"message": "maximum context length exceeded"}}
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def _http_500() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 500
    body = {"error": "internal"}
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def _run_with_responses(
    args: argparse.Namespace,
    responses: list[MagicMock | BaseException],
) -> RunState:
    """Patch httpx.Client so its .post returns/raises the given sequence.

    Also silences rich's Live updates by swapping the script's module-level
    ``CONSOLE`` for one that writes to an in-memory buffer — keeps test output
    clean without changing production behavior.
    """
    state = RunState()
    quiet_console = Console(file=io.StringIO(), force_terminal=False, width=120)
    with (
        patch("httpx.Client") as mock_client_cls,
        patch.object(chr_module, "CONSOLE", quiet_console),
    ):
        mock_client_cls.return_value.__enter__.return_value.post.side_effect = (
            responses
        )
        run_test(args, _provider(), _model(), state)
    return state


class TestRunTestIntegration:
    def test_happy_path_single_session(self) -> None:
        state = _run_with_responses(
            _args(total_requests=5),
            [_ok(1000, 500) for _ in range(5)],
        )
        assert len(state.results) == 5
        assert state.failures == []
        assert {r.session_id for r in state.results} == {1}
        assert [r.turn for r in state.results] == [1, 2, 3, 4, 5]
        # Hit rate math: 5*500 / 5*1000 = 0.5
        assert sum(r.prompt_tokens for r in state.results) == 5000
        assert sum(r.cached_tokens for r in state.results) == 2500

    def test_overflow_restarts_session(self) -> None:
        # 3 OK → overflow → 3 OK; need 6 successes total
        state = _run_with_responses(
            _args(total_requests=6),
            [_ok(), _ok(), _ok(), _overflow(), _ok(), _ok(), _ok()],
        )
        assert len(state.results) == 6
        assert len(state.failures) == 1
        assert state.failures[0].reason == "context_overflow"
        assert state.failures[0].session_id == 1
        # Session 1 had 3 turns, session 2 had 3 turns
        s1 = [r for r in state.results if r.session_id == 1]
        s2 = [r for r in state.results if r.session_id == 2]
        assert len(s1) == 3
        assert len(s2) == 3
        # Each session restarts the turn counter
        assert [r.turn for r in s1] == [1, 2, 3]
        assert [r.turn for r in s2] == [1, 2, 3]

    def test_isolated_failure_keeps_session(self) -> None:
        # 2 OK, 1 timeout, 2 OK — single failure should NOT restart session
        state = _run_with_responses(
            _args(total_requests=4),
            [_ok(), _ok(), httpx.TimeoutException("mock"), _ok(), _ok()],
        )
        assert len(state.results) == 4
        assert len(state.failures) == 1
        assert state.failures[0].reason == "timeout"
        assert {r.session_id for r in state.results} == {1}

    def test_three_consecutive_failures_restart_session(self) -> None:
        # 2 OK, 3 HTTP 500 (consecutive) → restart, then 4 OK
        state = _run_with_responses(
            _args(total_requests=6),
            [_ok(), _ok(),
             _http_500(), _http_500(), _http_500(),
             _ok(), _ok(), _ok(), _ok()],
        )
        assert len(state.results) == 6
        # 3 http_error failures recorded
        http_failures = [f for f in state.failures if f.reason == "http_error"]
        assert len(http_failures) == 3
        # Session restarted after the 3rd consecutive failure
        sessions = {r.session_id for r in state.results}
        assert sessions == {1, 2}

    def test_attempt_cap_prevents_infinite_loop(self) -> None:
        # Every request fails with overflow → loop terminates at the cap, not
        # at total_requests
        state = _run_with_responses(
            _args(total_requests=5, max_attempts_multiplier=2.0),
            [_overflow() for _ in range(20)],
        )
        assert len(state.results) == 0
        # Capped at 5 * 2 = 10 attempts
        assert len(state.failures) == 10

    def test_cached_field_missing_warning_does_not_break_run(self) -> None:
        # Server omits prompt_tokens_details → cached_tokens reads as 0 but
        # the run still completes
        no_details = MagicMock()
        no_details.status_code = 200
        body = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 10},
        }
        no_details.json.return_value = body
        no_details.text = json.dumps(body)
        state = _run_with_responses(_args(total_requests=3), [no_details] * 3)
        assert len(state.results) == 3
        assert all(r.cached_tokens == 0 for r in state.results)
        assert state.cached_field_missing_logged is True
