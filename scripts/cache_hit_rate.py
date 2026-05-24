"""Prompt cache hit-rate stress test.

Simulates a long-conversation workload (default: 500 sequential turns, ~28K
initial prompt, ~1.3K appended per turn) and reports the aggregate
``Σ cached_tokens / Σ prompt_tokens`` ratio for an OpenAI-format endpoint.

Independent of pytest — see docs/design/prompt-cache-hit-rate-script.md for the
full design rationale (independent entry, overflow → session restart, coarse
token generation, fixed-seed reproducibility).

Usage:
    uv run python scripts/cache_hit_rate.py --config config.yaml \\
        [--provider mgtv] [--model glm-5.1] [--total-requests 500] ...
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from tabulate import tabulate

if TYPE_CHECKING:
    from types import FrameType

CONSOLE = Console()
"""Single rich Console used for all output. ``Live`` shares this instance so
``CONSOLE.print(...)`` during a live display scrolls *above* the live region
without disrupting it — that's how overflow/restart notices interleave with
the progress bar."""

# Reuse SuiteConfig from src/ rather than re-parsing YAML; the script does
# not run under pytest so src/ is not on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import ModelConfig, ProviderConfig, SuiteConfig  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4
"""Coarse char-to-token ratio for *generating* prompt text. The hit-rate
math always uses the server-reported `usage.prompt_tokens`, so this
constant only affects how much padding text we emit — not the result."""

OVERFLOW_KEYWORDS = (
    "context_length",
    "maximum context",
    "context window",
    "context_window",
    "token limit",
    "too many tokens",
    "input is too long",
)
"""Substrings searched (case-insensitive) in 4xx error bodies to classify a
failure as 'context overflow' (→ trigger session restart) vs a transient
HTTP error (→ keep session, retry up to MAX_CONSECUTIVE_FAILURES)."""

MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_TIMEOUT_SEC = 120.0

# A ~500-char lorem ipsum block, repeated to ~10K-char corpus. Static content
# means identical fragments across turns → maximizes prefix-cache hits, which
# is exactly what this test is trying to measure.
_LOREM_BLOCK = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim "
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
    "commodo consequat. Duis aute irure dolor in reprehenderit in voluptate "
    "velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
    "occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
    "mollit anim id est laborum. "
)
_CORPUS = (_LOREM_BLOCK * 25)[:10000]


def make_text_for_tokens(token_count: int) -> str:
    """Return text of approximately ``token_count`` tokens (4 chars/token)."""
    char_count = max(1, token_count * CHARS_PER_TOKEN)
    repeats = (char_count // len(_CORPUS)) + 1
    return (_CORPUS * repeats)[:char_count]


# ── Distribution sampler ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Sampler:
    """Truncated-normal sampler: mean ~``avg``, hard clamp to ``[1, max_value]``.

    std = avg/3 puts ~99% of samples in [0, 2*avg] *before* clamping, so the
    rejection loop almost never iterates more than once for realistic specs.
    Empirical mean over 500 draws stays within ~5% of ``avg``.
    """

    avg: int
    max_value: int
    rng: random.Random

    def __call__(self) -> int:
        while True:
            s = int(self.rng.gauss(self.avg, self.avg / 3))
            if 1 <= s <= self.max_value:
                return s


# ── Records ───────────────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    request_idx: int          # 1-based, counts successes only
    session_id: int           # 1-based, increments on overflow restart
    turn: int                 # 1-based within session
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    elapsed_ms: float
    wall_time_iso: str
    initial_target_tokens: int  # what the sampler asked for (0 if not turn 1)
    append_target_tokens: int   # 0 if turn 1
    output_target_tokens: int


@dataclass
class FailureRecord:
    attempt_idx: int
    session_id: int
    turn: int  # turn that would have been (1-based)
    reason: str  # "context_overflow" | "http_error" | "timeout" | "network"
    detail: str
    elapsed_ms: float
    wall_time_iso: str


@dataclass
class RunState:
    results: list[RequestResult] = field(
        default_factory=lambda: list[RequestResult]()
    )
    failures: list[FailureRecord] = field(
        default_factory=lambda: list[FailureRecord]()
    )
    cached_field_missing_logged: bool = False
    stop_requested: bool = False


# ── Auth (kept aligned with conftest.client fixture's logic) ─────────────────


def build_headers(provider: ProviderConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    auth_type = provider.auth_type or {
        "openai": "bearer",
        "anthropic": "x-api-key",
        "gemini": "x-goog-api-key",
    }.get(str(provider.api_format), "bearer")

    if auth_type == "bearer":
        headers["Authorization"] = f"Bearer {provider.api_key}"
    elif auth_type == "x-api-key":
        headers["x-api-key"] = provider.api_key
    elif auth_type == "x-goog-api-key":
        headers["x-goog-api-key"] = provider.api_key
    return headers


# ── Overflow classification ──────────────────────────────────────────────────


def is_overflow(status: int, body: dict[str, Any] | str | None) -> bool:
    """True if a 4xx response indicates the prompt exceeded the context window.

    Different gateways word it differently; we match a few common substrings
    rather than an exact error code. False positives here mean an unnecessary
    session restart (harmless), false negatives mean we keep appending to an
    already-doomed session (retries will fail until MAX_CONSECUTIVE_FAILURES).
    """
    if status not in (400, 413, 422):
        return False
    if isinstance(body, dict):
        text = json.dumps(body)
    elif isinstance(body, str):
        text = body
    else:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in OVERFLOW_KEYWORDS)


# ── Provider/model selection ──────────────────────────────────────────────────


def select_target(
    suite: SuiteConfig,
    provider_name: str | None,
    model_name: str | None,
) -> tuple[ProviderConfig, ModelConfig]:
    providers = suite.providers
    if not providers:
        msg = "config has no providers"
        raise ValueError(msg)

    if provider_name is not None:
        matched = [p for p in providers if p.name == provider_name]
        if not matched:
            msg = (
                f"provider {provider_name!r} not found; "
                f"available: {[p.name for p in providers]}"
            )
            raise ValueError(msg)
        provider = matched[0]
    elif len(providers) == 1:
        provider = providers[0]
    else:
        msg = (
            f"multiple providers configured ({[p.name for p in providers]}); "
            "pass --provider to disambiguate"
        )
        raise ValueError(msg)

    if str(provider.api_format) != "openai":
        msg = (
            f"only openai api_format supported in v1; "
            f"provider {provider.name!r} is {provider.api_format}"
        )
        raise ValueError(msg)

    models = provider.models
    if not models:
        msg = f"provider {provider.name!r} has no models configured"
        raise ValueError(msg)

    if model_name is not None:
        matched_m = [m for m in models if m.name == model_name]
        if not matched_m:
            msg = (
                f"model {model_name!r} not found in provider {provider.name!r}; "
                f"available: {[m.name for m in models]}"
            )
            raise ValueError(msg)
        model_cfg = matched_m[0]
    elif len(models) == 1:
        model_cfg = models[0]
    else:
        msg = (
            f"multiple models in provider {provider.name!r} "
            f"({[m.name for m in models]}); pass --model to disambiguate"
        )
        raise ValueError(msg)

    return provider, model_cfg


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stress-test prompt cache hit rate on an OpenAI-format endpoint."
        ),
    )
    p.add_argument("--config", default="config.yaml",
                   help="Path to SuiteConfig YAML (default: config.yaml)")
    p.add_argument("--provider", default=None,
                   help="Provider name from config (required if config has >1)")
    p.add_argument("--model", default=None,
                   help="Model name from the provider (required if provider has >1)")
    p.add_argument("--total-requests", type=int, default=500,
                   help="Number of successful requests to collect (default: 500)")
    p.add_argument("--interval-sec", type=float, default=1.0,
                   help="Minimum wall-clock gap between request starts (default: 1.0)")
    p.add_argument("--initial-avg-tokens", type=int, default=28000)
    p.add_argument("--initial-max-tokens", type=int, default=80000)
    p.add_argument("--append-avg-tokens", type=int, default=1300)
    p.add_argument("--append-max-tokens", type=int, default=5000)
    p.add_argument("--output-avg-tokens", type=int, default=300)
    p.add_argument("--output-max-tokens", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for length-sampling reproducibility (default: 42)")
    p.add_argument("--output-dir", default="reports/cache_hit_rate",
                   help="Parent directory for run output")
    p.add_argument("--request-timeout-sec", type=float, default=DEFAULT_TIMEOUT_SEC,
                   help=f"Per-request timeout (default: {DEFAULT_TIMEOUT_SEC}s)")
    p.add_argument("--max-attempts-multiplier", type=float, default=2.0,
                   help="Cap attempts at total_requests * this (default: 2.0)")
    return p.parse_args()


# ── Runner ────────────────────────────────────────────────────────────────────


@dataclass
class _SendOutcome:
    """Outcome of one HTTP attempt — discriminated by which fields are set."""

    elapsed_ms: float
    wall_iso: str
    # exactly one of these is populated:
    status: int | None = None  # HTTP response received
    parsed: dict[str, Any] | str | None = None  # body of HTTP response
    error_kind: str | None = None  # "timeout" | "network" — transport failed
    error_detail: str = ""


def _send_one_request(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> _SendOutcome:
    t_start = time.monotonic()
    wall_iso = datetime.now().astimezone().isoformat()
    try:
        resp = client.post(url, headers=headers, json=body)
    except httpx.TimeoutException as e:
        return _SendOutcome(
            elapsed_ms=(time.monotonic() - t_start) * 1000,
            wall_iso=wall_iso,
            error_kind="timeout",
            error_detail=str(e)[:200],
        )
    except httpx.HTTPError as e:
        return _SendOutcome(
            elapsed_ms=(time.monotonic() - t_start) * 1000,
            wall_iso=wall_iso,
            error_kind="network",
            error_detail=str(e)[:200],
        )

    elapsed_ms = (time.monotonic() - t_start) * 1000
    parsed: dict[str, Any] | str
    try:
        raw_obj: object = resp.json()
    except (json.JSONDecodeError, ValueError):
        parsed = resp.text
    else:
        parsed = (
            cast("dict[str, Any]", raw_obj)
            if isinstance(raw_obj, dict) else resp.text
        )
    return _SendOutcome(
        elapsed_ms=elapsed_ms,
        wall_iso=wall_iso,
        status=resp.status_code,
        parsed=parsed,
    )


def _extract_assistant_content(parsed: dict[str, Any]) -> str:
    choices_raw: object = parsed.get("choices")
    if not isinstance(choices_raw, list) or not choices_raw:
        return ""
    first: object = cast("list[object]", choices_raw)[0]
    if not isinstance(first, dict):
        return ""
    msg_raw: object = cast("dict[str, Any]", first).get("message")
    if not isinstance(msg_raw, dict):
        return ""
    content: object = cast("dict[str, Any]", msg_raw).get("content")
    return content if isinstance(content, str) else ""


def _extract_usage(
    parsed: dict[str, Any], state: RunState,
) -> tuple[int, int, int]:
    """Pull ``(prompt_tokens, cached_tokens, completion_tokens)`` from a 200 body.

    Logs a one-time warning if ``prompt_tokens_details`` is absent (some
    third-party gateways drop it, in which case hit rate will read 0% even
    if caching is functional)."""
    usage_raw: object = parsed.get("usage")
    usage: dict[str, Any] = (
        cast("dict[str, Any]", usage_raw) if isinstance(usage_raw, dict) else {}
    )
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    details_raw: object = usage.get("prompt_tokens_details")
    if details_raw is None and not state.cached_field_missing_logged:
        _print(
            "[WARN] response.usage.prompt_tokens_details missing — "
            "cached_tokens will be treated as 0 for all requests"
        )
        state.cached_field_missing_logged = True
    details: dict[str, Any] = (
        cast("dict[str, Any]", details_raw) if isinstance(details_raw, dict) else {}
    )
    cached_tokens = int(details.get("cached_tokens", 0) or 0)
    return prompt_tokens, cached_tokens, completion_tokens


@dataclass
class _SessionState:
    """Mutable per-session state. Mutated in place by ``run_test``."""

    messages: list[dict[str, str]] = field(
        default_factory=lambda: list[dict[str, str]]()
    )
    session_id: int = 1
    turn_completed: int = 0
    consecutive_failures: int = 0

    def restart(self) -> None:
        self.session_id += 1
        self.turn_completed = 0
        self.messages = []
        self.consecutive_failures = 0


def _build_progress() -> Progress:
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(compact=True),
        console=CONSOLE,
        # Default 30s sample window means: if each request takes longer than
        # ~30s (large prompts always do), only 1 sample remains → speed
        # collapses to None → ETA shows "--:--". 1 hour keeps ≥120 samples
        # at 30s/req and ≥60 samples even at 60s/req.
        speed_estimate_period=3600.0,
    )


def _make_view(
    progress: Progress, state: RunState, sess: _SessionState,
) -> Group:
    sum_pt = sum(r.prompt_tokens for r in state.results)
    sum_ct = sum(r.cached_tokens for r in state.results)
    rate = (sum_ct / sum_pt * 100) if sum_pt > 0 else 0.0
    overflow_n = sum(1 for f in state.failures if f.reason == "context_overflow")
    other_n = len(state.failures) - overflow_n
    last = state.results[-1] if state.results else None

    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right", min_width=10)
    table.add_column()
    table.add_row(
        "Hit rate",
        f"[bold green]{rate:.2f}%[/]  ({sum_ct:,} / {sum_pt:,})",
    )
    table.add_row(
        "Session",
        f"{sess.session_id}  (turn {sess.turn_completed})",
    )
    table.add_row(
        "Failures",
        f"{overflow_n} overflow + {other_n} other",
    )
    if last is not None:
        table.add_row(
            "Last req",
            f"pt={last.prompt_tokens:,}  ct={last.cached_tokens:,}  "
            f"elapsed={last.elapsed_ms:.0f}ms",
        )
    return Group(progress, table)


def run_test(
    args: argparse.Namespace,
    provider: ProviderConfig,
    model_cfg: ModelConfig,
    state: RunState,
) -> None:
    rng = random.Random(args.seed)
    initial_sampler = Sampler(args.initial_avg_tokens, args.initial_max_tokens, rng)
    append_sampler = Sampler(args.append_avg_tokens, args.append_max_tokens, rng)
    output_sampler = Sampler(args.output_avg_tokens, args.output_max_tokens, rng)

    url = f"{provider.base_url.rstrip('/')}/v1/chat/completions"
    headers = build_headers(provider)

    sess = _SessionState()
    attempt_idx = 0
    max_attempts = int(args.total_requests * args.max_attempts_multiplier)
    last_send_monotonic = 0.0
    total: int = args.total_requests
    prev_success_count = 0

    progress = _build_progress()
    task_id = progress.add_task("Cache hit-rate", total=total)

    with (
        httpx.Client(
            timeout=args.request_timeout_sec, verify=provider.verify_ssl
        ) as client,
        Live(
            _make_view(progress, state, sess),
            console=CONSOLE,
            refresh_per_second=4,
            transient=False,
        ) as live,
    ):
        while len(state.results) < total and attempt_idx < max_attempts:
            if state.stop_requested:
                _print(
                    f"Interrupted at {len(state.results)}/{total}; "
                    "writing partial report."
                )
                return

            # Pacing: ≥ interval_sec between request *starts*. For 28K
            # prompts the response itself usually takes longer than the
            # interval, so this only sleeps when responses are unusually
            # fast (small first turn or fully cached prefix).
            if last_send_monotonic > 0:
                wait = last_send_monotonic + args.interval_sec - time.monotonic()
                if wait > 0:
                    time.sleep(wait)

            target_turn = sess.turn_completed + 1
            initial_target, append_target, user_text = _build_user_message(
                target_turn, initial_sampler, append_sampler,
            )
            output_target = output_sampler()
            body: dict[str, Any] = {
                "model": model_cfg.name,
                "messages": [*sess.messages, {"role": "user", "content": user_text}],
                "max_completion_tokens": output_target,
                "stream": False,
            }

            attempt_idx += 1
            last_send_monotonic = time.monotonic()
            outcome = _send_one_request(client, url, headers, body)

            _process_outcome(
                outcome=outcome,
                state=state,
                sess=sess,
                attempt_idx=attempt_idx,
                target_turn=target_turn,
                user_text=user_text,
                initial_target=initial_target,
                append_target=append_target,
                output_target=output_target,
            )
            if len(state.results) > prev_success_count:
                progress.advance(task_id)
                prev_success_count = len(state.results)
            live.update(_make_view(progress, state, sess))


def _build_user_message(
    target_turn: int,
    initial_sampler: Sampler,
    append_sampler: Sampler,
) -> tuple[int, int, str]:
    """Return ``(initial_target, append_target, user_text)`` for this turn.

    Exactly one of ``initial_target`` / ``append_target`` is non-zero,
    determined by whether this is the first turn in the session."""
    if target_turn == 1:
        initial_target = initial_sampler()
        return (
            initial_target,
            0,
            "Summarize the following text:\n\n"
            + make_text_for_tokens(initial_target),
        )
    append_target = append_sampler()
    return (
        0,
        append_target,
        "Continue with another paragraph:\n\n"
        + make_text_for_tokens(append_target),
    )


def _process_outcome(
    *,
    outcome: _SendOutcome,
    state: RunState,
    sess: _SessionState,
    attempt_idx: int,
    target_turn: int,
    user_text: str,
    initial_target: int,
    append_target: int,
    output_target: int,
) -> None:
    """Apply one ``_SendOutcome`` to ``state`` + ``sess`` (mutates both)."""
    if outcome.error_kind is not None:
        _append_failure(
            state, attempt_idx, sess.session_id, target_turn,
            outcome.error_kind, outcome.error_detail,
            outcome.elapsed_ms, outcome.wall_iso,
        )
        sess.consecutive_failures += 1
        _restart_if_too_many_failures(sess)
        return

    status = outcome.status
    parsed = outcome.parsed
    assert status is not None  # error_kind is None → status is set

    if status != 200:
        detail = (
            json.dumps(parsed)[:300] if isinstance(parsed, dict)
            else str(parsed)[:300]
        )
        if is_overflow(status, parsed):
            _append_failure(
                state, attempt_idx, sess.session_id, target_turn,
                "context_overflow", f"HTTP {status}: {detail}",
                outcome.elapsed_ms, outcome.wall_iso,
            )
            _print(
                f"[session {sess.session_id}] context overflow at turn "
                f"{target_turn}; restarting session"
            )
            sess.restart()
        else:
            _append_failure(
                state, attempt_idx, sess.session_id, target_turn,
                "http_error", f"HTTP {status}: {detail}",
                outcome.elapsed_ms, outcome.wall_iso,
            )
            sess.consecutive_failures += 1
            _restart_if_too_many_failures(sess)
        return

    if not isinstance(parsed, dict):
        _append_failure(
            state, attempt_idx, sess.session_id, target_turn,
            "http_error", "200 but non-JSON body",
            outcome.elapsed_ms, outcome.wall_iso,
        )
        sess.consecutive_failures += 1
        _restart_if_too_many_failures(sess)
        return

    prompt_tokens, cached_tokens, completion_tokens = _extract_usage(parsed, state)
    sess.turn_completed += 1
    sess.consecutive_failures = 0
    result = RequestResult(
        request_idx=len(state.results) + 1,
        session_id=sess.session_id,
        turn=sess.turn_completed,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        elapsed_ms=outcome.elapsed_ms,
        wall_time_iso=outcome.wall_iso,
        initial_target_tokens=initial_target,
        append_target_tokens=append_target,
        output_target_tokens=output_target,
    )
    state.results.append(result)

    sess.messages.append({"role": "user", "content": user_text})
    sess.messages.append({
        "role": "assistant",
        "content": _extract_assistant_content(parsed),
    })


def _append_failure(
    state: RunState,
    attempt_idx: int,
    session_id: int,
    target_turn: int,
    reason: str,
    detail: str,
    elapsed_ms: float,
    wall_iso: str,
) -> None:
    state.failures.append(FailureRecord(
        attempt_idx=attempt_idx,
        session_id=session_id,
        turn=target_turn,
        reason=reason,
        detail=detail,
        elapsed_ms=elapsed_ms,
        wall_time_iso=wall_iso,
    ))


def _restart_if_too_many_failures(sess: _SessionState) -> None:
    if sess.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        _print(
            f"[session {sess.session_id}] {sess.consecutive_failures} "
            "consecutive failures; restarting session"
        )
        sess.restart()


# ── Reports ───────────────────────────────────────────────────────────────────


def write_reports(
    output_dir: Path,
    args: argparse.Namespace,
    provider: ProviderConfig,
    model_cfg: ModelConfig,
    state: RunState,
    started_iso: str,
    ended_iso: str,
    duration_sec: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config_record = {
        "started_at": started_iso,
        "ended_at": ended_iso,
        "duration_sec": round(duration_sec, 2),
        "args": vars(args),
        "provider": {
            "name": provider.name,
            "base_url": provider.base_url,
            "api_format": str(provider.api_format),
            "verify_ssl": provider.verify_ssl,
        },
        "model": {
            "name": model_cfg.name,
            "profile": model_cfg.profile,
            "profile_snapshot": model_cfg.profile_snapshot,
        },
        "git_commit": _git_commit(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_record, indent=2, ensure_ascii=False)
    )

    _write_per_request_csv(output_dir / "per_request.csv", state.results)
    _write_summary(
        output_dir / "summary.md",
        args, provider, model_cfg, state,
        started_iso, ended_iso, duration_sec,
    )


def _write_per_request_csv(path: Path, results: list[RequestResult]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "request_idx", "session_id", "turn",
            "prompt_tokens", "cached_tokens", "completion_tokens",
            "elapsed_ms", "wall_time_iso",
            "initial_target_tokens", "append_target_tokens", "output_target_tokens",
        ])
        for r in results:
            w.writerow([
                r.request_idx, r.session_id, r.turn,
                r.prompt_tokens, r.cached_tokens, r.completion_tokens,
                f"{r.elapsed_ms:.1f}", r.wall_time_iso,
                r.initial_target_tokens, r.append_target_tokens, r.output_target_tokens,
            ])


def _write_summary(
    path: Path,
    args: argparse.Namespace,
    provider: ProviderConfig,
    model_cfg: ModelConfig,
    state: RunState,
    started_iso: str,
    ended_iso: str,
    duration_sec: float,
) -> None:
    results = state.results
    failures = state.failures
    total_target: int = args.total_requests
    attempts = len(results) + len(failures)

    sum_pt = sum(r.prompt_tokens for r in results)
    sum_ct = sum(r.cached_tokens for r in results)
    hit_rate = (sum_ct / sum_pt * 100) if sum_pt > 0 else 0.0

    overflow_restarts = sum(1 for f in failures if f.reason == "context_overflow")
    # Number of sessions actually started = max session_id seen across both
    # results and failures (sessions can start with a failure too).
    seen_sessions: set[int] = {r.session_id for r in results}
    seen_sessions.update(f.session_id for f in failures)
    num_sessions = max(seen_sessions) if seen_sessions else 0

    is_complete = len(results) >= total_target

    lines: list[str] = []
    lines.append("# Prompt Cache Hit Rate Report")
    lines.append("")
    lines.append(f"- Started: {started_iso}")
    lines.append(f"- Ended: {ended_iso}")
    lines.append(f"- Provider: {provider.name} ({provider.base_url})")
    lines.append(f"- Model: {model_cfg.name}  (profile: {model_cfg.profile})")
    lines.append(f"- Seed: {args.seed}")
    lines.append(f"- Git commit: {_git_commit()}")
    lines.append("")
    if not is_complete:
        lines.append(
            f"> **INCOMPLETE: {len(results)}/{total_target} successful "
            "requests collected before termination.**"
        )
        lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- **Cache hit rate: {hit_rate:.2f}%** "
        f"(Σ cached_tokens {sum_ct:,} / Σ prompt_tokens {sum_pt:,})"
    )
    lines.append(f"- Successful requests: {len(results)} / {attempts} attempted")
    lines.append(
        f"- Sessions: {num_sessions} ({overflow_restarts} restarts due to "
        "context overflow)"
    )
    lines.append(f"- Wall-clock duration: {_fmt_duration(duration_sec)}")
    if state.cached_field_missing_logged:
        lines.append(
            "- ⚠ `usage.prompt_tokens_details` was missing on at least one "
            "response → hit rate above may be 0 even if caching is working"
        )
    lines.append("")

    lines.append("## Per-session breakdown")
    lines.append("")
    by_session: dict[int, list[RequestResult]] = {}
    for r in results:
        by_session.setdefault(r.session_id, []).append(r)
    session_rows: list[list[str]] = []
    for sid in sorted(by_session):
        rs = by_session[sid]
        s_pt = sum(r.prompt_tokens for r in rs)
        s_ct = sum(r.cached_tokens for r in rs)
        s_rate = (s_ct / s_pt * 100) if s_pt > 0 else 0.0
        session_rows.append([
            str(sid), str(len(rs)),
            f"{s_pt:,}", f"{s_ct:,}", f"{s_rate:.2f}%",
        ])
    lines.append(tabulate(
        session_rows,
        headers=["Session", "Turns", "Σ prompt_tokens", "Σ cached_tokens", "Hit rate"],
        tablefmt="github",
    ))
    lines.append("")

    lines.append("## Sampling fidelity")
    lines.append("")
    lines.append("Compares the sampler's output (what we asked for) against the spec. "
                 "These are *target* tokens, not server-reported.")
    lines.append("")
    initials = [r.initial_target_tokens for r in results if r.initial_target_tokens > 0]
    appends = [r.append_target_tokens for r in results if r.append_target_tokens > 0]
    outputs = [r.output_target_tokens for r in results]
    fidelity_rows: list[list[str]] = []
    for label, spec_avg, spec_max, samples in (
        ("Initial prompt", args.initial_avg_tokens, args.initial_max_tokens, initials),
        ("Append per turn", args.append_avg_tokens, args.append_max_tokens, appends),
        ("Output per turn", args.output_avg_tokens, args.output_max_tokens, outputs),
    ):
        spec = f"{spec_avg:,} / {spec_max:,}"
        if samples:
            fidelity_rows.append([
                label, spec,
                f"{statistics.mean(samples):,.0f}", f"{max(samples):,}",
                str(len(samples)),
            ])
        else:
            fidelity_rows.append([label, spec, "—", "—", "0"])
    lines.append(tabulate(
        fidelity_rows,
        headers=["Stat", "Spec (avg / max)", "Empirical mean", "Empirical max", "N"],
        tablefmt="github",
    ))
    lines.append("")

    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.append("None.")
    else:
        by_reason: dict[str, list[FailureRecord]] = {}
        for f in failures:
            by_reason.setdefault(f.reason, []).append(f)
        notes = {
            "context_overflow": "Triggered session restart",
            "http_error": "Counted toward consecutive-failure cap",
            "timeout": f"Per-request timeout = {args.request_timeout_sec}s",
            "network": "httpx network exception",
        }
        failure_rows: list[list[str]] = [
            [reason, str(len(by_reason[reason])), notes.get(reason, "")]
            for reason in sorted(by_reason)
        ]
        lines.append(tabulate(
            failure_rows,
            headers=["Reason", "Count", "Notes"],
            tablefmt="github",
        ))
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _git_commit() -> str:
    try:
        out = subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"
    return out.decode().strip()


# ── Misc ──────────────────────────────────────────────────────────────────────


def _print(msg: str) -> None:
    """Log a line via the shared rich Console.

    Safe to call while ``Live`` is active — rich routes the line *above* the
    live region so progress and status stay pinned at the bottom of the
    terminal.
    """
    CONSOLE.print(msg)


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name).strip("_")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        _print(f"ERROR: config file not found: {config_path}")
        return 2
    suite = SuiteConfig.from_yaml(config_path)

    try:
        provider, model_cfg = select_target(suite, args.provider, args.model)
    except ValueError as e:
        _print(f"ERROR: {e}")
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_dir)
        / f"{timestamp}_{_slugify(provider.name)}_{_slugify(model_cfg.name)}"
    )

    _print(f"Provider: {provider.name} ({provider.base_url})")
    _print(f"Model:    {model_cfg.name}")
    _print(f"Output:   {run_dir}")
    _print(
        f"Target:   {args.total_requests} successful requests, "
        f"≥{args.interval_sec:g}s apart"
    )
    _print("")

    state = RunState()

    def _on_sigint(_signum: int, _frame: FrameType | None) -> None:
        state.stop_requested = True
        _print(
            "\n(SIGINT received — finishing current request "
            "then writing partial report)"
        )

    signal.signal(signal.SIGINT, _on_sigint)

    started_at = time.monotonic()
    started_iso = datetime.now().astimezone().isoformat()
    try:
        run_test(args, provider, model_cfg, state)
    finally:
        ended_iso = datetime.now().astimezone().isoformat()
        duration = time.monotonic() - started_at
        write_reports(
            run_dir, args, provider, model_cfg, state,
            started_iso, ended_iso, duration,
        )

    sum_pt = sum(r.prompt_tokens for r in state.results)
    sum_ct = sum(r.cached_tokens for r in state.results)
    rate = (sum_ct / sum_pt * 100) if sum_pt > 0 else 0.0
    _print("")
    _print(f"Done. Hit rate: {rate:.2f}%  ({sum_ct:,} / {sum_pt:,})")
    _print(f"Report: {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
