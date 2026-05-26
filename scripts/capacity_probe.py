"""Account capacity probe — measure RPM/TPM limits and business-traffic throughput.

Two subcommands:

- ``probe``    — find the account's RPM/TPM ceilings (Stage 0 header sniff,
                 then Stage 1 small-request ramp for RPM, then Stage 2
                 large-request ramp for TPM)
- ``capacity`` — under a given business request profile (avg input/output
                 tokens), find the sustained RPM/TPM the account can hold,
                 and classify which limit is the bottleneck

Independent of pytest — see docs/design/capacity-probe-script.md for full
design rationale (dual rate-limit problem, 429 three-tier classification,
asyncio dispatcher, self-adjusting feedback loop, safety mode).

Usage:
    uv run python scripts/capacity_probe.py probe \\
        --config config.yaml --provider mgtv --model glm-5.1 \\
        --max-context 32000

    uv run python scripts/capacity_probe.py capacity \\
        --config config.yaml --provider mgtv --model glm-5.1 \\
        --avg-input-tokens 1000 --avg-output-tokens 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
from _common import (
    CONSOLE,
    DEFAULT_TIMEOUT_SEC,
    ModelConfig,
    ProviderConfig,
    SuiteConfig,
    select_target,
    slugify,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "reports/capacity_probe"
DEFAULT_MAX_DURATION_MIN = 10
"""Safety cap (see design §12 / §7). Some providers run abuse detection on
sustained 429-heavy traffic — bail by default before tripping it."""
LONG_RUN_HARD_CAP_MIN = 120
"""Even with --allow-long-run, refuse to run longer than this — past 2h
the cost/benefit no longer pencils out for any realistic capacity question."""


# ── CLI ──────────────────────────────────────────────────────────────────────


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Args shared by both subcommands."""
    p.add_argument("--config", default="config.yaml",
                   help="Path to SuiteConfig YAML (default: config.yaml)")
    p.add_argument("--provider", default=None,
                   help="Provider name from config (required if config has >1)")
    p.add_argument("--model", default=None,
                   help="Model name from the provider (required if >1)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help=f"Parent directory for run output "
                        f"(default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--request-timeout-sec", type=float,
                   default=DEFAULT_TIMEOUT_SEC,
                   help=f"Per-request timeout (default: {DEFAULT_TIMEOUT_SEC}s)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for length-sampling reproducibility "
                        "(default: 42)")
    p.add_argument("--max-duration-minutes", type=int,
                   default=DEFAULT_MAX_DURATION_MIN,
                   help=f"Hard wall-clock cap; bail when reached even if "
                        f"not converged (default: {DEFAULT_MAX_DURATION_MIN})")
    p.add_argument("--allow-long-run", action="store_true",
                   help=f"Unlock --max-duration-minutes above "
                        f"{DEFAULT_MAX_DURATION_MIN}; hard cap is then "
                        f"{LONG_RUN_HARD_CAP_MIN}min")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="capacity_probe",
        description=(
            "Probe account RPM/TPM limits (probe) or measure sustained "
            "throughput under a given business request profile (capacity)."
        ),
    )
    subs = p.add_subparsers(dest="mode", required=True, metavar="{probe,capacity}")

    p_probe = subs.add_parser(
        "probe",
        help="Find account RPM/TPM ceilings",
        description=(
            "Stage 0: sniff response headers for upstream-declared limits. "
            "Stage 1: ramp small-request concurrency until 429 — measures RPM. "
            "Stage 2: ramp large-request load under RPM cap — measures TPM."
        ),
    )
    _add_common_args(p_probe)
    p_probe.add_argument("--max-context", type=int, required=True,
                         help="Model context-window upper bound (tokens); "
                              "Stage 2 builds prompts up to this size. "
                              "Required — passed explicitly rather than "
                              "read from a possibly-stale profile.")
    p_probe.add_argument("--stage1-workers", type=int,
                         default=STAGE_1_WORKERS_DEFAULT,
                         help=f"Stage 1 (RPM probe) concurrent workers "
                              f"(default: {STAGE_1_WORKERS_DEFAULT}). Raise "
                              f"for accounts with very high RPM ceilings.")
    p_probe.add_argument("--stage2-workers", type=int,
                         default=STAGE_2_WORKERS_DEFAULT,
                         help=f"Stage 2 (TPM probe) concurrent workers "
                              f"(default: {STAGE_2_WORKERS_DEFAULT}). "
                              f"Sized for ~10M TPM @ ~30K tokens/req; raise "
                              f"if reports show 'bounded_by_rpm' or no 429s.")
    p_probe.add_argument("--saturation-rpm", type=float,
                         default=SATURATION_RPM_DEFAULT,
                         help=f"Dispatcher RPM cap; queue back-pressure does "
                              f"the real throttling "
                              f"(default: {SATURATION_RPM_DEFAULT:.0f}).")

    p_capacity = subs.add_parser(
        "capacity",
        help="Measure sustained throughput under business request profile",
        description=(
            "Sample request shapes from --avg-input-tokens / --avg-output-tokens "
            "(truncated normal), apply self-adjusting RPM control "
            "(1.5x/0.7x feedback loop), collect 3-minute steady-state window."
        ),
    )
    _add_common_args(p_capacity)
    p_capacity.add_argument("--avg-input-tokens", type=int, required=True,
                            help="Mean input tokens per request "
                                 "(take from your business traffic P50)")
    p_capacity.add_argument("--avg-output-tokens", type=int, required=True,
                            help="Mean output tokens per request")
    p_capacity.add_argument("--max-input-tokens", type=int, default=None,
                            help="Sampler upper clamp on input "
                                 "(default: 2 * --avg-input-tokens)")
    p_capacity.add_argument("--max-output-tokens", type=int, default=None,
                            help="Sampler upper clamp on output "
                                 "(default: 2 * --avg-output-tokens)")
    p_capacity.add_argument("--start-rpm", type=int, default=100,
                            help="Initial target RPM if no probe data exists "
                                 "(default: 100). Raise for high-TPM "
                                 "accounts so the 1.5x ramp reaches steady "
                                 "state within --max-duration-minutes.")
    p_capacity.add_argument("--workers", type=int, default=None,
                            help="Override per-minute worker count "
                                 "(default: auto-sized from target RPM, "
                                 "see _auto_workers_for_rpm).")
    p_capacity.add_argument("--steady-state-minutes", type=int, default=3,
                            help="Consecutive minutes of stable 429-rate "
                                 "required to declare convergence (default: 3)")

    return p.parse_args()


# ── Request records ──────────────────────────────────────────────────────────


@dataclass
class RequestSpec:
    """Inputs for one chat-completion request.

    Generated by the dispatcher (probe or capacity mode), consumed by workers.
    Text is pre-computed so workers don't share an rng — keeps the
    dispatcher's sampling deterministic regardless of worker concurrency.
    """

    prompt_text: str
    input_tokens_target: int
    max_output_tokens: int


@dataclass
class RequestRecord:
    """One row in per_request.csv. Populated by ``_send_one``.

    ``classification`` is set by an injected classifier function so Step 4
    can plug in real rate-limit-source detection without changing this
    dataclass.
    """

    ts: float
    elapsed_ms: float
    status: int
    input_tokens: int
    output_tokens: int
    classification: str
    retry_after_sec: float | None
    error_detail: str = ""


# Type alias: a classifier maps (status, headers, parsed_body) → label.
# Default impl below; Step 4 replaces it.
Classifier = "Callable[[int, dict[str, str], dict[str, Any] | str | None], str]"


def classify_default(
    status: int,
    headers: dict[str, str],  # noqa: ARG001
    body: dict[str, Any] | str | None,  # noqa: ARG001
) -> str:
    """Minimal classifier — used as a fallback when ``classify_429`` doesn't
    fire. Treats all 429 as ``429_unclassified``.
    """
    if 200 <= status < 300:
        return "ok"
    if status == 429:
        return "429_unclassified"
    if 500 <= status < 600:
        return "5xx"
    return f"http_{status}"


# ── 429 classifier (three tiers, see design §6) ──────────────────────────────

RPM_BODY_KEYWORDS = (
    "limit type: requests",      # litellm-style (mgtv-confirmed)
    "requests per minute",
    "rpm",
    "request rate",
    "request limit",             # aliyun dashscope: "exceeded your current request limit"
    "exceeded your current request",
    "too many requests",
)
"""Substrings searched (case-insensitive) in 429 error bodies to identify
RPM-triggered limits when Tier 1/2 don't yield a match."""

TPM_BODY_KEYWORDS = (
    "limit type: tokens",        # litellm-style
    "tokens per minute",
    "tpm",
    "token rate",
    "token quota",
    "token limit per",
)
"""Substrings searched (case-insensitive) in 429 error bodies to identify
TPM-triggered limits when Tier 1/2 don't yield a match."""


def _tier1_rate_limit_type(headers: dict[str, str]) -> str | None:
    """litellm-style ``rate_limit_type: requests|tokens`` header (preferred
    discriminator — confirmed on mgtv)."""
    raw = headers.get("rate_limit_type") or headers.get("Rate-Limit-Type")
    if not raw:
        return None
    val = raw.strip().lower()
    if val == "requests":
        return "429_rpm"
    if val == "tokens":
        return "429_tpm"
    return None


def _tier2_ratelimit_headers(headers: dict[str, str]) -> str | None:
    """OpenAI / litellm ``x-ratelimit-*-remaining-{requests,tokens}=0``.

    Matches case-insensitively on the header name's *suffix* — covers both
    ``x-ratelimit-remaining-requests`` (OpenAI) and
    ``x-ratelimit-api_key-remaining-requests`` (litellm with api_key infix).
    """
    for name, value in headers.items():
        lower = name.lower()
        if "remaining-requests" in lower and _is_zero(value):
            return "429_rpm"
        if "remaining-tokens" in lower and _is_zero(value):
            return "429_tpm"
    return None


def _is_zero(value: str) -> bool:
    try:
        return int(value.strip()) == 0
    except (ValueError, AttributeError):
        return False


def _tier3_body_keywords(body: dict[str, Any] | str | None) -> str | None:
    """Substring match against ``error.message`` (or raw body text).

    Returns the first match; if both RPM and TPM keywords appear, RPM wins
    because providers tend to put the triggering limit type first.
    """
    if body is None:
        return None
    if isinstance(body, dict):
        err_raw = cast("Any", body.get("error"))
        if isinstance(err_raw, dict):
            err = cast("dict[str, Any]", err_raw)
            text = str(err.get("message") or err.get("code") or "")
        else:
            text = str(err_raw or body)
    else:
        text = body
    text_lower = text.lower()
    for kw in RPM_BODY_KEYWORDS:
        if kw in text_lower:
            return "429_rpm"
    for kw in TPM_BODY_KEYWORDS:
        if kw in text_lower:
            return "429_tpm"
    return None


def classify_429(
    status: int,
    headers: dict[str, str],
    body: dict[str, Any] | str | None,
) -> str:
    """Three-tier 429 classifier (see design §6 for full rationale).

    Non-429 status codes delegate to ``classify_default``. For 429:

      Tier 1 — ``rate_limit_type`` header (litellm-style, machine-readable)
      Tier 2 — ``x-ratelimit-*-remaining-{requests,tokens}=0`` (OpenAI/litellm)
      Tier 3 — body ``error.message`` keyword match (English + litellm phrasing)

    Returns ``429_rpm``, ``429_tpm``, or ``429_unclassified`` for 429s.
    """
    if status != 429:
        return classify_default(status, headers, body)
    for tier in (_tier1_rate_limit_type(headers),
                 _tier2_ratelimit_headers(headers),
                 _tier3_body_keywords(body)):
        if tier is not None:
            return tier
    return "429_unclassified"


def parse_retry_after(headers: httpx.Headers | dict[str, str]) -> float | None:
    """Read ``Retry-After`` header as seconds. HTTP-date form is not supported
    (servers we target use the integer-seconds form)."""
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# ── Response parsing ─────────────────────────────────────────────────────────


def _extract_usage_tokens(parsed: dict[str, Any] | str | None) -> tuple[int, int]:
    """Return ``(prompt_tokens, completion_tokens)`` from an OpenAI usage block.

    Returns ``(0, 0)`` on any structural mismatch — capacity stats treat
    missing usage as zero, which is conservative for throughput estimation.
    """
    if not isinstance(parsed, dict):
        return (0, 0)
    usage_raw = cast("Any", parsed.get("usage"))
    if not isinstance(usage_raw, dict):
        return (0, 0)
    usage = cast("dict[str, Any]", usage_raw)
    pt_raw = usage.get("prompt_tokens") or 0
    ct_raw = usage.get("completion_tokens") or 0
    try:
        return (int(pt_raw), int(ct_raw))
    except (TypeError, ValueError):
        return (0, 0)


def _extract_error_detail(parsed: dict[str, Any] | str | None) -> str:
    """Best-effort one-line error description from a non-2xx response body."""
    if parsed is None:
        return ""
    if isinstance(parsed, str):
        return parsed[:200]
    err_raw = cast("Any", parsed.get("error"))
    if isinstance(err_raw, dict):
        err = cast("dict[str, Any]", err_raw)
        msg = err.get("message") or err.get("code") or ""
        return str(msg)[:200]
    if err_raw:
        return str(err_raw)[:200]
    return str(parsed)[:200]


# ── Request core ─────────────────────────────────────────────────────────────


@dataclass
class _LoadState:
    """Mutable state shared across dispatcher + workers in one ``run_load``."""

    global_pause_until: float = 0.0
    """Monotonic time; all workers idle until this point. Set when any
    worker observes a 429 with ``Retry-After`` — cooperative back-off
    without needing a separate state machine."""

    stop_requested: bool = False
    """Set by SIGINT handler or convergence logic; dispatcher exits its
    loop and signals workers to drain."""


async def _send_one(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    model: str,
    spec: RequestSpec,
    classifier: Callable[[int, dict[str, str], dict[str, Any] | str | None], str],
) -> RequestRecord:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt_text}],
        "max_tokens": spec.max_output_tokens,
    }
    ts = time.time()
    t0 = time.monotonic()
    try:
        resp = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException as e:
        return RequestRecord(
            ts=ts, elapsed_ms=(time.monotonic() - t0) * 1000,
            status=0, input_tokens=0, output_tokens=0,
            classification="timeout", retry_after_sec=None,
            error_detail=str(e)[:200],
        )
    except httpx.HTTPError as e:
        return RequestRecord(
            ts=ts, elapsed_ms=(time.monotonic() - t0) * 1000,
            status=0, input_tokens=0, output_tokens=0,
            classification="network", retry_after_sec=None,
            error_detail=str(e)[:200],
        )

    elapsed_ms = (time.monotonic() - t0) * 1000
    parsed: dict[str, Any] | str
    try:
        raw_obj: object = resp.json()
    except ValueError:
        parsed = resp.text
    else:
        parsed = (
            cast("dict[str, Any]", raw_obj)
            if isinstance(raw_obj, dict) else resp.text
        )

    cls = classifier(resp.status_code, dict(resp.headers), parsed)
    retry_after = parse_retry_after(resp.headers)

    input_tokens, output_tokens = _extract_usage_tokens(parsed)
    detail = _extract_error_detail(parsed) if cls != "ok" else ""

    return RequestRecord(
        ts=ts, elapsed_ms=elapsed_ms, status=resp.status_code,
        input_tokens=input_tokens, output_tokens=output_tokens,
        classification=cls, retry_after_sec=retry_after,
        error_detail=detail,
    )


async def run_load(  # noqa: C901, PLR0913
    *,
    url: str,
    headers: dict[str, str],
    model: str,
    spec_fn: Callable[[], RequestSpec],
    target_rpm: float,
    duration_sec: float,
    n_workers: int,
    timeout_sec: float,
    classifier: Callable[
        [int, dict[str, str], dict[str, Any] | str | None], str,
    ] = classify_default,
    on_record: Callable[[RequestRecord], None] | None = None,
    state: _LoadState | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[RequestRecord]:
    """Drive concurrent load at ``target_rpm`` for up to ``duration_sec`` seconds.

    Dispatcher meters at fixed interval ``60/target_rpm`` using drift-free
    absolute scheduling. Workers pull from a bounded queue. 429s with
    ``Retry-After`` set a global pause that all workers honor before their
    next send — cooperative back-off without a separate state machine.

    Returns one ``RequestRecord`` per attempted send. ``ts`` is captured at
    dispatch time, so the list is approximately dispatch-ordered (workers
    may interleave by a few ms, but for analysis ``ts`` is the reference).

    ``state`` and ``transport`` are injection points: ``state`` lets a caller
    trigger early-stop (e.g. SIGINT, convergence); ``transport`` swaps in
    ``httpx.MockTransport`` for unit testing.
    """
    if target_rpm <= 0:
        msg = f"target_rpm must be positive, got {target_rpm!r}"
        raise ValueError(msg)
    if n_workers <= 0:
        msg = f"n_workers must be positive, got {n_workers!r}"
        raise ValueError(msg)

    records: list[RequestRecord] = []
    ls = state or _LoadState()
    queue: asyncio.Queue[RequestSpec | None] = asyncio.Queue(maxsize=n_workers * 2)

    async def worker(client: httpx.AsyncClient) -> None:
        while True:
            spec = await queue.get()
            try:
                if spec is None:
                    return
                pause = ls.global_pause_until - time.monotonic()
                if pause > 0:
                    await asyncio.sleep(pause)
                rec = await _send_one(client, url, headers, model, spec, classifier)
                records.append(rec)
                if on_record is not None:
                    on_record(rec)
                if rec.classification.startswith("429") and rec.retry_after_sec:
                    new_pause = time.monotonic() + rec.retry_after_sec
                    if new_pause > ls.global_pause_until:
                        ls.global_pause_until = new_pause
            finally:
                queue.task_done()

    async def dispatcher() -> None:
        interval = 60.0 / target_rpm
        next_dispatch = time.monotonic()
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline and not ls.stop_requested:
            now = time.monotonic()
            sleep_for = next_dispatch - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            await queue.put(spec_fn())
            next_dispatch += interval
        for _ in range(n_workers):
            await queue.put(None)

    async with httpx.AsyncClient(
        timeout=timeout_sec,
        transport=transport,
        limits=httpx.Limits(
            max_connections=n_workers * 2,
            max_keepalive_connections=n_workers,
        ),
    ) as client:
        await asyncio.gather(
            dispatcher(),
            *[worker(client) for _ in range(n_workers)],
        )

    return records


# ── Probe-mode helpers (see design §3) ───────────────────────────────────────


SATURATION_RPM_DEFAULT = 60000.0
"""Default dispatch RPM cap for probe stages. Set well above any realistic
provider limit so queue back-pressure (sized ``n_workers*2``) — not the
dispatcher interval — gates the actual rate. ``Retry-After`` back-off in
``run_load`` handles 429 bursts. Override with ``--saturation-rpm`` if
the dispatcher itself becomes a bottleneck."""

STAGE_1_WORKERS_DEFAULT = 200
"""Default Stage 1 worker count. Stage 1 sends small (~1s p50) requests to
find the RPM ceiling; each worker sustains ~60 RPM, so 200 covers RPM
limits up to ~12k. For accounts with higher RPM ceilings, raise via
``--stage1-workers`` (mind OS file-handle limits — n_workers*2 sockets)."""

STAGE_2_WORKERS_DEFAULT = 200
"""Default Stage 2 worker count. Stage 2 sends large requests
(~95 % of ``--max-context``); to test a TPM ceiling of ~10M with
30K-token requests at p50≈30s, we need ~167 in-flight requests, so 200
gives modest headroom. For very large contexts (≥128K) or higher TPM
ceilings, raise via ``--stage2-workers``."""

PROBE_RAMP_LATENCY_SLACK_SEC = 30.0
"""When budget is split across stages, reserve this much for the initial
ramp-up before the system reaches steady state. Stages below this are
clamped to a minimum useful duration."""


@dataclass
class StageResult:
    """Outcome of one probe stage. Reported into the run dir's summary."""

    stage: str
    elapsed_sec: float
    records: list[RequestRecord]
    measured_rpm: float
    measured_tpm: float
    class_counts: dict[str, int]
    discovered_from_headers: bool = False


@dataclass
class ProbeOutcome:
    """Final probe result; passed to report writer (Step 7)."""

    rpm_limit: float | None
    tpm_limit: float | None
    rpm_source: str  # "header" | "measured" | "unknown"
    tpm_source: str  # "header" | "measured" | "bounded_by_rpm" | "unknown"
    stage_0_headers: dict[str, str]
    stages: list[StageResult]


def _make_small_spec() -> RequestSpec:
    """~50-token prompt + max_tokens=10. RPM-bound by design — server
    almost certainly hits RPM ceiling long before TPM."""
    return RequestSpec(
        prompt_text="Reply with 'ok'.",
        input_tokens_target=10,
        max_output_tokens=10,
    )


def _make_big_spec_factory(max_context: int) -> Callable[[], RequestSpec]:
    """~70% of context as input + ~25% as max_tokens. Big enough that a
    handful of requests in 60s can exhaust any plausible TPM budget."""
    # Import here to keep top-level imports lean (only needed in probe mode).
    from _common import make_text_for_tokens  # noqa: PLC0415
    input_tokens = max(int(max_context * 0.70), 100)
    max_output = max(int(max_context * 0.25), 100)
    prompt = make_text_for_tokens(input_tokens)

    def _spec() -> RequestSpec:
        return RequestSpec(
            prompt_text=prompt,
            input_tokens_target=input_tokens,
            max_output_tokens=max_output,
        )
    return _spec


def _compute_throughput(
    records: list[RequestRecord],
    elapsed_sec: float,
) -> tuple[float, float]:
    """Return (RPM, TPM) from successful records.

    Anchors the rate to the *OK-request window* (first dispatch ts → last
    response finish ts) when there are ≥2 OK records, so ramp-up (queue
    filling) and ramp-down (in-flight requests after the dispatcher
    stops) don't dilute the average. Falls back to ``elapsed_sec`` when
    the OK window can't be derived (single record, or ts collapsed)."""
    ok = [r for r in records if r.classification == "ok"]
    if not ok:
        return (0.0, 0.0)
    if len(ok) >= 2:
        first_start = min(r.ts for r in ok)
        last_finish = max(r.ts + r.elapsed_ms / 1000.0 for r in ok)
        window = last_finish - first_start
        if window <= 0:
            window = elapsed_sec
    else:
        window = elapsed_sec
    if window <= 0:
        return (0.0, 0.0)
    rpm = len(ok) * 60.0 / window
    tpm = sum(r.input_tokens + r.output_tokens for r in ok) * 60.0 / window
    return (rpm, tpm)


def _count_classifications(records: list[RequestRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.classification] = counts.get(r.classification, 0) + 1
    return counts


def _parse_int_header(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


def _discover_limits_from_headers(
    headers: dict[str, str],
) -> tuple[int | None, int | None]:
    """Scan response headers for upstream-declared RPM/TPM limits.

    Recognized (case-insensitive):
      - ``x-litellm-key-rpm-limit`` / ``x-litellm-key-tpm-limit``
      - any header whose name contains ``limit-requests`` / ``limit-tokens``
        without ``remaining`` (covers OpenAI + litellm api_key infix variants)
    """
    rpm: int | None = None
    tpm: int | None = None
    for name, value in headers.items():
        lower = name.lower()
        if lower == "x-litellm-key-rpm-limit" and rpm is None:
            rpm = _parse_int_header(value)
        elif lower == "x-litellm-key-tpm-limit" and tpm is None:
            tpm = _parse_int_header(value)
        elif "limit-requests" in lower and "remaining" not in lower and rpm is None:
            rpm = _parse_int_header(value)
        elif "limit-tokens" in lower and "remaining" not in lower and tpm is None:
            tpm = _parse_int_header(value)
    return rpm, tpm


async def stage_0_header_sniff(
    url: str,
    headers: dict[str, str],
    model: str,
    timeout_sec: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[int | None, int | None, dict[str, str]]:
    """Send one minimal request, return ``(rpm_limit, tpm_limit, raw_headers)``.

    Falls back to ``(None, None, {})`` on any error — the caller treats this
    as "Stage 0 didn't help, fall through to ramp probes". ``transport``
    is the standard ``httpx.MockTransport`` injection point for tests."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply 'ok'."}],
        "max_tokens": 5,
    }
    async with httpx.AsyncClient(timeout=timeout_sec, transport=transport) as client:
        try:
            resp = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as e:
            CONSOLE.print(f"[yellow]Stage 0 request failed:[/yellow] {e}")
            return (None, None, {})

    response_headers = dict(resp.headers)
    rpm, tpm = _discover_limits_from_headers(response_headers)
    return (rpm, tpm, response_headers)


async def _run_stage(
    *,
    stage_name: str,
    url: str,
    headers: dict[str, str],
    model: str,
    spec_fn: Callable[[], RequestSpec],
    duration_sec: float,
    n_workers: int,
    timeout_sec: float,
    state: _LoadState,
    saturation_rpm: float,
) -> StageResult:
    """Execute one saturation-rate stage and compute its measured rates."""
    CONSOLE.print(
        f"[bold]{stage_name}[/bold] running for "
        f"{duration_sec:.0f}s at saturation rate "
        f"(target {saturation_rpm:.0f} RPM, {n_workers} workers)…"
    )
    t0 = time.monotonic()
    records = await run_load(
        url=url, headers=headers, model=model,
        spec_fn=spec_fn,
        target_rpm=saturation_rpm,
        duration_sec=duration_sec,
        n_workers=n_workers,
        timeout_sec=timeout_sec,
        classifier=classify_429,
        state=state,
    )
    elapsed = time.monotonic() - t0
    rpm, tpm = _compute_throughput(records, elapsed)
    counts = _count_classifications(records)
    CONSOLE.print(
        f"  → {len(records)} requests, "
        f"{counts.get('ok', 0)} ok, "
        f"{counts.get('429_rpm', 0)} rpm-429, "
        f"{counts.get('429_tpm', 0)} tpm-429, "
        f"{counts.get('429_unclassified', 0)} 429-unclassified"
    )
    CONSOLE.print(
        f"  → measured: {rpm:.1f} RPM, {tpm:,.0f} TPM"
    )
    return StageResult(
        stage=stage_name,
        elapsed_sec=elapsed,
        records=records,
        measured_rpm=rpm,
        measured_tpm=tpm,
        class_counts=counts,
    )


async def _probe_orchestrate(  # noqa: PLR0912, C901
    args: argparse.Namespace,
    url: str,
    headers: dict[str, str],
    model: str,
    state: _LoadState,
) -> ProbeOutcome:
    budget_sec = args.max_duration_minutes * 60.0

    # Stage 0
    CONSOLE.print("[bold]Stage 0[/bold] sniffing response headers for "
                  "upstream-declared limits…")
    rpm_from_hdr, tpm_from_hdr, hdr_snapshot = await stage_0_header_sniff(
        url, headers, model, args.request_timeout_sec,
    )
    if rpm_from_hdr is not None:
        CONSOLE.print(f"  → RPM limit from header: {rpm_from_hdr}")
    if tpm_from_hdr is not None:
        CONSOLE.print(f"  → TPM limit from header: {tpm_from_hdr}")
    if rpm_from_hdr is None and tpm_from_hdr is None:
        CONSOLE.print("  → no limits found in headers; falling through to "
                      "active probe")

    stages: list[StageResult] = []
    rpm_limit: float | None = float(rpm_from_hdr) if rpm_from_hdr else None
    tpm_limit: float | None = float(tpm_from_hdr) if tpm_from_hdr else None
    rpm_source = "header" if rpm_from_hdr else "unknown"
    tpm_source = "header" if tpm_from_hdr else "unknown"

    n_stages_remaining = (rpm_limit is None) + (tpm_limit is None)
    per_stage_sec = max(budget_sec / max(n_stages_remaining, 1), 60.0)

    # Stage 1
    if rpm_limit is None and not state.stop_requested:
        s1 = await _run_stage(
            stage_name="Stage 1 (RPM)",
            url=url, headers=headers, model=model,
            spec_fn=_make_small_spec,
            duration_sec=per_stage_sec,
            n_workers=args.stage1_workers,
            timeout_sec=args.request_timeout_sec,
            state=state,
            saturation_rpm=args.saturation_rpm,
        )
        stages.append(s1)
        if s1.measured_rpm > 0:
            rpm_limit = s1.measured_rpm
            rpm_source = "measured"

    # Stage 2
    if tpm_limit is None and not state.stop_requested:
        big_spec_fn = _make_big_spec_factory(args.max_context)
        s2 = await _run_stage(
            stage_name="Stage 2 (TPM)",
            url=url, headers=headers, model=model,
            spec_fn=big_spec_fn,
            duration_sec=per_stage_sec,
            n_workers=args.stage2_workers,
            timeout_sec=args.request_timeout_sec,
            state=state,
            saturation_rpm=args.saturation_rpm,
        )
        stages.append(s2)
        # If we saw TPM-triggered 429s, the measured TPM is the real ceiling.
        # If we only saw RPM-triggered 429s, TPM is bounded *below* by what
        # we measured (could be much higher in reality).
        tpm_429s = s2.class_counts.get("429_tpm", 0)
        if s2.measured_tpm > 0:
            tpm_limit = s2.measured_tpm
            tpm_source = "measured" if tpm_429s > 0 else "bounded_by_rpm"

    return ProbeOutcome(
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        rpm_source=rpm_source,
        tpm_source=tpm_source,
        stage_0_headers=hdr_snapshot,
        stages=stages,
    )


def _write_probe_records_csv(run_dir: Path, stages: list[StageResult]) -> None:
    import csv as _csv  # noqa: PLC0415
    path = run_dir / "per_request.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow([
            "stage", "ts", "elapsed_ms", "status", "input_tokens",
            "output_tokens", "classification", "retry_after_sec",
            "error_detail",
        ])
        for stage in stages:
            for r in stage.records:
                writer.writerow([
                    stage.stage, r.ts, f"{r.elapsed_ms:.1f}", r.status,
                    r.input_tokens, r.output_tokens, r.classification,
                    r.retry_after_sec if r.retry_after_sec is not None else "",
                    r.error_detail.replace("\n", " "),
                ])


def _format_headers_table(headers: dict[str, str]) -> str:
    """Pick the rate-limit-relevant subset for the report; full set is in
    the per_request.csv if anyone wants to dig."""
    keep_prefixes = ("x-litellm-key-", "x-ratelimit-", "rate_limit_", "retry-after")
    rows: list[tuple[str, str]] = []
    for name, value in sorted(headers.items()):
        lower = name.lower()
        if any(lower.startswith(p) or p in lower for p in keep_prefixes):
            rows.append((name, value))
    if not rows:
        return "_(none returned by upstream)_"
    lines = ["| Header | Value |", "|---|---|"]
    lines.extend(f"| `{k}` | `{v}` |" for k, v in rows)
    return "\n".join(lines)


def _write_probe_summary_md(
    run_dir: Path,
    args: argparse.Namespace,
    provider: ProviderConfig,
    model_cfg: ModelConfig,
    outcome: ProbeOutcome,
    started_iso: str,
    ended_iso: str,
    duration_sec: float,
) -> None:
    from _common import fmt_duration as _fmt_duration  # noqa: PLC0415
    from _common import git_commit as _git_commit  # noqa: PLC0415

    rpm_str = (f"**{outcome.rpm_limit:.0f}**" if outcome.rpm_limit
               else "_unknown_")
    if outcome.tpm_limit is None:
        tpm_str = "_unknown_"
    elif outcome.tpm_source == "bounded_by_rpm":
        # Marker that this is a *lower bound*, not the real ceiling — see
        # caveat below explaining why.
        tpm_str = f"**≥ {outcome.tpm_limit:,.0f}**"
    else:
        tpm_str = f"**{outcome.tpm_limit:,.0f}**"

    # Caveat banner — most common pitfall on a probe is "I never hit 429"
    caveats: list[str] = []
    for s in outcome.stages:
        had_429 = any(c.startswith("429") for c in s.class_counts)
        if not had_429:
            caveats.append(
                f"- **{s.stage}** ran for {s.elapsed_sec:.0f}s without "
                "triggering any 429. The measured rate is a **lower bound** "
                "(client concurrency was the bottleneck, not the server). "
                "True ceiling may be higher — re-run with more workers or "
                "a higher `--max-context` to push harder."
            )
    if outcome.tpm_source == "bounded_by_rpm":
        caveats.append(
            "- **TPM ceiling**: only RPM-triggered 429s were observed; TPM "
            "is bounded *below* by the measured value but the true ceiling "
            "may be much higher (or the provider may not enforce TPM at all)."
        )

    lines = [
        "# Capacity Probe Report (probe mode)",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Provider: `{provider.name}` (`{provider.base_url}`)",
        f"- Model: `{model_cfg.name}`",
        f"- Git commit: `{_git_commit()}`",
        f"- Started: `{started_iso}` · Ended: `{ended_iso}` "
        f"· Wall-clock: `{_fmt_duration(duration_sec)}`",
        f"- Probe config: "
        f"`--max-context={args.max_context}` · "
        f"`--stage1-workers={args.stage1_workers}` · "
        f"`--stage2-workers={args.stage2_workers}` · "
        f"`--saturation-rpm={args.saturation_rpm:.0f}`",
        "",
        "## Headline",
        "",
        f"- **RPM limit**: {rpm_str} (source: `{outcome.rpm_source}`)",
        f"- **TPM limit**: {tpm_str} (source: `{outcome.tpm_source}`)",
        "",
        "Plug your real business request size into "
        "`effective_tpm = min(TPM_limit, RPM_limit × avg_tokens_per_request)` "
        "to project actual capacity — or use `capacity` mode for this directly.",
        "",
        "## Stage 0 — header sniff",
        "",
        _format_headers_table(outcome.stage_0_headers),
        "",
    ]
    if outcome.stages:
        lines.extend([
            "## Stage breakdown",
            "",
            "| Stage | Duration | Requests | OK | 429 RPM | 429 TPM | "
            "Unclassified | Measured RPM | Measured TPM |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        for s in outcome.stages:
            c = s.class_counts
            lines.append(
                f"| {s.stage} | {s.elapsed_sec:.0f}s | "
                f"{len(s.records)} | {c.get('ok', 0)} | "
                f"{c.get('429_rpm', 0)} | {c.get('429_tpm', 0)} | "
                f"{c.get('429_unclassified', 0)} | "
                f"{s.measured_rpm:.1f} | {s.measured_tpm:,.0f} |"
            )
        lines.append("")
    if caveats:
        lines.extend(["## Caveats", "", *caveats, ""])
    failures = _failure_summary([r for s in outcome.stages for r in s.records])
    if failures:
        lines.extend(["## Failures", "", *failures, ""])

    lines.extend([
        "## Run config",
        "",
        "Inputs preserved in `config.json` next to this file. "
        "Per-request rows in `per_request.csv`.",
        "",
    ])
    _ = args  # config snapshot lives in config.json, no need to duplicate here
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _failure_summary(records: list[RequestRecord]) -> list[str]:
    """Markdown table of failure-classification counts, or empty list if none."""
    counts = _count_classifications(records)
    fail_keys = [k for k in counts if k not in ("ok",)]
    if not fail_keys:
        return []
    rows = ["| Classification | Count |", "|---|---|"]
    rows.extend(f"| `{k}` | {counts[k]} |" for k in sorted(fail_keys))
    return rows


def _write_config_json(
    run_dir: Path,
    args: argparse.Namespace,
    provider: ProviderConfig,
    model_cfg: ModelConfig,
    started_iso: str,
    ended_iso: str,
) -> None:
    from _common import git_commit as _git_commit  # noqa: PLC0415
    snapshot = {
        "mode": args.mode,
        "provider": provider.name,
        "base_url": provider.base_url,
        "model": model_cfg.name,
        "git_commit": _git_commit(),
        "started_at": started_iso,
        "ended_at": ended_iso,
        # vars() pulls all argparse fields uniformly — sub-command-specific
        # args (e.g. --max-context for probe, --avg-input-tokens for capacity)
        # appear automatically without needing per-mode plumbing.
        "args": _serializable_args(args),
    }
    (run_dir / "config.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in vars(args).items():
        # argparse may return Path-like or other non-trivial values;
        # coerce anything not natively JSON-serializable to str.
        if v is None or isinstance(v, (str, int, float, bool, list, dict)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


# ── Entry points ─────────────────────────────────────────────────────────────


def run_probe(args: argparse.Namespace, run_dir: Path) -> int:
    from _common import build_headers  # noqa: PLC0415
    suite = SuiteConfig.from_yaml(Path(args.config))
    provider, model_cfg = select_target(suite, args.provider, args.model)
    url = f"{provider.base_url}/v1/chat/completions"
    headers = build_headers(provider)

    started_iso = datetime.now().astimezone().isoformat()
    t0 = time.monotonic()
    state = _LoadState()
    outcome = asyncio.run(_probe_orchestrate(args, url, headers, model_cfg.name, state))
    duration_sec = time.monotonic() - t0
    ended_iso = datetime.now().astimezone().isoformat()

    _write_probe_records_csv(run_dir, outcome.stages)
    _write_probe_summary_md(
        run_dir, args, provider, model_cfg, outcome,
        started_iso, ended_iso, duration_sec,
    )
    _write_config_json(run_dir, args, provider, model_cfg, started_iso, ended_iso)

    CONSOLE.print("")
    CONSOLE.print("[bold green]Probe complete[/bold green]")
    CONSOLE.print(f"  RPM limit: {outcome.rpm_limit} ({outcome.rpm_source})")
    CONSOLE.print(f"  TPM limit: {outcome.tpm_limit} ({outcome.tpm_source})")
    CONSOLE.print(f"  Report:    {run_dir / 'summary.md'}")
    return 0


def run_capacity(args: argparse.Namespace, run_dir: Path) -> int:
    import random as _random  # noqa: PLC0415

    from _common import Sampler, build_headers, make_text_for_tokens  # noqa: PLC0415

    suite = SuiteConfig.from_yaml(Path(args.config))
    provider, model_cfg = select_target(suite, args.provider, args.model)
    url = f"{provider.base_url}/v1/chat/completions"
    headers = build_headers(provider)

    rng = _random.Random(args.seed)
    in_max = args.max_input_tokens or args.avg_input_tokens * 2
    out_max = args.max_output_tokens or args.avg_output_tokens * 2
    input_sampler = Sampler(args.avg_input_tokens, in_max, rng)
    output_sampler = Sampler(args.avg_output_tokens, out_max, rng)

    def spec_fn() -> RequestSpec:
        input_tokens = input_sampler()
        output_tokens = output_sampler()
        return RequestSpec(
            prompt_text=make_text_for_tokens(input_tokens),
            input_tokens_target=input_tokens,
            max_output_tokens=output_tokens,
        )

    started_iso = datetime.now().astimezone().isoformat()
    t0 = time.monotonic()
    state = _LoadState()
    outcome = asyncio.run(_capacity_orchestrate(
        args=args,
        url=url,
        headers=headers,
        model=model_cfg.name,
        spec_fn=spec_fn,
        state=state,
    ))
    duration_sec = time.monotonic() - t0
    ended_iso = datetime.now().astimezone().isoformat()

    _write_capacity_records_csv(run_dir, outcome)
    _write_capacity_per_minute_csv(run_dir, outcome)
    _write_capacity_summary_md(
        run_dir, args, provider, model_cfg, outcome,
        started_iso, ended_iso, duration_sec,
    )
    _write_config_json(run_dir, args, provider, model_cfg, started_iso, ended_iso)

    CONSOLE.print("")
    CONSOLE.print("[bold green]Capacity test complete[/bold green]")
    CONSOLE.print(f"  Converged:      {outcome.converged}")
    CONSOLE.print(f"  Sustained RPM:  {outcome.sustained_rpm:.1f}")
    CONSOLE.print(f"  Sustained TPM:  {outcome.sustained_tpm:,.0f}")
    CONSOLE.print(f"  Bottleneck:     {outcome.bottleneck}")
    CONSOLE.print(f"  Report:         {run_dir / 'summary.md'}")
    return 0


# ── Capacity-mode controller (see design §4) ─────────────────────────────────

STEADY_STATE_LOW_BAND = 0.01
"""Below this 429 rate → headroom exists, ramp target RPM up by RAMP_UP_FACTOR."""
STEADY_STATE_HIGH_BAND = 0.10
"""Above this 429 rate → we're hammering the limit, back off by BACKOFF_FACTOR."""
RAMP_UP_FACTOR = 1.5
BACKOFF_FACTOR = 0.7
MIN_TARGET_RPM = 1.0
BOTTLENECK_THRESHOLD = 0.70
"""If ≥70% of 429s are one class (rpm vs tpm), declare it the bottleneck;
otherwise classification is 'MIXED'."""


@dataclass
class MinuteSample:
    """One minute of capacity testing — fed back into the controller."""

    minute_idx: int
    target_rpm: float
    n_workers: int
    records: list[RequestRecord]
    rate_429: float
    decision: str  # "ramp_up" | "hold" | "back_off"


@dataclass
class CapacityOutcome:
    converged: bool
    sustained_rpm: float
    sustained_tpm: float
    sustained_input_tpm: float
    sustained_output_tpm: float
    bottleneck: str  # "RPM" | "TPM" | "MIXED" | "NO_429"
    final_target_rpm: float
    minutes: list[MinuteSample]
    steady_minutes_collected: int


def _compute_429_rate(records: list[RequestRecord]) -> float:
    if not records:
        return 0.0
    n_429 = sum(1 for r in records if r.classification.startswith("429"))
    return n_429 / len(records)


def _classify_bottleneck(records: list[RequestRecord]) -> str:
    """Look at the 429 distribution and pick the dominant cause.

    Returns ``RPM`` / ``TPM`` / ``MIXED`` / ``NO_429``. ``MIXED`` means both
    are within a factor of each other — capacity is genuinely close to the
    boundary of both ceilings."""
    rpm = sum(1 for r in records if r.classification == "429_rpm")
    tpm = sum(1 for r in records if r.classification == "429_tpm")
    total = rpm + tpm
    if total == 0:
        return "NO_429"
    if rpm / total >= BOTTLENECK_THRESHOLD:
        return "RPM"
    if tpm / total >= BOTTLENECK_THRESHOLD:
        return "TPM"
    return "MIXED"


def _auto_workers_for_rpm(target_rpm: float) -> int:
    """Crude sizing: assume ~5s p50 latency; each worker can sustain 12 RPM.
    Floor of 10 keeps small-RPM minutes from being concurrency-starved
    (queue back-pressure handles the upper bound). For high-TPM accounts
    this scales smoothly into the hundreds; pass ``--workers`` to pin a
    fixed count if the auto sizing is off for your latency profile."""
    return max(int(target_rpm / 12) + 1, 10)


def _make_decision(rate_429: float) -> str:
    if rate_429 < STEADY_STATE_LOW_BAND:
        return "ramp_up"
    if rate_429 > STEADY_STATE_HIGH_BAND:
        return "back_off"
    return "hold"


async def _run_capacity_minute(
    *,
    minute_idx: int,
    target_rpm: float,
    n_workers: int,
    url: str,
    headers: dict[str, str],
    model: str,
    spec_fn: Callable[[], RequestSpec],
    timeout_sec: float,
    state: _LoadState,
) -> MinuteSample:
    CONSOLE.print(
        f"[bold]Minute {minute_idx + 1}[/bold] target={target_rpm:.1f} RPM, "
        f"workers={n_workers}…"
    )
    records = await run_load(
        url=url, headers=headers, model=model,
        spec_fn=spec_fn,
        target_rpm=target_rpm,
        duration_sec=60.0,
        n_workers=n_workers,
        timeout_sec=timeout_sec,
        classifier=classify_429,
        state=state,
    )
    rate_429 = _compute_429_rate(records)
    decision = _make_decision(rate_429)
    ok_count = sum(1 for r in records if r.classification == "ok")
    CONSOLE.print(
        f"  → {len(records)} req, {ok_count} ok, "
        f"429 rate {rate_429*100:.1f}% → {decision}"
    )
    return MinuteSample(
        minute_idx=minute_idx,
        target_rpm=target_rpm,
        n_workers=n_workers,
        records=records,
        rate_429=rate_429,
        decision=decision,
    )


def _aggregate_steady_state(
    steady_minutes: list[MinuteSample],
) -> tuple[float, float, float, float, str]:
    """Mean RPM / TPM / input-TPM / output-TPM + bottleneck label across the
    steady-state window."""
    all_records = [r for m in steady_minutes for r in m.records]
    if not all_records:
        return (0.0, 0.0, 0.0, 0.0, "NO_429")
    ok = [r for r in all_records if r.classification == "ok"]
    minutes_count = len(steady_minutes)
    if minutes_count == 0:
        return (0.0, 0.0, 0.0, 0.0, "NO_429")
    rpm = len(ok) / minutes_count
    in_tpm = sum(r.input_tokens for r in ok) / minutes_count
    out_tpm = sum(r.output_tokens for r in ok) / minutes_count
    bottleneck = _classify_bottleneck(all_records)
    return (rpm, in_tpm + out_tpm, in_tpm, out_tpm, bottleneck)


async def _capacity_orchestrate(  # noqa: PLR0913
    *,
    args: argparse.Namespace,
    url: str,
    headers: dict[str, str],
    model: str,
    spec_fn: Callable[[], RequestSpec],
    state: _LoadState,
) -> CapacityOutcome:
    target_rpm = float(args.start_rpm)
    minutes: list[MinuteSample] = []
    steady_window: list[MinuteSample] = []
    minute_idx = 0
    budget_minutes = args.max_duration_minutes
    required_steady = args.steady_state_minutes

    while minute_idx < budget_minutes and not state.stop_requested:
        n_workers = args.workers or _auto_workers_for_rpm(target_rpm)
        sample = await _run_capacity_minute(
            minute_idx=minute_idx,
            target_rpm=target_rpm,
            n_workers=n_workers,
            url=url, headers=headers, model=model,
            spec_fn=spec_fn,
            timeout_sec=args.request_timeout_sec,
            state=state,
        )
        minutes.append(sample)

        if sample.decision == "hold":
            steady_window.append(sample)
            if len(steady_window) >= required_steady:
                CONSOLE.print(
                    f"[green]Steady state reached[/green] "
                    f"({required_steady} minutes within "
                    f"{STEADY_STATE_LOW_BAND*100:.0f}–"
                    f"{STEADY_STATE_HIGH_BAND*100:.0f}% 429 band)"
                )
                break
        elif sample.decision == "ramp_up":
            target_rpm = max(target_rpm * RAMP_UP_FACTOR, MIN_TARGET_RPM)
            steady_window = []
        else:  # back_off
            target_rpm = max(target_rpm * BACKOFF_FACTOR, MIN_TARGET_RPM)
            steady_window = []

        minute_idx += 1

    converged = len(steady_window) >= required_steady
    if steady_window:
        rpm, tpm, in_tpm, out_tpm, bottleneck = _aggregate_steady_state(steady_window)
    else:
        # Didn't converge — report last minute's data so the run isn't wasted
        last = minutes[-1] if minutes else None
        if last:
            rpm, tpm, in_tpm, out_tpm, bottleneck = _aggregate_steady_state([last])
        else:
            rpm, tpm, in_tpm, out_tpm, bottleneck = (0.0, 0.0, 0.0, 0.0, "NO_429")

    return CapacityOutcome(
        converged=converged,
        sustained_rpm=rpm,
        sustained_tpm=tpm,
        sustained_input_tpm=in_tpm,
        sustained_output_tpm=out_tpm,
        bottleneck=bottleneck,
        final_target_rpm=target_rpm,
        minutes=minutes,
        steady_minutes_collected=len(steady_window),
    )


def _write_capacity_records_csv(run_dir: Path, outcome: CapacityOutcome) -> None:
    import csv as _csv  # noqa: PLC0415
    path = run_dir / "per_request.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow([
            "minute", "target_rpm", "ts", "elapsed_ms", "status",
            "input_tokens", "output_tokens", "classification",
            "retry_after_sec", "error_detail",
        ])
        for m in outcome.minutes:
            for r in m.records:
                writer.writerow([
                    m.minute_idx, f"{m.target_rpm:.1f}",
                    r.ts, f"{r.elapsed_ms:.1f}", r.status,
                    r.input_tokens, r.output_tokens, r.classification,
                    r.retry_after_sec if r.retry_after_sec is not None else "",
                    r.error_detail.replace("\n", " "),
                ])


def _write_capacity_per_minute_csv(run_dir: Path, outcome: CapacityOutcome) -> None:
    """Per-minute rollup — needed both for the controller feedback log and as
    a plot-friendly time series."""
    import csv as _csv  # noqa: PLC0415
    path = run_dir / "per_minute.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow([
            "minute", "target_rpm", "n_workers", "n_requests", "n_ok",
            "n_429_rpm", "n_429_tpm", "n_429_unclassified", "n_other",
            "input_tokens", "output_tokens", "rate_429", "decision",
        ])
        for m in outcome.minutes:
            c = _count_classifications(m.records)
            ok = c.get("ok", 0)
            n_429_rpm = c.get("429_rpm", 0)
            n_429_tpm = c.get("429_tpm", 0)
            n_429_unc = c.get("429_unclassified", 0)
            n_other = len(m.records) - ok - n_429_rpm - n_429_tpm - n_429_unc
            in_tok = sum(r.input_tokens for r in m.records
                         if r.classification == "ok")
            out_tok = sum(r.output_tokens for r in m.records
                          if r.classification == "ok")
            writer.writerow([
                m.minute_idx + 1, f"{m.target_rpm:.1f}", m.n_workers,
                len(m.records), ok, n_429_rpm, n_429_tpm, n_429_unc, n_other,
                in_tok, out_tok, f"{m.rate_429:.4f}", m.decision,
            ])


def _write_capacity_summary_md(  # noqa: PLR0913
    run_dir: Path,
    args: argparse.Namespace,
    provider: ProviderConfig,
    model_cfg: ModelConfig,
    outcome: CapacityOutcome,
    started_iso: str,
    ended_iso: str,
    duration_sec: float,
) -> None:
    from _common import fmt_duration as _fmt_duration  # noqa: PLC0415
    from _common import git_commit as _git_commit  # noqa: PLC0415

    converged_str = ("**yes**" if outcome.converged
                     else "**no** (hit `--max-duration-minutes` first; "
                          "results below are from the last attempted minute)")

    # 429-source breakdown across steady-state window
    steady_records = [
        r for m in outcome.minutes[-outcome.steady_minutes_collected:]
        for r in m.records
    ] if outcome.steady_minutes_collected > 0 else []
    sc = _count_classifications(steady_records)
    total_429 = sc.get("429_rpm", 0) + sc.get("429_tpm", 0) + sc.get(
        "429_unclassified", 0)
    if total_429 > 0:
        rpm_pct = sc.get("429_rpm", 0) / total_429 * 100
        tpm_pct = sc.get("429_tpm", 0) / total_429 * 100
        unc_pct = sc.get("429_unclassified", 0) / total_429 * 100
        breakdown = (f"RPM-triggered {rpm_pct:.0f}% · "
                     f"TPM-triggered {tpm_pct:.0f}% · "
                     f"Unclassified {unc_pct:.0f}%")
    else:
        breakdown = "no 429s observed in steady state"

    lines = [
        "# Capacity Probe Report (capacity mode)",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Provider: `{provider.name}` (`{provider.base_url}`)",
        f"- Model: `{model_cfg.name}`",
        f"- Git commit: `{_git_commit()}`",
        f"- Started: `{started_iso}` · Ended: `{ended_iso}` "
        f"· Wall-clock: `{_fmt_duration(duration_sec)}`",
        "",
        "## Headline",
        "",
        f"- Sustained throughput: **{outcome.sustained_rpm:.1f} RPM** / "
        f"**{outcome.sustained_tpm:,.0f} TPM** "
        f"(input {outcome.sustained_input_tpm:,.0f} + "
        f"output {outcome.sustained_output_tpm:,.0f})",
        f"- Bottleneck: **{outcome.bottleneck}** ({breakdown})",
        f"- Converged: {converged_str}",
        f"- Steady-state minutes used: "
        f"{outcome.steady_minutes_collected} / "
        f"`--steady-state-minutes={args.steady_state_minutes}`",
        f"- Final target RPM: {outcome.final_target_rpm:.1f}",
        "",
        "## Business request profile",
        "",
        f"- `--avg-input-tokens`: {args.avg_input_tokens}",
        f"- `--avg-output-tokens`: {args.avg_output_tokens}",
        f"- `--max-input-tokens`: "
        f"{args.max_input_tokens or args.avg_input_tokens * 2} "
        "(sampler upper clamp)",
        f"- `--max-output-tokens`: "
        f"{args.max_output_tokens or args.avg_output_tokens * 2}",
        f"- `--seed`: {args.seed} (length-sampling reproducibility)",
        f"- `--start-rpm`: {args.start_rpm} · "
        f"`--workers`: "
        f"{f'{args.workers} (pinned)' if args.workers else 'auto (target_rpm/12)'}",
        "",
        "## Minute-by-minute trace",
        "",
        "| Minute | Target RPM | Workers | Requests | OK | 429 RPM | "
        "429 TPM | Unclass. | Input tok | Output tok | 429 rate | Decision |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in outcome.minutes:
        c = _count_classifications(m.records)
        ok_in = sum(r.input_tokens for r in m.records
                    if r.classification == "ok")
        ok_out = sum(r.output_tokens for r in m.records
                     if r.classification == "ok")
        lines.append(
            f"| {m.minute_idx + 1} | {m.target_rpm:.1f} | {m.n_workers} | "
            f"{len(m.records)} | {c.get('ok', 0)} | "
            f"{c.get('429_rpm', 0)} | {c.get('429_tpm', 0)} | "
            f"{c.get('429_unclassified', 0)} | {ok_in:,} | {ok_out:,} | "
            f"{m.rate_429*100:.1f}% | `{m.decision}` |"
        )
    lines.append("")

    failures = _failure_summary([r for m in outcome.minutes for r in m.records])
    if failures:
        lines.extend(["## Failures", "", *failures, ""])

    lines.extend([
        "## Run config",
        "",
        "Full inputs in `config.json`; per-request rows in `per_request.csv`; "
        "per-minute rollup in `per_minute.csv`.",
        "",
    ])
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()

    # Safety: cap --max-duration-minutes unless --allow-long-run is set
    if args.max_duration_minutes > DEFAULT_MAX_DURATION_MIN and not args.allow_long_run:
        CONSOLE.print(
            f"[red]ERROR:[/red] --max-duration-minutes "
            f"{args.max_duration_minutes} exceeds default cap "
            f"{DEFAULT_MAX_DURATION_MIN}; pass --allow-long-run to unlock"
        )
        return 2
    if args.max_duration_minutes > LONG_RUN_HARD_CAP_MIN:
        CONSOLE.print(
            f"[red]ERROR:[/red] --max-duration-minutes "
            f"{args.max_duration_minutes} exceeds hard cap "
            f"{LONG_RUN_HARD_CAP_MIN}min"
        )
        return 2

    config_path = Path(args.config)
    if not config_path.is_file():
        CONSOLE.print(f"[red]ERROR:[/red] config file not found: {config_path}")
        return 2
    suite = SuiteConfig.from_yaml(config_path)

    try:
        provider, model_cfg = select_target(suite, args.provider, args.model)
    except ValueError as e:
        CONSOLE.print(f"[red]ERROR:[/red] {e}")
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_dir)
        / f"{timestamp}_{args.mode}_{slugify(provider.name)}_{slugify(model_cfg.name)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    CONSOLE.print(f"Mode:     {args.mode}")
    CONSOLE.print(f"Provider: {provider.name} ({provider.base_url})")
    CONSOLE.print(f"Model:    {model_cfg.name}")
    CONSOLE.print(f"Output:   {run_dir}")
    CONSOLE.print("")

    if args.mode == "probe":
        return run_probe(args, run_dir)
    if args.mode == "capacity":
        return run_capacity(args, run_dir)
    CONSOLE.print(f"[red]ERROR:[/red] unknown mode {args.mode!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
