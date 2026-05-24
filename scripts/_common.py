"""Shared helpers for the standalone stress-test scripts.

Both [cache_hit_rate.py](cache_hit_rate.py) and [capacity_probe.py](capacity_probe.py)
reuse:

- Config loading (re-exports ``SuiteConfig`` / ``ProviderConfig`` / ``ModelConfig``
  from ``src/config.py`` so importers don't repeat the sys.path trick)
- Auth header construction (``build_headers``)
- Provider/model selection (``select_target``)
- Coarse text generation (~4 chars/token) and truncated-normal length sampling
  (``make_text_for_tokens``, ``Sampler``)
- Misc utilities (``slugify``, ``git_commit``, ``fmt_duration``, shared
  ``CONSOLE``)
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    import random

# Scripts run as `python scripts/foo.py`, not as a package. Inject src/ so we
# can re-export config types and let callers `from _common import SuiteConfig`
# without each script repeating this dance.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from config import ModelConfig, ProviderConfig, SuiteConfig  # noqa: E402

__all__ = [
    "CHARS_PER_TOKEN",
    "CONSOLE",
    "DEFAULT_TIMEOUT_SEC",
    "ModelConfig",
    "ProviderConfig",
    "Sampler",
    "SuiteConfig",
    "build_headers",
    "fmt_duration",
    "git_commit",
    "make_text_for_tokens",
    "select_target",
    "slugify",
]

CONSOLE = Console()
"""Single rich Console shared across all scripts. ``Live`` displays share this
instance so ``CONSOLE.print(...)`` during a live region scrolls *above* the
region instead of disrupting it."""

CHARS_PER_TOKEN = 4
"""Coarse char-to-token ratio for *generating* prompt text. Statistics
always use the server-reported ``usage.prompt_tokens``, so this constant
only affects how much padding text we emit — never the result."""

DEFAULT_TIMEOUT_SEC = 120.0


# ── Text generation ──────────────────────────────────────────────────────────

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


# ── Distribution sampler ─────────────────────────────────────────────────────


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


# ── Provider/model selection ─────────────────────────────────────────────────


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


# ── Misc ─────────────────────────────────────────────────────────────────────


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def git_commit() -> str:
    try:
        out = subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"
    return out.decode().strip()


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name).strip("_")
