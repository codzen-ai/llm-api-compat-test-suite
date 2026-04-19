"""Shared test fixtures (image bytes, etc.) used across provider test suites."""

from __future__ import annotations

import base64
from pathlib import Path

_TINY_PNG_PATH = Path(__file__).parent / "tiny.png"

TINY_PNG_BYTES: bytes = _TINY_PNG_PATH.read_bytes()
"""Raw PNG bytes. 64x64 RGB with a red circle and blue square — large enough
that Anthropic and OpenAI vision endpoints both accept it (1x1 gets rejected
with 'Could not process image')."""

TINY_PNG_B64: str = base64.b64encode(TINY_PNG_BYTES).decode("ascii")
"""Base64-encoded PNG for APIs that want the image inline as a data URL or
the raw base64 payload."""
