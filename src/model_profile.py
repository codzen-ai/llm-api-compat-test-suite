"""Model capability profiles — the ground-truth benchmark for compat testing.

A profile records what an official model snapshot supports (e.g. OpenAI's
`gpt-4o-2024-08-06`). The test runner filters test cases to the capabilities
listed in the profile, so third-party providers claiming to emulate that
model are measured against the same subset.

Profiles live in ``model_profiles/{api_format}/{model_name}/{snapshot}.yaml``
and are hand-authored (AI-assisted). No automated recording — "PASS = supported"
is the wrong inference; see docs/profile-based-compatibility-testing.md.
"""

from __future__ import annotations

import datetime  # noqa: TC003  # pydantic needs runtime access for validation
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict

from config import (  # noqa: TC001  # pydantic needs runtime access
    ApiFormat,
    ModelConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    FallbackHandler = Callable[[ModelConfig, str], None]


DEFAULT_PROFILES_ROOT = Path(__file__).resolve().parent.parent / "model_profiles"


class ProfileNotFoundError(Exception):
    """Raised when a requested profile (or snapshot) does not exist on disk."""


class ModelProfile(BaseModel):
    """Ground-truth capability surface for one model snapshot."""

    # `model` is a real field name in the YAML; disable pydantic's default
    # `model_`-prefix namespace protection. `extra="forbid"` catches typos
    # in YAML early.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str
    """Model identifier (possibly an alias), e.g. ``gpt-4o``."""

    snapshot: str
    """Dated, immutable snapshot name, e.g. ``gpt-4o-2024-08-06``."""

    api_format: ApiFormat

    created_at: datetime.date
    """Date the profile was authored — used to pick the latest snapshot."""

    source_endpoint: str
    """Official endpoint the profile was validated against."""

    capabilities: list[str]
    """Capability markers (see ``pyproject.toml`` markers list) the snapshot
    is known to support. Tests marked with a capability not listed here are
    skipped for this model."""

    # Populated by the registry after validation; not part of the YAML schema.
    # Excluded from dumps so a round-trip through YAML stays clean.
    source_path: Path | None = None


class ProfileRegistry:
    """Filesystem-backed loader for :class:`ModelProfile` YAML files."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root if root is not None else DEFAULT_PROFILES_ROOT

    # ── Discovery ────────────────────────────────────────────────────────

    def _format_dir(self, api_format: ApiFormat | str) -> Path:
        return self.root / str(api_format)

    def _model_dir(self, api_format: ApiFormat | str, name: str) -> Path:
        return self._format_dir(api_format) / name

    def list_profile_names(self, api_format: ApiFormat | str) -> list[str]:
        d = self._format_dir(api_format)
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    def list_snapshot_files(
        self, api_format: ApiFormat | str, name: str
    ) -> list[Path]:
        d = self._model_dir(api_format, name)
        if not d.is_dir():
            return []
        return sorted(d.glob("*.yaml"))

    # ── Loading ──────────────────────────────────────────────────────────

    def load(
        self,
        api_format: ApiFormat | str,
        name: str,
        snapshot: str | None = None,
    ) -> ModelProfile:
        """Load a profile. ``snapshot=None`` → latest by ``created_at``.

        Raises :class:`ProfileNotFoundError` if the directory or snapshot is
        missing; the message lists viable alternatives at the same level so
        the user can correct their config.
        """
        model_dir = self._model_dir(api_format, name)
        if not model_dir.is_dir():
            available = self.list_profile_names(api_format)
            msg = (
                f"Profile directory not found: {model_dir}. "
                f"Available profiles for api_format={api_format!s}: "
                f"{available or 'none'}"
            )
            raise ProfileNotFoundError(msg)

        if snapshot is not None:
            path = model_dir / f"{snapshot}.yaml"
            if not path.is_file():
                available_snaps = [p.stem for p in self.list_snapshot_files(
                    api_format, name
                )]
                msg = (
                    f"Profile snapshot not found: {path}. "
                    f"Available snapshots: {available_snaps or 'none'}"
                )
                raise ProfileNotFoundError(msg)
            return self._load_file(path)

        candidates = self.list_snapshot_files(api_format, name)
        if not candidates:
            msg = (
                f"No snapshot YAML files in {model_dir}. "
                "Add one like `2024-08-06.yaml`."
            )
            raise ProfileNotFoundError(msg)

        loaded = [self._load_file(p) for p in candidates]
        loaded.sort(key=lambda prof: prof.created_at, reverse=True)
        return loaded[0]

    def _load_file(self, path: Path) -> ModelProfile:
        with path.open() as f:  # noqa: PTH123
            data = yaml.safe_load(f)
        profile = ModelProfile.model_validate(data)
        return profile.model_copy(update={"source_path": path})

    # ── Introspection ────────────────────────────────────────────────────

    def iter_all(self) -> Iterable[ModelProfile]:
        """Yield every profile in the registry (latest per model)."""
        if not self.root.is_dir():
            return
        for format_dir in sorted(self.root.iterdir()):
            if not format_dir.is_dir():
                continue
            api_format = format_dir.name
            for model_dir in sorted(format_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                try:
                    yield self.load(api_format, model_dir.name)
                except ProfileNotFoundError:
                    continue


# ── Resolved model (config + profile pairing) ────────────────────────────


class ResolvedModel(BaseModel):
    """A configured model paired with the profile that defines its benchmark.

    Introduced in TODO 2 of the profile-based compat plan. During the
    transitional period ``profile`` may be ``None``: either the user has not
    yet migrated their config, or profile loading failed. Capability checks
    then fall back to ``config.capabilities``.
    """

    config: ModelConfig
    profile: ModelProfile | None = None

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def capabilities(self) -> list[str]:
        """Effective capabilities for skip logic.

        Prefers the profile (authoritative ground truth). Falls back to the
        user-declared capabilities when no profile is attached — this path is
        removed in TODO 7.
        """
        if self.profile is not None:
            return self.profile.capabilities
        return self.config.capabilities


def resolve_models(
    api_format: ApiFormat | str,
    models: list[ModelConfig],
    registry: ProfileRegistry | None = None,
    *,
    on_fallback: FallbackHandler | None = None,
) -> list[ResolvedModel]:
    """Pair each :class:`ModelConfig` with its loaded :class:`ModelProfile`.

    Models without a ``profile`` field, or whose profile fails to load, are
    returned with ``profile=None`` and ``on_fallback`` is invoked so the
    caller can surface a warning. The loader never raises here — the transition
    plan explicitly tolerates missing profiles until TODO 7.
    """
    reg = registry if registry is not None else ProfileRegistry()
    resolved: list[ResolvedModel] = []
    for m in models:
        profile: ModelProfile | None = None
        if m.profile is not None:
            try:
                profile = reg.load(api_format, m.profile, m.profile_snapshot)
            except ProfileNotFoundError as err:
                if on_fallback is not None:
                    on_fallback(m, str(err))
        resolved.append(ResolvedModel(config=m, profile=profile))
    return resolved
