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
    from collections.abc import Iterable


DEFAULT_PROFILES_ROOT = Path(__file__).resolve().parent.parent / "model_profiles"


class ProfileNotFoundError(LookupError):
    """Raised when a requested profile (or snapshot) does not exist on disk."""


class PerformanceBudget(BaseModel):
    """Latency budget for a model snapshot.

    Both fields are upper bounds: the median of N samples must be ≤ the
    budget for the test to pass.
    """

    model_config = ConfigDict(extra="forbid")

    ttft_ms: float
    """Time-to-first-token upper bound, in milliseconds."""

    tpot_ms: float
    """Time-per-output-token upper bound, in milliseconds/token."""


class ModelProfile(BaseModel):
    """Ground-truth capability surface for one model snapshot.

    Pure YAML schema — round-trips cleanly to/from disk. The path the YAML
    was loaded from is tracked separately (see :class:`ProfileRegistry.load`'s
    return type and :class:`ResolvedModel.source_path`).
    """

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

    performance: PerformanceBudget | None = None
    """Default TTFT/TPOT budget. Users may override per-model in their config
    file. ``None`` means no budget is defined; performance tests skip."""


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
    ) -> tuple[ModelProfile, Path]:
        """Load a profile. ``snapshot=None`` → latest by ``created_at``.

        Returns the parsed profile and the YAML file it came from. The path
        is returned separately rather than embedded in :class:`ModelProfile`
        so the model stays a pure YAML schema.

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
            return self._load_file(path), path

        candidates = self.list_snapshot_files(api_format, name)
        if not candidates:
            msg = (
                f"No snapshot YAML files in {model_dir}. "
                "Add one like `2024-08-06.yaml`."
            )
            raise ProfileNotFoundError(msg)

        loaded = [(self._load_file(p), p) for p in candidates]
        loaded.sort(key=lambda pair: pair[0].created_at, reverse=True)
        return loaded[0]

    def _load_file(self, path: Path) -> ModelProfile:
        with path.open() as f:
            data = yaml.safe_load(f)
        return ModelProfile.model_validate(data)

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
                    profile, _ = self.load(api_format, model_dir.name)
                except ProfileNotFoundError:
                    continue
                yield profile


# ── Resolved model (config + profile pairing) ────────────────────────────


class ResolvedModel(BaseModel):
    """A configured model paired with the profile that defines its benchmark."""

    config: ModelConfig
    profile: ModelProfile
    source_path: Path
    """Path to the YAML the profile was loaded from — reported back so a
    reader of ``reports/{ts}/summary.md`` can reconstruct the exact ground
    truth used for a run."""

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def capabilities(self) -> list[str]:
        return self.profile.capabilities

    @property
    def performance_budget(self) -> PerformanceBudget | None:
        """Effective performance budget: config override fields win, missing
        fields fall back to the profile default. Returns ``None`` if neither
        source supplies both ``ttft_ms`` and ``tpot_ms``."""
        profile_perf = self.profile.performance
        override = self.config.performance

        ttft = override.ttft_ms if override and override.ttft_ms is not None else None
        if ttft is None and profile_perf is not None:
            ttft = profile_perf.ttft_ms

        tpot = override.tpot_ms if override and override.tpot_ms is not None else None
        if tpot is None and profile_perf is not None:
            tpot = profile_perf.tpot_ms

        if ttft is None or tpot is None:
            return None
        return PerformanceBudget(ttft_ms=ttft, tpot_ms=tpot)


def resolve_models(
    api_format: ApiFormat | str,
    models: list[ModelConfig],
    registry: ProfileRegistry | None = None,
) -> list[ResolvedModel]:
    """Pair each :class:`ModelConfig` with its loaded :class:`ModelProfile`.

    Raises :class:`ProfileNotFoundError` for any model whose profile can't be
    located — the caller is expected to translate that into a surfaced error
    (``pytest.UsageError`` from conftest) rather than silently skipping.
    """
    reg = registry if registry is not None else ProfileRegistry()
    resolved: list[ResolvedModel] = []
    for m in models:
        profile, path = reg.load(api_format, m.profile, m.profile_snapshot)
        resolved.append(
            ResolvedModel(config=m, profile=profile, source_path=path)
        )
    return resolved
