"""Unit tests for ModelProfile loading and ProfileRegistry lookup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import ModelConfig  # noqa: E402
from model_profile import (  # noqa: E402
    ModelProfile,
    ProfileNotFoundError,
    ProfileRegistry,
    ResolvedModel,
    resolve_models,
)

VALID_YAML = """\
model: gpt-4o
snapshot: gpt-4o-2024-08-06
api_format: openai
created_at: 2024-08-06
source_endpoint: https://api.openai.com
capabilities:
  - chat
  - streaming
"""


def _write_profile(root: Path, relpath: str, content: str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestSchema:
    def test_valid_yaml_parses(self, tmp_path: Path) -> None:
        path = _write_profile(
            tmp_path, "openai/gpt-4o/2024-08-06.yaml", VALID_YAML
        )
        registry = ProfileRegistry(root=tmp_path)
        profile = registry.load("openai", "gpt-4o", "2024-08-06")
        assert isinstance(profile, ModelProfile)
        assert profile.model == "gpt-4o"
        assert profile.snapshot == "gpt-4o-2024-08-06"
        assert profile.capabilities == ["chat", "streaming"]
        assert profile.source_path == path

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        bad = VALID_YAML.replace("snapshot: gpt-4o-2024-08-06\n", "")
        _write_profile(tmp_path, "openai/gpt-4o/2024-08-06.yaml", bad)
        registry = ProfileRegistry(root=tmp_path)
        with pytest.raises(ValidationError):
            registry.load("openai", "gpt-4o", "2024-08-06")

    def test_extra_field_forbidden(self, tmp_path: Path) -> None:
        bad = VALID_YAML + "unknown_key: oops\n"
        _write_profile(tmp_path, "openai/gpt-4o/2024-08-06.yaml", bad)
        registry = ProfileRegistry(root=tmp_path)
        with pytest.raises(ValidationError):
            registry.load("openai", "gpt-4o", "2024-08-06")


class TestLatestResolution:
    def test_picks_latest_by_created_at(self, tmp_path: Path) -> None:
        older = VALID_YAML
        newer = VALID_YAML.replace(
            "snapshot: gpt-4o-2024-08-06", "snapshot: gpt-4o-2025-03-15"
        ).replace("created_at: 2024-08-06", "created_at: 2025-03-15")
        _write_profile(tmp_path, "openai/gpt-4o/2024-08-06.yaml", older)
        newer_path = _write_profile(
            tmp_path, "openai/gpt-4o/2025-03-15.yaml", newer
        )

        registry = ProfileRegistry(root=tmp_path)
        latest = registry.load("openai", "gpt-4o")
        assert latest.snapshot == "gpt-4o-2025-03-15"
        assert latest.source_path == newer_path

    def test_pin_snapshot(self, tmp_path: Path) -> None:
        _write_profile(tmp_path, "openai/gpt-4o/2024-08-06.yaml", VALID_YAML)
        newer = VALID_YAML.replace(
            "created_at: 2024-08-06", "created_at: 2025-03-15"
        )
        _write_profile(tmp_path, "openai/gpt-4o/2025-03-15.yaml", newer)

        registry = ProfileRegistry(root=tmp_path)
        pinned = registry.load("openai", "gpt-4o", "2024-08-06")
        assert pinned.snapshot == "gpt-4o-2024-08-06"


class TestMissing:
    def test_missing_model_dir(self, tmp_path: Path) -> None:
        _write_profile(
            tmp_path, "openai/gpt-4o/2024-08-06.yaml", VALID_YAML
        )
        registry = ProfileRegistry(root=tmp_path)
        with pytest.raises(ProfileNotFoundError) as exc:
            registry.load("openai", "does-not-exist")
        assert "Available profiles" in str(exc.value)
        assert "gpt-4o" in str(exc.value)

    def test_missing_snapshot(self, tmp_path: Path) -> None:
        _write_profile(
            tmp_path, "openai/gpt-4o/2024-08-06.yaml", VALID_YAML
        )
        registry = ProfileRegistry(root=tmp_path)
        with pytest.raises(ProfileNotFoundError) as exc:
            registry.load("openai", "gpt-4o", "1999-01-01")
        assert "Available snapshots" in str(exc.value)
        assert "2024-08-06" in str(exc.value)

    def test_empty_model_dir(self, tmp_path: Path) -> None:
        (tmp_path / "openai" / "gpt-4o").mkdir(parents=True)
        registry = ProfileRegistry(root=tmp_path)
        with pytest.raises(ProfileNotFoundError):
            registry.load("openai", "gpt-4o")


class TestSampleProfile:
    """The committed sample must parse — it's the documented schema example."""

    def test_default_registry_loads_sample(self) -> None:
        registry = ProfileRegistry()
        profile = registry.load("openai", "gpt-5.4-mini")
        assert profile.model == "gpt-5.4-mini"
        assert "chat" in profile.capabilities
        assert profile.source_path is not None


# ── ResolvedModel / resolve_models (TODO 2) ───────────────────────────────


class TestModelConfigProfileFields:
    def test_accepts_profile_fields(self) -> None:
        cfg = ModelConfig.model_validate({
            "name": "openai/gpt-5.4-mini",
            "profile": "gpt-5.4-mini",
            "profile_snapshot": "2026-04-18",
        })
        assert cfg.profile == "gpt-5.4-mini"
        assert cfg.profile_snapshot == "2026-04-18"

    def test_legacy_config_still_loads(self) -> None:
        """PR1-era configs (no profile, capabilities only) keep working."""
        cfg = ModelConfig.model_validate({
            "name": "gpt-4o",
            "capabilities": ["chat", "streaming"],
        })
        assert cfg.profile is None
        assert cfg.profile_snapshot is None
        assert cfg.capabilities == ["chat", "streaming"]


class TestResolvedModel:
    def test_prefers_profile_when_present(self, tmp_path: Path) -> None:
        _write_profile(
            tmp_path, "openai/gpt-4o/2024-08-06.yaml", VALID_YAML
        )
        reg = ProfileRegistry(root=tmp_path)
        profile = reg.load("openai", "gpt-4o")
        cfg = ModelConfig(
            name="gpt-4o", capabilities=["chat"], profile="gpt-4o"
        )
        resolved = ResolvedModel(config=cfg, profile=profile)
        # Profile wins over config.capabilities
        assert resolved.capabilities == ["chat", "streaming"]
        assert resolved.name == "gpt-4o"

    def test_falls_back_to_config_when_no_profile(self) -> None:
        cfg = ModelConfig(name="gpt-4o", capabilities=["chat", "tools"])
        resolved = ResolvedModel(config=cfg, profile=None)
        assert resolved.capabilities == ["chat", "tools"]


class TestResolveModels:
    def test_loads_profiles(self, tmp_path: Path) -> None:
        _write_profile(
            tmp_path, "openai/gpt-4o/2024-08-06.yaml", VALID_YAML
        )
        reg = ProfileRegistry(root=tmp_path)
        cfg = ModelConfig(name="gpt-4o", profile="gpt-4o")
        resolved = resolve_models("openai", [cfg], registry=reg)
        assert len(resolved) == 1
        assert resolved[0].profile is not None
        assert resolved[0].profile.snapshot == "gpt-4o-2024-08-06"

    def test_missing_profile_triggers_fallback_callback(
        self, tmp_path: Path
    ) -> None:
        """Missing profile → ResolvedModel.profile is None and callback fires."""
        reg = ProfileRegistry(root=tmp_path)
        cfg = ModelConfig(
            name="weird", capabilities=["chat"], profile="nope"
        )
        calls: list[tuple[str, str]] = []
        resolved = resolve_models(
            "openai",
            [cfg],
            registry=reg,
            on_fallback=lambda m, msg: calls.append((m.name, msg)),
        )
        assert resolved[0].profile is None
        # Fallback capability path uses config.capabilities
        assert resolved[0].capabilities == ["chat"]
        assert len(calls) == 1
        assert calls[0][0] == "weird"
        assert "not found" in calls[0][1].lower()

    def test_no_profile_field_skips_loading(self, tmp_path: Path) -> None:
        """A model with profile=None isn't a fallback — don't fire the callback."""
        reg = ProfileRegistry(root=tmp_path)
        cfg = ModelConfig(name="legacy", capabilities=["chat"])
        calls: list[str] = []
        resolved = resolve_models(
            "openai", [cfg], registry=reg, on_fallback=lambda m, _: calls.append(m.name)
        )
        assert resolved[0].profile is None
        assert calls == []
