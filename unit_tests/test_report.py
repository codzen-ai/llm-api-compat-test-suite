"""Unit tests for the report's Profile metadata rendering."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import ApiFormat, ModelConfig, ProviderConfig  # noqa: E402
from model_profile import ModelProfile, ResolvedModel  # noqa: E402
from report import ReportCollector, TestResult  # noqa: E402


def _make_provider(models: list[ModelConfig]) -> ProviderConfig:
    return ProviderConfig(
        name="test-provider",
        base_url="http://localhost:9999",
        api_key="k",
        api_format=ApiFormat.OPENAI,
        models=models,
    )


def _make_profile(snapshot: str, caps: list[str]) -> ModelProfile:
    return ModelProfile(
        model="m",
        snapshot=snapshot,
        api_format=ApiFormat.OPENAI,
        created_at=datetime.date(2026, 4, 18),
        source_endpoint="https://api.openai.com",
        capabilities=caps,
        source_path=Path("model_profiles/openai/m/2026-04-18.yaml"),
    )


def _collector(tmp_path: Path) -> ReportCollector:
    col = ReportCollector(report_dir=tmp_path)
    col.add_result(
        TestResult(
            node_id="tests/openai_compat/t.py::test_x",
            outcome="passed",
            duration=0.1,
        )
    )
    return col


def test_report_lists_profile_metadata_auto_latest(tmp_path: Path) -> None:
    """A model resolved without pinning shows `auto-latest` resolution."""
    # no profile_snapshot → auto-latest
    m = ModelConfig(name="m", profile="m")
    provider = _make_provider([m])
    profile = _make_profile("m-2026-04-18", ["chat"])
    resolved = [ResolvedModel(config=m, profile=profile)]

    path = _collector(tmp_path).generate_summary(
        provider, resolved_models=resolved
    )
    text = path.read_text()

    assert "## Models" in text
    assert "m-2026-04-18" in text
    assert "2026-04-18" in text
    assert "auto-latest" in text
    assert "pinned" not in text
    assert "model_profiles/openai/m/2026-04-18.yaml" in text


def test_report_marks_pinned_when_snapshot_set(tmp_path: Path) -> None:
    m = ModelConfig(name="m", profile="m", profile_snapshot="m-2026-04-18")
    provider = _make_provider([m])
    profile = _make_profile("m-2026-04-18", ["chat"])
    resolved = [ResolvedModel(config=m, profile=profile)]

    path = _collector(tmp_path).generate_summary(
        provider, resolved_models=resolved
    )
    text = path.read_text()

    assert "pinned" in text


def test_report_handles_missing_resolved_models(tmp_path: Path) -> None:
    """``resolved_models=None`` is degenerate (no conftest pairing happened);
    rendering must not crash, and no per-model row should be fabricated."""
    provider = _make_provider([ModelConfig(name="m", profile="m")])
    text = (
        _collector(tmp_path)
        .generate_summary(provider, resolved_models=None)
        .read_text()
    )

    assert "## Models" in text
    # Header present, but no data row for model "m"
    assert "| m |" not in text


@pytest.mark.parametrize("count", [2, 3])
def test_report_renders_one_row_per_model(tmp_path: Path, count: int) -> None:
    configs = [
        ModelConfig(name=f"m{i}", profile=f"m{i}") for i in range(count)
    ]
    provider = _make_provider(configs)
    resolved = [
        ResolvedModel(config=c, profile=_make_profile(f"{c.name}-2026-04-18", ["chat"]))
        for c in configs
    ]

    text = (
        _collector(tmp_path)
        .generate_summary(provider, resolved_models=resolved)
        .read_text()
    )

    for c in configs:
        assert f"{c.name}-2026-04-18" in text
