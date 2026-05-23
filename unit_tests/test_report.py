"""Unit tests for the report's Profile metadata rendering."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from config import ApiFormat, ModelConfig, ProviderConfig
from model_profile import ModelProfile, ResolvedModel
from report import ReportCollector, TestResult


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
    )


_FAKE_SOURCE = Path("model_profiles/openai/m/2026-04-18.yaml")


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
    resolved = [
        ResolvedModel(config=m, profile=profile, source_path=_FAKE_SOURCE)
    ]

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
    # profile_snapshot is the YAML filename stem (date), not the inner
    # `snapshot:` field — see config.example.yaml.
    m = ModelConfig(name="m", profile="m", profile_snapshot="2026-04-18")
    provider = _make_provider([m])
    profile = _make_profile("m-2026-04-18", ["chat"])
    resolved = [
        ResolvedModel(config=m, profile=profile, source_path=_FAKE_SOURCE)
    ]

    path = _collector(tmp_path).generate_summary(
        provider, resolved_models=resolved
    )
    text = path.read_text()

    assert "pinned" in text


def test_report_omits_models_section_without_resolved(tmp_path: Path) -> None:
    """No resolved models → no `## Models` section at all (a header with
    zero rows would be visual noise)."""
    provider = _make_provider([ModelConfig(name="m", profile="m")])
    text = (
        _collector(tmp_path)
        .generate_summary(provider, resolved_models=None)
        .read_text()
    )

    assert "## Models" not in text


def test_report_renders_test_details(tmp_path: Path) -> None:
    """A TestResult with ``details`` produces a ``## Test Details`` section
    keyed by the test's last ``::`` segment."""
    col = ReportCollector(report_dir=tmp_path)
    col.add_result(
        TestResult(
            node_id="tests/openai_compat/test_performance.py::TestPerformance::test_streaming_latency[m]",
            outcome="passed",
            duration=27.5,
            details="- TTFT median: 420 ms\n- TPOT median: 19.2 ms/tok",
        )
    )
    text = col.generate_summary().read_text()

    assert "## Test Details" in text
    assert "### test_streaming_latency[m]" in text
    assert "TTFT median: 420 ms" in text
    assert "TPOT median: 19.2 ms/tok" in text


def test_report_omits_test_details_when_no_details(tmp_path: Path) -> None:
    """No results with ``details`` → no header at all (avoid an empty section)."""
    col = ReportCollector(report_dir=tmp_path)
    col.add_result(
        TestResult(node_id="t::x", outcome="passed", duration=0.1)
    )
    text = col.generate_summary().read_text()
    assert "## Test Details" not in text


@pytest.mark.parametrize("count", [2, 3])
def test_report_renders_one_row_per_model(tmp_path: Path, count: int) -> None:
    configs = [
        ModelConfig(name=f"m{i}", profile=f"m{i}") for i in range(count)
    ]
    provider = _make_provider(configs)
    resolved = [
        ResolvedModel(
            config=c,
            profile=_make_profile(f"{c.name}-2026-04-18", ["chat"]),
            source_path=_FAKE_SOURCE,
        )
        for c in configs
    ]

    text = (
        _collector(tmp_path)
        .generate_summary(provider, resolved_models=resolved)
        .read_text()
    )

    for c in configs:
        assert f"{c.name}-2026-04-18" in text
