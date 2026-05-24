"""Unit tests for the report's Profile metadata rendering."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from config import ApiFormat, ModelConfig, ProviderConfig
from model_profile import ModelProfile, ResolvedModel
from report import ReportCollector, TestResult, write_index


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
            model_name="m",
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
            model_name="m",
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
        TestResult(
            node_id="t::x",
            model_name="m",
            outcome="passed",
            duration=0.1,
        )
    )
    text = col.generate_summary().read_text()
    assert "## Test Details" not in text


def _build_collectors(
    tmp_path: Path,
    specs: list[tuple[str, list[str]]],
) -> tuple[list[ResolvedModel], dict[str, ReportCollector]]:
    """specs: list of ``(model_name, [outcomes])``. Returns the resolved
    models and a dict of populated collectors, mirroring the layout that
    conftest produces (each collector rooted at ``tmp_path/<model>``)."""
    resolved: list[ResolvedModel] = []
    collectors: dict[str, ReportCollector] = {}
    for name, outcomes in specs:
        cfg = ModelConfig(name=name, profile=name)
        profile = _make_profile(f"{name}-2026-04-18", ["chat", "streaming"])
        rm = ResolvedModel(config=cfg, profile=profile, source_path=_FAKE_SOURCE)
        resolved.append(rm)
        col_dir = tmp_path / name
        col_dir.mkdir()
        col = ReportCollector(report_dir=col_dir)
        for i, outcome in enumerate(outcomes):
            col.add_result(
                TestResult(
                    node_id=f"tests/openai_compat/t.py::test_{i}",
                    model_name=name,
                    outcome=outcome,
                    duration=0.1,
                )
            )
        collectors[name] = col
    return resolved, collectors


def test_index_lists_each_configured_model_with_counts(tmp_path: Path) -> None:
    resolved, collectors = _build_collectors(
        tmp_path,
        [
            ("alpha", ["passed", "passed", "failed"]),
            ("beta", ["passed", "skipped"]),
        ],
    )
    provider = _make_provider([rm.config for rm in resolved])

    path = write_index(tmp_path, provider, resolved, collectors)
    text = path.read_text()

    assert path == tmp_path / "index.md"
    assert "## Models" in text
    assert "| alpha |" in text
    assert "| beta |" in text
    # Both per-model summaries are linked relative to the run dir.
    assert "[summary](alpha/summary.md)" in text
    assert "[summary](beta/summary.md)" in text


def test_index_includes_capability_count_column(tmp_path: Path) -> None:
    """Capabilities column tells readers the test sets are not directly
    comparable — keep it present and populated from the resolved model."""
    resolved, collectors = _build_collectors(tmp_path, [("alpha", ["passed"])])
    provider = _make_provider([rm.config for rm in resolved])

    text = write_index(tmp_path, provider, resolved, collectors).read_text()

    assert "Capabilities" in text
    # The profile created by _build_collectors declares 2 capabilities.
    assert "| 2 |" in text


def test_index_omits_pass_rate_column(tmp_path: Path) -> None:
    """Cross-model pass-rate comparison is misleading when models have
    different capability sets; the index deliberately doesn't render it."""
    resolved, collectors = _build_collectors(
        tmp_path,
        [("alpha", ["passed", "failed"])],
    )
    provider = _make_provider([rm.config for rm in resolved])

    text = write_index(tmp_path, provider, resolved, collectors).read_text()

    lowered = text.lower()
    assert "pass rate" not in lowered
    assert "%" not in text


def test_index_shows_configured_model_with_no_results(tmp_path: Path) -> None:
    """A model whose tests were all filtered out by capability matching
    still appears in the index — silent disappearance would hide config
    mistakes."""
    resolved, collectors = _build_collectors(
        tmp_path,
        [("alpha", []), ("beta", ["passed"])],
    )
    provider = _make_provider([rm.config for rm in resolved])

    text = write_index(tmp_path, provider, resolved, collectors).read_text()

    # alpha row exists with zero counts; the row still has 8 cells.
    alpha_rows = [line for line in text.splitlines() if line.startswith("| alpha |")]
    assert len(alpha_rows) == 1
    assert "| 0 | 0 | 0 | 0 |" in alpha_rows[0]


def test_index_preserves_configured_model_order(tmp_path: Path) -> None:
    """Models should appear in YAML/config order so the "primary" model
    sits at the top, not in alphabetical order."""
    resolved, collectors = _build_collectors(
        tmp_path,
        [("zeta", ["passed"]), ("alpha", ["passed"])],
    )
    provider = _make_provider([rm.config for rm in resolved])

    text = write_index(tmp_path, provider, resolved, collectors).read_text()

    zeta_idx = text.find("| zeta |")
    alpha_idx = text.find("| alpha |")
    assert zeta_idx != -1 and alpha_idx != -1
    assert zeta_idx < alpha_idx


def test_collector_writes_summary_even_with_no_results(tmp_path: Path) -> None:
    """Empty collector still produces a ``summary.md`` so that
    "configured but ran nothing" is auditable from the file tree alone."""
    cfg = ModelConfig(name="m", profile="m")
    provider = _make_provider([cfg])
    profile = _make_profile("m-2026-04-18", ["chat"])
    resolved = [ResolvedModel(config=cfg, profile=profile, source_path=_FAKE_SOURCE)]
    col = ReportCollector(report_dir=tmp_path)

    path = col.generate_summary(provider, resolved_models=resolved)

    assert path.exists()
    text = path.read_text()
    assert "# LLM API Compatibility Test Report" in text


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
