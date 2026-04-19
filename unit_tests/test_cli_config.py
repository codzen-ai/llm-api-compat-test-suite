"""Unit tests for CLI config options (--config, --base-url, etc.)."""

from __future__ import annotations

from pathlib import Path

import pytest

# pytester is a built-in pytest plugin for testing pytest itself
pytest_plugins = ["pytester"]

PROJECT_ROOT = Path(__file__).parent.parent
CONFTEST_PATH = PROJECT_ROOT / "conftest.py"
SRC_DIR = PROJECT_ROOT / "src"

# Minimal test file that uses the `model` fixture from conftest
DUMMY_TEST = """\
def test_dummy(model):
    assert isinstance(model, str)
"""


def _make_conftest(pytester: pytest.Pytester) -> None:
    """Copy the project conftest into pytester's tmpdir."""
    conftest_content = CONFTEST_PATH.read_text()
    # Patch sys.path to point to the real src/ directory
    conftest_content = conftest_content.replace(
        'sys.path.insert(0, str(Path(__file__).parent / "src"))',
        f'sys.path.insert(0, r"{SRC_DIR}")',
    )
    pytester.makeconftest(conftest_content)


def _make_config(pytester: pytest.Pytester, filename: str = "config.yaml") -> Path:
    """Create a minimal valid config.yaml in pytester's tmpdir.

    Uses the committed ``gpt-5.4-mini`` profile as the ground truth — it's
    always available in the real ``model_profiles/`` tree the pytester'd
    conftest still points at via SRC_DIR.
    """
    content = """\
providers:
  - name: "test-provider"
    base_url: "http://localhost:9999"
    api_key: "test-key"
    api_format: "openai"
    models:
      - name: "test-model"
        profile: "gpt-5.4-mini"
"""
    config_path = pytester.path / filename
    config_path.write_text(content)
    return config_path


class TestNoArgs:
    def test_no_args_skips_all(self, pytester: pytest.Pytester) -> None:
        """pytest with no config args → all tests skip."""
        _make_conftest(pytester)
        pytester.makepyfile(test_sample=DUMMY_TEST)

        result = pytester.runpytest("-v")

        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(["*No models configured*"])


class TestConfigFlag:
    def test_config_flag_loads_default(
        self, pytester: pytest.Pytester
    ) -> None:
        """pytest --config (no value) → loads config.yaml."""
        _make_conftest(pytester)
        _make_config(pytester, "config.yaml")
        pytester.makepyfile(test_sample=DUMMY_TEST)

        result = pytester.runpytest("--config", "-v")

        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test-model*"])

    def test_config_with_custom_path(
        self, pytester: pytest.Pytester
    ) -> None:
        """pytest --config=custom.yaml → loads specified file."""
        _make_conftest(pytester)
        _make_config(pytester, "custom.yaml")
        pytester.makepyfile(test_sample=DUMMY_TEST)

        result = pytester.runpytest("--config=custom.yaml", "-v")

        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test-model*"])

    def test_config_file_not_found(
        self, pytester: pytest.Pytester
    ) -> None:
        """pytest --config=nonexistent.yaml → error."""
        _make_conftest(pytester)
        pytester.makepyfile(test_sample=DUMMY_TEST)

        result = pytester.runpytest("--config=nonexistent.yaml", "-v")

        result.stderr.fnmatch_lines(["*Config file not found*"])
        assert result.ret != 0


class TestIgnoreProfile:
    """`--ignore-profile` bypasses capability filtering for profile authoring."""

    # A test that only runs when the model has the "special_cap" capability —
    # exercises the skip path unless --ignore-profile is set.
    MARKED_TEST = """\
import pytest

@pytest.mark.capability("special_cap")
def test_requires_special_cap(model):
    assert isinstance(model, str)
"""

    def test_without_flag_skips(self, pytester: pytest.Pytester) -> None:
        _make_conftest(pytester)
        _make_config(pytester, "config.yaml")  # model only has "chat"
        pytester.makepyfile(test_marked=self.MARKED_TEST)

        # -rs surfaces the full skip reason in the summary line
        result = pytester.runpytest("--config", "-v", "-rs")

        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(["*lacks capability 'special_cap'*"])

    def test_with_flag_runs(self, pytester: pytest.Pytester) -> None:
        _make_conftest(pytester)
        _make_config(pytester, "config.yaml")  # same config, still lacks cap
        pytester.makepyfile(test_marked=self.MARKED_TEST)

        result = pytester.runpytest("--config", "--ignore-profile", "-v")

        # With the flag, skip is bypassed → the test body runs and passes.
        result.assert_outcomes(passed=1)


class TestCliArgs:
    def test_cli_base_url(self, pytester: pytest.Pytester) -> None:
        """pytest --base-url=... --api-format=openai → CLI mode works."""
        _make_conftest(pytester)
        pytester.makepyfile(test_sample=DUMMY_TEST)

        result = pytester.runpytest(
            "--base-url=http://localhost:9999",
            "--api-key=test-key",
            "--api-format=openai",
            "--model=cli-model",
            "--profile=gpt-5.4-mini",
            "-v",
        )

        # Model is parametrized from CLI, test shows up with model name
        result.stdout.fnmatch_lines(["*cli-model*"])

    def test_cli_model_without_profile_errors(
        self, pytester: pytest.Pytester
    ) -> None:
        """--model without --profile → UsageError (no implicit ground truth)."""
        _make_conftest(pytester)
        pytester.makepyfile(test_sample=DUMMY_TEST)

        result = pytester.runpytest(
            "--base-url=http://localhost:9999",
            "--api-key=test-key",
            "--api-format=openai",
            "--model=cli-model",
            "-v",
        )

        result.stderr.fnmatch_lines(["*--profile is required*"])
        assert result.ret != 0

    def test_cli_missing_profile_errors(
        self, pytester: pytest.Pytester
    ) -> None:
        """--profile=nonexistent → UsageError listing available profiles."""
        _make_conftest(pytester)
        pytester.makepyfile(test_sample=DUMMY_TEST)

        result = pytester.runpytest(
            "--base-url=http://localhost:9999",
            "--api-key=test-key",
            "--api-format=openai",
            "--model=cli-model",
            "--profile=does-not-exist",
            "-v",
        )

        result.stderr.fnmatch_lines(["*Profile directory not found*"])
        result.stderr.fnmatch_lines(["*Available profiles*"])
        assert result.ret != 0
