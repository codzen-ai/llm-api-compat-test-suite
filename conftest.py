from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from config import ModelConfig, ProviderConfig, SuiteConfig
from http_client import LoggingHttpClient
from model_profile import (
    ProfileNotFoundError,
    ResolvedModel,
    resolve_models,
)
from report import ReportCollector, TestResult

if TYPE_CHECKING:
    from collections.abc import Generator

# ── Global state ──────────────────────────────────────────────────────────────

_suite_config: SuiteConfig | None = None
_active_provider: ProviderConfig | None = None
_active_models: list[ModelConfig] = []
_resolved_models: list[ResolvedModel] = []
_report_dir: Path = Path("reports")
_collector = ReportCollector()


# ── CLI options ───────────────────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llm-compat", "LLM API Compatibility Testing")
    group.addoption(
        "--config",
        dest="config_file",
        nargs="?",
        const="config.yaml",
        default=None,
        help="Path to YAML config file (default: config.yaml)",
    )
    group.addoption(
        "--base-url",
        dest="base_url",
        default=None,
        help="API base URL (CLI override)",
    )
    group.addoption(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key (CLI override)",
    )
    group.addoption(
        "--api-format",
        dest="api_format",
        default=None,
        choices=["openai", "anthropic", "gemini"],
        help="API format to test: openai, anthropic, or gemini",
    )
    group.addoption(
        "--model",
        dest="model",
        default=None,
        help="Model name to test (CLI override)",
    )
    group.addoption(
        "--profile",
        dest="profile",
        default=None,
        help=(
            "Profile name (e.g. gpt-5.4-mini) to benchmark the model against. "
            "Required in CLI mode when --model is given."
        ),
    )
    group.addoption(
        "--profile-snapshot",
        dest="profile_snapshot",
        default=None,
        help=(
            "Pin a specific profile snapshot — the YAML filename stem under "
            "model_profiles/<api_format>/<profile>/ (e.g. 2025-03-15). "
            "Omit for the latest by `created_at`."
        ),
    )
    group.addoption(
        "--auth-type",
        dest="auth_type",
        default=None,
        choices=["bearer", "x-api-key", "x-goog-api-key"],
        help="Auth header type (default: auto from api-format)",
    )
    group.addoption(
        "--no-verify-ssl",
        dest="no_verify_ssl",
        action="store_true",
        default=False,
        help="Disable SSL certificate verification",
    )
    group.addoption(
        "--ignore-profile",
        dest="ignore_profile",
        action="store_true",
        default=False,
        help=(
            "Recording mode: run every capability-marked test regardless of "
            "profile, so you can observe what a model really supports before "
            "authoring its profile YAML. See "
            "docs/profile-based-compatibility-testing.md."
        ),
    )


# ── Configuration loading ─────────────────────────────────────────────────────


def pytest_configure(config: pytest.Config) -> None:
    global _suite_config, _active_provider, _active_models  # noqa: PLW0603
    global _resolved_models, _report_dir, _collector  # noqa: PLW0603

    config_file: str | None = config.getoption("config_file")
    base_url: str | None = config.getoption("base_url")

    if config_file:
        path = Path(config_file)
        if not path.exists():
            msg = f"Config file not found: {config_file}"
            raise pytest.UsageError(msg)
        _suite_config = SuiteConfig.from_yaml(path)
    elif base_url:
        api_key: str = config.getoption("api_key") or ""
        api_format: str = config.getoption("api_format") or "openai"
        model_name: str | None = config.getoption("model")
        profile_name: str | None = config.getoption("profile")
        profile_snapshot: str | None = config.getoption("profile_snapshot")
        try:
            _suite_config = SuiteConfig.from_cli(
                base_url=base_url,
                api_key=api_key,
                api_format=api_format,
                model=model_name,
                profile=profile_name,
                profile_snapshot=profile_snapshot,
            )
        except ValueError as err:
            raise pytest.UsageError(str(err)) from err
    else:
        return

    if _suite_config and _suite_config.providers:
        _active_provider = _suite_config.providers[0]
        _active_models = _active_provider.models

        # Filter to specific model if --model is provided
        model_filter: str | None = config.getoption("model")
        if model_filter and config_file:
            _active_models = [
                m for m in _active_models if m.name == model_filter
            ]

        try:
            _resolved_models = resolve_models(
                _active_provider.api_format,
                _active_models,
            )
        except ProfileNotFoundError as err:
            raise pytest.UsageError(str(err)) from err

    # Set up report directory
    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    _report_dir = Path("reports") / timestamp
    _report_dir.mkdir(parents=True, exist_ok=True)
    (_report_dir / "logs").mkdir(exist_ok=True)
    _collector = ReportCollector(report_dir=_report_dir)


# ── Test collection filtering ─────────────────────────────────────────────────


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not _active_provider:
        for item in items:
            test_path = str(item.path)
            if any(
                d in test_path
                for d in ("openai_compat", "anthropic_compat", "gemini_compat")
            ):
                item.add_marker(
                    pytest.mark.skip(reason="No provider configured")
                )
        return

    api_format = _active_provider.api_format
    to_remove: list[pytest.Item] = []

    for item in items:
        # Skip tests not matching the current api_format
        test_path = str(item.path)
        if "openai_compat" in test_path and api_format != "openai":
            to_remove.append(item)
            continue
        if "anthropic_compat" in test_path and api_format != "anthropic":
            to_remove.append(item)
            continue
        if "gemini_compat" in test_path and api_format != "gemini":
            to_remove.append(item)
            continue

    for item in to_remove:
        items.remove(item)


def _should_skip_for_capability(
    item: pytest.Item, model: ResolvedModel
) -> str | None:
    """Return skip reason if the test's capability marker is absent from the
    model's profile-declared capabilities."""
    effective = model.capabilities
    for marker in item.iter_markers("capability"):
        required: str | None = marker.args[0] if marker.args else None
        if required and required not in effective:
            return f"Model '{model.name}' lacks capability '{required}'"
    return None


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def provider_config() -> ProviderConfig:
    if not _active_provider:
        pytest.skip("No provider configured")
    return _active_provider


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize resolved_model fixture based on loaded config."""
    if "resolved_model" in metafunc.fixturenames:
        models = _resolved_models if _resolved_models else [None]
        ids = [m.name if m else "no-model" for m in models]
        metafunc.parametrize("resolved_model", models, ids=ids, indirect=True)


@pytest.fixture
def resolved_model(request: pytest.FixtureRequest) -> ResolvedModel:
    model: ResolvedModel | None = request.param
    if model is None:
        pytest.skip("No models configured")
    return model


@pytest.fixture
def model(
    resolved_model: ResolvedModel, request: pytest.FixtureRequest
) -> str:
    if request.config.getoption("ignore_profile"):
        # Recording mode — bypass capability filtering so every marked test
        # runs; results feed profile authoring.
        return resolved_model.name
    # pytest's stubs type `request.node` loosely (Item | Collector | Unknown);
    # we know the fixture only runs against Items because the marker scan
    # needs item-level metadata.
    node = cast(
        "pytest.Item",
        request.node,  # pyright: ignore[reportUnknownMemberType]
    )
    skip_reason = _should_skip_for_capability(node, resolved_model)
    if skip_reason:
        pytest.skip(skip_reason)
    return resolved_model.name


@pytest.fixture
def client(
    provider_config: ProviderConfig,
    resolved_model: ResolvedModel,
    request: pytest.FixtureRequest,
) -> Generator[LoggingHttpClient]:
    api_format = provider_config.api_format
    headers: dict[str, str] = {"Content-Type": "application/json"}

    # Determine auth type: CLI override > config > default from api_format
    auth_type: str | None = request.config.getoption("auth_type")
    if auth_type is None:
        auth_type = provider_config.auth_type
    if auth_type is None:
        auth_type = {
            "openai": "bearer",
            "anthropic": "x-api-key",
            "gemini": "x-goog-api-key",
        }.get(api_format, "bearer")

    if auth_type == "bearer":
        headers["Authorization"] = f"Bearer {provider_config.api_key}"
    elif auth_type == "x-api-key":
        headers["x-api-key"] = provider_config.api_key
    elif auth_type == "x-goog-api-key":
        headers["x-goog-api-key"] = provider_config.api_key

    if api_format == "anthropic":
        headers["anthropic-version"] = "2023-06-01"

    # Determine verify_ssl: CLI override > config
    cli_no_verify = bool(request.config.getoption("no_verify_ssl"))
    verify_ssl = False if cli_no_verify else provider_config.verify_ssl

    http_client = LoggingHttpClient(
        base_url=provider_config.base_url,
        default_headers=headers,
        verify_ssl=verify_ssl,
    )

    yield http_client

    # Write per-test log file on teardown
    if http_client.records:
        node_id = str(request.node.nodeid)  # type: ignore[union-attr]
        safe_name = node_id.replace("/", "__").replace("::", "__")
        log_path = _report_dir / "logs" / f"{safe_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"Test: {node_id}",
            f"Model: {resolved_model.name}",
            f"Provider: {provider_config.name}",
            f"Base URL: {provider_config.base_url}",
            "=" * 72,
            "",
        ]
        for i, record in enumerate(http_client.records):
            lines.extend([
                f"--- Request #{i + 1} ---",
                "",
                record.format_log(),
                "",
            ])

        log_path.write_text("\n".join(lines), encoding="utf-8")


# ── Report hooks ──────────────────────────────────────────────────────────────


def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> None:
    if call.when != "call":
        return

    outcome = "passed" if call.excinfo is None else "failed"
    duration = call.duration

    safe_name = item.nodeid.replace("/", "__").replace("::", "__")
    log_path = _report_dir / "logs" / f"{safe_name}.log"

    failure_message = ""
    if call.excinfo is not None:
        failure_message = str(call.excinfo.getrepr(style="short"))

    _collector.add_result(
        TestResult(
            node_id=item.nodeid,
            outcome=outcome,
            duration=duration,
            log_file=log_path if log_path.exists() else None,
            failure_message=failure_message,
        )
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _suite_config is None:
        return
    if _collector.results:
        summary_path = _collector.generate_summary(
            provider=_active_provider,
            resolved_models=_resolved_models,
        )
        print(f"\n{'=' * 72}")  # noqa: T201
        print(f"Report: {summary_path}")  # noqa: T201
        print(f"Logs:   {_report_dir / 'logs'}")  # noqa: T201
