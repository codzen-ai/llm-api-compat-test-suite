# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install deps (Python 3.13+)
uv run ruff check .              # Lint
uv run pyright                   # Type check (strict mode)
uv run pytest unit_tests/ -v     # Unit tests (no API key needed)
uv run pytest --config -v        # Run compat tests with config.yaml
uv run pytest tests/openai_compat/ --config -v   # Single format
uv run pytest tests/openai_compat/test_chat_basic.py::TestChatBasic::test_simple_message --config -v  # Single test
uv run pytest --collect-only --config  # Preview which tests will run
uv run pytest --config --ignore-profile -v   # Recording mode: bypass capability filtering to author a new profile
```

## Architecture

**This project's "product" is the compatibility test cases themselves** — `tests/` contains the deliverables, `unit_tests/` tests the framework.

### Data flow

1. **Config** (`--config` loads YAML, or `--base-url`/`--api-key`/`--api-format` for CLI mode) → `src/config.py` Pydantic models
2. **conftest.py** reads config into globals, dynamically parametrizes tests per model via `pytest_generate_tests()`, builds auth headers, creates `LoggingHttpClient`, and creates one `ReportCollector` per configured model
3. **Test files** receive `client: LoggingHttpClient` and `model: str` fixtures, make raw HTTP requests, assert on response structure
4. **On teardown**, client fixture writes per-test `.log` files into `reports/{timestamp}/{model}/logs/`; `pytest_sessionfinish` writes one `summary.md` per model under its subdirectory plus a top-level `reports/{timestamp}/index.md` listing all models — see [docs/multi-model-report-layout.md](docs/multi-model-report-layout.md) for why the per-model split exists and why the index deliberately omits a pass-rate column

### Key non-obvious patterns

- **`pytest_collection_modifyitems` removes items** (not just marks skip) for non-matching api_format — so test counts reflect only relevant tests
- **`pytest_generate_tests`** handles dynamic model parametrization because `@pytest.fixture(params=...)` evaluates at import time before config is loaded
- **Auth override chain**: CLI `--auth-type` > config `auth_type` field > default from `api_format` (bearer/x-api-key/x-goog-api-key)
- **`--config` with no value** defaults to `config.yaml` (uses `nargs="?"` + `const`)
- `--profile <name>` is required (YAML per-model or CLI flag); capabilities are read from `model_profiles/{api_format}/{profile}/…yaml`, not user-declared — see [docs/profile-based-compatibility-testing.md](docs/profile-based-compatibility-testing.md)
- **Reports are split per-model**: `reports/{ts}/{model}/{summary.md + logs/}` plus a top-level `index.md`. The index lists models but does not compute pass rates because different models typically have different capability sets — see [docs/multi-model-report-layout.md](docs/multi-model-report-layout.md)
- `src/` and `tests/` are on `sys.path` via `pythonpath = ["src", "tests"]` in [pyproject.toml](pyproject.toml), so modules import as `from config import ...` / `from fixtures import ...` (not package imports)

### Adding a new test

Put it in the right `tests/{format}_compat/` directory. Attach `@pytest.mark.capability("X")` for **every** parameter/behavior the test exercises — marker granularity is "one capability per independently-toggleable model behavior" (see [docs/profile-based-compatibility-testing.md](docs/profile-based-compatibility-testing.md)). Register new marker names in `pyproject.toml`'s `markers` list. Request `client` and `model` fixtures:

```python
@pytest.mark.capability("streaming")
class TestNewFeature:
    def test_feature(self, client: LoggingHttpClient, model: str) -> None:
        status, lines = client.request_stream("POST", "/v1/chat/completions", json_body={...})
        assert status == 200
```

Whether the test runs for a given model is decided by the model's **profile** (`model_profiles/{api_format}/{profile}/…yaml`), not by anything the user declares. If you introduce a new marker, update the relevant profile files — ideally by re-running `--ignore-profile` against the official endpoint and reclassifying PASS/FAIL results (do not assume PASS means supported). See [model_profiles/README.md](model_profiles/README.md) for the authoring workflow.

## Lint/type rules

- Ruff: strict rule set. Tests exempt from S101 (assert) and TCH (runtime imports). conftest.py exempt from T201 (print).
- Pyright: strict mode globally. `tests/` and `unit_tests/` suppress `reportUnknown{Variable,Member,Argument}Type` because dynamic JSON from `dict[str, Any]` propagates `Unknown`.
