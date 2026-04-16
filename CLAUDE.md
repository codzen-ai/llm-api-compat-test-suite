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
```

## Architecture

**This project's "product" is the compatibility test cases themselves** — `tests/` contains the deliverables, `unit_tests/` tests the framework.

### Data flow

1. **Config** (`--config` loads YAML, or `--base-url`/`--api-key`/`--api-format` for CLI mode) → `src/config.py` Pydantic models
2. **conftest.py** reads config into globals, dynamically parametrizes tests per model via `pytest_generate_tests()`, builds auth headers, creates `LoggingHttpClient`
3. **Test files** receive `client: LoggingHttpClient` and `model: str` fixtures, make raw HTTP requests, assert on response structure
4. **On teardown**, client fixture writes per-test `.log` files; `pytest_sessionfinish` generates `reports/{timestamp}/summary.md`

### Key non-obvious patterns

- **`pytest_collection_modifyitems` removes items** (not just marks skip) for non-matching api_format — so test counts reflect only relevant tests
- **`pytest_generate_tests`** handles dynamic model parametrization because `@pytest.fixture(params=...)` evaluates at import time before config is loaded
- **Auth override chain**: CLI `--auth-type` > config `auth_type` field > default from `api_format` (bearer/x-api-key/x-goog-api-key)
- **`--config` with no value** defaults to `config.yaml` (uses `nargs="?"` + `const`)
- CLI mode auto-enables all capabilities; YAML config lets you restrict per model
- `src/` modules are on `sys.path` via conftest.py line 14, imported as `from config import ...` (not package imports)

### Adding a new test

Put it in the right `tests/{format}_compat/` directory. Use `@pytest.mark.capability("tools")` if it needs specific capabilities. Request `client` and `model` fixtures:

```python
@pytest.mark.capability("streaming")
class TestNewFeature:
    def test_feature(self, client: LoggingHttpClient, model: str) -> None:
        status, lines = client.request_stream("POST", "/v1/chat/completions", json_body={...})
        assert status == 200
```

## Lint/type rules

- Ruff: strict rule set. Tests exempt from S101 (assert) and TCH (runtime imports). conftest.py exempt from T201 (print).
- Pyright: strict mode globally. `tests/` and `unit_tests/` suppress `reportUnknown{Variable,Member,Argument}Type` because dynamic JSON from `dict[str, Any]` propagates `Unknown`.
