# LLM API Compatibility Test Suite

A pytest-based test suite for verifying whether third-party LLM APIs are truly compatible with official OpenAI, Anthropic (Claude), and Gemini APIs.

Tests send raw HTTP requests in the official API format to the third-party endpoint, then validate response format and functionality.

## Features

- **3 API formats**: OpenAI, Anthropic, Gemini — 43 test cases total
- **Raw HTTP testing**: Uses `httpx` directly (not SDKs) to verify HTTP-level compatibility
- **Pydantic config validation**: Configuration is validated with Pydantic models, invalid values fail fast
- **Model capability filtering**: Auto-skips tests the model doesn't support (e.g., vision, tools)
- **Per-test HTTP logs**: Every request/response pair saved to individual `.log` files
- **Markdown summary report**: Generated automatically after each run

## Test Coverage

| API Format | Tests | Categories |
|---|---|---|
| OpenAI | 14 | chat, streaming, tool calling, vision, embeddings |
| Anthropic | 13 | messages, streaming, tool use, vision |
| Gemini | 11 | generateContent, streaming, function calling |

## Quick Start

### Install

```bash
# Requires Python 3.13+
uv sync
```

### Configure

**Option A: YAML config file** (recommended for multiple models/providers)

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your provider details
```

```yaml
providers:
  - name: "my-provider"
    base_url: "https://api.example.com"
    api_key_env: "MY_PROVIDER_API_KEY"  # Read from env var
    api_format: "openai"                 # openai | anthropic | gemini
    models:
      - name: "gpt-4o"
        capabilities: ["chat", "streaming", "tools", "vision"]
      - name: "gpt-3.5-turbo"
        capabilities: ["chat", "streaming", "tools"]
```

**Option B: CLI arguments** (quick single-provider testing)

```bash
pytest --base-url=https://api.example.com --api-key=sk-xxx --api-format=openai --model=gpt-4o -v
```

Additional CLI options:

- `--auth-type=bearer|x-api-key|x-goog-api-key` — Override auth header type (default: auto from api-format)
- `--no-verify-ssl` — Disable SSL certificate verification (for self-signed certs)

In CLI mode, all capabilities are enabled by default.

### CLI Options

| Option | Required | Default | Description |
|---|---|---|---|
| `--config` | No | - | Path to YAML configuration file |
| `--base-url` | Yes* | - | API base URL |
| `--api-key` | Yes* | - | API key |
| `--api-format` | Yes* | - | `openai`, `anthropic`, or `gemini` |
| `--model` | No | All models in config | Model name to test |
| `--auth-type` | No | Auto from `api-format` | `bearer`, `x-api-key`, or `x-goog-api-key` |
| `--no-verify-ssl` | No | `false` | Disable SSL certificate verification |

\* Required when not using `--config`.

### Run Tests

```bash
# Run all tests with config file
pytest --config=config.yaml -v

# Run only OpenAI-compatible tests
pytest tests/openai_compat/ --config=config.yaml -v

# Run only Anthropic-compatible tests
pytest tests/anthropic_compat/ --config=config.yaml -v

# Run only Gemini-compatible tests
pytest tests/gemini_compat/ --config=config.yaml -v

# Filter to a specific model
pytest --config=config.yaml --model=gpt-4o -v

# CLI mode (no config file needed)
pytest --base-url=https://api.example.com \
       --api-key=sk-xxx \
       --api-format=openai \
       --model=gpt-4o -v

# Use Bearer auth with Anthropic format (e.g., floodgate)
pytest tests/anthropic_compat/ \
       --base-url=https://example.com/api/anthropic \
       --api-key="$API_KEY" \
       --api-format=anthropic \
       --auth-type=bearer \
       --no-verify-ssl \
       --model=claude-haiku -v
```

## Model Capabilities

Each model in the config declares which capabilities it supports. Tests requiring a capability the model lacks are automatically skipped.

| Capability | Description | Supported Formats |
|---|---|---|
| `chat` | Basic chat/message completion | all |
| `streaming` | Server-sent events streaming | all |
| `tools` | Function/tool calling | all |
| `vision` | Image input processing | all |
| `embeddings` | Text embedding API | OpenAI only |

## Reports

After each run, reports are generated in:

```
reports/{timestamp}/
├── summary.md          # Pass/fail table grouped by API format
└── logs/
    ├── tests__openai_compat__test_chat_basic.py__TestChatBasic__test_simple_message[gpt-4o].log
    └── ...
```

Each `.log` file contains:

- Test name, model, and provider info
- Full HTTP request (method, URL, headers, body)
- Full HTTP response (status code, headers, body)
- Elapsed time

## Project Structure

```
├── conftest.py                 # CLI options, fixtures, report hooks
├── config.example.yaml         # Configuration template
├── src/
│   ├── config.py               # YAML config loading & validation
│   ├── http_client.py          # httpx wrapper with request/response capture
│   └── report.py               # Markdown report generation
├── tests/                      # Compatibility tests (core deliverable)
│   ├── openai_compat/
│   ├── anthropic_compat/
│   └── gemini_compat/
└── unit_tests/                 # Unit tests for the project itself
    └── test_cli_config.py
```

## Development

### Unit Tests

```bash
# Run unit tests (no API key needed)
pytest unit_tests/ -v
```
