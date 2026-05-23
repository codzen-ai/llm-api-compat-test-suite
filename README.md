# LLM API Compatibility Test Suite

A pytest-based test suite for verifying whether third-party LLM APIs are truly compatible with official OpenAI, Anthropic (Claude), and Gemini APIs.

Tests send raw HTTP requests in the official API format to the third-party endpoint, then validate response format and functionality.

## Features

- **3 API formats**: OpenAI, Anthropic, Gemini — 58 test cases total
- **Raw HTTP testing**: Uses `httpx` directly (not SDKs) to verify HTTP-level compatibility
- **Pydantic config validation**: Configuration is validated with Pydantic models, invalid values fail fast
- **Profile-based capability filtering**: Each model is benchmarked against a hand-authored ground-truth profile (e.g. OpenAI's `gpt-5.4-mini`). Tests are filtered to the capability subset the reference model actually supports. See [docs/profile-based-compatibility-testing.md](docs/profile-based-compatibility-testing.md).
- **Per-test HTTP logs**: Every request/response pair saved to individual `.log` files
- **Markdown summary report**: Generated automatically after each run, annotated with the profile baseline used

## Test Coverage

| API Format | Tests | Categories |
|---|---|---|
| OpenAI | 28 | chat, streaming, tool calling, vision, embeddings |
| Anthropic | 16 | messages, streaming, tool use, vision |
| Gemini | 14 | generateContent, streaming, function calling |

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
    api_key_env: "MY_PROVIDER_API_KEY"   # Read from env var
    api_format: "openai"                  # openai | anthropic | gemini
    models:
      - name: "openai/gpt-5.4-mini"       # third-party's model identifier
        profile: "gpt-5.4-mini"            # which OpenAI ground-truth to benchmark against
        # profile_snapshot: "2025-03-15"   # optional, pin a specific snapshot; omit for latest
```

`profile` points at a YAML under `model_profiles/{api_format}/{profile}/…yaml` that records what the official reference model supports. Capability filtering is driven by that file — users do not declare capabilities themselves.

Ready-to-use configs live in [`configs/`](configs/): `configs/openai.yaml`, `configs/anthropic.yaml`, `configs/openrouter.yaml`. Set the matching `*_API_KEY` env var and run `pytest --config configs/openai.yaml -v` (etc.).

**Option B: CLI arguments** (quick single-provider testing)

```bash
pytest --base-url=https://api.example.com \
       --api-key=sk-xxx \
       --api-format=openai \
       --model=gpt-4o \
       --profile=gpt-5.4-mini -v
```

`--profile` is required whenever `--model` is given. Additional CLI options:

- `--profile-snapshot=YYYY-MM-DD` — Pin a specific profile snapshot file; omit for latest
- `--auth-type=bearer|x-api-key|x-goog-api-key` — Override auth header type (default: auto from api-format)
- `--no-verify-ssl` — Disable SSL certificate verification (for self-signed certs)
- `--ignore-profile` — Recording mode: run every capability-marked test regardless of profile. Used when authoring a new profile; see [model_profiles/README.md](model_profiles/README.md).

### CLI Options

| Option | Required | Default | Description |
|---|---|---|---|
| `--config` | No | - | Path to YAML configuration file (default filename: `config.yaml` when flag is bare) |
| `--base-url` | Yes* | - | API base URL |
| `--api-key` | Yes* | - | API key |
| `--api-format` | Yes* | - | `openai`, `anthropic`, or `gemini` |
| `--model` | No | All models in config | Model name to test |
| `--profile` | Yes** | - | Profile name to benchmark against (e.g. `gpt-5.4-mini`) |
| `--profile-snapshot` | No | Latest by `created_at` | Pin a snapshot file (e.g. `2025-03-15`) |
| `--auth-type` | No | Auto from `api-format` | `bearer`, `x-api-key`, or `x-goog-api-key` |
| `--no-verify-ssl` | No | `false` | Disable SSL certificate verification |
| `--ignore-profile` | No | `false` | Bypass capability filtering (profile authoring mode) |

\* Required when not using `--config`.
\*\* Required whenever `--model` is given; in YAML mode each model must declare `profile:`.

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
       --model=gpt-4o \
       --profile=gpt-5.4-mini -v

# Use Bearer auth with Anthropic format (e.g., floodgate)
pytest tests/anthropic_compat/ \
       --base-url=https://example.com/api/anthropic \
       --api-key="$API_KEY" \
       --api-format=anthropic \
       --auth-type=bearer \
       --no-verify-ssl \
       --model=claude-haiku-4-5 \
       --profile=claude-haiku-4-5 -v
```

## Model Capabilities

Capabilities are declared by **profiles**, not by users. A profile records what an official model snapshot (e.g. `gpt-5.4-mini`) actually supports; tests carrying a `@pytest.mark.capability("X")` marker are skipped if `X` is absent from the configured profile.

Fine-grained markers currently in use: `chat`, `streaming`, `tools`, `vision`, `embeddings`, `max_tokens`, `max_completion_tokens`, `stop_sequences`, `n_multi`, `logprobs`, `seed`, `json_mode`, `system_message`, `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` (full list in [pyproject.toml](pyproject.toml)).

Authoring a new profile → see [model_profiles/README.md](model_profiles/README.md). Design rationale → see [docs/profile-based-compatibility-testing.md](docs/profile-based-compatibility-testing.md).

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
├── config.yaml                 # Personal config (gitignored; copy from config.example.yaml)
├── configs/                    # Curated configs for the 3 reference providers
│   ├── openai.yaml
│   ├── anthropic.yaml
│   └── openrouter.yaml
├── src/
│   ├── config.py               # YAML config loading & validation
│   ├── http_client.py          # httpx wrapper with request/response capture
│   ├── model_profile.py        # Ground-truth profile loader
│   └── report.py               # Markdown report generation
├── model_profiles/             # Hand-authored ground-truth YAMLs
│   └── openai/{model}/{YYYY-MM-DD}.yaml
├── docs/                       # Design docs (profile-based compat testing)
├── tests/                      # Compatibility tests (core deliverable)
│   ├── openai_compat/
│   ├── anthropic_compat/
│   └── gemini_compat/
└── unit_tests/                 # Unit tests for the project itself
```

## Development

### Unit Tests

```bash
# Run unit tests (no API key needed)
pytest unit_tests/ -v
```
