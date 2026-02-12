# Contributing

Guide for setting up a local development environment and running tests.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`

## Setup

```bash
git clone https://github.com/Zipstack/llm-whisperer-python-client.git
cd llm-whisperer-python-client

# Create a virtual environment and install all dependencies
uv sync --group dev --group test

# Either prefix commands with `uv run` (used throughout this guide),
# or activate the venv manually:
#   source .venv/bin/activate        # Linux / macOS
#   .venv\Scripts\activate           # Windows
```

If you prefer pip:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,test]"
```

## Environment variables

The project uses [`pytest-dotenv`](https://pypi.org/project/pytest-dotenv/) to auto-load environment variables from a `.env` file in the project root.

```bash
cp sample.env .env
```

Edit `.env` and fill in your API key:

| Variable | Description | Required for |
|---|---|---|
| `LLMWHISPERER_BASE_URL_V2` | API base URL (v2) | Integration tests |
| `LLMWHISPERER_API_KEY` | Your LLMWhisperer API key | Integration tests |
| `LLMWHISPERER_LOG_LEVEL` | Log level (`DEBUG`, `INFO`, etc.) | Optional |

Unit tests are fully mocked and do **not** require a `.env` file or API key.

## Running tests

### Unit tests (no API key needed)

```bash
uv run pytest tests/unit/ -v
```

### Integration tests (requires valid `.env`)

```bash
uv run pytest tests/integration/ -v
```

### All tests

```bash
uv run pytest -v
```

### Via tox

```bash
uv run tox
```

This runs tests under Python 3.12 using `uv sync --group test` to install dependencies.

### Via poe

```bash
uv run poe test
```

## Linting and formatting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting (configured in `pyproject.toml`).

```bash
# Lint
uv run ruff check src/ tests/

# Auto-fix lint issues
uv run ruff check --fix src/ tests/

# Format
uv run ruff format src/ tests/
```

## Project structure

```
src/unstract/llmwhisperer/
    __init__.py          # Package version
    client_v2.py         # Main client (LLMWhispererClientV2)
    utils.py             # Utility helpers

tests/
    conftest.py          # Shared fixtures
    unit/                # Mocked tests (no network calls)
    integration/         # Tests against the live API
    test_data/           # Sample files used by tests
```
