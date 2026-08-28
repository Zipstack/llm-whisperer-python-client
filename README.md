# Unstract LLMWhisperer Python Client

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llmwhisperer-client)](https://pypi.org/project/llmwhisperer-client/)
[![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FZipstack%2Fllm-whisperer-python-client%2Fmain%2Fpyproject.toml)
](https://pypi.org/project/llmwhisperer-client/)
[![PyPI - Version](https://img.shields.io/pypi/v/llmwhisperer-client)](https://pypi.org/project/llmwhisperer-client/)

LLMs are powerful, but their output is as good as the input you provide. LLMWhisperer is a technology that presents data from complex documents (different designs and formats) to LLMs in a way that they can best understand. LLMWhisperer features include Layout Preserving Mode, Auto-switching between native text and OCR modes, proper representation of radio buttons and checkboxes in PDF forms as raw text, among other features. You can now extract raw text from complex PDF documents or images without having to worry about whether the document is a native text document, a scanned image or just a picture clicked on a smartphone. Extraction of raw text from invoices, purchase orders, bank statements, etc works easily for structured data extraction with LLMs powered by LLMWhisperer's Layout Preserving mode. 

Refer to the client documentation for more information: [LLMWhisperer Client Documentation](https://docs.unstract.com/llmwhisperer/llm_whisperer/python_client/llm_whisperer_python_client_intro/)

## Client

This package provides **LLMWhispererClientV2**, the client for LLMWhisperer API v2. It is required for all users on API version 2.0.0 and above.

Documentation is available [here](https://docs.unstract.com/llmwhisperer/).

### Covered surface

`specs/llmwhisperer.json` describes the whole service. It is a copy of the
service's own published spec, taken at the revision `tools/gen_sdk.sh` records —
never edited here, because an edit is reverted by the next refresh. A fix goes
upstream first and comes back through a re-copy.

This client wraps a subset of that surface, unchanged from what it has always
wrapped. The operations it deliberately does not expose are listed as
`UNWRAPPED_OPERATIONS` in `tests/unit/compat_test.py`, which fails if the two
disagree — so that list, not this paragraph, is what to read.

### Page separator

The query parameter carrying the page separator was renamed in LLMWhisperer
v2.64.2; before that release the service read only the earlier, misspelled
spelling. The client sends the value under **both** spellings, so a custom
`page_separator` applies whatever version the service is on, and no version
check is needed against a self-hosted deployment. The behaviour is pinned by
`tests/unit/client_v2_test.py`; do not drop either spelling as redundant.

## Running Tests

Install test dependencies and run all tests:

```bash
uv run --group test pytest
```

To run only unit tests (skipping integration tests):

```bash
uv run --group test pytest tests/unit tests/utils_test.py
```

To run only integration tests:

```bash
uv run --group test pytest tests/integration
```

Integration tests require a valid API key. Copy `sample.env` to `.env` and fill in your credentials before running them.

## Questions and Feedback

On Slack, [join great conversations](https://join-slack.unstract.com/) around LLMs, their ecosystem and leveraging them to automate the previously unautomatable!

[LLMWhisperer Playground](https://pg.llmwhisperer.unstract.com/): Test drive LLMWhisperer with your own documents. No sign up needed!

[LLMWhisperer developer documentation and playground](https://dev-pg.llmwhisperer.unstract.com/): Learn more about LLMWhisperer and its API.
