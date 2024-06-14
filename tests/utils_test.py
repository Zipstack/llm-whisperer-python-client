import pytest

from unstract.llmwhisperer.utils import LLMWhispererUtils


def test_redact_key_normal():
    api_key = "1234567890abcdef"  # gitleaks:allow
    expected = "1234xxxxxxxxxxxx"
    assert LLMWhispererUtils.redact_key(api_key) == expected


def test_redact_key_different_reveal_length():
    api_key = "1234567890abcdef"  # gitleaks:allow
    expected = "123456xxxxxxxxxx"
    assert LLMWhispererUtils.redact_key(api_key, reveal_length=6) == expected


def test_redact_key_non_string_input():
    with pytest.raises(ValueError, match="API key must be a string"):
        LLMWhispererUtils.redact_key(12345)
