from typing import Any
from unittest.mock import MagicMock

import pytest
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException, LLMWhispererClientV2

WEBHOOK_URL = "http://test-webhook.com/callback"
AUTH_TOKEN = "dummy-auth-token"
WEBHOOK_NAME = "test_webhook"
WEBHOOK_RESPONSE = {"message": "Webhook registered successfully"}
WHISPER_RESPONSE = {"status_code": 200, "extraction": {"result_text": "Test result"}}


def test_register_webhook(mocker: Any, client_v2: LLMWhispererClientV2) -> None:
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.text = '{"message": "Webhook registered successfully"}'  # noqa: E501
    mock_send.return_value = mock_response

    response = client_v2.register_webhook(WEBHOOK_URL, AUTH_TOKEN, WEBHOOK_NAME)

    assert response == WEBHOOK_RESPONSE
    mock_send.assert_called_once()


def test_get_webhook_details(mocker: Any, client_v2: LLMWhispererClientV2) -> None:
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"status": "success", "webhook_details": {"url": "http://test-webhook.com/callback"}}'  # noqa: E501
    mock_send.return_value = mock_response

    response = client_v2.get_webhook_details(WEBHOOK_NAME)

    assert response["status"] == "success"
    assert response["webhook_details"]["url"] == WEBHOOK_URL


def test_whisper_json_string_response_error(mocker: Any, client_v2: LLMWhispererClientV2) -> None:
    """Test whisper method handles JSON string responses correctly for error
    cases."""
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = '"Error message as JSON string"'
    mock_response.encoding = "utf-8"
    mock_send.return_value = mock_response

    with pytest.raises(LLMWhispererClientException) as exc_info:
        client_v2.whisper(url="https://example.com/test.pdf")

    error = exc_info.value.args[0]
    assert error["message"] == "Error message as JSON string"
    assert error["status_code"] == 400
    assert error["extraction"] == {}


def test_whisper_json_string_response_202(mocker: Any, client_v2: LLMWhispererClientV2) -> None:
    """Test whisper method handles JSON string responses correctly for 202
    status."""
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 202
    mock_response.text = '"Processing in progress"'
    mock_response.encoding = "utf-8"
    mock_send.return_value = mock_response

    response = client_v2.whisper(url="https://example.com/test.pdf", wait_for_completion=False)

    assert response["message"] == "Processing in progress"
    assert response["status_code"] == 202
    assert response["extraction"] == {}


def test_whisper_invalid_json_response_error(mocker: Any, client_v2: LLMWhispererClientV2) -> None:
    """Test whisper method handles invalid JSON responses correctly for error
    cases."""
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Invalid JSON response"
    mock_response.encoding = "utf-8"
    mock_send.return_value = mock_response

    with pytest.raises(LLMWhispererClientException) as exc_info:
        client_v2.whisper(url="https://example.com/test.pdf")

    error = exc_info.value.args[0]
    assert error["message"] == "Invalid JSON response"
    assert error["status_code"] == 500
    assert error["extraction"] == {}


def test_whisper_invalid_json_response_202(mocker: Any, client_v2: LLMWhispererClientV2) -> None:
    """Test whisper method handles invalid JSON responses correctly for 202
    status."""
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 202
    mock_response.text = "Invalid JSON response"
    mock_response.encoding = "utf-8"
    mock_send.return_value = mock_response

    response = client_v2.whisper(url="https://example.com/test.pdf", wait_for_completion=False)

    assert response["message"] == "Invalid JSON response"
    assert response["status_code"] == 202
    assert response["extraction"] == {}
