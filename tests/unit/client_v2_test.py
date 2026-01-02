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


def test_get_usage_with_tag_only(mocker: Any, client_v2: LLMWhispererClientV2) -> None:
    """Test get_usage method with only required tag parameter."""
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = """{
        "end_date": "2024-01-31",
        "start_date": "2024-01-01",
        "subscription_id": "test-subscription-id",
        "tag": "test-tag",
        "usage": [
            {"service_type": "form", "pages_processed": 100},
            {"service_type": "high_quality", "pages_processed": 50}
        ]
    }"""
    mock_send.return_value = mock_response

    response = client_v2.get_usage(tag="test-tag")

    assert response["tag"] == "test-tag"
    assert response["subscription_id"] == "test-subscription-id"
    assert "usage" in response
    assert len(response["usage"]) == 2
    mock_send.assert_called_once()


def test_get_usage_with_date_range(mocker: Any, client_v2: LLMWhispererClientV2) -> None:
    """Test get_usage method with tag and date range parameters."""
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = """{
        "end_date": "2024-01-31",
        "start_date": "2024-01-01",
        "subscription_id": "test-subscription-id",
        "tag": "production",
        "usage": [
            {"service_type": "form", "pages_processed": 500}
        ]
    }"""
    mock_send.return_value = mock_response

    response = client_v2.get_usage(tag="production", from_date="2024-01-01", to_date="2024-01-31")

    assert response["start_date"] == "2024-01-01"
    assert response["end_date"] == "2024-01-31"
    assert response["tag"] == "production"
    mock_send.assert_called_once()


def test_get_usage_error_response(mocker: Any, client_v2: LLMWhispererClientV2) -> None:
    """Test get_usage method handles error responses correctly."""
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = '{"message": "Invalid tag parameter"}'
    mock_send.return_value = mock_response

    with pytest.raises(LLMWhispererClientException) as exc_info:
        client_v2.get_usage(tag="invalid-tag")

    error = exc_info.value.args[0]
    assert error["message"] == "Invalid tag parameter"
    assert error["status_code"] == 400
