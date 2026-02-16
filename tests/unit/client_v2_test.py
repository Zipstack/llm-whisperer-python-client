import time
from unittest.mock import MagicMock

import pytest
import requests
from pytest_mock import MockerFixture
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException, LLMWhispererClientV2

WEBHOOK_URL = "http://test-webhook.com/callback"
AUTH_TOKEN = "dummy-auth-token"
WEBHOOK_NAME = "test_webhook"
WEBHOOK_RESPONSE = {"message": "Webhook registered successfully"}
WHISPER_RESPONSE = {"status_code": 200, "extraction": {"result_text": "Test result"}}


def test_register_webhook(mocker: MockerFixture, client_v2: LLMWhispererClientV2) -> None:
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.text = '{"message": "Webhook registered successfully"}'  # noqa: E501
    mock_send.return_value = mock_response

    response = client_v2.register_webhook(WEBHOOK_URL, AUTH_TOKEN, WEBHOOK_NAME)

    assert response == WEBHOOK_RESPONSE
    mock_send.assert_called_once()


def test_get_webhook_details(mocker: MockerFixture, client_v2: LLMWhispererClientV2) -> None:
    mock_send = mocker.patch("requests.Session.send")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"status": "success", "webhook_details": {"url": "http://test-webhook.com/callback"}}'  # noqa: E501
    mock_send.return_value = mock_response

    response = client_v2.get_webhook_details(WEBHOOK_NAME)

    assert response["status"] == "success"
    assert response["webhook_details"]["url"] == WEBHOOK_URL


def test_whisper_json_string_response_error(mocker: MockerFixture, client_v2: LLMWhispererClientV2) -> None:
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


def test_whisper_json_string_response_202(mocker: MockerFixture, client_v2: LLMWhispererClientV2) -> None:
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


def test_whisper_invalid_json_response_error(mocker: MockerFixture, client_v2: LLMWhispererClientV2) -> None:
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


def test_whisper_invalid_json_response_202(mocker: MockerFixture, client_v2: LLMWhispererClientV2) -> None:
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


# --- Retry behavior tests ---


@pytest.fixture(name="retry_client")
def llm_whisperer_retry_client() -> LLMWhispererClientV2:
    return LLMWhispererClientV2(
        base_url="http://localhost",
        api_key="test-key",
        logging_level="ERROR",
        max_retries=3,
        retry_min_wait=0.01,
        retry_max_wait=0.02,
    )


@pytest.fixture(name="no_retry_client")
def llm_whisperer_no_retry_client() -> LLMWhispererClientV2:
    return LLMWhispererClientV2(
        base_url="http://localhost",
        api_key="test-key",
        logging_level="ERROR",
        max_retries=0,
    )


def _mock_response(status_code: int = 200, text: str = '{"status": "ok"}') -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.headers = {}
    return resp


def test_retry_on_connection_error(mocker: MockerFixture, retry_client: LLMWhispererClientV2) -> None:
    """ConnectionError triggers retry, succeeds on 3rd attempt."""
    success_resp = _mock_response(200, '{"pages_processed": 1}')
    mock_send = mocker.patch(
        "requests.Session.send",
        side_effect=[
            requests.ConnectionError("connection refused"),
            requests.ConnectionError("connection refused"),
            success_resp,
        ],
    )

    result = retry_client.get_usage_info()

    assert result == {"pages_processed": 1}
    assert mock_send.call_count == 3


def test_retry_on_timeout(mocker: MockerFixture, retry_client: LLMWhispererClientV2) -> None:
    """Timeout triggers retry, succeeds on 2nd attempt."""
    success_resp = _mock_response(200, '{"pages_processed": 1}')
    mock_send = mocker.patch(
        "requests.Session.send",
        side_effect=[
            requests.Timeout("request timed out"),
            success_resp,
        ],
    )

    result = retry_client.get_usage_info()

    assert result == {"pages_processed": 1}
    assert mock_send.call_count == 2


def test_retry_on_429(mocker: MockerFixture, retry_client: LLMWhispererClientV2) -> None:
    """HTTP 429 triggers retry, succeeds on 2nd attempt."""
    rate_limit_resp = _mock_response(429, '{"error": "rate limited"}')
    success_resp = _mock_response(200, '{"pages_processed": 1}')
    mock_send = mocker.patch(
        "requests.Session.send",
        side_effect=[rate_limit_resp, success_resp],
    )

    result = retry_client.get_usage_info()

    assert result == {"pages_processed": 1}
    assert mock_send.call_count == 2


def test_retry_on_500(mocker: MockerFixture, retry_client: LLMWhispererClientV2) -> None:
    """HTTP 500 triggers retry, succeeds on 2nd attempt."""
    server_err_resp = _mock_response(500, '{"error": "internal server error"}')
    success_resp = _mock_response(200, '{"pages_processed": 1}')
    mock_send = mocker.patch(
        "requests.Session.send",
        side_effect=[server_err_resp, success_resp],
    )

    result = retry_client.get_usage_info()

    assert result == {"pages_processed": 1}
    assert mock_send.call_count == 2


def test_no_retry_on_400(mocker: MockerFixture, retry_client: LLMWhispererClientV2) -> None:
    """HTTP 400 does NOT retry (client error)."""
    bad_request_resp = _mock_response(400, '{"error": "bad request"}')
    mock_send = mocker.patch("requests.Session.send", return_value=bad_request_resp)

    with pytest.raises(LLMWhispererClientException):
        retry_client.get_usage_info()

    assert mock_send.call_count == 1


def test_no_retry_on_401(mocker: MockerFixture, retry_client: LLMWhispererClientV2) -> None:
    """HTTP 401 does NOT retry (auth error)."""
    unauth_resp = _mock_response(401, '{"error": "unauthorized"}')
    mock_send = mocker.patch("requests.Session.send", return_value=unauth_resp)

    with pytest.raises(LLMWhispererClientException):
        retry_client.get_usage_info()

    assert mock_send.call_count == 1


def test_retries_exhausted_raises(mocker: MockerFixture, retry_client: LLMWhispererClientV2) -> None:
    """After all retries exhausted on ConnectionError, raises the exception."""
    mock_send = mocker.patch(
        "requests.Session.send",
        side_effect=requests.ConnectionError("connection refused"),
    )

    with pytest.raises(requests.ConnectionError):
        retry_client.get_usage_info()

    # 1 initial + 3 retries = 4 total attempts
    assert mock_send.call_count == 4


def test_retries_exhausted_500_returns_response(mocker: MockerFixture, retry_client: LLMWhispererClientV2) -> None:
    """After all retries exhausted on 500, returns the error response (caller
    raises exception)."""
    server_err_resp = _mock_response(500, '{"error": "internal server error"}')
    mock_send = mocker.patch("requests.Session.send", return_value=server_err_resp)

    with pytest.raises(LLMWhispererClientException):
        retry_client.get_usage_info()

    # 1 initial + 3 retries = 4 total attempts
    assert mock_send.call_count == 4


def test_retry_disabled(mocker: MockerFixture, no_retry_client: LLMWhispererClientV2) -> None:
    """max_retries=0 means single attempt only, no retry on failure."""
    mock_send = mocker.patch(
        "requests.Session.send",
        side_effect=requests.ConnectionError("connection refused"),
    )

    with pytest.raises(requests.ConnectionError):
        no_retry_client.get_usage_info()

    assert mock_send.call_count == 1


# --- Deadline and whisper timeout budget tests ---


def test_whisper_post_uses_min_of_api_timeout_and_wait_timeout(
    mocker: MockerFixture,
) -> None:
    """Whisper() POST should use min(api_timeout, wait_timeout), not raw
    wait_timeout."""
    client = LLMWhispererClientV2(
        base_url="http://localhost",
        api_key="test-key",
        logging_level="ERROR",
        max_retries=0,
    )
    # api_timeout defaults to 120, wait_timeout will be 180
    # So POST timeout should be min(120, 180) = 120
    mock_response = _mock_response(200, '{"status_code": 200, "extraction": {"text": "ok"}}')
    mock_send = mocker.patch("requests.Session.send", return_value=mock_response)

    client.whisper(url="https://example.com/test.pdf", wait_timeout=180)

    call_kwargs = mock_send.call_args
    # timeout should be 120 (api_timeout), not 180 (wait_timeout)
    assert call_kwargs.kwargs.get("timeout") is not None or call_kwargs[1].get("timeout") is not None
    actual_timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
    assert actual_timeout <= 120, f"Expected timeout <= 120 (api_timeout), got {actual_timeout}"


def test_whisper_post_uses_wait_timeout_when_smaller(
    mocker: MockerFixture,
) -> None:
    """When wait_timeout < api_timeout, POST timeout should use
    wait_timeout."""
    client = LLMWhispererClientV2(
        base_url="http://localhost",
        api_key="test-key",
        logging_level="ERROR",
        max_retries=0,
    )
    mock_response = _mock_response(200, '{"status_code": 200, "extraction": {"text": "ok"}}')
    mock_send = mocker.patch("requests.Session.send", return_value=mock_response)

    client.whisper(url="https://example.com/test.pdf", wait_timeout=10)

    call_kwargs = mock_send.call_args
    actual_timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
    assert actual_timeout <= 10, f"Expected timeout <= 10 (wait_timeout), got {actual_timeout}"


def test_send_request_deadline_caps_timeout(mocker: MockerFixture) -> None:
    """_send_request with deadline should cap individual request timeout to
    remaining time."""
    client = LLMWhispererClientV2(
        base_url="http://localhost",
        api_key="test-key",
        logging_level="ERROR",
        max_retries=0,
    )
    mock_response = _mock_response(200)
    mock_send = mocker.patch("requests.Session.send", return_value=mock_response)

    # Set deadline 2 seconds from now, but request timeout=300
    deadline = time.time() + 2.0
    req = requests.Request("GET", "http://localhost/test", headers={})
    prepared = req.prepare()

    client._send_request(prepared, timeout=300, deadline=deadline)

    call_kwargs = mock_send.call_args
    actual_timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
    # Should be capped to ~2s (remaining time), not 300
    assert actual_timeout <= 3.0, f"Expected timeout <= ~2s (deadline), got {actual_timeout}"


def test_send_request_deadline_stops_retries(mocker: MockerFixture) -> None:
    """_send_request with an already-expired deadline should stop retrying
    early."""
    client = LLMWhispererClientV2(
        base_url="http://localhost",
        api_key="test-key",
        logging_level="ERROR",
        max_retries=10,
        retry_min_wait=0.1,
        retry_max_wait=0.2,
    )

    server_err_resp = _mock_response(500, '{"error": "internal server error"}')
    mock_send = mocker.patch("requests.Session.send", return_value=server_err_resp)

    # Deadline is 0.3s from now — with 0.1-0.2s waits between retries,
    # only a few attempts should fit before the deadline expires
    deadline = time.time() + 0.3
    req = requests.Request("GET", "http://localhost/test", headers={})
    prepared = req.prepare()

    response = client._send_request(prepared, timeout=1, deadline=deadline)

    # Should have stopped well before 11 attempts (1 initial + 10 retries)
    assert response.status_code == 500
    assert mock_send.call_count < 11, f"Expected fewer than 11 attempts due to deadline, got {mock_send.call_count}"
