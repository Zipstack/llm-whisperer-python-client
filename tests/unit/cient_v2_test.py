WEBHOOK_URL = "http://test-webhook.com/callback"
AUTH_TOKEN = "dummy-auth-token"
WEBHOOK_NAME = "test_webhook"
WEBHOOK_RESPONSE = {"status": "success", "message": "Webhook registered successfully"}
WHISPER_RESPONSE = {"status_code": 200, "extraction": {"result_text": "Test result"}}


def test_register_webhook(mocker, client_v2):
    mock_send = mocker.patch("requests.Session.send")
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"status": "success", "message": "Webhook registered successfully"}'  # noqa: E501
    mock_send.return_value = mock_response

    response = client_v2.register_webhook(WEBHOOK_URL, AUTH_TOKEN, WEBHOOK_NAME)

    assert response == WEBHOOK_RESPONSE
    mock_send.assert_called_once()


def test_get_webhook_details(mocker, client_v2):
    mock_send = mocker.patch("requests.Session.send")
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.text = (
        '{"status": "success", "webhook_details": {"url": "http://test-webhook.com/callback"}}'  # noqa: E501
    )
    mock_send.return_value = mock_response

    response = client_v2.get_webhook_details(WEBHOOK_NAME)

    assert response["status"] == "success"
    assert response["webhook_details"]["url"] == WEBHOOK_URL
    mock_send.assert_called_once()
