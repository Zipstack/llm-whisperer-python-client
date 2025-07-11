import logging
import os
from difflib import SequenceMatcher, unified_diff
from pathlib import Path

import pytest
from unstract.llmwhisperer.client_v2 import (
    LLMWhispererClientException,
    LLMWhispererClientV2,
)

logger = logging.getLogger(__name__)

# Test tolerance constants for better maintainability
COORDINATE_TOLERANCE = 2
PERCENTAGE_TOLERANCE = 0.05
PAGE_HEIGHT_TOLERANCE = 5
OCR_SIMILARITY_THRESHOLD = 0.90


def test_get_usage_info(client_v2: LLMWhispererClientV2) -> None:
    usage_info = client_v2.get_usage_info()
    logger.info(usage_info)
    assert isinstance(usage_info, dict), "usage_info should be a dictionary"
    expected_keys = [
        "current_page_count",
        "current_page_count_low_cost",
        "current_page_count_form",
        "current_page_count_high_quality",
        "current_page_count_native_text",
        "current_page_count_excel",
        "daily_quota",
        "monthly_quota",
        "overage_page_count",
        "subscription_plan",
        "today_page_count",
        "current_page_count_table",
    ]
    assert set(usage_info.keys()) == set(expected_keys), f"usage_info {usage_info} does not contain the expected keys"


@pytest.mark.parametrize(
    "output_mode, mode, input_file",
    [
        ("layout_preserving", "native_text", "credit_card.pdf"),
        ("layout_preserving", "low_cost", "credit_card.pdf"),
        ("layout_preserving", "high_quality", "restaurant_invoice_photo.pdf"),
        ("layout_preserving", "form", "handwritten-form.pdf"),
        ("text", "native_text", "credit_card.pdf"),
        ("text", "low_cost", "credit_card.pdf"),
        ("text", "high_quality", "restaurant_invoice_photo.pdf"),
        ("text", "form", "handwritten-form.pdf"),
        ("layout_preserving", "high_quality", "utf_8_chars.pdf"),
    ],
)
def test_whisper_v2(
    client_v2: LLMWhispererClientV2,
    data_dir: str,
    output_mode: str,
    mode: str,
    input_file: str,
) -> None:
    file_path = os.path.join(data_dir, input_file)
    whisper_result = client_v2.whisper(
        mode=mode,
        output_mode=output_mode,
        file_path=file_path,
        wait_for_completion=True,
    )
    logger.debug(f"Result for '{output_mode}', '{mode}', " f"'{input_file}: {whisper_result}")

    exp_basename = f"{Path(input_file).stem}.{mode}.{output_mode}.txt"
    exp_file = os.path.join(data_dir, "expected", exp_basename)
    # verify extracted text
    assert_extracted_text(exp_file, whisper_result, mode, output_mode)


@pytest.mark.parametrize(
    "input_file",
    [
        ("credit_card.pdf"),
    ],
)
def test_highlight(client_v2: LLMWhispererClientV2, data_dir: str, input_file: str) -> None:
    file_path = os.path.join(data_dir, input_file)

    whisper_result = client_v2.whisper(
        add_line_nos=True,
        file_path=file_path,
        wait_for_completion=True,
    )
    whisper_hash = whisper_result["whisper_hash"]
    highlight_data = client_v2.get_highlight_data(whisper_hash=whisper_hash, lines="1-2")

    # Assert the structure and content of highlight_data
    assert isinstance(highlight_data, dict)
    assert len(highlight_data) == 2
    assert "1" in highlight_data
    assert "2" in highlight_data

    # Assert line 1 data
    line1 = highlight_data["1"]
    assert line1["base_y"] == 0
    assert line1["base_y_percent"] == 0
    assert line1["height"] == 0
    assert line1["height_percent"] == 0
    assert line1["page"] == 0
    assert line1["page_height"] == 0
    assert line1["raw"] == [0, 0, 0, 0]

    # Assert line 2 data
    line2 = highlight_data["2"]
    assert line2["base_y"] == pytest.approx(155, abs=COORDINATE_TOLERANCE)
    assert line2["base_y_percent"] == pytest.approx(4.8927, abs=PERCENTAGE_TOLERANCE)
    assert line2["height"] == pytest.approx(51, abs=COORDINATE_TOLERANCE)
    assert line2["height_percent"] == pytest.approx(1.6098, abs=PERCENTAGE_TOLERANCE)
    assert line2["page"] == 0
    assert line2["page_height"] == pytest.approx(3168, abs=PAGE_HEIGHT_TOLERANCE)


@pytest.mark.parametrize(
    "output_mode, mode, url, input_file, page_count",
    [
        (
            "layout_preserving",
            "native_text",
            "https://unstractpocstorage.blob.core.windows.net/public/Amex.pdf",
            "credit_card.pdf",
            7,
        ),
        (
            "layout_preserving",
            "low_cost",
            "https://unstractpocstorage.blob.core.windows.net/public/Amex.pdf",
            "credit_card.pdf",
            7,
        ),
        (
            "layout_preserving",
            "high_quality",
            "https://unstractpocstorage.blob.core.windows.net/public/scanned_bill.pdf",
            "restaurant_invoice_photo.pdf",
            1,
        ),
        (
            "layout_preserving",
            "form",
            "https://unstractpocstorage.blob.core.windows.net/public/scanned_form.pdf",
            "handwritten-form.pdf",
            1,
        ),
    ],
)
def test_whisper_v2_url_in_post(
    client_v2: LLMWhispererClientV2,
    data_dir: str,
    output_mode: str,
    mode: str,
    url: str,
    input_file: str,
    page_count: int,
) -> None:
    usage_before = client_v2.get_usage_info()
    whisper_result = client_v2.whisper(mode=mode, output_mode=output_mode, url=url, wait_for_completion=True)
    logger.debug(f"Result for '{output_mode}', '{mode}', " f"'{input_file}: {whisper_result}")

    exp_basename = f"{Path(input_file).stem}.{mode}.{output_mode}.txt"
    exp_file = os.path.join(data_dir, "expected", exp_basename)
    # verify extracted text
    assert_extracted_text(exp_file, whisper_result, mode, output_mode)
    usage_after = client_v2.get_usage_info()
    # Verify usage after extraction
    verify_usage(usage_before, usage_after, page_count, mode)


@pytest.mark.parametrize(
    "url,token,webhook_name",
    [
        (
            os.getenv("WEBHOOK_TEST_URL", "https://httpbin.org/post"),  # configurable via env var, defaults to httpbin.org
            "",
            "client_v2_test",
        ),
    ],
)
def test_webhook(client_v2: LLMWhispererClientV2, url: str, token: str, webhook_name: str) -> None:
    """Tests the registration, retrieval, update, and deletion of a webhook.

    This test method performs the following steps:
    1. Registers a new webhook with the provided URL, token, and webhook name.
    2. Retrieves the details of the registered webhook and verifies the URL, token, and webhook name.
    3. Updates the webhook details with a new token.
    4. Deletes the webhook and verifies the deletion.

    Args:
        client_v2 (LLMWhispererClientV2): The client instance for making API requests.
        url (str): The URL of the webhook.
        token (str): The authentication token for the webhook.
        webhook_name (str): The name of the webhook.

    Returns:
        None
    """
    result = client_v2.register_webhook(url, token, webhook_name)
    assert isinstance(result, dict)
    assert result["message"] == "Webhook created successfully"

    result = client_v2.get_webhook_details(webhook_name)
    assert isinstance(result, dict)
    assert result["url"] == url
    assert result["auth_token"] == token
    assert result["webhook_name"] == webhook_name

    result = client_v2.update_webhook_details(webhook_name, url, "new_token")
    assert isinstance(result, dict)
    assert result["message"] == "Webhook updated successfully"

    result = client_v2.get_webhook_details(webhook_name)
    assert isinstance(result, dict)
    assert result["auth_token"] == "new_token"

    result = client_v2.delete_webhook(webhook_name)
    assert isinstance(result, dict)
    assert result["message"] == "Webhook deleted successfully"

    try:
        client_v2.get_webhook_details(webhook_name)
    except LLMWhispererClientException as e:
        assert e.error_message()["message"] == "Webhook details not found"
        assert e.error_message()["status_code"] == 404


def assert_error_message(whisper_result: dict) -> None:
    assert isinstance(whisper_result, dict)
    assert whisper_result["status"] == "error"
    assert "error" in whisper_result["message"]


def assert_extracted_text(file_path: str, whisper_result: dict, mode: str, output_mode: str) -> None:
    with open(file_path, encoding="utf-8") as f:
        exp = f.read()

    assert isinstance(whisper_result, dict)
    assert whisper_result["status_code"] == 200

    # For OCR based processing
    threshold = OCR_SIMILARITY_THRESHOLD

    # For text based processing
    if mode == "native_text" and output_mode == "text":
        threshold = 0.99
    elif mode == "low_cost":
        threshold = OCR_SIMILARITY_THRESHOLD
    extracted_text = whisper_result["extraction"]["result_text"]
    similarity = SequenceMatcher(None, extracted_text, exp).ratio()

    if similarity < threshold:
        diff = "\n".join(
            unified_diff(
                exp.splitlines(),
                extracted_text.splitlines(),
                fromfile="Expected",
                tofile="Extracted",
            )
        )
        pytest.fail(f"Diff:\n{diff}.\n Texts are not similar enough: {similarity * 100:.2f}% similarity. ")


def verify_usage(before_extract: dict, after_extract: dict, page_count: int, mode: str = "form") -> None:
    all_modes = ["form", "high_quality", "low_cost", "native_text"]
    all_modes.remove(mode)
    assert (
        after_extract["today_page_count"] == before_extract["today_page_count"] + page_count
    ), "today_page_count calculation is wrong"
    assert (
        after_extract["current_page_count"] == before_extract["current_page_count"] + page_count
    ), "current_page_count calculation is wrong"
    if after_extract["overage_page_count"] > 0:
        assert (
            after_extract["overage_page_count"] == before_extract["overage_page_count"] + page_count
        ), "overage_page_count calculation is wrong"
    assert (
        after_extract[f"current_page_count_{mode}"] == before_extract[f"current_page_count_{mode}"] + page_count
    ), f"{mode} mode calculation is wrong"
    for i in range(len(all_modes)):
        assert (
            after_extract[f"current_page_count_{all_modes[i]}"] == before_extract[f"current_page_count_{all_modes[i]}"]
        ), f"{all_modes[i]} mode calculation is wrong"
