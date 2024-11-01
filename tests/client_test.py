import logging
import os
from difflib import SequenceMatcher, unified_diff
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


def test_get_usage_info(client):
    usage_info = client.get_usage_info()
    logger.info(usage_info)
    assert isinstance(usage_info, dict), "usage_info should be a dictionary"
    expected_keys = [
        "current_page_count",
        "daily_quota",
        "monthly_quota",
        "overage_page_count",
        "subscription_plan",
        "today_page_count",
    ]
    assert set(usage_info.keys()) == set(expected_keys), f"usage_info {usage_info} does not contain the expected keys"


@pytest.mark.parametrize(
    "processing_mode, output_mode, input_file",
    [
        ("ocr", "line-printer", "restaurant_invoice_photo.pdf"),
        ("ocr", "line-printer", "credit_card.pdf"),
        ("ocr", "line-printer", "handwritten-form.pdf"),
        ("ocr", "text", "restaurant_invoice_photo.pdf"),
        ("text", "line-printer", "restaurant_invoice_photo.pdf"),
        ("text", "text", "handwritten-form.pdf"),
    ],
)
def test_whisper(client, data_dir, processing_mode, output_mode, input_file):
    file_path = os.path.join(data_dir, input_file)
    whisper_result = client.whisper(
        processing_mode=processing_mode,
        output_mode=output_mode,
        file_path=file_path,
        timeout=200,
    )
    logger.debug(whisper_result)

    exp_basename = f"{Path(input_file).stem}.{processing_mode}.{output_mode}.txt"
    exp_file = os.path.join(data_dir, "expected", exp_basename)
    assert_extracted_text(exp_file, whisper_result, processing_mode, output_mode)


def assert_extracted_text(file_path, whisper_result, mode, output_mode):
    with open(file_path, encoding="utf-8") as f:
        exp = f.read()

    assert isinstance(whisper_result, dict)
    assert whisper_result["status_code"] == 200

    # For OCR based processing
    threshold = 0.97

    # For text based processing
    if mode == "native_text" and output_mode == "text":
        threshold = 0.99
    extracted_text = whisper_result["extracted_text"]
    similarity = SequenceMatcher(None, extracted_text, exp).ratio()

    if similarity < threshold:
        diff = "\n".join(
            unified_diff(exp.splitlines(), extracted_text.splitlines(), fromfile="Expected", tofile="Extracted")
        )
        pytest.fail(f"Texts are not similar enough: {similarity * 100:.2f}% similarity. Diff:\n{diff}")
