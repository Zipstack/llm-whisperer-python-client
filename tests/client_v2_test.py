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
    ],
)
def test_whisper_v2(client_v2, data_dir, output_mode, mode, input_file):
    file_path = os.path.join(data_dir, input_file)
    whisper_result = client_v2.whisper(
        mode=mode, output_mode=output_mode, file_path=file_path, wait_for_completion=True
    )
    logger.debug(f"Result for '{output_mode}', '{mode}', " f"'{input_file}: {whisper_result}")

    exp_basename = f"{Path(input_file).stem}.{mode}.{output_mode}.txt"
    exp_file = os.path.join(data_dir, "expected", exp_basename)
    with open(exp_file, encoding="utf-8") as f:
        exp = f.read()

    assert isinstance(whisper_result, dict)
    assert whisper_result["status_code"] == 200

    # For text based processing, perform a strict match
    if mode == "native_text" and output_mode == "text":
        assert whisper_result["extraction"]["result_text"] == exp
    # For OCR based processing, perform a fuzzy match
    else:
        extracted_text = whisper_result["extraction"]["result_text"]
        similarity = SequenceMatcher(None, extracted_text, exp).ratio()
        threshold = 0.97

        if similarity < threshold:
            diff = "\n".join(
                unified_diff(exp.splitlines(), extracted_text.splitlines(), fromfile="Expected", tofile="Extracted")
            )
            pytest.fail(f"Texts are not similar enough: {similarity * 100:.2f}% similarity. Diff:\n{diff}")

    assert whisper_result["extraction"]["result_text"] == exp
