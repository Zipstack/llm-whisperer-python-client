import logging
import os
from difflib import SequenceMatcher, unified_diff
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


def test_get_usage_info(client_v2):
    usage_info = client_v2.get_usage_info()
    logger.info(usage_info)
    assert isinstance(usage_info, dict), "usage_info should be a dictionary"
    expected_keys = [
        "current_page_count",
        "current_page_count_low_cost",
        "current_page_count_form",
        "current_page_count_high_quality",
        "current_page_count_native_text",
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
        ("layout_preserving", "high_quality", "utf_8_chars.pdf"),
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
    # verify extracted text
    assert_extracted_text(exp_file, whisper_result, mode, output_mode)


@pytest.mark.parametrize(
    "output_mode, mode, url, input_file, page_count",
    [
        ("layout_preserving", "native_text", "https://unstractpocstorage.blob.core.windows.net/public/Amex.pdf",
         "credit_card.pdf", 7),
        ("layout_preserving", "low_cost", "https://unstractpocstorage.blob.core.windows.net/public/Amex.pdf",
         "credit_card.pdf", 7),
        ("layout_preserving", "high_quality", "https://unstractpocstorage.blob.core.windows.net/public/scanned_bill.pdf",
         "restaurant_invoice_photo.pdf", 1),
        ("layout_preserving", "form", "https://unstractpocstorage.blob.core.windows.net/public/scanned_form.pdf",
         "handwritten-form.pdf", 1),
    ]
)
def test_whisper_v2_url_in_post(client_v2, data_dir, output_mode, mode, url, input_file, page_count):
    usage_before = client_v2.get_usage_info()
    whisper_result = client_v2.whisper(
        mode=mode, output_mode=output_mode, url=url, wait_for_completion=True
    )
    logger.debug(f"Result for '{output_mode}', '{mode}', " f"'{input_file}: {whisper_result}")

    exp_basename = f"{Path(input_file).stem}.{mode}.{output_mode}.txt"
    exp_file = os.path.join(data_dir, "expected", exp_basename)
    # verify extracted text
    assert_extracted_text(exp_file, whisper_result, mode, output_mode)
    usage_after = client_v2.get_usage_info()
    # Verify usage after extraction
    verify_usage(usage_before, usage_after, page_count, mode)


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
    extracted_text = whisper_result["extraction"]["result_text"]
    similarity = SequenceMatcher(None, extracted_text, exp).ratio()

    if similarity < threshold:
        diff = "\n".join(
            unified_diff(exp.splitlines(), extracted_text.splitlines(), fromfile="Expected", tofile="Extracted")
        )
        pytest.fail(f"Texts are not similar enough: {similarity * 100:.2f}% similarity. Diff:\n{diff}")


def verify_usage(before_extract, after_extract, page_count, mode='form'):
    all_modes = ['form', 'high_quality', 'low_cost', 'native_text']
    all_modes.remove(mode)
    assert (after_extract['today_page_count'] == before_extract['today_page_count'] + page_count), \
        "today_page_count calculation is wrong"
    assert (after_extract['current_page_count'] == before_extract['current_page_count'] + page_count), \
        "current_page_count calculation is wrong"
    if after_extract['overage_page_count'] > 0:
        assert (after_extract['overage_page_count'] == before_extract['overage_page_count'] + page_count), \
            "overage_page_count calculation is wrong"
    assert (after_extract[f'current_page_count_{mode}'] == before_extract[f'current_page_count_{mode}'] + page_count), \
        f"{mode} mode calculation is wrong"
    for i in range(len(all_modes)):
        assert (after_extract[f'current_page_count_{all_modes[i]}'] ==
                before_extract[f'current_page_count_{all_modes[i]}']), \
            f"{all_modes[i]} mode calculation is wrong"
