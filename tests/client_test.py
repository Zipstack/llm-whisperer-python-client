import logging
import os
import unittest
from pathlib import Path

import pytest

from unstract.llmwhisperer import LLMWhispererClient

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
    response = client.whisper(
        processing_mode=processing_mode,
        output_mode=output_mode,
        file_path=file_path,
        timeout=200,
    )
    logger.debug(response)

    exp_basename = f"{Path(input_file).stem}.{processing_mode}.{output_mode}.txt"
    exp_file = os.path.join(data_dir, "expected", exp_basename)
    with open(exp_file, encoding="utf-8") as f:
        exp = f.read()

    assert isinstance(response, dict)
    assert response["status_code"] == 200
    assert response["extracted_text"] == exp


# TODO: Review and port to pytest based tests
class TestLLMWhispererClient(unittest.TestCase):
    @unittest.skip("Skipping test_whisper")
    def test_whisper(self):
        client = LLMWhispererClient()
        # response = client.whisper(
        #     url="https://storage.googleapis.com/pandora-static/samples/bill.jpg.pdf"
        # )
        response = client.whisper(
            file_path="test_files/restaurant_invoice_photo.pdf",
            timeout=200,
            store_metadata_for_highlighting=True,
        )
        logger.info(response)
        # self.assertIsInstance(response, dict)

    @unittest.skip("Skipping test_whisper_status")
    def test_whisper_status(self):
        client = LLMWhispererClient()
        response = client.whisper_status(whisper_hash="7cfa5cbb|5f1d285a7cf18d203de7af1a1abb0a3a")
        logger.info(response)
        self.assertIsInstance(response, dict)

    @unittest.skip("Skipping test_whisper_retrieve")
    def test_whisper_retrieve(self):
        client = LLMWhispererClient()
        response = client.whisper_retrieve(whisper_hash="7cfa5cbb|5f1d285a7cf18d203de7af1a1abb0a3a")
        logger.info(response)
        self.assertIsInstance(response, dict)

    @unittest.skip("Skipping test_whisper_highlight_data")
    def test_whisper_highlight_data(self):
        client = LLMWhispererClient()
        response = client.highlight_data(
            whisper_hash="9924d865|5f1d285a7cf18d203de7af1a1abb0a3a",
            search_text="Indiranagar",
        )
        logger.info(response)
        self.assertIsInstance(response, dict)


if __name__ == "__main__":
    unittest.main()
