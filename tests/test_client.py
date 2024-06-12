import logging
import unittest

import pytest

from llmwhisperer.client import LLMWhispererClient

logger = logging.getLogger(__name__)


@pytest.fixture
def llm_whisperer_client():
    # Create an instance of the client
    client = LLMWhispererClient()
    return client


def test_get_usage_info(llm_whisperer_client):
    usage_info = llm_whisperer_client.get_usage_info()
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
