import unittest

from llmwhisperer.client import LLMWhispererClient


class TestLLMWhispererClient(unittest.TestCase):
    @unittest.skip("Skipping test_get_usage_info")
    def test_get_usage_info(self):
        client = LLMWhispererClient()
        usage_info = client.get_usage_info()
        print(usage_info)
        self.assertIsInstance(usage_info, dict)

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
        print(response)
        # self.assertIsInstance(response, dict)

    @unittest.skip("Skipping test_whisper_status")
    def test_whisper_status(self):
        client = LLMWhispererClient()
        response = client.whisper_status(whisper_hash="7cfa5cbb|5f1d285a7cf18d203de7af1a1abb0a3a")
        print(response)
        self.assertIsInstance(response, dict)

    @unittest.skip("Skipping test_whisper_retrieve")
    def test_whisper_retrieve(self):
        client = LLMWhispererClient()
        response = client.whisper_retrieve(whisper_hash="7cfa5cbb|5f1d285a7cf18d203de7af1a1abb0a3a")
        print(response)
        self.assertIsInstance(response, dict)

    def test_whisper_highlight_data(self):
        client = LLMWhispererClient()
        response = client.highlight_data(
            whisper_hash="9924d865|5f1d285a7cf18d203de7af1a1abb0a3a",
            search_text="Indiranagar",
        )
        print(response)
        self.assertIsInstance(response, dict)


if __name__ == "__main__":
    unittest.main()
