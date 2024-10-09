import json
import logging
import os
import time
from client_v2 import LLMWhispererClientV2, LLMWhispererClientException


if __name__ == "__main__":
    client = LLMWhispererClientV2()

    try:
        # result = client.whisper(
        #     mode="native_text",
        #     output_mode="layout_preserving",
        #     file_path="../../../tests/test_data/credit_card.pdf",
        # )
        # result = client.whisper(
        #     mode="high_quality",
        #     output_mode="layout_preserving",
        #     file_path="../../../tests/test_data/credit_card.pdf",
        # )
        # result = client.whisper(
        #     mode="low_cost",
        #     output_mode="layout_preserving",
        #     file_path="../../../tests/test_data/credit_card.pdf",
        # )

        # result = client.register_webhook(
        #     url="https://webhook.site/15422328-2a5e-4a1d-9a20-f78313ca5007",
        #     auth_token="my_auth_token",
        #     webhook_name="wb3",
        # )
        # print(result)

        # result = client.get_webhook_details(webhook_name="wb3")
        # print(result)

        result = client.whisper(
            mode="high_quality",
            output_mode="layout_preserving",
            file_path="../../../tests/test_data/credit_card.pdf",
            use_webhook="wb3",
            webhook_metadata="Dummy metadata for webhook",
        )

        # result = client.whisper(
        #     mode="form",
        #     output_mode="layout_preserving",
        #     file_path="../../../tests/test_data/credit_card.pdf",
        # )

        # if result["status_code"] == 202:
        #     print("Whisper request accepted.")
        #     print(f"Whisper hash: {result['whisper_hash']}")
        #     while True:
        #         print("Polling for whisper status...")
        #         status = client.whisper_status(whisper_hash=result["whisper_hash"])
        #         print(status)
        #         if status["status"] == "processing":
        #             print("STATUS: processing...")
        #         elif status["status"] == "delivered":
        #             print("STATUS: Already delivered!")
        #             break
        #         elif status["status"] == "unknown":
        #             print("STATUS: unknown...")
        #             break
        #         elif status["status"] == "processed":
        #             print("STATUS: processed!")
        #             print("Let's retrieve the result of the extraction...")
        #             resultx = client.whisper_retrieve(
        #                 whisper_hash=result["whisper_hash"]
        #             )
        #             print(resultx)
        #             break
        #         time.sleep(2)
    except LLMWhispererClientException as e:
        print(e)
