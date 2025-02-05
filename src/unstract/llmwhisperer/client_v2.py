"""This module provides a Python client for interacting with the LLMWhisperer
API.

Note: This is for the LLMWhisperer API v2.x

Prepare documents for LLM consumption
LLMs are powerful, but their output is as good as the input you provide.
LLMWhisperer is a technology that presents data from complex documents
(different designs and formats) to LLMs in a way that they can best understand.

LLMWhisperer is available as an API that can be integrated into your existing
systems to preprocess your documents before they are fed into LLMs. It can handle
a variety of document types, including PDFs, images, and scanned documents.

This client simplifies the process of making requests to the API and handling the responses.

Classes:
    LLMWhispererClientException: Exception raised for errors in the LLMWhispererClient.
"""

import copy
import json
import logging
import os
import time
from typing import IO

import requests

BASE_URL_V2 = "https://llmwhisperer-api.us-central.unstract.com/api/v2"


class LLMWhispererClientException(Exception):
    """Exception raised for errors in the LLMWhispererClient.

    Attributes:
        message (str): Explanation of the error.
        status_code (int): HTTP status code returned by the LLMWhisperer API.

    Args:
        message (str): Explanation of the error.
        status_code (int, optional): HTTP status code returned by the LLMWhisperer API. Defaults to None.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

    def error_message(self):
        return self.value


class LLMWhispererClientV2:
    """A client for interacting with the LLMWhisperer API.

    Note: This is for the LLMWhisperer API v2.x

    This client uses the requests library to make HTTP requests to the
    LLMWhisperer API. It also includes a logger for tracking the
    client's activities and errors.
    """

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    log_stream_handler = logging.StreamHandler()
    log_stream_handler.setFormatter(formatter)
    logger.addHandler(log_stream_handler)

    api_key = ""
    base_url = ""
    api_timeout = 120

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        logging_level: str = "",
    ):
        """Initializes the LLMWhispererClient with the given parameters.

        Args:
            base_url (str, optional): The base URL for the LLMWhisperer API. Defaults to "".
                                      If the base_url is not provided, the client will use
                                      the value of the LLMWHISPERER_BASE_URL_V2 environment
                                      variable,or the default value.
            api_key (str, optional): The API key for the LLMWhisperer API. Defaults to "".
                                     If the api_key is not provided, the client will use the
                                     value of the LLMWHISPERER_API_KEY environment variable.
            logging_level (str, optional): The logging level for the client. Can be "DEBUG",
                                           "INFO", "WARNING" or "ERROR". Defaults to the
                                           value of the LLMWHISPERER_LOGGING_LEVEL
                                           environment variable, or "DEBUG" if the
                                           environment variable is not set.
        """
        if logging_level == "":
            logging_level = os.getenv("LLMWHISPERER_LOGGING_LEVEL", "DEBUG")
        if logging_level == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
        elif logging_level == "INFO":
            self.logger.setLevel(logging.INFO)
        elif logging_level == "WARNING":
            self.logger.setLevel(logging.WARNING)
        elif logging_level == "ERROR":
            self.logger.setLevel(logging.ERROR)
        self.logger.setLevel(logging_level)
        self.logger.debug("logging_level set to %s", logging_level)
        if base_url == "":
            self.base_url = os.getenv("LLMWHISPERER_BASE_URL_V2", BASE_URL_V2)
        else:
            self.base_url = base_url
        self.logger.debug("base_url set to %s", self.base_url)

        if api_key == "":
            self.api_key = os.getenv("LLMWHISPERER_API_KEY", "")
        else:
            self.api_key = api_key

        self.headers = {"unstract-key": self.api_key}
        # For test purpose
        # self.headers = {
        #     "Subscription-Id": "python-client",
        #     "Subscription-Name": "python-client",
        #     "User-Id": "python-client-user",
        #     "Product-Id": "python-client-product",
        #     "Product-Name": "python-client-product",
        #     "Start-Date": "2024-07-09",
        # }

    def get_usage_info(self) -> dict:
        """Retrieves the usage information of the LLMWhisperer API.

        This method sends a GET request to the '/get-usage-info' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the usage information.
        Refer to https://docs.unstract.com/llm_whisperer/apis/llm_whisperer_usage_api

        Returns:
            dict: A dictionary containing the usage information.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                          the error message and status code returned by the API.
        """
        self.logger.debug("get_usage_info called")
        url = f"{self.base_url}/get-usage-info"
        self.logger.debug("url: %s", url)
        req = requests.Request("GET", url, headers=self.headers)
        prepared = req.prepare()
        s = requests.Session()
        response = s.send(prepared, timeout=self.api_timeout)
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)
        return json.loads(response.text)

    def whisper(
        self,
        file_path: str = "",
        stream: IO[bytes] = None,
        url: str = "",
        mode: str = "form",
        output_mode: str = "layout_preserving",
        page_seperator: str = "<<<",
        pages_to_extract: str = "",
        median_filter_size: int = 0,
        gaussian_blur_radius: int = 0,
        line_splitter_tolerance: float = 0.4,
        horizontal_stretch_factor: float = 1.0,
        mark_vertical_lines: bool = False,
        mark_horizontal_lines: bool = False,
        line_spitter_strategy: str = "left-priority",
        lang="eng",
        tag="default",
        filename="",
        webhook_metadata="",
        use_webhook="",
        wait_for_completion=False,
        wait_timeout=180,
        encoding: str = "utf-8",
    ) -> dict:
        """
        Sends a request to the LLMWhisperer API to process a document.
        Refer to https://docs.unstract.com/llm_whisperer/apis/llm_whisperer_text_extraction_api

        Args:
            file_path (str, optional): The path to the file to be processed. Defaults to "".
            stream (IO[bytes], optional): A stream of bytes to be processed. Defaults to None.
            url (str, optional): The URL of the file to be processed. Defaults to "".
            mode (str, optional): The processing mode. Can be "high_quality", "form", "low_cost" or "native_text".
                Defaults to "high_quality".
            output_mode (str, optional): The output mode. Can be "layout_preserving" or "text".
                Defaults to "layout_preserving".
            page_seperator (str, optional): The page separator. Defaults to "<<<".
            pages_to_extract (str, optional): The pages to extract. Defaults to "".
            median_filter_size (int, optional): The size of the median filter. Defaults to 0.
            gaussian_blur_radius (int, optional): The radius of the Gaussian blur. Defaults to 0.
            line_splitter_tolerance (float, optional): The line splitter tolerance. Defaults to 0.4.
            horizontal_stretch_factor (float, optional): The horizontal stretch factor. Defaults to 1.0.
            mark_vertical_lines (bool, optional): Whether to mark vertical lines. Defaults to False.
            mark_horizontal_lines (bool, optional): Whether to mark horizontal lines. Defaults to False.
            line_spitter_strategy (str, optional): The line splitter strategy. Defaults to "left-priority".
            lang (str, optional): The language of the document. Defaults to "eng".
            tag (str, optional): The tag for the document. Defaults to "default".
            filename (str, optional): The name of the file to store in reports. Defaults to "".
            webhook_metadata (str, optional): The webhook metadata. This data will be passed to the webhook if
                webhooks are used Defaults to "".
            use_webhook (str, optional): Webhook name to call. Defaults to "". If not provided, then
                no webhook will be called.
            wait_for_completion (bool, optional): Whether to wait for the whisper operation to complete.
                Defaults to False.
            wait_timeout (int, optional): The number of seconds to wait for the whisper operation to complete.
                Defaults to 180.
            encoding (str): The character encoding to use for processing the text. Defaults to "utf-8".

        Returns:
            dict: The response from the API as a dictionary.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                          the error message and status code returned by the API.
        """
        self.logger.debug("whisper called")
        api_url = f"{self.base_url}/whisper"
        params = {
            "mode": mode,
            "output_mode": output_mode,
            "page_seperator": page_seperator,
            "pages_to_extract": pages_to_extract,
            "median_filter_size": median_filter_size,
            "gaussian_blur_radius": gaussian_blur_radius,
            "line_splitter_tolerance": line_splitter_tolerance,
            "horizontal_stretch_factor": horizontal_stretch_factor,
            "mark_vertical_lines": mark_vertical_lines,
            "mark_horizontal_lines": mark_horizontal_lines,
            "line_spitter_strategy": line_spitter_strategy,
            "lang": lang,
            "tag": tag,
            "filename": filename,
            "webhook_metadata": webhook_metadata,
            "use_webhook": use_webhook,
        }

        self.logger.debug("api_url: %s", api_url)
        self.logger.debug("params: %s", params)

        if use_webhook != "" and wait_for_completion:
            raise LLMWhispererClientException(
                {
                    "status_code": -1,
                    "message": "Cannot wait for completion when using webhook",
                }
            )

        if url == "" and file_path == "" and stream is None:
            raise LLMWhispererClientException(
                {
                    "status_code": -1,
                    "message": "Either url, stream or file_path must be provided",
                }
            )

        should_stream = False
        if url == "":
            if stream is not None:
                should_stream = True

                def generate():
                    yield from stream

                req = requests.Request(
                    "POST",
                    api_url,
                    params=params,
                    headers=self.headers,
                    data=generate(),
                )

            else:
                with open(file_path, "rb") as f:
                    data = f.read()
                req = requests.Request(
                    "POST",
                    api_url,
                    params=params,
                    headers=self.headers,
                    data=data,
                )
        else:
            params["url_in_post"] = True
            req = requests.Request("POST", api_url, params=params, headers=self.headers, data=url)
        prepared = req.prepare()
        s = requests.Session()
        response = s.send(prepared, timeout=wait_timeout, stream=should_stream)
        response.encoding = encoding
        if response.status_code != 200 and response.status_code != 202:
            message = json.loads(response.text)
            message["status_code"] = response.status_code
            message["extraction"] = {}
            raise LLMWhispererClientException(message)
        if response.status_code == 202:
            message = json.loads(response.text)
            message["status_code"] = response.status_code
            message["extraction"] = {}
            if not wait_for_completion:
                return message
            whisper_hash = message["whisper_hash"]
            start_time = time.time()
            while time.time() - start_time < wait_timeout:
                status = self.whisper_status(whisper_hash=whisper_hash)
                if status["status_code"] != 200:
                    message["status_code"] = -1
                    message["message"] = "Whisper client operation failed"
                    message["extraction"] = {}
                    return message
                if status["status"] == "accepted":
                    self.logger.debug(f'Whisper-hash:{whisper_hash} | STATUS: {status["status"]}...')
                if status["status"] == "processing":
                    self.logger.debug(f"Whisper-hash:{whisper_hash} | STATUS: processing...")

                elif status["status"] == "error":
                    self.logger.debug(f"Whisper-hash:{whisper_hash} | STATUS: failed...")
                    self.logger.error(f'Whisper-hash:{whisper_hash} | STATUS: failed with {status["message"]}')
                    message["status_code"] = -1
                    message["message"] = status["message"]
                    message["status"] = "error"
                    message["extraction"] = {}
                    return message
                elif "error" in status["status"]:
                    # for backward compatabity
                    self.logger.debug(f"Whisper-hash:{whisper_hash} | STATUS: failed...")
                    self.logger.error(f'Whisper-hash:{whisper_hash} | STATUS: failed with {status["status"]}')
                    message["status_code"] = -1
                    message["message"] = status["status"]
                    message["status"] = "error"
                    message["extraction"] = {}
                    return message
                elif status["status"] == "processed":
                    self.logger.debug(f"Whisper-hash:{whisper_hash} | STATUS: processed!")
                    resultx = self.whisper_retrieve(whisper_hash=whisper_hash)
                    if resultx["status_code"] == 200:
                        message["status_code"] = 200
                        message["message"] = "Whisper operation completed"
                        message["status"] = "processed"
                        message["extraction"] = resultx["extraction"]
                    else:
                        message["status_code"] = -1
                        message["message"] = "Whisper client operation failed"
                        message["extraction"] = {}
                    return message
                time.sleep(5)
            message["status_code"] = -1
            message["message"] = "Whisper client operation timed out"
            message["extraction"] = {}
            return message

        # Will not reach here if status code is 202
        message = json.loads(response.text)
        message["status_code"] = response.status_code
        return message

    def whisper_status(self, whisper_hash: str) -> dict:
        """Retrieves the status of the whisper operation from the LLMWhisperer
        API.

        This method sends a GET request to the '/whisper-status' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the status of the whisper operation.

        Refer https://docs.unstract.com/llm_whisperer/apis/llm_whisperer_text_extraction_status_api

        Args:
            whisper_hash (str): The hash of the whisper (returned by whisper method)

        Returns:
            dict: A dictionary containing the status of the whisper operation. The keys in the
                  dictionary include 'status_code' and the status details.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                          the error message and status code returned by the API.
        """
        self.logger.debug("whisper_status called")
        url = f"{self.base_url}/whisper-status"
        params = {"whisper_hash": whisper_hash}
        self.logger.debug("url: %s", url)
        req = requests.Request("GET", url, headers=self.headers, params=params)
        prepared = req.prepare()
        s = requests.Session()
        response = s.send(prepared, timeout=self.api_timeout)
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)
        message = json.loads(response.text)
        message["status_code"] = response.status_code
        return message

    def whisper_retrieve(self, whisper_hash: str, encoding: str = "utf-8") -> dict:
        """Retrieves the result of the whisper operation from the LLMWhisperer
        API.

        This method sends a GET request to the '/whisper-retrieve' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the result of the whisper operation.

        Refer to https://docs.unstract.com/llm_whisperer/apis/llm_whisperer_text_extraction_retrieve_api

        Args:
            whisper_hash (str): The hash of the whisper operation.
            encoding (str): The character encoding to use for processing the text. Defaults to "utf-8".

        Returns:
            dict: A dictionary containing the status code and the extracted text from the whisper operation.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                          the error message and status code returned by the API.
        """
        self.logger.debug("whisper_retrieve called")
        url = f"{self.base_url}/whisper-retrieve"
        params = {"whisper_hash": whisper_hash}
        self.logger.debug("url: %s", url)
        req = requests.Request("GET", url, headers=self.headers, params=params)
        prepared = req.prepare()
        s = requests.Session()
        response = s.send(prepared, timeout=self.api_timeout)
        response.encoding = encoding
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)

        return {
            "status_code": response.status_code,
            "extraction": json.loads(response.text),
        }

    def register_webhook(self, url: str, auth_token: str, webhook_name: str) -> dict:
        """Registers a webhook with the LLMWhisperer API.

        This method sends a POST request to the '/whisper-manage-callback' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the status of the webhook registration.

        Refer to https://docs.unstract.com/llm_whisperer/apis/

        Args:
            url (str): The URL of the webhook.
            auth_token (str): The authentication token for the webhook.
            webhook_name (str): The name of the webhook.

        Returns:
            dict: A dictionary containing the status code and the response from the API.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                            the error message and status code returned by the API.
        """

        data = {
            "url": url,
            "auth_token": auth_token,
            "webhook_name": webhook_name,
        }
        url = f"{self.base_url}/whisper-manage-callback"
        headersx = copy.deepcopy(self.headers)
        headersx["Content-Type"] = "application/json"
        req = requests.Request("POST", url, headers=headersx, json=data)
        prepared = req.prepare()
        s = requests.Session()
        response = s.send(prepared, timeout=self.api_timeout)
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)
        return json.loads(response.text)

    def get_webhook_details(self, webhook_name: str) -> dict:
        """Retrieves the details of a webhook from the LLMWhisperer API.

        This method sends a GET request to the '/whisper-manage-callback' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the details of the webhook.

        Refer to https://docs.unstract.com/llm_whisperer/apis/

        Args:
            webhook_name (str): The name of the webhook.

        Returns:
            dict: A dictionary containing the status code and the response from the API.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                            the error message and status code returned by the API.
        """

        url = f"{self.base_url}/whisper-manage-callback"
        params = {"webhook_name": webhook_name}
        req = requests.Request("GET", url, headers=self.headers, params=params)
        prepared = req.prepare()
        s = requests.Session()
        response = s.send(prepared, timeout=self.api_timeout)
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)
        return json.loads(response.text)

    def get_highlight_rect(
        self,
        line_metadata: list[int],
        target_width: int,
        target_height: int,
    ) -> tuple[int, int, int, int, int]:
        """Given the line metadata and the line number, this function returns
        the bounding box of the line in the format (page,x1,y1,x2,y2)

        Args:
            line_metadata (list[int]): The line metadata returned by the LLMWhisperer API.
            target_width (int): The width of your target image/page in UI.
            target_height (int): The height of your target image/page in UI.

        Returns:
            tuple: The bounding box of the line in the format (page,x1,y1,x2,y2)
        """

        page = line_metadata[0]
        x1 = 0
        y1 = line_metadata[1] - line_metadata[2]
        x2 = target_width
        y2 = line_metadata[1]
        original_height = line_metadata[3]

        y1 = int((float(y1) / float(original_height)) * float(target_height))
        y2 = int((float(y2) / float(original_height)) * float(target_height))

        return (page, x1, y1, x2, y2)
