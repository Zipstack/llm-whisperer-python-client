"""This module provides a Python client for interacting with the LLMWhisperer
API.

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

import json
import logging
import os
from typing import IO

import requests

from unstract.llmwhisperer.utils import LLMWhispererUtils

BASE_URL = "https://llmwhisperer-api.unstract.com/v1"


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


class LLMWhispererClient:
    """A client for interacting with the LLMWhisperer API.

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
        api_timeout: int = 120,
        logging_level: str = "",
    ):
        """Initializes the LLMWhispererClient with the given parameters.

        Args:
            base_url (str, optional): The base URL for the LLMWhisperer API. Defaults to "".
                                      If the base_url is not provided, the client will use
                                      the value of the LLMWHISPERER_BASE_URL environment
                                      variable,or the default value.
            api_key (str, optional): The API key for the LLMWhisperer API. Defaults to "".
                                     If the api_key is not provided, the client will use the
                                     value of the LLMWHISPERER_API_KEY environment variable.
            api_timeout (int, optional): The timeout for API requests. Defaults to 120s.
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
            self.base_url = os.getenv("LLMWHISPERER_BASE_URL", BASE_URL)
        else:
            self.base_url = base_url
        self.logger.debug("base_url set to %s", self.base_url)

        if api_key == "":
            self.api_key = os.getenv("LLMWHISPERER_API_KEY", "")
        else:
            self.api_key = api_key
        self.logger.debug("api_key set to %s", LLMWhispererUtils.redact_key(self.api_key))

        self.api_timeout = api_timeout

        self.headers = {"unstract-key": self.api_key}

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
        processing_mode: str = "ocr",
        output_mode: str = "line-printer",
        page_seperator: str = "<<<",
        force_text_processing: bool = False,
        pages_to_extract: str = "",
        timeout: int = 200,
        store_metadata_for_highlighting: bool = False,
        median_filter_size: int = 0,
        gaussian_blur_radius: int = 0,
        ocr_provider: str = "advanced",
        line_splitter_tolerance: float = 0.4,
        horizontal_stretch_factor: float = 1.0,
        encoding: str = "utf-8",
    ) -> dict:
        """
        Sends a request to the LLMWhisperer API to process a document.
        Refer to https://docs.unstract.com/llm_whisperer/apis/llm_whisperer_text_extraction_api

        Args:
            file_path (str, optional): The path to the file to be processed. Defaults to "".
            stream (IO[bytes], optional): A stream of bytes to be processed. Defaults to None.
            url (str, optional): The URL of the file to be processed. Defaults to "".
            processing_mode (str, optional): The processing mode. Can be "ocr" or "text". Defaults to "ocr".
            output_mode (str, optional): The output mode. Can be "line-printer" or "text". Defaults to "line-printer".
            page_seperator (str, optional): The page separator. Defaults to "<<<".
            force_text_processing (bool, optional): Whether to force text processing. Defaults to False.
            pages_to_extract (str, optional): The pages to extract. Defaults to "".
            timeout (int, optional): The timeout for the request in seconds. Defaults to 200.
            store_metadata_for_highlighting (bool, optional): Whether to store metadata for highlighting. Def False.
            median_filter_size (int, optional): The size of the median filter. Defaults to 0.
            gaussian_blur_radius (int, optional): The radius of the Gaussian blur. Defaults to 0.
            ocr_provider (str, optional): The OCR provider. Can be "advanced" or "basic". Defaults to "advanced".
            line_splitter_tolerance (float, optional): The line splitter tolerance. Defaults to 0.4.
            horizontal_stretch_factor (float, optional): The horizontal stretch factor. Defaults to 1.0.
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
            "url": url,
            "processing_mode": processing_mode,
            "output_mode": output_mode,
            "page_seperator": page_seperator,
            "force_text_processing": force_text_processing,
            "pages_to_extract": pages_to_extract,
            "timeout": timeout,
            "store_metadata_for_highlighting": store_metadata_for_highlighting,
            "median_filter_size": median_filter_size,
            "gaussian_blur_radius": gaussian_blur_radius,
            "ocr_provider": ocr_provider,
            "line_splitter_tolerance": line_splitter_tolerance,
            "horizontal_stretch_factor": horizontal_stretch_factor,
        }

        self.logger.debug("api_url: %s", api_url)
        self.logger.debug("params: %s", params)

        if url == "" and file_path == "" and stream is None:
            raise LLMWhispererClientException(
                {
                    "status_code": -1,
                    "message": "Either url, stream or file_path must be provided",
                }
            )

        if timeout < 0 or timeout > 200:
            raise LLMWhispererClientException(
                {
                    "status_code": -1,
                    "message": "timeout must be between 0 and 200",
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
            req = requests.Request("POST", api_url, params=params, headers=self.headers)
        prepared = req.prepare()
        s = requests.Session()
        response = s.send(prepared, timeout=timeout, stream=should_stream)
        response.encoding = encoding
        if response.status_code != 200 and response.status_code != 202:
            message = json.loads(response.text)
            message["status_code"] = response.status_code
            raise LLMWhispererClientException(message)
        if response.status_code == 202:
            message = json.loads(response.text)
            message["status_code"] = response.status_code
            return message
        return {
            "status_code": response.status_code,
            "extracted_text": response.text,
            "whisper_hash": response.headers["whisper-hash"],
        }

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
        params = {"whisper-hash": whisper_hash}
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
        params = {"whisper-hash": whisper_hash}
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
            "extracted_text": response.text,
        }

    def highlight_data(self, whisper_hash: str, search_text: str) -> dict:
        """
        Highlights the specified text in the result of a whisper operation.
        Note: The whisper operation must have been performed with the
        store_metadata_for_highlighting parameter set to True.

        Refer to https://docs.unstract.com/llm_whisperer/apis/llm_whisperer_text_extraction_highlight_api

        This method sends a POST request to the '/highlight-data' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the highlighted text information.

        Args:
            whisper_hash (str): The hash of the whisper operation.
            search_text (str): The text to be highlighted.

        Returns:
            dict: A dictionary containing the status code and the highlighted text.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                          the error message and status code returned by the API.
        """
        self.logger.debug("highlight_data called")
        url = f"{self.base_url}/highlight-data"
        params = {"whisper-hash": whisper_hash}
        self.logger.debug("url: %s", url)
        req = requests.Request(
            "POST",
            url,
            headers=self.headers,
            params=params,
            data=search_text,
        )
        prepared = req.prepare()
        s = requests.Session()
        response = s.send(prepared, timeout=self.api_timeout)
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)
        result = json.loads(response.text)
        result["status_code"] = response.status_code
        return result
