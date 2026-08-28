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

import io
import json
import logging
import os
import threading
import time
import warnings
from types import ModuleType
from typing import IO, Any

import httpx

# `requests` remains a dependency for its exception classes. Callers catch
# ConnectionError and Timeout by name around these calls, and the httpx
# equivalents are not subclasses, so they are translated at the transport seam.
import requests
import tenacity
from tenacity import retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential_jitter

from unstract.llmwhisperer.sdk_llmwhisperer.api.account import usage_info
from unstract.llmwhisperer.sdk_llmwhisperer.api.webhook import (
    webhook_delete,
    webhook_get,
    webhook_post,
    webhook_put,
)
from unstract.llmwhisperer.sdk_llmwhisperer.api.whisper import detail, extract, highlights, retrieve, status
from unstract.llmwhisperer.sdk_llmwhisperer.models import WebhookConfig
from unstract.llmwhisperer.sdk_llmwhisperer.types import File

BASE_URL_V2 = "https://llmwhisperer-api.us-central.unstract.com/api/v2"

#: The spec's paths are absolute from the service root, while ``base_url``
#: already carries this prefix. Stripping it keeps the URL identical to the one
#: the client built by hand, for any ``base_url`` a caller configures.
_SPEC_PREFIX = "/api/v2"

#: Query parameters each call sends. The generated builder writes every
#: spec-declared parameter, including ones this client has never sent, and
#: sending a default is not the same as omitting it: it pins a value the service
#: would otherwise choose. A new parameter must be added here to be sent at all.
_SEND_ONLY: dict[str, frozenset[str]] = {
    "extract": frozenset(
        {
            "mode",
            "output_mode",
            "page_separator",
            "page_seperator",
            "pages_to_extract",
            "median_filter_size",
            "gaussian_blur_radius",
            "line_splitter_tolerance",
            "horizontal_stretch_factor",
            "mark_vertical_lines",
            "mark_horizontal_lines",
            "line_splitter_strategy",
            "add_line_nos",
            "include_line_confidence",
            "word_confidence_threshold",
            "lang",
            "tag",
            "file_name",
            "webhook_metadata",
            "use_webhook",
            "allow_rotated_text",
            "watermark_angle_threshold",
            "ignore_vertical_text",
            "derotate_threshold",
            "checkbox_confidence_threshold",
            "min_table_width",
            # In URL mode the URL travels in the body; it is not also a query
            # parameter, so `url` is deliberately absent here.
            "url_in_post",
        }
    ),
    "status": frozenset({"whisper_hash"}),
    "detail": frozenset({"whisper_hash"}),
    "retrieve": frozenset({"whisper_hash"}),
    "highlights": frozenset({"whisper_hash", "lines", "extract_all_lines"}),
    "usage_info": frozenset(),
    "webhook_get": frozenset({"webhook_name"}),
    "webhook_delete": frozenset({"webhook_name"}),
    "webhook_post": frozenset(),
    "webhook_put": frozenset(),
}


#: Headers the released client put on the wire without being asked, which httpx
#: spells differently. `Accept-Encoding` is load-bearing: it asks for no
#: compression, and a service that gzips its response has never been exercised
#: against this client. Overridable via `custom_headers` under any casing.
_TRANSPORT_HEADERS = {"Accept-Encoding": "identity"}


#: httpx failure -> the ``requests`` class callers catch. First match wins, so a
#: subclass has to precede the base it derives from, and ``RequestError`` last is
#: what keeps a novel httpx failure from escaping untranslated.
_TRANSLATIONS: tuple[tuple[type[Exception], type[Exception]], ...] = (
    # requests.ConnectTimeout is both a ConnectionError and a Timeout; the plain
    # Timeout httpx implies would stop matching half the callers.
    (httpx.ConnectTimeout, requests.ConnectTimeout),
    (httpx.ReadTimeout, requests.ReadTimeout),
    # Neither had a Timeout equivalent: a send that failed and a pool that could
    # not hand out a connection both surfaced as ConnectionError.
    (httpx.WriteTimeout, requests.ConnectionError),
    (httpx.PoolTimeout, requests.ConnectionError),
    (httpx.TimeoutException, requests.Timeout),
    # A URL rejected before any socket is opened. Deliberately not a
    # ConnectionError: retrying a malformed URL cannot start working.
    (httpx.UnsupportedProtocol, requests.exceptions.MissingSchema),
    (httpx.ProxyError, requests.exceptions.ProxyError),
    (httpx.ConnectError, requests.ConnectionError),
    (httpx.TooManyRedirects, requests.TooManyRedirects),
    (httpx.DecodingError, requests.exceptions.ContentDecodingError),
    (httpx.InvalidURL, requests.exceptions.InvalidURL),
    # A request that can never be sent as written -- an illegal header value is
    # the usual cause. Deliberately not a ConnectionError: it is permanent, and
    # the catch-all below would have it retried as a network blip.
    (httpx.LocalProtocolError, requests.exceptions.InvalidHeader),
    (httpx.RequestError, requests.ConnectionError),
)


#: Failures whose httpx message quotes the value that caused them. Every header
#: this client sends carries the API key, so the message is replaced with one
#: that names where to look instead of reproducing the credential.
_REPLACED_MESSAGES: dict[type[Exception], str] = {
    httpx.LocalProtocolError: (
        "The request could not be sent: a header value is not valid. Check the API key and any "
        "custom headers for stray whitespace or line breaks."
    ),
}


def _translate_transport_errors(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Re-raise httpx transport failures as their ``requests`` equivalents.

    Callers document and catch the ``requests`` classes, and the retry policy
    keys off them too, so the class chosen here decides whether a failure is
    retried.

    ``RequestError`` is the catch-all for the transport subtree, which is where
    a novel failure appears. httpx puts three families outside it: ``InvalidURL``,
    translated here because ``requests`` raised its own, and ``StreamError`` and
    ``CookieConflict``, which propagate as themselves.

    A closed transport is signalled by a bare ``RuntimeError``, which is in
    neither family; it becomes the exception every method already documents.
    """
    try:
        return fn(*args, **kwargs)
    except (httpx.RequestError, httpx.InvalidURL) as e:
        for failure, equivalent in _TRANSLATIONS:
            if isinstance(e, failure):
                raise equivalent(_REPLACED_MESSAGES.get(failure, str(e))) from e
        raise
    except RuntimeError as e:
        raise LLMWhispererClientException(str(e), 1) from e


def _wire_value(value: Any) -> Any:
    """Render a query value the way ``requests`` did: httpx lowercases
    bools."""
    return str(value) if isinstance(value, bool) else value


class LLMWhispererClientException(Exception):
    """Exception raised for errors in the LLMWhispererClient.

    Attributes:
        message (str): Explanation of the error.
        status_code (int): HTTP status code returned by the LLMWhisperer API.

    Args:
        message (str): Explanation of the error.
        status_code (int, optional): HTTP status code returned by the LLMWhisperer API. Defaults to None.
    """

    def __init__(self, value: str, status_code: int | None = None) -> None:
        """Initialize the LLMWhispererClientException.

        Args:
            value: The error message or value.
            status_code: The HTTP status code returned by the LLMWhisperer API.
        """
        self.value = value
        self.status_code = status_code

    def __str__(self) -> str:
        """Return string representation of the exception.

        Returns:
            String representation of the error value.
        """
        return repr(self.value)

    def error_message(self) -> str:
        return self.value


class _RetryableHTTPError(Exception):
    """Internal exception wrapping an HTTP response with a retryable status
    code (429, 5xx)."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


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

    api_key: str = ""
    base_url: str = ""
    api_timeout: int = 120

    #: Built on first use, and again after ``close()`` gives its sockets back.
    _transport_client: httpx.Client | None = None

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        logging_level: str = "",
        custom_headers: dict[str, str] | None = None,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 60.0,
    ) -> None:
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
            custom_headers (Optional[Dict[str, str]], optional): Custom headers to add to
                                                                every request. These will
                                                                be merged with default
                                                                headers, with custom
                                                                headers taking precedence.
                                                                Defaults to None.
            max_retries (int, optional): Maximum number of retry attempts for transient
                                         HTTP errors. Set to 0 to disable retries.
                                         Defaults to 3.
            retry_min_wait (float, optional): Minimum backoff wait in seconds. Defaults to 1.0.
            retry_max_wait (float, optional): Maximum backoff wait in seconds. Defaults to 60.0.
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
        if custom_headers:
            self.headers.update(custom_headers)

        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self._transport_lock = threading.Lock()

    @property
    def _transport(self) -> httpx.Client:
        """The HTTP client, built on first use.

        ``follow_redirects`` is on because the previous transport followed them
        by default; without it a 30x from a proxy or an http->https upgrade
        surfaces as an empty response body. Timeouts and headers are set per
        request.

        The build is locked: unguarded, two threads racing the first call each
        open a pool and the loser is dropped without ever being closed.
        """
        if self._transport_client is None:
            with self._transport_lock:
                if self._transport_client is None:
                    self._transport_client = httpx.Client(
                        follow_redirects=True,
                        timeout=httpx.Timeout(None),
                    )
        return self._transport_client

    def close(self) -> None:
        """Release the pooled connections.

        Safe to call more than once, and the client keeps working
        afterwards -- the next request opens a new pool. A call already in
        flight is not waited for: it fails with
        ``LLMWhispererClientException``.
        """
        with self._transport_lock:
            transport, self._transport_client = self._transport_client, None
        if transport is not None:
            transport.close()

    def __enter__(self) -> "LLMWhispererClientV2":
        """Enter a scope that closes the transport on the way out."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the transport, whether the scope ended well or badly."""
        self.close()

    def _build_request(
        self, module: ModuleType, send_only: frozenset[str] | None = None, **kwargs: Any
    ) -> httpx.Request:
        """Build a request from the generated builder for an operation.

        The generated code owns the URL, the parameter names and the body
        encoding. What it must not own is which parameters go out: it writes
        every spec-declared default, so the set is narrowed to what this client
        actually sets. Its ``Content-Type`` is dropped too — the transport
        derives the same one the previous client sent.

        ``send_only`` narrows further for a call whose parameter set varies with
        its arguments; it must stay within the operation's declared set.

        Headers are read per request rather than held by the transport, so
        assigning ``headers`` or rotating the key reaches the next call the way
        it did when every call passed them itself. They are merged
        case-insensitively: a plain dict merge keeps ``Unstract-Key`` and
        ``unstract-key`` as two headers, so an override would travel alongside
        the value it was meant to replace.
        """
        declared = _SEND_ONLY[module.__name__.rsplit(".", 1)[-1]]
        if send_only is None:
            send_only = declared
        elif not send_only <= declared:
            raise LLMWhispererClientException(f"Undeclared parameters: {sorted(send_only - declared)}", 1)
        built = module._get_kwargs(**kwargs)
        params = {k: _wire_value(v) for k, v in built.pop("params", {}).items() if k in send_only}
        built.pop("headers", None)
        url = self.base_url + built.pop("url").removeprefix(_SPEC_PREFIX)
        headers = httpx.Headers(_TRANSPORT_HEADERS)
        for name, value in self.headers.items():
            # Assigned one at a time: this is what drops a differently-cased
            # entry already present, which ``update`` keeps alongside it.
            headers[name] = value
        # Built through the translation seam: a malformed base_url is rejected
        # here, before any socket is opened.
        request: httpx.Request = _translate_transport_errors(
            self._transport.build_request,
            built.pop("method").upper(),
            url,
            params=params,
            headers=headers,
            **built,
        )
        return request

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """Return True if the exception represents a transient/retryable
        error."""
        if isinstance(exc, requests.ConnectionError | requests.Timeout):
            return True
        if isinstance(exc, _RetryableHTTPError):
            return bool(exc.response.status_code == 429 or exc.response.status_code >= 500)
        return False

    def _log_retry(self, retry_state: tenacity.RetryCallState) -> None:
        """Log a warning before each retry sleep."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        attempt = retry_state.attempt_number
        if isinstance(exc, _RetryableHTTPError):
            self.logger.warning("Retry attempt %d: HTTP %d", attempt, exc.response.status_code)
        elif isinstance(exc, requests.ConnectionError | requests.Timeout):
            self.logger.warning("Retry attempt %d: %s", attempt, type(exc).__name__)

    def _retry_wait(self, retry_state: tenacity.RetryCallState) -> float:
        """Compute wait time, respecting Retry-After header on 429
        responses."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, _RetryableHTTPError) and exc.response.status_code == 429:
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
        return wait_exponential_jitter(
            initial=self.retry_min_wait,
            max=self.retry_max_wait,
        )(retry_state=retry_state)

    def _send(self, request: httpx.Request, *, timeout: float, stream: bool = False) -> httpx.Response:
        """Issue one request, translating transport failures on the way out.

        Translation happens here rather than around the retry loop, so
        the retry predicate still sees the exception types it is
        configured to retry.
        """
        request.extensions = {**request.extensions, "timeout": httpx.Timeout(timeout).as_dict()}
        response: httpx.Response = _translate_transport_errors(self._transport.send, request, stream=stream)
        if stream:
            _translate_transport_errors(response.read)
        return response

    def _send_request(
        self,
        prepared: httpx.Request,
        timeout: int | None = None,
        stream: bool = False,
        deadline: float | None = None,
    ) -> httpx.Response:
        """Send an HTTP request with optional tenacity retry on transient
        errors.

        Args:
            prepared: The prepared request to send.
            timeout: Request timeout in seconds. Defaults to self.api_timeout.
            stream: Whether to stream the response. Defaults to False.
            deadline: Absolute time (time.time()) by which all attempts must finish.
                When set, each attempt's HTTP timeout is capped to the remaining time
                and retries stop once the deadline is exceeded. Defaults to None
                (no deadline).

        Returns:
            The HTTP response.

        Raises:
            requests.ConnectionError: If connection fails after all retries.
            requests.Timeout: If request times out after all retries.
        """
        if timeout is None:
            timeout = self.api_timeout
        req_timeout: int = timeout

        def _effective_timeout() -> int | float:
            if deadline is not None:
                remaining = max(0.1, deadline - time.time())
                return min(req_timeout, remaining)
            return req_timeout

        if self.max_retries == 0:
            return self._send(prepared, timeout=_effective_timeout(), stream=stream)

        def _attempt() -> httpx.Response:
            response = self._send(prepared, timeout=_effective_timeout(), stream=stream)
            if response.status_code == 429 or response.status_code >= 500:
                raise _RetryableHTTPError(response)
            return response

        stop_condition: tenacity.stop.stop_base = stop_after_attempt(self.max_retries + 1)
        if deadline is not None:
            max_duration = max(0, deadline - time.time())
            stop_condition = stop_condition | stop_after_delay(max_duration)

        retrying = tenacity.Retrying(
            retry=retry_if_exception(self._is_retryable),
            stop=stop_condition,
            wait=self._retry_wait,
            before_sleep=self._log_retry,
            reraise=True,
        )
        try:
            return retrying(_attempt)
        except _RetryableHTTPError as e:
            return e.response

    def get_usage_info(self) -> Any:
        """Retrieves the usage information of the LLMWhisperer API.

        This method sends a GET request to the '/get-usage-info' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the usage information.
        Refer to https://docs.unstract.com/llm_whisperer/apis/llm_whisperer_usage_api

        Returns:
            Dict[Any, Any]: A dictionary containing the usage information.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                          the error message and status code returned by the API.
        """
        self.logger.debug("get_usage_info called")
        prepared = self._build_request(usage_info)
        self.logger.debug("url: %s", prepared.url)
        response = self._send_request(prepared)
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)
        return json.loads(response.text)

    def get_highlight_data(self, whisper_hash: str, lines: str, extract_all_lines: bool = False) -> Any:
        """Retrieves the highlight information of the LLMWhisperer API.

        This method sends a GET request to the '/highlights' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the usage information.
        Refer to https://docs.unstract.com/llm_whisperer/apis/llm_whisperer_usage_api

        Args:
            whisper_hash (str): The hash of the whisper operation.
            lines (str): Define which lines metadata to retrieve.
                You can specify which lines metadata to retrieve with this parameter.
                Example 1-5,7,21- will retrieve lines metadata 1,2,3,4,5,7,21,22,23,24...
                till the last line meta data.

        Returns:
            Dict[Any, Any]: A dictionary containing the highlight information.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                          the error message and status code returned by the API.
        """
        self.logger.debug("highlight called")
        prepared = self._build_request(
            highlights,
            whisper_hash=whisper_hash,
            lines=lines,
            extract_all_lines=extract_all_lines,
        )
        self.logger.debug("url: %s", prepared.url)
        response = self._send_request(prepared)
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)
        return json.loads(response.text)

    def whisper_detail(self, whisper_hash: str) -> Any:
        """Retrieves the details of a text extraction process.

        This method sends a GET request to the '/whisper-detail' endpoint of the LLMWhisperer API.
        The response is a JSON object containing metadata about the extraction job.
        Refer to https://docs.unstract.com/llmwhisperer/llm_whisperer/apis/llm_whisperer_text_extraction_detail_api

        Args:
            whisper_hash (str): The identifier returned when starting the extraction process.

        Returns:
            Dict[Any, Any]: A dictionary containing the extraction details including
                completed_at, mode, processed_pages, processing_started_at,
                processing_time_in_seconds, requested_pages, tag, total_pages,
                upload_file_size_in_kb, and whisper_hash.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                          the error message and status code returned by the API.
        """
        self.logger.debug("whisper_detail called")
        prepared = self._build_request(detail, whisper_hash=whisper_hash)
        self.logger.debug("url: %s", prepared.url)
        self.logger.debug("whisper_hash: %s", whisper_hash)

        response = self._send_request(prepared)
        if response.status_code != 200:
            if not (response.text or "").strip():
                raise LLMWhispererClientException("API error: empty response body", response.status_code)
            try:
                err = json.loads(response.text)
            except json.JSONDecodeError as e:
                response_preview = response.text[:500] + "..." if len(response.text) > 500 else response.text
                raise LLMWhispererClientException(
                    f"API error: non-JSON response - {response_preview}", response.status_code
                ) from e
            raise LLMWhispererClientException(err, response.status_code)
        return json.loads(response.text)

    def _resolve_deprecated_param(
        self,
        name: str,
        value: str | None,
        deprecated_name: str,
        deprecated_value: str | None,
        default: str,
        *,
        forward: bool,
    ) -> str:
        """Resolves a renamed parameter, warning when the old name is used.

        Args:
            name: The supported parameter name.
            value: Value passed under the supported name, None when unset.
            deprecated_name: The deprecated parameter name.
            deprecated_value: Value passed under the deprecated name, None when unset.
            default: Value to use when neither name is passed.
            forward: Whether the deprecated value is honoured. False for parameters the
                service never received, where applying the value now would silently
                change extraction output.

        Returns:
            The resolved value.

        Raises:
            LLMWhispererClientException: If both names are passed.
        """
        if deprecated_value is None:
            return default if value is None else value
        if value is not None:
            raise LLMWhispererClientException(
                f"Cannot pass both '{deprecated_name}' and '{name}', use '{name}' only",
                1,
            )
        message = f"'{deprecated_name}' is deprecated and will be removed in a future release, use '{name}' instead"
        if not forward:
            message += f". The value passed is ignored: '{deprecated_name}' never reached the service"
        self.logger.warning(message)
        warnings.warn(message, DeprecationWarning, stacklevel=3)
        return deprecated_value if forward else default

    def whisper(  # noqa: C901
        self,
        file_path: str = "",
        stream: IO[bytes] | None = None,
        url: str = "",
        mode: str = "form",
        output_mode: str = "layout_preserving",
        page_seperator: str | None = None,
        pages_to_extract: str = "",
        median_filter_size: int = 0,
        gaussian_blur_radius: int = 0,
        line_splitter_tolerance: float = 0.4,
        horizontal_stretch_factor: float = 1.0,
        mark_vertical_lines: bool = False,
        mark_horizontal_lines: bool = False,
        line_spitter_strategy: str | None = None,
        add_line_nos: bool = False,
        include_line_confidence: bool = False,
        word_confidence_threshold: float = 0.3,
        lang: str = "eng",
        tag: str = "default",
        filename: str | None = None,
        webhook_metadata: str = "",
        use_webhook: str = "",
        wait_for_completion: bool = False,
        wait_timeout: int = 180,
        encoding: str = "utf-8",
        page_separator: str | None = None,
        line_splitter_strategy: str | None = None,
        file_name: str | None = None,
        *,
        allow_rotated_text: bool | None = None,
        watermark_angle_threshold: float | None = None,
        ignore_vertical_text: bool | None = None,
        derotate_threshold: float | None = None,
        checkbox_confidence_threshold: float | None = None,
        min_table_width: float | None = None,
    ) -> Any:
        """Sends a request to the LLMWhisperer API to process a document.
        Refer to https://docs.unstract.com/llm_whisperer/apis/llm_whisperer_text_extraction_api.

        Args:
            file_path (str, optional): The path to the file to be processed. Defaults to "".
            stream (IO[bytes], optional): A stream of bytes to be processed. Defaults to None.
            url (str, optional): The URL of the file to be processed. Defaults to "".
            mode (str, optional): The processing mode. Can be "high_quality", "form", "low_cost", "native_text"
                or "table". Defaults to "high_quality".
            output_mode (str, optional): The output mode. Can be "layout_preserving" or "text".
                Defaults to "layout_preserving".
            page_seperator (str, optional): Deprecated misspelling of page_separator, still
                honoured. Defaults to None.
            pages_to_extract (str, optional): The pages to extract. Defaults to "".
            median_filter_size (int, optional): The size of the median filter. Defaults to 0.
            gaussian_blur_radius (int, optional): The radius of the Gaussian blur. Defaults to 0.
            line_splitter_tolerance (float, optional): The line splitter tolerance. Defaults to 0.4.
                This client pins its own default below the service's, and has always sent it, so
                the two are expected to differ.
            horizontal_stretch_factor (float, optional): The horizontal stretch factor. Defaults to 1.0.
            mark_vertical_lines (bool, optional): Whether to mark vertical lines. Defaults to False.
            mark_horizontal_lines (bool, optional): Whether to mark horizontal lines. Defaults to False.
            line_spitter_strategy (str, optional): Deprecated misspelling of
                line_splitter_strategy. The value is ignored, since it was never sent under a
                name the service reads. Defaults to None.
            add_line_nos (bool, optional): Adds line numbers to the extracted text and saves line metadata,
              which can be queried later using the highlights API.
            include_line_confidence (bool, optional): Adds line confidence to the line metadata returned by
              the highlights API. Requires add_line_nos to be enabled. Defaults to False.
            word_confidence_threshold (float, optional): The minimum OCR confidence score a word must have to be
              included in the extracted text. Accepts a value in the range [0.0, 1.0], where higher values are
              stricter. Any word whose confidence value falls below the configured threshold is ignored and
              excluded from the final output. This parameter works only with "form", "high_quality" and "table"
              modes. Defaults to 0.3.
            lang (str, optional): The language of the document. Defaults to "eng".
            tag (str, optional): The tag for the document. Defaults to "default".
            filename (str, optional): Deprecated name for file_name, still honoured.
                Defaults to None.
            webhook_metadata (str, optional): The webhook metadata. This data will be passed to the webhook if
                webhooks are used Defaults to "".
            use_webhook (str, optional): Webhook name to call. Defaults to "". If not provided, then
                no webhook will be called.
            wait_for_completion (bool, optional): Whether to wait for the whisper operation to complete.
                Defaults to False.
            wait_timeout (int, optional): The number of seconds to wait for the whisper operation to complete.
                Defaults to 180.
            encoding (str): The character encoding to use for processing the text. Defaults to "utf-8".
            page_separator (str, optional): The page separator. Defaults to "<<<".
            line_splitter_strategy (str, optional): The line splitter strategy.
                Defaults to "left-priority".
            file_name (str, optional): The name of the file to store in reports. Defaults to "".
            allow_rotated_text (bool, optional): Whether to keep words whose own orientation is
              rotated. With this off, a word angled further than watermark_angle_threshold is
              treated as a watermark and excluded. Defaults to True.
            watermark_angle_threshold (float, optional): The angle in degrees beyond which a
              rotated word counts as a watermark. Only applies when allow_rotated_text is off.
              Defaults to 25.0.
            ignore_vertical_text (bool, optional): Whether to drop vertically oriented text
              instead of extracting it. Defaults to False.
            derotate_threshold (float, optional): The page rotation in degrees beyond which the
              page is straightened and re-read. Defaults to 10.0.
            checkbox_confidence_threshold (float, optional): The minimum confidence a detected
              checkbox mark must have to be reported as marked. Accepts a value in the range
              [0.0, 1.0]. Defaults to 0.3.
            min_table_width (float, optional): The minimum width a table must span, as a
              fraction of the page width, to be extracted as a table. Defaults to 0.

        Returns:
            Dict[Any, Any]: The response from the API as a dictionary.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                          the error message and status code returned by the API.
                                          Also raised when a parameter is passed under both its
                                          deprecated and its supported name.
            LookupError: If ``encoding`` does not name a codec Python knows. The extraction has
                         already run and been counted by the time this is raised.
        """
        self.logger.debug("whisper called")
        page_separator = self._resolve_deprecated_param(
            "page_separator", page_separator, "page_seperator", page_seperator, "<<<", forward=True
        )
        line_splitter_strategy = self._resolve_deprecated_param(
            "line_splitter_strategy",
            line_splitter_strategy,
            "line_spitter_strategy",
            line_spitter_strategy,
            "left-priority",
            forward=False,
        )
        file_name = self._resolve_deprecated_param("file_name", file_name, "filename", filename, "", forward=True)
        api_url = f"{self.base_url}/whisper"
        params = {
            "mode": mode,
            "output_mode": output_mode,
            # Both spellings are sent: services older than v2.64.2 read only the misspelled one
            "page_separator": page_separator,
            "page_seperator": page_separator,
            "pages_to_extract": pages_to_extract,
            "median_filter_size": median_filter_size,
            "gaussian_blur_radius": gaussian_blur_radius,
            "line_splitter_tolerance": line_splitter_tolerance,
            "horizontal_stretch_factor": horizontal_stretch_factor,
            "mark_vertical_lines": mark_vertical_lines,
            "mark_horizontal_lines": mark_horizontal_lines,
            "line_splitter_strategy": line_splitter_strategy,
            "add_line_nos": add_line_nos,
            "include_line_confidence": include_line_confidence,
            "word_confidence_threshold": word_confidence_threshold,
            "lang": lang,
            "tag": tag,
            "file_name": file_name,
            "webhook_metadata": webhook_metadata,
            "use_webhook": use_webhook,
        }
        # Only what the caller asked for. These have no default here on purpose:
        # sending one pins a value the service would otherwise choose, and the
        # two diverge the moment the service's own default moves. ``None`` means
        # unset rather than null -- a query string carries no null, so it would
        # otherwise go out as the literal string "None".
        params.update(
            {
                name: value
                for name, value in (
                    ("allow_rotated_text", allow_rotated_text),
                    ("watermark_angle_threshold", watermark_angle_threshold),
                    ("ignore_vertical_text", ignore_vertical_text),
                    ("derotate_threshold", derotate_threshold),
                    ("checkbox_confidence_threshold", checkbox_confidence_threshold),
                    ("min_table_width", min_table_width),
                )
                if value is not None
            }
        )

        self.logger.debug("api_url: %s", api_url)
        self.logger.debug("params: %s", params)

        if use_webhook != "" and wait_for_completion:
            raise LLMWhispererClientException("Cannot wait for completion when using webhook", 1)

        if url == "" and file_path == "" and stream is None:
            raise LLMWhispererClientException(
                "Either url, stream or file_path must be provided",
                1,
            )

        should_stream = False
        if url == "":
            if stream is not None:
                should_stream = True
                data = b"".join(stream)
            else:
                with open(file_path, "rb") as f:
                    data = f.read()
        else:
            # The URL travels in the body, not on the query string; url_in_post
            # is what tells the service to read it from there.
            params["url_in_post"] = True
            data = url.encode()
        # The wire carries exactly the parameters assembled above — url_in_post
        # only exists in URL mode, and the generated default would otherwise
        # send it on every upload.
        prepared = self._build_request(extract, frozenset(params), body=File(payload=io.BytesIO(data)), **params)
        start_time = time.time()
        deadline = start_time + wait_timeout
        post_timeout = min(self.api_timeout, wait_timeout)
        response = self._send_request(prepared, timeout=post_timeout, stream=should_stream, deadline=deadline)
        response.encoding = encoding
        if response.status_code not in (200, 202):
            try:
                message = json.loads(response.text)
                if not isinstance(message, dict):
                    message = {"message": str(message)}
            except (json.JSONDecodeError, ValueError):
                message = {"message": response.text}
            message["status_code"] = response.status_code
            message["extraction"] = {}
            raise LLMWhispererClientException(message)
        if response.status_code == 202:
            try:
                message = json.loads(response.text)
                if not isinstance(message, dict):
                    message = {"message": str(message)}
            except (json.JSONDecodeError, ValueError):
                message = {"message": response.text}
            message["status_code"] = response.status_code
            message["extraction"] = {}
            if not wait_for_completion:
                return message
            whisper_hash = message["whisper_hash"]
            while time.time() - start_time < wait_timeout:
                status = self.whisper_status(whisper_hash=whisper_hash)
                if status["status_code"] != 200:
                    message["status_code"] = -1
                    message["message"] = "Whisper client operation failed"
                    message["extraction"] = {}
                    return message
                if status["status"] == "accepted":
                    self.logger.debug(f"Whisper-hash:{whisper_hash} | STATUS: {status['status']}...")
                if status["status"] == "processing":
                    self.logger.debug(f"Whisper-hash:{whisper_hash} | STATUS: processing...")

                elif status["status"] == "error":
                    self.logger.debug(f"Whisper-hash:{whisper_hash} | STATUS: failed...")
                    self.logger.error(f"Whisper-hash:{whisper_hash} | STATUS: failed with {status['message']}")
                    message["status_code"] = -1
                    message["message"] = status["message"]
                    message["status"] = "error"
                    message["extraction"] = {}
                    return message
                elif "error" in status["status"]:
                    # for backward compatabity
                    self.logger.debug(f"Whisper-hash:{whisper_hash} | STATUS: failed...")
                    self.logger.error(f"Whisper-hash:{whisper_hash} | STATUS: failed with {status['status']}")
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

    def whisper_status(self, whisper_hash: str) -> Any:
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
        prepared = self._build_request(status, whisper_hash=whisper_hash)
        self.logger.debug("url: %s", prepared.url)
        response = self._send_request(prepared)
        if response.status_code != 200:
            if not (response.text or "").strip():
                self.logger.error(f"API error - empty response body, status code: {response.status_code}")
                raise LLMWhispererClientException("API error: empty response body", response.status_code)
            try:
                err = json.loads(response.text)
            except json.JSONDecodeError as e:
                # Truncate response text if too long to avoid log pollution
                response_preview = response.text[:500] + "..." if len(response.text) > 500 else response.text
                self.logger.error(f"API error - JSON decode failed: {e}; Response preview: {response_preview!r}")
                raise LLMWhispererClientException(
                    f"API error: non-JSON response - {response_preview}", response.status_code
                ) from e
            raise LLMWhispererClientException(err, response.status_code)
        message = json.loads(response.text)
        message["status_code"] = response.status_code
        return message

    def whisper_retrieve(self, whisper_hash: str, encoding: str = "utf-8") -> Any:
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
            LookupError: If ``encoding`` does not name a codec Python knows. The extraction has
                         already run and been counted by the time this is raised.
        """
        self.logger.debug("whisper_retrieve called")
        prepared = self._build_request(retrieve, whisper_hash=whisper_hash)
        self.logger.debug("url: %s", prepared.url)
        response = self._send_request(prepared)
        response.encoding = encoding
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)

        return {
            "status_code": response.status_code,
            "extraction": json.loads(response.text),
        }

    def register_webhook(self, url: str, auth_token: str, webhook_name: str) -> Any:
        """Registers a webhook with the LLMWhisperer API.

        This method sends a POST request to the '/whisper-manage-callback' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the status of the webhook registration.

        Refer to https://docs.unstract.com/llm_whisperer/apis/

        Args:
            url (str): The URL of the webhook.
            auth_token (str): The authentication token for the webhook.
            webhook_name (str): The name of the webhook.

        Returns:
            Any: A dictionary containing the status code and the response from the API.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                            the error message and status code returned by the API.
        """
        body = WebhookConfig(url=url, auth_token=auth_token, webhook_name=webhook_name)
        prepared = self._build_request(webhook_post, body=body)
        response = self._send_request(prepared)
        if response.status_code != 201:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)
        return json.loads(response.text)

    def update_webhook_details(self, webhook_name: str, url: str, auth_token: str) -> Any:
        """Updates the details of a webhook from the LLMWhisperer API.

        This method sends a PUT request to the '/whisper-manage-callback' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the status of the webhook update.

        Refer to https://docs.unstract.com/llm_whisperer/apis/

        Args:
            webhook_name (str): The name of the webhook.
            url (str): The URL of the webhook.
            auth_token (str): The authentication token for the webhook.

        Returns:
            dict: A dictionary containing the status code and the response from the API.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                            the error message and status code returned by the API.
        """
        body = WebhookConfig(url=url, auth_token=auth_token, webhook_name=webhook_name)
        prepared = self._build_request(webhook_put, body=body)
        response = self._send_request(prepared)
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)
        return json.loads(response.text)

    def get_webhook_details(self, webhook_name: str) -> Any:
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
        prepared = self._build_request(webhook_get, webhook_name=webhook_name)
        response = self._send_request(prepared)
        if response.status_code != 200:
            err = json.loads(response.text)
            err["status_code"] = response.status_code
            raise LLMWhispererClientException(err)
        return json.loads(response.text)

    def delete_webhook(self, webhook_name: str) -> Any:
        """Deletes a webhook from the LLMWhisperer API.

        This method sends a DELETE request to the '/whisper-manage-callback' endpoint of the LLMWhisperer API.
        The response is a JSON object containing the status of the webhook deletion.

        Refer to https://docs.unstract.com/llm_whisperer/apis/

        Args:
            webhook_name (str): The name of the webhook.

        Returns:
            dict: A dictionary containing the status code and the response from the API.

        Raises:
            LLMWhispererClientException: If the API request fails, it raises an exception with
                                            the error message and status code returned by the API.
        """
        prepared = self._build_request(webhook_delete, webhook_name=webhook_name)
        response = self._send_request(prepared)
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
        the bounding box of the line in the format (page,x1,y1,x2,y2).

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
