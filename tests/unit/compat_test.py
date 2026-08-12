"""Parity tests against the published client.

The transport underneath ``LLMWhispererClientV2`` changed; its published
behaviour must not. These tests pin the seams where that could silently break:
the constructor and method signatures, what goes out on the wire, which
exceptions come back out, and what each method returns — the last two by running
the published client side by side over the same responses.

The baseline is vendored under ``tests/baseline`` rather than imported from an
installed distribution, so the comparison is against a fixed published client
instead of whatever the working tree currently says.
"""

import ast
import importlib.util
import inspect
import io
import json
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import requests
from unstract.llmwhisperer.client_v2 import (
    _SEND_ONLY,
    LLMWhispererClientException,
    LLMWhispererClientV2,
)

BASELINE_REF = "0e9fda3"
BASELINE_PATH = Path(__file__).parents[1] / "baseline" / "client_v2_pr34.py"
SPEC_PATH = Path(__file__).parents[2] / "specs" / "llmwhisperer.json"

Call = Callable[[Any, str], Any]

BASE_URL = "https://x.test/api/v2"

# Operations the spec declares that this client does not expose. The published
# client exposes none of them either, so the facade is at parity — the spec
# surface is simply wider than the hand-written one. Listing them here keeps the
# coverage check honest instead of passing on whatever happens to be wrapped.
UNWRAPPED_OPERATIONS = frozenset(
    {
        "usage",
        "test_connection",
        "pdf_to_images",
        "pdf_to_images_status",
        "pdf_to_images_retrieve",
        "document_insights",
        "document_insights_retrieve",
        "convert_to_pdf",
        "convert_xlsb_to_xlsx",
    }
)


def _load_baseline() -> ModuleType:
    """Import the vendored published client under its own module name."""
    spec = importlib.util.spec_from_file_location("baseline_client_v2", BASELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = _load_baseline()


def _client(**kwargs: Any) -> LLMWhispererClientV2:
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("logging_level", "ERROR")
    kwargs.setdefault("max_retries", 0)
    return LLMWhispererClientV2(**kwargs)


def _baseline_client(**kwargs: Any) -> Any:
    kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("logging_level", "ERROR")
    kwargs.setdefault("max_retries", 0)
    return baseline.LLMWhispererClientV2(**kwargs)


def _mock_response(status_code: int = 200, text: str = '{"ok": true}') -> MagicMock:
    """A response both transports can be fed, so only the client differs."""
    response = MagicMock()
    response.status_code = status_code
    response.text = text
    response.headers = {}
    return response


@pytest.fixture
def sample_file(tmp_path: Path) -> str:
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"PDFBYTES")
    return str(path)


# --------------------------------------------------------------------------
# What goes out on the wire, compared against the published client
# --------------------------------------------------------------------------

# Every call the client can make, as (name, callable). Each runs against both
# clients with the same mocked response, and the two requests must agree.
CALLS = {
    "usage_info": (lambda c, f: c.get_usage_info(), 200),
    "whisper_status": (lambda c, f: c.whisper_status("hash-1"), 200),
    "whisper_retrieve": (lambda c, f: c.whisper_retrieve("hash-1"), 200),
    "whisper_detail": (lambda c, f: c.whisper_detail("hash-1"), 200),
    "highlights": (lambda c, f: c.get_highlight_data("hash-1", "1-5"), 200),
    "highlights_all_lines": (lambda c, f: c.get_highlight_data("hash-1", "-1", True), 200),
    "webhook_get": (lambda c, f: c.get_webhook_details("wh"), 200),
    "webhook_delete": (lambda c, f: c.delete_webhook("wh"), 200),
    "webhook_put": (lambda c, f: c.update_webhook_details("wh", "http://cb", "tok"), 200),
    "webhook_post": (lambda c, f: c.register_webhook("http://cb", "tok", "wh"), 201),
    "whisper_file": (lambda c, f: c.whisper(file_path=f, wait_for_completion=False), 200),
    "whisper_stream": (lambda c, f: c.whisper(stream=io.BytesIO(b"STREAM"), wait_for_completion=False), 200),
    "whisper_url": (lambda c, f: c.whisper(url="https://e.test/a.pdf", wait_for_completion=False), 200),
    "whisper_every_param": (
        lambda c, f: c.whisper(
            file_path=f,
            wait_for_completion=False,
            mode="high_quality",
            output_mode="text",
            page_separator="---",
            pages_to_extract="1-2",
            median_filter_size=2,
            gaussian_blur_radius=1,
            line_splitter_tolerance=0.9,
            horizontal_stretch_factor=1.5,
            mark_vertical_lines=True,
            mark_horizontal_lines=True,
            line_splitter_strategy="mid-priority",
            add_line_nos=True,
            include_line_confidence=True,
            word_confidence_threshold=0.75,
            lang="deu",
            tag="t",
            file_name="invoice.pdf",
            webhook_metadata="meta",
            use_webhook="wh",
        ),
        200,
    ),
}

_WHISPER_OK = '{"status_code": 200, "extraction": {"result_text": "ok"}}'


def _sent(call: Call, status_code: int, sample_file: str) -> tuple[httpx.Request, Any]:
    """Run the call against both clients and return the two requests."""
    text = _WHISPER_OK
    with patch.object(LLMWhispererClientV2, "_send", return_value=_mock_response(status_code, text)) as ours:
        call(_client(), sample_file)
    with patch("requests.Session.send", return_value=_mock_response(status_code, text)) as theirs:
        call(_baseline_client(), sample_file)
    return ours.call_args[0][0], theirs.call_args[0][0]


@pytest.mark.parametrize("name", list(CALLS))
def test_request_matches_the_published_client(name: str, sample_file: str) -> None:
    """Method, path, query and body must be identical.

    Query-parameter *order* is not compared: httpx sorts what the
    previous transport emitted in insertion order, and order carries no
    meaning.
    """
    call, status_code = CALLS[name]
    ours, theirs = _sent(call, status_code, sample_file)

    ours_url, theirs_url = urlparse(str(ours.url)), urlparse(theirs.url)
    assert ours.method.upper() == theirs.method.upper()
    assert (ours_url.scheme, ours_url.netloc, ours_url.path) == (
        theirs_url.scheme,
        theirs_url.netloc,
        theirs_url.path,
    )
    assert parse_qs(ours_url.query) == parse_qs(theirs_url.query)

    their_body = theirs.body.encode() if isinstance(theirs.body, str) else theirs.body
    if ours.headers.get("content-type") == "application/json":
        # Both send the same object; the encoders differ on separators and key
        # order, neither of which a JSON reader can observe.
        assert json.loads(ours.read()) == json.loads(their_body)
    else:
        assert ours.read() == (their_body or b"")


def test_the_auth_header_is_unchanged(sample_file: str) -> None:
    ours, theirs = _sent(*CALLS["usage_info"], sample_file)
    assert ours.headers["unstract-key"] == theirs.headers["unstract-key"] == "test-key"


def test_custom_headers_still_reach_the_request() -> None:
    client = _client(custom_headers={"x-trace": "abc", "unstract-key": "override"})
    with patch.object(LLMWhispererClientV2, "_send", return_value=_mock_response()) as send:
        client.get_usage_info()
    request = send.call_args[0][0]
    assert request.headers["x-trace"] == "abc"
    assert request.headers["unstract-key"] == "override"


def test_url_mode_does_not_put_the_url_on_the_query_string() -> None:
    """The URL travels in the body.

    The spec declares a `url` parameter, and
    sending it as well would be a parameter the published client never sent.
    """
    client = _client()
    with patch.object(LLMWhispererClientV2, "_send", return_value=_mock_response(200, _WHISPER_OK)) as send:
        client.whisper(url="https://e.test/a.pdf", wait_for_completion=False)
    request = send.call_args[0][0]
    assert "url" not in parse_qs(urlparse(str(request.url)).query)
    assert request.read() == b"https://e.test/a.pdf"


def test_upload_mode_does_not_send_url_in_post(sample_file: str) -> None:
    """`url_in_post` only exists in URL mode; its generated default would
    otherwise ride along on every upload."""
    client = _client()
    with patch.object(LLMWhispererClientV2, "_send", return_value=_mock_response(200, _WHISPER_OK)) as send:
        client.whisper(file_path=sample_file, wait_for_completion=False)
    assert "url_in_post" not in parse_qs(urlparse(str(send.call_args[0][0].url)).query)


def test_booleans_are_sent_the_way_the_previous_transport_sent_them() -> None:
    """Httpx renders bools lowercase; the previous transport used
    `str(bool)`."""
    client = _client()
    with patch.object(LLMWhispererClientV2, "_send", return_value=_mock_response(200, _WHISPER_OK)) as send:
        client.whisper(url="https://e.test/a.pdf", wait_for_completion=False, add_line_nos=True)
    query = parse_qs(urlparse(str(send.call_args[0][0].url)).query)
    assert query["add_line_nos"] == ["True"]
    assert query["url_in_post"] == ["True"]
    assert query["mark_vertical_lines"] == ["False"]


def test_send_only_covers_every_parameter_whisper_builds(sample_file: str) -> None:
    """The guard is only as good as its declared set; this pins the two
    together so a new whisper parameter cannot reach the wire undeclared."""
    client = _client()
    with patch.object(LLMWhispererClientV2, "_send", return_value=_mock_response(200, _WHISPER_OK)) as send:
        client.whisper(url="https://e.test/a.pdf", wait_for_completion=False)
    query = set(parse_qs(urlparse(str(send.call_args[0][0].url)).query))
    assert query <= _SEND_ONLY["extract"]


_ADDED_PARAMS = {
    "allow_rotated_text": (False, "False"),
    "watermark_angle_threshold": (0.0, "0.0"),
    "ignore_vertical_text": (True, "True"),
    "derotate_threshold": (0, "0"),
    "checkbox_confidence_threshold": (0.0, "0.0"),
    "min_table_width": (0.5, "0.5"),
}


def _whisper_query(client: Any, **kwargs: Any) -> dict[str, list[str]]:
    with patch.object(LLMWhispererClientV2, "_send", return_value=_mock_response(200, _WHISPER_OK)) as send:
        client.whisper(url="https://e.test/a.pdf", wait_for_completion=False, **kwargs)
    return parse_qs(urlparse(str(send.call_args[0][0].url)).query)


def test_an_unrequested_parameter_is_not_sent() -> None:
    """Sending one pins a value the service would otherwise choose, and the two
    diverge the moment the service's own default moves."""
    assert not set(_whisper_query(_client())) & set(_ADDED_PARAMS)


@pytest.mark.parametrize(("name", "value", "expected"), [(n, v, e) for n, (v, e) in _ADDED_PARAMS.items()])
def test_a_requested_parameter_is_sent(name: str, value: Any, expected: str) -> None:
    """Every value here is falsy or off: a truthiness filter would drop them and
    hand the decision back to the service without saying so."""
    assert _whisper_query(_client(), **{name: value})[name] == [expected]


def test_undeclared_parameters_are_refused() -> None:
    from unstract.llmwhisperer.sdk_llmwhisperer.api.whisper import status as status_module

    client = _client()
    with pytest.raises(LLMWhispererClientException):
        client._build_request(status_module, frozenset({"whisper_hash", "made_up"}), whisper_hash="h")


def test_no_operation_sends_a_spec_default_the_client_never_set() -> None:
    """The generated builder writes every declared parameter; each call must
    narrow that to what it asked for."""
    from unstract.llmwhisperer.sdk_llmwhisperer.api.whisper import extract, retrieve

    assert "text_only" in inspect.signature(retrieve._get_kwargs).parameters
    assert "text_only" not in _SEND_ONLY["retrieve"]
    declared = set(inspect.signature(extract._get_kwargs).parameters) - {"body"}
    # In URL mode the URL travels in the body, so the query parameter the
    # generator writes for it is the one extract must never send.
    assert declared - _SEND_ONLY["extract"] == {"url_query"}
    assert _SEND_ONLY["extract"] <= declared


# --------------------------------------------------------------------------
# Return values and exceptions, compared against the published client
# --------------------------------------------------------------------------

ERROR_BODIES = [
    ("json_error", '{"message": "denied"}'),
    ("nested_json", '{"message": {"detail": "denied"}}'),
    ("json_string", '"denied"'),
    ("html", "<html>gateway</html>"),
    ("empty", ""),
]


def _outcome(call: Call, client: Any, sample_file: str, status_code: int, text: str) -> tuple[Any, ...]:
    """Normalise a call into a comparable value: returned, or raised."""
    with (
        patch.object(LLMWhispererClientV2, "_send", return_value=_mock_response(status_code, text)),
        patch("requests.Session.send", return_value=_mock_response(status_code, text)),
    ):
        try:
            return ("return", call(client, sample_file))
        except Exception as exc:  # noqa: BLE001 - the comparison is the point
            return ("raise", type(exc).__name__, getattr(exc, "status_code", None), str(exc)[:160])


@pytest.mark.parametrize("status_code", [200, 201, 400, 401, 404, 500])
@pytest.mark.parametrize("name", list(CALLS))
def test_return_value_matches_the_published_client(name: str, status_code: int, sample_file: str) -> None:
    call = CALLS[name][0]
    ours = _outcome(call, _client(), sample_file, status_code, '{"message": "x", "status": "processed"}')
    theirs = _outcome(call, _baseline_client(), sample_file, status_code, '{"message": "x", "status": "processed"}')
    assert ours == theirs


@pytest.mark.parametrize(("body_name", "text"), ERROR_BODIES, ids=[b[0] for b in ERROR_BODIES])
@pytest.mark.parametrize("name", list(CALLS))
def test_error_handling_matches_the_published_client(name: str, body_name: str, text: str, sample_file: str) -> None:
    """Including the published client's own rough edges: several methods parse
    an error body unguarded and leak ``JSONDecodeError``.

    A drop-in inherits the contract, bugs included.
    """
    call = CALLS[name][0]
    ours = _outcome(call, _client(), sample_file, 500, text)
    theirs = _outcome(call, _baseline_client(), sample_file, 500, text)
    assert ours == theirs


def test_whisper_poll_loop_matches_the_published_client(sample_file: str) -> None:
    """wait_for_completion drives status and retrieve; the assembled result
    dict must be identical."""
    accepted = '{"whisper_hash": "h-1", "message": "queued"}'
    processed = '{"status": "processed", "message": "done"}'
    extraction = '{"result_text": "hello"}'

    def run(client: Any, patch_target: Any) -> Any:
        responses = [
            _mock_response(202, accepted),
            _mock_response(200, processed),
            _mock_response(200, extraction),
        ]
        with patch_target(side_effect=responses):
            return client.whisper(file_path=sample_file, wait_for_completion=True, wait_timeout=30)

    ours = run(_client(), lambda **kw: patch.object(LLMWhispererClientV2, "_send", **kw))
    theirs = run(_baseline_client(), lambda **kw: patch("requests.Session.send", **kw))
    assert ours == theirs
    assert ours["extraction"] == {"result_text": "hello"}


# --------------------------------------------------------------------------
# Transport behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (httpx.ConnectTimeout("connect timed out"), requests.ConnectTimeout),
        (httpx.ReadTimeout("read timed out"), requests.ReadTimeout),
        (httpx.WriteTimeout("write timed out"), requests.Timeout),
        (httpx.PoolTimeout("pool timed out"), requests.Timeout),
        (httpx.ConnectError("refused"), requests.ConnectionError),
        (httpx.ReadError("reset"), requests.ConnectionError),
        (httpx.WriteError("broken pipe"), requests.ConnectionError),
        (httpx.ProtocolError("bad framing"), requests.ConnectionError),
        (httpx.ProxyError("proxy exploded"), requests.ConnectionError),
    ],
)
def test_transport_errors_are_translated(raised: Exception, expected: type[Exception]) -> None:
    """Callers catch the ``requests`` classes by name; httpx's are not
    subclasses of them.

    The exact class matters, not just the family: a caller that catches
    ``ReadTimeout`` sees nothing if a broader ``Timeout`` is raised in its
    place.
    """
    client = _client()
    with patch.object(client._transport, "send", side_effect=raised):
        with pytest.raises(expected) as caught:
            client.get_usage_info()
    assert type(caught.value) is expected


def test_a_connect_timeout_is_still_a_connection_error() -> None:
    """``requests.ConnectTimeout`` is both a ``ConnectionError`` and a
    ``Timeout``.

    Mapping it to a plain ``Timeout`` would stop matching every
    caller that catches the connection family.
    """
    client = _client()
    with patch.object(client._transport, "send", side_effect=httpx.ConnectTimeout("connect timed out")):
        with pytest.raises(requests.ConnectionError):
            client.get_usage_info()


def test_translated_errors_keep_the_original_cause() -> None:
    client = _client()
    original = httpx.ConnectError("refused")
    with patch.object(client._transport, "send", side_effect=original):
        with pytest.raises(requests.ConnectionError) as excinfo:
            client.get_usage_info()
    assert excinfo.value.__cause__ is original


def test_transport_failures_are_still_retried() -> None:
    """Retry matches on the ``requests`` types, so translation has to happen
    inside the retried call or transport-error retry silently stops working."""
    client = _client(max_retries=2, retry_min_wait=0.001, retry_max_wait=0.002)
    with patch.object(client._transport, "send", side_effect=httpx.ConnectError("refused")) as send:
        with pytest.raises(requests.ConnectionError):
            client.get_usage_info()
    assert send.call_count == 3


def test_redirects_are_followed() -> None:
    """The previous transport followed them by default.

    Without this a 30x surfaces as 'API error: empty response body'.
    """
    assert _client()._transport.follow_redirects is True


def test_the_request_timeout_reaches_the_transport() -> None:
    """Unlike the other client's ``api_timeout``, this one is a real socket
    timeout and must not be dropped on the way down."""
    client = _client()
    with patch.object(client._transport, "send", return_value=_mock_response()) as send:
        client.get_usage_info()
    request = send.call_args[0][0]
    assert request.extensions["timeout"]["read"] == client.api_timeout


def test_the_deadline_still_caps_each_attempt() -> None:
    client = _client()
    request = httpx.Request("GET", "http://localhost/test")
    with patch.object(client._transport, "send", return_value=_mock_response()) as send:
        client._send_request(request, timeout=300, deadline=time.time() + 2.0)
    assert send.call_args[0][0].extensions["timeout"]["read"] <= 3.0


def test_the_deadline_still_stops_retries() -> None:
    client = _client(max_retries=10, retry_min_wait=0.1, retry_max_wait=0.2)
    request = httpx.Request("GET", "http://localhost/test")
    with patch.object(client._transport, "send", return_value=_mock_response(500, '{"e": 1}')) as send:
        response = client._send_request(request, timeout=1, deadline=time.time() + 0.3)
    assert response.status_code == 500
    assert send.call_count < 11


def test_encoding_is_applied_to_a_real_response(sample_file: str) -> None:
    """``whisper`` and ``whisper_retrieve`` set the response encoding before
    reading text; on a real response that has to still be legal."""
    body = '{"result_text": "café"}'.encode("latin-1")
    client = _client()
    with patch.object(LLMWhispererClientV2, "_send", return_value=httpx.Response(200, content=body)):
        result = client.whisper_retrieve("hash-1", encoding="latin-1")
    assert result["extraction"] == {"result_text": "café"}


# --------------------------------------------------------------------------
# Construction and surface
# --------------------------------------------------------------------------


def _baseline_class_node() -> ast.ClassDef:
    tree = ast.parse(BASELINE_PATH.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LLMWhispererClientV2":
            return node
    raise AssertionError("LLMWhispererClientV2 not found in the baseline")


def _baseline_method(name: str) -> ast.FunctionDef:
    for node in _baseline_class_node().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in the baseline")


def _params(node: ast.FunctionDef) -> list[tuple[str, object]]:
    args = node.args.args[1:] + node.args.kwonlyargs
    defaults = [None] * (len(node.args.args[1:]) - len(node.args.defaults)) + [
        ast.literal_eval(d) for d in node.args.defaults
    ]
    defaults += [ast.literal_eval(d) if d is not None else None for d in node.args.kw_defaults]
    return list(zip([a.arg for a in args], defaults, strict=False))


def _live_params(func: Any, *, keyword_only: bool = True) -> list[tuple[str, object]]:
    """Parameters in order, with their defaults.

    ``keyword_only=False`` drops keyword-only parameters, for the comparisons
    where a keyword-only addition is allowed: none is reachable from a released
    call shape, so adding one leaves every existing call intact.
    """
    return [
        (name, None if p.default is inspect.Parameter.empty else p.default)
        for name, p in inspect.signature(func).parameters.items()
        if name not in ("self", "cls") and (keyword_only or p.kind is not inspect.Parameter.KEYWORD_ONLY)
    ]


def test_constructor_is_unchanged() -> None:
    """Names, order and defaults all matter: callers pass some positionally."""
    assert _live_params(LLMWhispererClientV2.__init__) == _params(_baseline_method("__init__"))


def test_public_methods_are_unchanged() -> None:
    methods = {
        node.name: node
        for node in _baseline_class_node().body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    assert len(methods) == 11, "the published surface is 11 public methods"

    for name, node in methods.items():
        live = getattr(LLMWhispererClientV2, name, None)
        assert live is not None, f"{name} disappeared from the client"
        assert _live_params(live, keyword_only=False) == _params(node), name


def test_the_deprecated_parameter_resolver_is_unchanged() -> None:
    """Which renames forward and which stay dead is a service-side fact, not a
    style choice: applying a dead one would change extraction output.
    """
    assert _live_params(LLMWhispererClientV2._resolve_deprecated_param) == _params(
        _baseline_method("_resolve_deprecated_param")
    )
    assert _body_dump(_baseline_method("_resolve_deprecated_param")) == _body_dump(
        _method_node(LLMWhispererClientV2, "_resolve_deprecated_param")
    )


def _method_node(cls: type, name: str) -> ast.FunctionDef:
    source = inspect.getsource(getattr(cls, name))
    node = ast.parse(inspect.cleandoc(source)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def _body_dump(node: ast.FunctionDef) -> str:
    """The statements, minus the docstring — re-indenting changes its text."""
    body = node.body[1:] if isinstance(node.body[0], ast.Expr) else node.body
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def test_get_highlight_rect_is_unchanged() -> None:
    """Pure geometry, no request: it should survive the port verbatim."""
    assert _body_dump(_baseline_method("get_highlight_rect")) == _body_dump(
        _method_node(LLMWhispererClientV2, "get_highlight_rect")
    )


def test_class_attributes_are_unchanged() -> None:
    for node in _baseline_class_node().body:
        target, assigned = None, None
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            target, assigned = getattr(node.target, "id", None), node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, assigned = getattr(node.targets[0], "id", None), node.value
        if target is None or assigned is None:
            continue
        try:
            value = ast.literal_eval(assigned)
        except ValueError:
            continue  # logger and friends: identity, not value
        assert getattr(LLMWhispererClientV2, target) == value, target


def test_retry_policy_attributes_are_unchanged() -> None:
    """Retry is ungenerated and untested by a signature sweep, and has drifted
    before."""
    ours, theirs = _client(max_retries=3), _baseline_client(max_retries=3)
    for attribute in ("max_retries", "retry_min_wait", "retry_max_wait", "api_timeout", "base_url", "headers"):
        assert getattr(ours, attribute) == getattr(theirs, attribute), attribute
    assert LLMWhispererClientV2._is_retryable(requests.ConnectionError()) is True
    assert LLMWhispererClientV2._is_retryable(requests.Timeout()) is True
    assert LLMWhispererClientV2._is_retryable(ValueError()) is False


def test_defaults_match_a_default_constructed_published_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMWHISPERER_BASE_URL_V2", raising=False)
    monkeypatch.setenv("LLMWHISPERER_API_KEY", "env-key")
    ours = LLMWhispererClientV2(logging_level="ERROR")
    theirs = baseline.LLMWhispererClientV2(logging_level="ERROR")
    for attribute in ("base_url", "api_key", "api_timeout", "headers", "max_retries"):
        assert getattr(ours, attribute) == getattr(theirs, attribute), attribute


def test_every_wrapped_operation_is_covered() -> None:
    """A new spec operation shows up here as a failure, not as silence."""
    spec = json.loads(SPEC_PATH.read_text())
    declared = {
        operation["operationId"]
        for path in spec["paths"].values()
        for method, operation in path.items()
        if method in {"get", "post", "put", "patch", "delete"}
    }
    # Checked before the subtraction: an entry excusing an operation the spec no
    # longer declares keeps this passing forever, and nothing about a green run
    # says the list is still describing anything.
    assert UNWRAPPED_OPERATIONS <= declared
    assert declared - UNWRAPPED_OPERATIONS == set(_SEND_ONLY)


def test_the_baseline_is_pinned() -> None:
    assert BASELINE_REF in BASELINE_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert "DO NOT EDIT" in BASELINE_PATH.read_text(encoding="utf-8")
