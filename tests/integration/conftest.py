import pytest
from unstract.llmwhisperer.client_v2 import LLMWhispererClientV2


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--max-retries",
        action="store",
        default=3,
        type=int,
        help="Max retry attempts for LLMWhispererClientV2 (0 to disable)",
    )


@pytest.fixture(name="client_v2")
def llm_whisperer_client_v2(request: pytest.FixtureRequest) -> LLMWhispererClientV2:
    max_retries = request.config.getoption("--max-retries")
    client = LLMWhispererClientV2(
        max_retries=max_retries,
        retry_min_wait=1.0,
        retry_max_wait=60.0,
    )
    return client
