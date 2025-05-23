import os

import pytest
from unstract.llmwhisperer.client_v2 import LLMWhispererClientV2


@pytest.fixture(name="client_v2")
def llm_whisperer_client_v2() -> LLMWhispererClientV2:
    client = LLMWhispererClientV2()
    return client


@pytest.fixture(name="data_dir", scope="session")
def test_data_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "test_data")
