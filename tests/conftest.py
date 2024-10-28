import os

import pytest
from dotenv import load_dotenv

from unstract.llmwhisperer.client import LLMWhispererClient
from unstract.llmwhisperer.client_v2 import LLMWhispererClientV2

load_dotenv()


@pytest.fixture(name="client")
def llm_whisperer_client():
    client = LLMWhispererClient()
    return client


@pytest.fixture(name="client_v2")
def llm_whisperer_client_v2():
    client = LLMWhispererClientV2()
    return client


@pytest.fixture(name="data_dir", scope="session")
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), "test_data")
