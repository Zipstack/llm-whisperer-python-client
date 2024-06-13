import os

import pytest

from llmwhisperer.client import LLMWhispererClient


@pytest.fixture(name="client")
def llm_whisperer_client():
    # Create an instance of the client
    client = LLMWhispererClient()
    return client


@pytest.fixture(name="data_dir", scope="session")
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), "test_data")
