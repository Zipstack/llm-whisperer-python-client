__version__ = "2.6.1"

from .client_v2 import LLMWhispererClientV2  # noqa: F401


def get_llmw_py_client_version() -> str:
    """Returns the SDK version."""
    return __version__
