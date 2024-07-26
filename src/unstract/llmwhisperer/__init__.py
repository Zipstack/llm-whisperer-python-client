__version__ = "0.21.0"

from .client import LLMWhispererClient  # noqa: F401


def get_sdk_version():
    """Returns the SDK version."""
    return __version__
