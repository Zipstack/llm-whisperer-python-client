class LLMWhispererUtils:
    @staticmethod
    def redact_key(api_key: str, reveal_length=4) -> str:
        """Hides sensitive information partially. Useful for logging keys.

        Args:
            api_key (str): API key to redact

        Returns:
            str: Redacted API key
        """
        if not isinstance(api_key, str):
            raise ValueError("API key must be a string")

        if reveal_length < 0:
            raise ValueError("Reveal length must be a non-negative integer")

        redacted_length = max(len(api_key) - reveal_length, 0)
        revealed_part = api_key[:reveal_length]
        redacted_part = "x" * redacted_length
        return revealed_part + redacted_part
