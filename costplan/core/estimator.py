"""Token estimation for LLM inputs."""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TokenEstimator:
    """Estimates token counts for text inputs."""

    def __init__(self, estimation_mode: str = "tiktoken"):
        """Initialize the token estimator.

        Args:
            estimation_mode: Estimation mode - "tiktoken" or "heuristic"
        """
        self.estimation_mode = estimation_mode
        self._encoders = {}  # Cache for tiktoken encoders

    def estimate_tokens(self, text: str, model: str) -> int:
        """Estimate token count for text.

        Args:
            text: Input text to estimate
            model: Model name (used for tiktoken encoding)

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Try tiktoken first if enabled
        if self.estimation_mode == "tiktoken":
            try:
                return self._estimate_with_tiktoken(text, model)
            except Exception as e:
                logger.warning(
                    f"Tiktoken estimation failed for model {model}: {e}. "
                    "Falling back to heuristic."
                )
                return self._estimate_with_heuristic(text)
        else:
            return self._estimate_with_heuristic(text)

    def estimate_from_messages(
        self,
        messages: List[Dict[str, Any]],
        model: str
    ) -> int:
        """Estimate token count for chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name

        Returns:
            Estimated token count including message formatting overhead
        """
        if not messages:
            return 0

        # Try tiktoken with proper message formatting
        if self.estimation_mode == "tiktoken":
            try:
                return self._estimate_messages_with_tiktoken(messages, model)
            except Exception as e:
                logger.warning(
                    f"Tiktoken message estimation failed: {e}. "
                    "Falling back to heuristic."
                )

        # Fallback: estimate each message content + overhead
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += self._estimate_with_heuristic(content)
            # Add overhead for message structure (role, formatting, etc.)
            total_tokens += 4  # Approximate overhead per message

        return total_tokens

    def _estimate_with_tiktoken(self, text: str, model: str) -> int:
        """Estimate using tiktoken library.

        Args:
            text: Input text
            model: Model name

        Returns:
            Token count from tiktoken
        """
        import tiktoken

        # Get or create encoder for this model
        if model not in self._encoders:
            try:
                self._encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Model not recognized, use cl100k_base (GPT-4 default)
                logger.info(
                    f"Model {model} not recognized by tiktoken, "
                    "using cl100k_base encoding"
                )
                self._encoders[model] = tiktoken.get_encoding("cl100k_base")

        encoder = self._encoders[model]
        return len(encoder.encode(text))

    def _estimate_messages_with_tiktoken(
        self,
        messages: List[Dict[str, Any]],
        model: str
    ) -> int:
        """Estimate messages using tiktoken with proper formatting.

        Args:
            messages: List of message dicts
            model: Model name

        Returns:
            Token count including message formatting overhead
        """
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        # Token counting logic based on OpenAI's guidelines
        tokens_per_message = 3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = 1  # If there's a name, the role is omitted

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def _estimate_with_heuristic(self, text: str) -> int:
        """Estimate using character-based heuristic.

        Args:
            text: Input text

        Returns:
            Estimated token count (characters / 4)
        """
        # Conservative estimate: 1 token ≈ 4 characters
        return max(1, len(text) // 4)

    def batch_estimate(
        self,
        texts: List[str],
        model: str
    ) -> List[int]:
        """Estimate tokens for multiple texts.

        Args:
            texts: List of input texts
            model: Model name

        Returns:
            List of token counts
        """
        return [self.estimate_tokens(text, model) for text in texts]
