"""Provider execution wrapper for LLM requests."""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of LLM execution."""

    response_text: str
    usage: Dict[str, int]  # Contains prompt_tokens, completion_tokens, total_tokens
    raw_response: Any
    model: str
    success: bool = True
    error_message: Optional[str] = None

    def __repr__(self) -> str:
        if self.success:
            return (
                f"ExecutionResult(model={self.model}, "
                f"tokens={self.usage.get('total_tokens', 0)}, "
                f"response_length={len(self.response_text)})"
            )
        else:
            return f"ExecutionResult(model={self.model}, error={self.error_message})"


class ProviderExecutor:
    """Wrapper for executing LLM requests via OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """Initialize the provider executor.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API requests (None = OpenAI default)
            organization: Organization ID (optional)
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
        )

    def execute(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute a single prompt completion.

        Args:
            prompt: Input prompt text
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional arguments to pass to the API

        Returns:
            ExecutionResult with response and usage data
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Extract response text
            response_text = response.choices[0].message.content or ""

            # Extract usage data
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return ExecutionResult(
                response_text=response_text,
                usage=usage,
                raw_response=response,
                model=model,
                success=True,
            )

        except RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=f"Rate limit exceeded: {str(e)}",
            )

        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=f"Connection error: {str(e)}",
            )

        except APIError as e:
            logger.error(f"API error: {e}")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=f"API error: {str(e)}",
            )

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    def execute_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ExecutionResult:
        """Execute a chat completion with messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional arguments to pass to the API

        Returns:
            ExecutionResult with response and usage data
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Extract response text
            response_text = response.choices[0].message.content or ""

            # Extract usage data
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return ExecutionResult(
                response_text=response_text,
                usage=usage,
                raw_response=response,
                model=model,
                success=True,
            )

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=str(e),
            )

    def validate_connection(self) -> bool:
        """Validate API connection and credentials.

        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # Try to list models as a connection test
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
