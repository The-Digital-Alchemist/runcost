"""Anthropic Claude provider implementation.

Pricing (pricing.json anthropic slice) and token estimation are internal to this module.
"""

import logging
import os
from typing import Optional, Tuple, List

from costplan.core.provider import BaseProvider, TokenPrediction
from costplan.core.executor import ExecutionResult
from costplan.core.pricing import PricingRegistry
from costplan.config.settings import Settings

logger = logging.getLogger(__name__)


def _heuristic_tokens(text: str) -> int:
    """Rough token estimate (Claude ~4 chars per token)."""
    return max(1, len(text) // 4)


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider. Uses heuristic token estimation (no tiktoken for Claude)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        pricing_registry: Optional[PricingRegistry] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key (or ANTHROPIC_API_KEY env)
            timeout: Request timeout in seconds
            pricing_registry: Uses default pricing.json anthropic slice if None
            settings: For default output ratio (default if None)
        """
        self._api_key = api_key
        self._timeout = timeout
        self._pricing = pricing_registry or PricingRegistry(provider_name="anthropic")
        self._settings = settings or Settings()
        self._client = None  # Lazy init so predict-only usage doesn't require key

    def _get_client(self):
        """Lazy-init Anthropic client (needed only for execute)."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install with: pip install anthropic"
                ) from e
            key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "Anthropic API key not set. Use api_key= or ANTHROPIC_API_KEY."
                )
            self._client = anthropic.Anthropic(api_key=key, timeout=self._timeout)
        return self._client

    def predict_tokens(
        self,
        prompt: str,
        model: str,
        output_ratio: Optional[float] = None,
    ) -> TokenPrediction:
        """Heuristic token estimate (Claude tokenizer not public)."""
        input_tokens = _heuristic_tokens(prompt)
        ratio = output_ratio or self._settings.default_output_ratio
        output_tokens = int(input_tokens * ratio)
        return TokenPrediction(input_tokens=input_tokens, output_tokens=output_tokens)

    def execute(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ExecutionResult:
        """Execute via Anthropic Messages API."""
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except ImportError as e:
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=str(e),
            )
        except Exception as e:
            logger.exception("Anthropic API error")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=str(e),
            )

        # Map Anthropic usage to our ExecutionResult shape
        usage = getattr(resp, "usage", None)
        if usage is not None:
            inp = getattr(usage, "input_tokens", 0) or 0
            out = getattr(usage, "output_tokens", 0) or 0
            usage_dict = {
                "prompt_tokens": inp,
                "completion_tokens": out,
                "total_tokens": inp + out,
            }
        else:
            usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        text = ""
        if getattr(resp, "content", None):
            for block in resp.content:
                if getattr(block, "text", None):
                    text += block.text

        return ExecutionResult(
            response_text=text,
            usage=usage_dict,
            raw_response=resp,
            model=model,
            success=True,
        )

    def get_pricing(self, model: str) -> Tuple[float, float]:
        """Return (input $/1k, output $/1k) from internal pricing registry."""
        return self._pricing.get_model_pricing(model)

    def list_models(self) -> List[str]:
        """Return model names from internal pricing (anthropic slice)."""
        return self._pricing.list_supported_models()

    def execute_with_messages(
        self,
        messages: list,
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ExecutionResult:
        """Execute with message list (Anthropic format)."""
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )
        except Exception as e:
            logger.exception("Anthropic API error")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=str(e),
            )

        usage = getattr(resp, "usage", None)
        if usage is not None:
            inp = getattr(usage, "input_tokens", 0) or 0
            out = getattr(usage, "output_tokens", 0) or 0
            usage_dict = {
                "prompt_tokens": inp,
                "completion_tokens": out,
                "total_tokens": inp + out,
            }
        else:
            usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        text = ""
        if getattr(resp, "content", None):
            for block in resp.content:
                if getattr(block, "text", None):
                    text += block.text

        return ExecutionResult(
            response_text=text,
            usage=usage_dict,
            raw_response=resp,
            model=model,
            success=True,
        )
