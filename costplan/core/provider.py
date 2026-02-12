"""Provider abstraction for LLM backends.

Core logic depends on BaseProvider, not on any specific vendor (OpenAI, etc.).
Pricing and token estimation are provider-scoped; the core never opens pricing.json
or uses TokenEstimator directly.
Implementations live under costplan.core.providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from costplan.core.executor import ExecutionResult
from costplan.core.predictor import (
    PredictionResult,
    build_prediction_result_from_tokens_and_pricing,
)


@dataclass
class TokenPrediction:
    """Predicted token counts for a request (input + output)."""

    input_tokens: int
    output_tokens: int


class BaseProvider(ABC):
    """Abstract base for LLM providers. Implement this to add support for new models/backends."""

    @abstractmethod
    def predict_tokens(
        self,
        prompt: str,
        model: str,
        output_ratio: Optional[float] = None,
    ) -> TokenPrediction:
        """Predict input and output token counts before execution.

        Args:
            prompt: Input prompt text
            model: Model name
            output_ratio: Optional ratio of output tokens to input tokens (e.g. 0.6)

        Returns:
            TokenPrediction with input_tokens and output_tokens
        """
        pass

    @abstractmethod
    def execute(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a completion and return response and usage.

        Args:
            prompt: Input prompt text
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate (optional)
            **kwargs: Provider-specific options

        Returns:
            ExecutionResult with response_text, usage dict, success, etc.
        """
        pass

    @abstractmethod
    def get_pricing(self, model: str) -> Tuple[float, float]:
        """Return (input_price_per_1k_tokens, output_price_per_1k_tokens) in dollars.

        Args:
            model: Model name

        Returns:
            (input $/1k tokens, output $/1k tokens)
        """
        pass

    def predict(
        self,
        prompt: str,
        model: str,
        output_ratio: Optional[float] = None,
    ) -> PredictionResult:
        """Predict cost for a prompt (tokens + pricing via provider). Core uses this; no direct pricing/token logic."""
        token_pred = self.predict_tokens(prompt, model, output_ratio=output_ratio)
        input_price, output_price = self.get_pricing(model)
        return build_prediction_result_from_tokens_and_pricing(
            model=model,
            input_tokens=token_pred.input_tokens,
            output_tokens=token_pred.output_tokens,
            input_price_per_1k=input_price,
            output_price_per_1k=output_price,
        )

    def list_models(self) -> List[str]:
        """Return list of model names this provider supports (for pricing/list). Default: empty."""
        return []

    def execute_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a chat completion with messages. Override for chat-style APIs."""
        # Default: concat message contents as a single prompt (subclasses can override)
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("content"))
        return self.execute(
            prompt, model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
