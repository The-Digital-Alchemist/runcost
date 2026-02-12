"""OpenAI (and OpenAI-compatible) provider implementation.

Pricing (pricing.json) and token estimation (tiktoken) are internal to this module.
The core never opens pricing.json or uses TokenEstimator directly.
"""

from typing import Optional, Tuple, List

from costplan.core.provider import BaseProvider, TokenPrediction
from costplan.core.executor import ProviderExecutor, ExecutionResult
from costplan.core.estimator import TokenEstimator
from costplan.core.pricing import PricingRegistry
from costplan.config.settings import Settings


class OpenAIProvider(BaseProvider):
    """OpenAI and OpenAI-compatible API provider. Uses tiktoken for token prediction."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: float = 60.0,
        pricing_registry: Optional[PricingRegistry] = None,
        token_estimator: Optional[TokenEstimator] = None,
        settings: Optional[Settings] = None,
    ):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI (or compatible) API key
            base_url: API base URL (None = OpenAI default)
            organization: Optional organization ID
            timeout: Request timeout in seconds
            pricing_registry: Pricing lookup (uses default pricing.json if None)
            token_estimator: Token estimation (tiktoken/heuristic; default if None)
            settings: Settings for default output ratio (default if None)
        """
        self._executor = ProviderExecutor(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
        )
        self._pricing = pricing_registry or PricingRegistry(provider_name="openai")
        self._estimator = token_estimator or TokenEstimator(
            estimation_mode=(settings or Settings()).token_estimation_mode
        )
        self._settings = settings or Settings()

    def predict_tokens(
        self,
        prompt: str,
        model: str,
        output_ratio: Optional[float] = None,
    ) -> TokenPrediction:
        """Use tiktoken (or heuristic) to predict input tokens; derive output from ratio."""
        input_tokens = self._estimator.estimate_tokens(prompt, model)
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
        """Execute via OpenAI-compatible API."""
        return self._executor.execute(
            prompt, model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    def get_pricing(self, model: str) -> Tuple[float, float]:
        """Return (input $/1k tokens, output $/1k tokens) from internal pricing registry."""
        return self._pricing.get_model_pricing(model)

    def list_models(self) -> List[str]:
        """Return model names supported by this provider (from internal pricing)."""
        return self._pricing.list_supported_models()

    def execute_with_messages(
        self,
        messages: list,
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ExecutionResult:
        """Execute chat completion with messages."""
        return self._executor.execute_with_messages(
            messages, model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
