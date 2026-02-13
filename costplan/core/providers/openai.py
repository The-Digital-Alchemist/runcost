"""OpenAI (and OpenAI-compatible) provider implementation.

Pricing (pricing.json) and token estimation (tiktoken) are internal to this module.
The core never opens pricing.json or uses TokenEstimator directly.
"""

import logging
from typing import Optional, Tuple, List, Any, Dict

from costplan.core.provider import BaseProvider, TokenPrediction
from costplan.core.executor import ProviderExecutor, ExecutionResult
from costplan.core.estimator import TokenEstimator
from costplan.core.pricing import PricingRegistry
from costplan.config.settings import Settings

logger = logging.getLogger(__name__)


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
        self._api_key = api_key
        self._base_url = base_url
        self._organization = organization
        self._timeout = timeout
        self._executor = ProviderExecutor(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout,
        )
        self._pricing = pricing_registry or PricingRegistry(provider_name="openai")
        _settings = settings or Settings()
        self._estimator = token_estimator or TokenEstimator(
            estimation_mode=_settings.token_estimation_mode,
            allow_heuristic_fallback=False,  # Production: provider-accurate tokenizer only
        )
        self._settings = settings or Settings()
        self._async_client = None  # Lazy init

    def _get_async_client(self):
        """Lazy-init AsyncOpenAI client."""
        if self._async_client is None:
            from openai import AsyncOpenAI
            self._async_client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                organization=self._organization,
                timeout=self._timeout,
            )
        return self._async_client

    def predict_tokens(
        self,
        prompt: str,
        model: str,
        output_ratio: Optional[float] = None,
    ) -> TokenPrediction:
        """Input tokens from tiktoken only (no fallback). Output tokens from ratio for cost prediction."""
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

    # --- Native async methods ---

    async def async_execute(
        self,
        prompt: str,
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Async execute via AsyncOpenAI."""
        try:
            client = self._get_async_client()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            response_text = response.choices[0].message.content or ""
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
            logger.error(f"Async OpenAI error: {e}")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=str(e),
            )

    async def async_execute_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Async execute chat completion with messages."""
        try:
            client = self._get_async_client()
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            response_text = response.choices[0].message.content or ""
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
            logger.error(f"Async OpenAI messages error: {e}")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=str(e),
            )
