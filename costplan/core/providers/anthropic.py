"""Anthropic Claude provider implementation.

Pricing (pricing.json anthropic slice) and token counting (API count_tokens or heuristic) internal.
"""

import logging
import os
from typing import Optional, Tuple, List, Any, Dict

from costplan.core.provider import BaseProvider, TokenPrediction
from costplan.core.executor import ExecutionResult
from costplan.core.pricing import PricingRegistry
from costplan.config.settings import Settings

logger = logging.getLogger(__name__)

# Strict mode: no client -> raise. Non-strict: no client -> allow heuristic fallback.
STRICT_TOKEN_COUNT_DEFAULT = True


def _heuristic_tokens(text: str) -> int:
    """Rough token estimate (Claude ~4 chars per token). Only used in non-strict mode when no client."""
    return max(1, len(text) // 4)


def _has_api_key(api_key: Optional[str]) -> bool:
    return bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))


def _extract_response_text(resp: Any) -> str:
    """Extract text from Anthropic response content blocks."""
    text = ""
    if getattr(resp, "content", None):
        for block in resp.content:
            if getattr(block, "text", None):
                text += block.text
    return text


def _extract_usage_dict(resp: Any) -> Dict[str, int]:
    """Extract usage dict from Anthropic response, mapping to our standard format."""
    usage = getattr(resp, "usage", None)
    if usage is not None:
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        return {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "total_tokens": inp + out,
        }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider. Accurate input tokens via count_tokens() when client exists; strict mode raises when no client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        pricing_registry: Optional[PricingRegistry] = None,
        settings: Optional[Settings] = None,
        strict_token_count: bool = STRICT_TOKEN_COUNT_DEFAULT,
    ):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key (or ANTHROPIC_API_KEY env)
            timeout: Request timeout in seconds
            pricing_registry: Uses default pricing.json anthropic slice if None
            settings: For default output ratio (default if None)
            strict_token_count: If True, require API client for predict_tokens (use count_tokens()); if no key, raise. If False, fall back to heuristic when no client.
        """
        self._api_key = api_key
        self._timeout = timeout
        self._pricing = pricing_registry or PricingRegistry(provider_name="anthropic")
        self._settings = settings or Settings()
        self._strict_token_count = strict_token_count
        self._client = None  # Lazy init
        self._async_client = None  # Lazy init

    def _get_client(self):
        """Lazy-init Anthropic client. Raises if no API key."""
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

    def _get_async_client(self):
        """Lazy-init AsyncAnthropic client."""
        if self._async_client is None:
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
            self._async_client = anthropic.AsyncAnthropic(api_key=key, timeout=self._timeout)
        return self._async_client

    def _count_input_tokens_via_api(self, prompt: str, model: str) -> int:
        """Call Anthropic count_tokens API. Raises on failure or missing client."""
        client = self._get_client()
        resp = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return getattr(resp, "input_tokens", 0) or 0

    def predict_tokens(
        self,
        prompt: str,
        model: str,
        output_ratio: Optional[float] = None,
    ) -> TokenPrediction:
        """Input tokens: from count_tokens() API when client exists; else strict -> raise, non-strict -> heuristic."""
        if _has_api_key(self._api_key):
            try:
                input_tokens = self._count_input_tokens_via_api(prompt, model)
            except Exception as e:
                if self._strict_token_count:
                    raise RuntimeError(
                        f"Anthropic count_tokens failed (strict mode). {e}"
                    ) from e
                logger.warning("count_tokens failed, falling back to heuristic: %s", e)
                input_tokens = _heuristic_tokens(prompt)
        else:
            if self._strict_token_count:
                raise RuntimeError(
                    "Anthropic API key required for token count in strict mode. "
                    "Set ANTHROPIC_API_KEY or pass api_key=, or use strict_token_count=False to allow heuristic fallback."
                )
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

        return ExecutionResult(
            response_text=_extract_response_text(resp),
            usage=_extract_usage_dict(resp),
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

        return ExecutionResult(
            response_text=_extract_response_text(resp),
            usage=_extract_usage_dict(resp),
            raw_response=resp,
            model=model,
            success=True,
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
        """Async execute via AsyncAnthropic."""
        try:
            client = self._get_async_client()
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except Exception as e:
            logger.exception("Async Anthropic API error")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=str(e),
            )

        return ExecutionResult(
            response_text=_extract_response_text(resp),
            usage=_extract_usage_dict(resp),
            raw_response=resp,
            model=model,
            success=True,
        )

    async def async_execute_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Async execute with message list."""
        try:
            client = self._get_async_client()
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )
        except Exception as e:
            logger.exception("Async Anthropic messages error")
            return ExecutionResult(
                response_text="",
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_response=None,
                model=model,
                success=False,
                error_message=str(e),
            )

        return ExecutionResult(
            response_text=_extract_response_text(resp),
            usage=_extract_usage_dict(resp),
            raw_response=resp,
            model=model,
            success=True,
        )
