"""Factory for creating providers by name. Core uses this; no direct OpenAI/Claude imports in callers."""

from typing import Optional, Any

from costplan.core.provider import BaseProvider
from costplan.config.settings import Settings


def create(
    provider_name: str,
    settings: Optional[Settings] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> BaseProvider:
    """Create a provider by name. Pricing and token logic stay inside the returned provider.

    Args:
        provider_name: e.g. "openai"
        settings: Optional settings (for API key, base URL, output ratio, etc.)
        api_key: Override API key (else from settings/env)
        base_url: Override base URL (else from settings/env)
        **kwargs: Passed to the provider constructor

    Returns:
        BaseProvider implementation

    Raises:
        ValueError: If provider_name is not supported
    """
    name = provider_name.lower().strip()
    _settings = settings or Settings()

    if name == "openai":
        from costplan.core.providers.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=api_key or _settings.get_api_key(),
            base_url=base_url or _settings.get_base_url(),
            settings=_settings,
            **kwargs,
        )

    if name == "anthropic":
        from costplan.core.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            api_key=api_key or _settings.get_anthropic_api_key(),
            settings=_settings,
            **kwargs,
        )

    raise ValueError(f"Unknown provider: {provider_name}. Supported: openai, anthropic")
