"""Cache-aware cost calculation for proxy responses.

Handles Anthropic's token pricing model including cache_read and cache_creation tokens,
as well as standard OpenAI token pricing. Falls back to conservative estimates when
a model isn't in the pricing registry so budget tracking is never skipped.
"""

import logging

from costplan.core.pricing import PricingRegistry, PricingNotFoundError

logger = logging.getLogger(__name__)

# Fallback pricing (per 1k tokens) used when a model isn't in the registry.
# Uses Opus-tier rates — deliberately expensive so the budget errs on the
# side of caution rather than letting unknown models slip through for free.
_ANTHROPIC_FALLBACK = {
    "input_cost_per_1k_tokens": 0.015,
    "output_cost_per_1k_tokens": 0.075,
    "cache_read_cost_per_1k_tokens": 0.0015,
    "cache_creation_cost_per_1k_tokens": 0.01875,
}
_OPENAI_FALLBACK_INPUT = 0.03   # GPT-4 tier
_OPENAI_FALLBACK_OUTPUT = 0.06


def _resolve_anthropic_pricing(model: str, pricing: PricingRegistry | None) -> dict:
    """Return full pricing dict, falling back to conservative defaults for unknown models."""
    if pricing is None:
        pricing = PricingRegistry(provider_name="anthropic")
    try:
        return pricing.get_full_pricing(model)
    except PricingNotFoundError:
        logger.debug(
            "Model %r not in pricing registry — using Opus-tier fallback pricing. "
            "Add it to pricing.json for accurate tracking.",
            model,
        )
        return dict(_ANTHROPIC_FALLBACK)


def compute_anthropic_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    pricing: PricingRegistry | None = None,
) -> float:
    """Compute actual cost for an Anthropic API call with full cache awareness.

    Falls back to conservative (Opus-tier) pricing when the model isn't
    in the registry so that budget tracking is never skipped.

    Args:
        model: Model name (e.g. "claude-sonnet-4-20250514").
        input_tokens: Regular (non-cached) input tokens.
        output_tokens: Output tokens.
        cache_read_tokens: Tokens read from cache.
        cache_creation_tokens: Tokens written to cache.
        pricing: PricingRegistry for Anthropic. Uses default if None.

    Returns:
        Total cost in dollars.
    """
    full_pricing = _resolve_anthropic_pricing(model, pricing)
    cost = (
        (input_tokens / 1000) * full_pricing["input_cost_per_1k_tokens"]
        + (output_tokens / 1000) * full_pricing["output_cost_per_1k_tokens"]
        + (cache_read_tokens / 1000) * full_pricing.get("cache_read_cost_per_1k_tokens", 0)
        + (cache_creation_tokens / 1000) * full_pricing.get("cache_creation_cost_per_1k_tokens", 0)
    )
    return cost


def compute_openai_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: PricingRegistry | None = None,
) -> float:
    """Compute actual cost for an OpenAI API call.

    Falls back to conservative (GPT-4 tier) pricing for unknown models.

    Args:
        model: Model name (e.g. "gpt-4o").
        prompt_tokens: Input/prompt tokens.
        completion_tokens: Output/completion tokens.
        pricing: PricingRegistry for OpenAI. Uses default if None.

    Returns:
        Total cost in dollars.
    """
    if pricing is None:
        pricing = PricingRegistry(provider_name="openai")

    try:
        input_price, output_price = pricing.get_model_pricing(model)
    except PricingNotFoundError:
        logger.debug(
            "Model %r not in pricing registry — using GPT-4 fallback pricing.", model,
        )
        input_price, output_price = _OPENAI_FALLBACK_INPUT, _OPENAI_FALLBACK_OUTPUT

    return (prompt_tokens / 1000) * input_price + (completion_tokens / 1000) * output_price


def heuristic_input_cost_estimate(
    message_text_length: int,
    model: str,
    pricing: PricingRegistry,
) -> float:
    """Quick heuristic cost estimate from message text length (for pre-check only).

    Uses ~3 chars per token as a rough estimate. This is intentionally conservative
    (overestimates) to avoid false negatives on the pre-check.
    Falls back to Opus-tier pricing for unknown models.

    Args:
        message_text_length: Total character count of all message content.
        model: Model name.
        pricing: PricingRegistry instance.

    Returns:
        Estimated input cost in dollars.
    """
    estimated_tokens = max(1, message_text_length // 3)  # Conservative: 3 chars/token
    full_pricing = _resolve_anthropic_pricing(model, pricing)
    return (estimated_tokens / 1000) * full_pricing["input_cost_per_1k_tokens"]
