"""Cache-aware cost calculation for proxy responses.

Handles Anthropic's token pricing model including cache_read and cache_creation tokens,
as well as standard OpenAI token pricing.
"""

from costplan.core.pricing import PricingRegistry


def compute_anthropic_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    pricing: PricingRegistry | None = None,
) -> float:
    """Compute actual cost for an Anthropic API call with full cache awareness.

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
    if pricing is None:
        pricing = PricingRegistry(provider_name="anthropic")

    full_pricing = pricing.get_full_pricing(model)
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

    input_price, output_price = pricing.get_model_pricing(model)
    return (prompt_tokens / 1000) * input_price + (completion_tokens / 1000) * output_price


def heuristic_input_cost_estimate(
    message_text_length: int,
    model: str,
    pricing: PricingRegistry,
) -> float:
    """Quick heuristic cost estimate from message text length (for pre-check only).

    Uses ~4 chars per token as a rough estimate. This is intentionally conservative
    (overestimates) to avoid false negatives on the pre-check.

    Args:
        message_text_length: Total character count of all message content.
        model: Model name.
        pricing: PricingRegistry instance.

    Returns:
        Estimated input cost in dollars.
    """
    estimated_tokens = max(1, message_text_length // 3)  # Conservative: 3 chars/token
    full_pricing = pricing.get_full_pricing(model)
    return (estimated_tokens / 1000) * full_pricing["input_cost_per_1k_tokens"]
