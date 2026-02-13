"""Actual cost calculation from LLM execution results."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from costplan.core.pricing import PricingRegistry


@dataclass
class ActualCostResult:
    """Result of actual cost calculation."""

    model: str
    actual_input_tokens: int
    actual_output_tokens: int
    actual_input_cost: float
    actual_output_cost: float
    actual_total_cost: float
    # Cache-aware fields (Anthropic)
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_cost: float = 0.0
    cache_creation_cost: float = 0.0

    def __repr__(self) -> str:
        parts = [
            f"ActualCostResult(model={self.model}",
            f"total_cost=${self.actual_total_cost:.4f}",
            f"tokens={self.actual_input_tokens}+{self.actual_output_tokens}",
        ]
        if self.cache_read_tokens or self.cache_creation_tokens:
            parts.append(
                f"cache_read={self.cache_read_tokens}, cache_create={self.cache_creation_tokens}"
            )
        return ", ".join(parts) + ")"


class CostCalculator:
    """Calculates actual costs from LLM execution results."""

    def __init__(self, pricing_registry: Optional[PricingRegistry] = None):
        """Initialize the cost calculator.

        Args:
            pricing_registry: Pricing registry instance (creates default if None)
        """
        self.pricing_registry = pricing_registry or PricingRegistry()

    def calculate(
        self,
        usage: Dict[str, Any],
        model: str,
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
    ) -> ActualCostResult:
        """Calculate actual cost from usage data.

        Args:
            usage: Usage dict from LLM response (with prompt_tokens, completion_tokens)
            model: Model name
            cache_read_tokens: Override cache read tokens (else from usage dict)
            cache_creation_tokens: Override cache creation tokens (else from usage dict)

        Returns:
            ActualCostResult with actual costs

        Raises:
            ValueError: If usage dict is missing required fields
        """
        # Extract token counts
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

        if prompt_tokens is None or completion_tokens is None:
            raise ValueError(
                "Usage dict must contain 'prompt_tokens' and 'completion_tokens'. "
                f"Got: {usage}"
            )

        # Extract cache tokens from usage dict if not overridden
        cr_tokens = cache_read_tokens if cache_read_tokens is not None else usage.get("cache_read_input_tokens", 0)
        cc_tokens = cache_creation_tokens if cache_creation_tokens is not None else usage.get("cache_creation_input_tokens", 0)

        # Get pricing (full pricing includes cache rates if available)
        full_pricing = self.pricing_registry.get_full_pricing(model)
        input_price_per_1k = full_pricing["input_cost_per_1k_tokens"]
        output_price_per_1k = full_pricing["output_cost_per_1k_tokens"]
        cache_read_price = full_pricing.get("cache_read_cost_per_1k_tokens", 0.0)
        cache_creation_price = full_pricing.get("cache_creation_cost_per_1k_tokens", 0.0)

        # Calculate actual costs
        input_cost = (prompt_tokens / 1000) * input_price_per_1k
        output_cost = (completion_tokens / 1000) * output_price_per_1k
        cr_cost = (cr_tokens / 1000) * cache_read_price
        cc_cost = (cc_tokens / 1000) * cache_creation_price
        total_cost = input_cost + output_cost + cr_cost + cc_cost

        return ActualCostResult(
            model=model,
            actual_input_tokens=prompt_tokens,
            actual_output_tokens=completion_tokens,
            actual_input_cost=input_cost,
            actual_output_cost=output_cost,
            actual_total_cost=total_cost,
            cache_read_tokens=cr_tokens,
            cache_creation_tokens=cc_tokens,
            cache_read_cost=cr_cost,
            cache_creation_cost=cc_cost,
        )

    def calculate_from_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> ActualCostResult:
        """Calculate cost from token counts directly.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
            cache_read_tokens: Number of cache read tokens
            cache_creation_tokens: Number of cache creation tokens

        Returns:
            ActualCostResult with costs
        """
        full_pricing = self.pricing_registry.get_full_pricing(model)
        input_price_per_1k = full_pricing["input_cost_per_1k_tokens"]
        output_price_per_1k = full_pricing["output_cost_per_1k_tokens"]
        cache_read_price = full_pricing.get("cache_read_cost_per_1k_tokens", 0.0)
        cache_creation_price = full_pricing.get("cache_creation_cost_per_1k_tokens", 0.0)

        input_cost = (input_tokens / 1000) * input_price_per_1k
        output_cost = (output_tokens / 1000) * output_price_per_1k
        cr_cost = (cache_read_tokens / 1000) * cache_read_price
        cc_cost = (cache_creation_tokens / 1000) * cache_creation_price
        total_cost = input_cost + output_cost + cr_cost + cc_cost

        return ActualCostResult(
            model=model,
            actual_input_tokens=input_tokens,
            actual_output_tokens=output_tokens,
            actual_input_cost=input_cost,
            actual_output_cost=output_cost,
            actual_total_cost=total_cost,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_cost=cr_cost,
            cache_creation_cost=cc_cost,
        )

    def calculate_error(
        self,
        predicted_cost: float,
        actual_cost: float
    ) -> float:
        """Calculate prediction error percentage.

        Args:
            predicted_cost: Predicted cost
            actual_cost: Actual cost

        Returns:
            Error percentage (positive = overestimated, negative = underestimated)
        """
        return calculate_error_percent(predicted_cost, actual_cost)


def calculate_error_percent(predicted_cost: float, actual_cost: float) -> float:
    """Calculate prediction error percentage (positive = overestimated, negative = underestimated)."""
    if actual_cost == 0:
        return 0.0
    return ((predicted_cost - actual_cost) / actual_cost) * 100
