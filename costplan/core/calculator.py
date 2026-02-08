"""Actual cost calculation from LLM execution results."""

from dataclasses import dataclass
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

    def __repr__(self) -> str:
        return (
            f"ActualCostResult(model={self.model}, "
            f"total_cost=${self.actual_total_cost:.4f}, "
            f"tokens={self.actual_input_tokens}+{self.actual_output_tokens})"
        )


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
        model: str
    ) -> ActualCostResult:
        """Calculate actual cost from usage data.

        Args:
            usage: Usage dict from LLM response (with prompt_tokens, completion_tokens)
            model: Model name

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

        # Get pricing
        input_price_per_1k, output_price_per_1k = \
            self.pricing_registry.get_model_pricing(model)

        # Calculate actual costs
        input_cost = (prompt_tokens / 1000) * input_price_per_1k
        output_cost = (completion_tokens / 1000) * output_price_per_1k
        total_cost = input_cost + output_cost

        return ActualCostResult(
            model=model,
            actual_input_tokens=prompt_tokens,
            actual_output_tokens=completion_tokens,
            actual_input_cost=input_cost,
            actual_output_cost=output_cost,
            actual_total_cost=total_cost,
        )

    def calculate_from_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str
    ) -> ActualCostResult:
        """Calculate cost from token counts directly.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name

        Returns:
            ActualCostResult with costs
        """
        # Get pricing
        input_price_per_1k, output_price_per_1k = \
            self.pricing_registry.get_model_pricing(model)

        # Calculate costs
        input_cost = (input_tokens / 1000) * input_price_per_1k
        output_cost = (output_tokens / 1000) * output_price_per_1k
        total_cost = input_cost + output_cost

        return ActualCostResult(
            model=model,
            actual_input_tokens=input_tokens,
            actual_output_tokens=output_tokens,
            actual_input_cost=input_cost,
            actual_output_cost=output_cost,
            actual_total_cost=total_cost,
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
        if actual_cost == 0:
            return 0.0

        error = ((predicted_cost - actual_cost) / actual_cost) * 100
        return error
