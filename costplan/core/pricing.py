"""Pricing Registry for managing model pricing information."""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


class PricingNotFoundError(Exception):
    """Raised when pricing information is not found for a model."""
    pass


class PricingRegistry:
    """Registry for storing and retrieving model pricing information."""

    def __init__(self, pricing_file_path: Optional[str] = None):
        """Initialize the pricing registry.

        Args:
            pricing_file_path: Path to the pricing JSON file. If None, uses default.
        """
        self._pricing_data: Dict[str, Dict[str, float]] = {}
        self._pricing_file_path = pricing_file_path
        self.load_pricing()

    def load_pricing(self) -> None:
        """Load pricing data from JSON file.

        Raises:
            FileNotFoundError: If the pricing file doesn't exist.
            json.JSONDecodeError: If the pricing file is invalid JSON.
        """
        if self._pricing_file_path:
            pricing_path = Path(self._pricing_file_path)
        else:
            # Default to the package's config directory
            package_dir = Path(__file__).parent.parent
            pricing_path = package_dir / "config" / "pricing.json"

        if not pricing_path.exists():
            raise FileNotFoundError(f"Pricing file not found: {pricing_path}")

        with open(pricing_path, "r", encoding="utf-8") as f:
            self._pricing_data = json.load(f)

    def get_model_pricing(self, model_name: str) -> Tuple[float, float]:
        """Get pricing information for a specific model.

        Args:
            model_name: Name of the model (e.g., "gpt-4", "gpt-3.5-turbo")

        Returns:
            Tuple of (input_cost_per_1k_tokens, output_cost_per_1k_tokens)

        Raises:
            PricingNotFoundError: If pricing for the model is not found.
        """
        if model_name not in self._pricing_data:
            raise PricingNotFoundError(
                f"Pricing not found for model: {model_name}. "
                f"Available models: {', '.join(self.list_supported_models())}"
            )

        pricing = self._pricing_data[model_name]
        return (
            pricing["input_cost_per_1k_tokens"],
            pricing["output_cost_per_1k_tokens"]
        )

    def list_supported_models(self) -> list[str]:
        """List all models with pricing information.

        Returns:
            List of model names.
        """
        return sorted(self._pricing_data.keys())

    def add_model_pricing(
        self,
        model_name: str,
        input_cost_per_1k: float,
        output_cost_per_1k: float
    ) -> None:
        """Add or update pricing for a model.

        Args:
            model_name: Name of the model
            input_cost_per_1k: Cost per 1000 input tokens
            output_cost_per_1k: Cost per 1000 output tokens
        """
        self._pricing_data[model_name] = {
            "input_cost_per_1k_tokens": input_cost_per_1k,
            "output_cost_per_1k_tokens": output_cost_per_1k
        }

    def has_model(self, model_name: str) -> bool:
        """Check if pricing exists for a model.

        Args:
            model_name: Name of the model

        Returns:
            True if pricing exists, False otherwise.
        """
        return model_name in self._pricing_data
