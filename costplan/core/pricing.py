"""Pricing Registry for managing model pricing information."""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple


class PricingNotFoundError(Exception):
    """Raised when pricing information is not found for a model."""
    pass


def _looks_like_provider_map(data: dict) -> bool:
    """True if data is { "openai": { "gpt-4": {...} }, ... } (multi-provider)."""
    if not isinstance(data, dict) or not data:
        return False
    first_val = next(iter(data.values()))
    if not isinstance(first_val, dict) or not first_val:
        return False
    # Skip the model_aliases key when checking structure
    for key, val in data.items():
        if key == "model_aliases":
            continue
        if not isinstance(val, dict):
            return False
        first_model = next(iter(val.values()), None)
        if first_model is not None:
            return isinstance(first_model, dict) and "input_cost_per_1k_tokens" in first_model
    return False


class PricingRegistry:
    """Registry for one provider's model pricing. Loads from a slice of pricing.json."""

    def __init__(
        self,
        pricing_file_path: Optional[str] = None,
        provider_name: Optional[str] = None,
    ):
        """Initialize the pricing registry.

        Args:
            pricing_file_path: Path to the pricing JSON file. If None, uses default config/pricing.json.
            provider_name: Top-level key in the JSON (e.g. "openai", "anthropic"). If the file
                is multi-provider, this selects the slice. If None, the file is treated as flat
                (single-provider format).
        """
        self._pricing_data: Dict[str, Dict[str, float]] = {}
        self._aliases: Dict[str, str] = {}
        self._pricing_file_path = pricing_file_path
        self._provider_name = provider_name
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
            package_dir = Path(__file__).parent.parent
            pricing_path = package_dir / "config" / "pricing.json"

        if not pricing_path.exists():
            raise FileNotFoundError(f"Pricing file not found: {pricing_path}")

        with open(pricing_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load model aliases if present
        if isinstance(data, dict) and "model_aliases" in data:
            self._aliases = data["model_aliases"]

        # Multi-provider: {"openai": { "gpt-4": {...} }, "anthropic": {...} }
        if _looks_like_provider_map(data):
            provider = self._provider_name
            if provider is None:
                provider = "openai"  # default for legacy callers (CostPredictor, etc.)
            if provider not in data:
                raise FileNotFoundError(
                    f"Provider '{provider}' not found in {pricing_path}. "
                    f"Top-level keys: {[k for k in data.keys() if k != 'model_aliases']}"
                )
            self._pricing_data = data[provider]
        else:
            # Flat format (backward compat): { "gpt-4": {...} }
            self._pricing_data = data

    def _resolve_model(self, model_name: str) -> str:
        """Resolve model aliases to canonical names."""
        return self._aliases.get(model_name, model_name)

    def get_model_pricing(self, model_name: str) -> Tuple[float, float]:
        """Get pricing information for a specific model.

        Args:
            model_name: Name of the model (e.g., "gpt-4", "claude-sonnet-4")

        Returns:
            Tuple of (input_cost_per_1k_tokens, output_cost_per_1k_tokens)

        Raises:
            PricingNotFoundError: If pricing for the model is not found.
        """
        resolved = self._resolve_model(model_name)
        if resolved not in self._pricing_data:
            raise PricingNotFoundError(
                f"Pricing not found for model: {model_name}. "
                f"Available models: {', '.join(self.list_supported_models())}"
            )

        pricing = self._pricing_data[resolved]
        return (
            pricing["input_cost_per_1k_tokens"],
            pricing["output_cost_per_1k_tokens"]
        )

    def get_full_pricing(self, model_name: str) -> Dict[str, float]:
        """Get full pricing dict for a model, including cache pricing if available.

        Args:
            model_name: Name of the model (supports aliases)

        Returns:
            Dict with keys: input_cost_per_1k_tokens, output_cost_per_1k_tokens,
            and optionally cache_read_cost_per_1k_tokens, cache_creation_cost_per_1k_tokens.

        Raises:
            PricingNotFoundError: If pricing for the model is not found.
        """
        resolved = self._resolve_model(model_name)
        if resolved not in self._pricing_data:
            raise PricingNotFoundError(
                f"Pricing not found for model: {model_name}. "
                f"Available models: {', '.join(self.list_supported_models())}"
            )
        return dict(self._pricing_data[resolved])

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
        output_cost_per_1k: float,
        cache_read_cost_per_1k: Optional[float] = None,
        cache_creation_cost_per_1k: Optional[float] = None,
    ) -> None:
        """Add or update pricing for a model.

        Args:
            model_name: Name of the model
            input_cost_per_1k: Cost per 1000 input tokens
            output_cost_per_1k: Cost per 1000 output tokens
            cache_read_cost_per_1k: Cost per 1000 cache read tokens (optional)
            cache_creation_cost_per_1k: Cost per 1000 cache creation tokens (optional)
        """
        pricing: Dict[str, float] = {
            "input_cost_per_1k_tokens": input_cost_per_1k,
            "output_cost_per_1k_tokens": output_cost_per_1k,
        }
        if cache_read_cost_per_1k is not None:
            pricing["cache_read_cost_per_1k_tokens"] = cache_read_cost_per_1k
        if cache_creation_cost_per_1k is not None:
            pricing["cache_creation_cost_per_1k_tokens"] = cache_creation_cost_per_1k
        self._pricing_data[model_name] = pricing

    def has_model(self, model_name: str) -> bool:
        """Check if pricing exists for a model.

        Args:
            model_name: Name of the model (supports aliases)

        Returns:
            True if pricing exists, False otherwise.
        """
        resolved = self._resolve_model(model_name)
        return resolved in self._pricing_data
