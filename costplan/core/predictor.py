"""Cost prediction engine for LLM requests."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from costplan.core.estimator import TokenEstimator
from costplan.core.pricing import PricingRegistry
from costplan.config.settings import Settings


@dataclass
class PredictionResult:
    """Result of a cost prediction."""

    model: str
    predicted_input_tokens: int
    predicted_output_tokens: int
    predicted_input_cost: float
    predicted_output_cost: float
    predicted_total_cost: float
    confidence_level: str  # "High", "Medium", "Low"
    confidence_percent: Optional[float] = None  # Historical error percentage

    def __repr__(self) -> str:
        return (
            f"PredictionResult(model={self.model}, "
            f"total_cost=${self.predicted_total_cost:.4f}, "
            f"confidence={self.confidence_level})"
        )


def build_prediction_result_from_tokens_and_pricing(
    model: str,
    input_tokens: int,
    output_tokens: int,
    input_price_per_1k: float,
    output_price_per_1k: float,
    confidence_level: str = "Medium",
    confidence_percent: Optional[float] = None,
) -> PredictionResult:
    """Build a PredictionResult from token counts and per-1k pricing. Used by providers."""
    input_cost = (input_tokens / 1000) * input_price_per_1k
    output_cost = (output_tokens / 1000) * output_price_per_1k
    total_cost = input_cost + output_cost
    return PredictionResult(
        model=model,
        predicted_input_tokens=input_tokens,
        predicted_output_tokens=output_tokens,
        predicted_input_cost=input_cost,
        predicted_output_cost=output_cost,
        predicted_total_cost=total_cost,
        confidence_level=confidence_level,
        confidence_percent=confidence_percent,
    )


class CostPredictor:
    """Predicts costs for LLM requests before execution."""

    def __init__(
        self,
        pricing_registry: Optional[PricingRegistry] = None,
        settings: Optional[Settings] = None,
        token_estimator: Optional[TokenEstimator] = None,
    ):
        """Initialize the cost predictor.

        Args:
            pricing_registry: Pricing registry instance (creates default if None)
            settings: Settings instance (creates default if None)
            token_estimator: Token estimator instance (creates default if None)
        """
        self.pricing_registry = pricing_registry or PricingRegistry()
        self.settings = settings or Settings()
        self.token_estimator = token_estimator or TokenEstimator(
            estimation_mode=self.settings.token_estimation_mode
        )
        self._historical_errors = {}  # Cache for historical errors

    def predict(
        self,
        prompt: str,
        model: str,
        output_ratio: Optional[float] = None
    ) -> PredictionResult:
        """Predict cost for a single prompt.

        Args:
            prompt: Input prompt text
            model: Model name
            output_ratio: Override default output token ratio (optional)

        Returns:
            PredictionResult with cost predictions
        """
        # Estimate input tokens
        input_tokens = self.token_estimator.estimate_tokens(prompt, model)

        # Calculate predicted output tokens
        ratio = output_ratio or self.settings.default_output_ratio
        output_tokens = int(input_tokens * ratio)

        # Get pricing
        input_price_per_1k, output_price_per_1k = \
            self.pricing_registry.get_model_pricing(model)

        # Calculate costs
        input_cost = (input_tokens / 1000) * input_price_per_1k
        output_cost = (output_tokens / 1000) * output_price_per_1k
        total_cost = input_cost + output_cost

        # Get confidence level from historical data
        confidence_level, confidence_percent = self._calculate_confidence(model)

        return PredictionResult(
            model=model,
            predicted_input_tokens=input_tokens,
            predicted_output_tokens=output_tokens,
            predicted_input_cost=input_cost,
            predicted_output_cost=output_cost,
            predicted_total_cost=total_cost,
            confidence_level=confidence_level,
            confidence_percent=confidence_percent,
        )

    def predict_from_messages(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        output_ratio: Optional[float] = None
    ) -> PredictionResult:
        """Predict cost for chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            output_ratio: Override default output token ratio (optional)

        Returns:
            PredictionResult with cost predictions
        """
        # Estimate input tokens from messages
        input_tokens = self.token_estimator.estimate_from_messages(messages, model)

        # Calculate predicted output tokens
        ratio = output_ratio or self.settings.default_output_ratio
        output_tokens = int(input_tokens * ratio)

        # Get pricing
        input_price_per_1k, output_price_per_1k = \
            self.pricing_registry.get_model_pricing(model)

        # Calculate costs
        input_cost = (input_tokens / 1000) * input_price_per_1k
        output_cost = (output_tokens / 1000) * output_price_per_1k
        total_cost = input_cost + output_cost

        # Get confidence level
        confidence_level, confidence_percent = self._calculate_confidence(model)

        return PredictionResult(
            model=model,
            predicted_input_tokens=input_tokens,
            predicted_output_tokens=output_tokens,
            predicted_input_cost=input_cost,
            predicted_output_cost=output_cost,
            predicted_total_cost=total_cost,
            confidence_level=confidence_level,
            confidence_percent=confidence_percent,
        )

    def batch_predict(
        self,
        prompts: List[str],
        model: str,
        output_ratio: Optional[float] = None
    ) -> List[PredictionResult]:
        """Predict costs for multiple prompts.

        Args:
            prompts: List of input prompts
            model: Model name
            output_ratio: Override default output token ratio (optional)

        Returns:
            List of PredictionResult objects
        """
        return [
            self.predict(prompt, model, output_ratio)
            for prompt in prompts
        ]

    def _calculate_confidence(self, model: str) -> tuple[str, Optional[float]]:
        """Calculate confidence level based on historical error.

        Args:
            model: Model name

        Returns:
            Tuple of (confidence_level, error_percent)
        """
        # Check if we have historical error data
        error_percent = self._historical_errors.get(model)

        if error_percent is None:
            # No historical data, use default medium confidence
            return "Medium", None

        # Calculate confidence based on error threshold
        threshold = self.settings.confidence_threshold

        if error_percent <= threshold * 0.5:
            return "High", error_percent
        elif error_percent <= threshold:
            return "Medium", error_percent
        else:
            return "Low", error_percent

    def update_historical_error(self, model: str, error_percent: float) -> None:
        """Update historical error for a model.

        Args:
            model: Model name
            error_percent: Average error percentage (0-100)
        """
        self._historical_errors[model] = error_percent

    def set_historical_errors(self, errors: Dict[str, float]) -> None:
        """Set historical errors for multiple models.

        Args:
            errors: Dict mapping model names to error percentages
        """
        self._historical_errors = errors.copy()
