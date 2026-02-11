"""Budget policy and session tracking for LLM cost control."""

from typing import Optional, List, Dict, Any, TYPE_CHECKING

from costplan.core.executor import ExecutionResult
from costplan.core.predictor import PredictionResult
from costplan.core.calculator import ActualCostResult

if TYPE_CHECKING:
    from costplan.core.provider import BaseProvider


class BudgetExceededError(Exception):
    """Raised when a call or session would exceed the configured budget."""

    def __init__(self, message: str, limit_type: str = "budget"):
        self.limit_type = limit_type  # "per_call" or "per_session"
        super().__init__(message)


class BudgetPolicy:
    """Limits for cost per call and/or per session."""

    def __init__(
        self,
        per_call: Optional[float] = None,
        per_session: Optional[float] = None,
    ):
        """Initialize budget policy.

        Args:
            per_call: Maximum cost allowed for a single call (dollars). None = no limit.
            per_session: Maximum total cost allowed for the session (dollars). None = no limit.
        """
        self.per_call = per_call
        self.per_session = per_session


class BudgetSession:
    """Tracks total spend for the current session."""

    def __init__(self) -> None:
        self.total_spent: float = 0.0

    def reset(self) -> None:
        """Reset session spend to zero."""
        self.total_spent = 0.0


def _prediction_from_tokens_and_pricing(
    model: str,
    input_tokens: int,
    output_tokens: int,
    input_price_per_1k: float,
    output_price_per_1k: float,
) -> PredictionResult:
    """Build PredictionResult from token counts and pricing."""
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
        confidence_level="Medium",
        confidence_percent=None,
    )


def _actual_from_usage_and_pricing(
    model: str,
    usage: Dict[str, int],
    input_price_per_1k: float,
    output_price_per_1k: float,
) -> ActualCostResult:
    """Build ActualCostResult from usage dict and pricing."""
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
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


class BudgetedClient:
    """Uses a BaseProvider with optional budget checks. Decoupled from any specific LLM vendor."""

    def __init__(
        self,
        provider: "BaseProvider",
        policy: Optional[BudgetPolicy] = None,
        session: Optional[BudgetSession] = None,
    ):
        """Initialize the budgeted client.

        Args:
            provider: LLM provider (OpenAI, Anthropic, etc.) implementing BaseProvider
            policy: Optional budget limits (per_call, per_session). None = no limits.
            session: Optional session to track spend. If policy is set and session is None, one is created.
        """
        self.provider = provider
        self.policy = policy
        self.session = session if session is not None else (BudgetSession() if policy else None)

    def _check_budget(self, predicted_cost: float) -> None:
        """Raise BudgetExceededError if predicted cost would exceed policy."""
        if self.policy is None or self.session is None:
            return
        if self.policy.per_call is not None and predicted_cost > self.policy.per_call:
            raise BudgetExceededError(
                f"Predicted cost ${predicted_cost:.4f} exceeds per-call limit ${self.policy.per_call:.4f}",
                limit_type="per_call",
            )
        if self.policy.per_session is not None:
            would_be = self.session.total_spent + predicted_cost
            if would_be > self.policy.per_session:
                raise BudgetExceededError(
                    f"Predicted cost ${predicted_cost:.4f} would exceed session limit "
                    f"${self.policy.per_session:.4f} (current session: ${self.session.total_spent:.4f})",
                    limit_type="per_session",
                )

    def execute(
        self,
        prompt: str,
        model: str,
        output_ratio: Optional[float] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[PredictionResult, ExecutionResult, ActualCostResult]:
        """Predict cost via provider, enforce budget, execute, then update session with actual cost.

        Returns:
            (prediction, execution_result, actual_cost)
        """
        token_pred = self.provider.predict_tokens(prompt, model, output_ratio=output_ratio)
        input_price, output_price = self.provider.get_pricing(model)
        prediction = _prediction_from_tokens_and_pricing(
            model,
            token_pred.input_tokens,
            token_pred.output_tokens,
            input_price,
            output_price,
        )
        predicted_cost = prediction.predicted_total_cost

        self._check_budget(predicted_cost)

        execution = self.provider.execute(
            prompt, model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        if not execution.success:
            return prediction, execution, ActualCostResult(
                model=model,
                actual_input_tokens=0,
                actual_output_tokens=0,
                actual_input_cost=0.0,
                actual_output_cost=0.0,
                actual_total_cost=0.0,
            )

        actual = _actual_from_usage_and_pricing(
            model, execution.usage, input_price, output_price
        )
        if self.session is not None:
            self.session.total_spent += actual.actual_total_cost

        return prediction, execution, actual

    def execute_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: str,
        output_ratio: Optional[float] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[PredictionResult, ExecutionResult, ActualCostResult]:
        """Same as execute but with chat messages. Predicts from combined message text."""
        prompt_for_prediction = "\n".join(
            m.get("content", "") for m in messages if m.get("content")
        )
        token_pred = self.provider.predict_tokens(
            prompt_for_prediction, model, output_ratio=output_ratio
        )
        input_price, output_price = self.provider.get_pricing(model)
        prediction = _prediction_from_tokens_and_pricing(
            model,
            token_pred.input_tokens,
            token_pred.output_tokens,
            input_price,
            output_price,
        )
        predicted_cost = prediction.predicted_total_cost

        self._check_budget(predicted_cost)

        execution = self.provider.execute_with_messages(
            messages, model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

        if not execution.success:
            return prediction, execution, ActualCostResult(
                model=model,
                actual_input_tokens=0,
                actual_output_tokens=0,
                actual_input_cost=0.0,
                actual_output_cost=0.0,
                actual_total_cost=0.0,
            )

        actual = _actual_from_usage_and_pricing(
            model, execution.usage, input_price, output_price
        )
        if self.session is not None:
            self.session.total_spent += actual.actual_total_cost

        return prediction, execution, actual
