"""Budget policy and session tracking for LLM cost control.

Production API: BudgetedLLM — lightweight execution wrapper that guarantees LLM calls
stay within defined economic constraints. Hard per-call and per-session budget,
deterministic BudgetExceededError, dynamic max_tokens from remaining budget.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union

from costplan.core.executor import ExecutionResult
from costplan.core.predictor import (
    PredictionResult,
    build_prediction_result_from_tokens_and_pricing,
)
from costplan.core.calculator import ActualCostResult

if TYPE_CHECKING:
    from costplan.core.provider import BaseProvider


# Epsilon for post-execution assertion (pricing drift, rounding).
BUDGET_ASSERTION_EPSILON = 1e-6


class BudgetExceededError(Exception):
    """Deterministic exception when a call or session would exceed budget. Catch this in agent loops."""

    def __init__(
        self,
        message: str,
        limit_type: str = "budget",
        remaining_budget: Optional[float] = None,
        allowed_budget: Optional[float] = None,
    ):
        self.limit_type = limit_type  # "per_call" | "per_session" | "session_locked"
        self.remaining_budget = remaining_budget
        self.allowed_budget = allowed_budget
        super().__init__(message)


class BudgetViolationError(Exception):
    """Post-execution invariant violation: actual cost exceeded allowed budget (pricing drift, rounding, token drift)."""

    def __init__(
        self,
        message: str,
        actual_cost: float,
        allowed_budget: float,
    ):
        self.actual_cost = actual_cost
        self.allowed_budget = allowed_budget
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
        prediction = build_prediction_result_from_tokens_and_pricing(
            model=model,
            input_tokens=token_pred.input_tokens,
            output_tokens=token_pred.output_tokens,
            input_price_per_1k=input_price,
            output_price_per_1k=output_price,
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
        prediction = build_prediction_result_from_tokens_and_pricing(
            model=model,
            input_tokens=token_pred.input_tokens,
            output_tokens=token_pred.output_tokens,
            input_price_per_1k=input_price,
            output_price_per_1k=output_price,
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


# -----------------------------------------------------------------------------
# BudgetedLLM: production circuit breaker. Hard limits, dynamic max_tokens, lock.
# -----------------------------------------------------------------------------

def _resolve_provider(provider: Union[str, "BaseProvider"], settings: Optional[Any]) -> "BaseProvider":
    """Return BaseProvider from str (via factory) or use as-is."""
    if isinstance(provider, str):
        from costplan.core.factory import create
        from costplan.config.settings import Settings
        return create(provider_name=provider, settings=settings or Settings())
    return provider


class BudgetedLLM:
    """
    Lightweight execution wrapper that guarantees LLM calls stay within economic constraints.
    Hard per-call and per-session budget, deterministic BudgetExceededError, dynamic max_tokens.
    Use in agent loops: response = client.generate(prompt); catch BudgetExceededError.
    """

    def __init__(
        self,
        provider: Union[str, "BaseProvider"],
        model: str,
        per_call_budget: float,
        session_budget: float,
        settings: Optional[Any] = None,
        output_ratio: Optional[float] = None,
    ):
        """
        Args:
            provider: Provider name (str) or BaseProvider instance.
            model: Model id (e.g. "claude-3-5-sonnet-20241022", "gpt-4o").
            per_call_budget: Max dollars per call; enforced before execute.
            session_budget: Max dollars for session; when exceeded, client locks.
            settings: Optional Settings (used when provider is str).
            output_ratio: Optional override for output token prediction (default from settings).
        """
        self._provider = _resolve_provider(provider, settings)
        self._model = model
        self._per_call_budget = per_call_budget
        self._session_budget = session_budget
        self._session_spent = 0.0
        self._locked = False
        self._settings = settings
        self._output_ratio = output_ratio

    def remaining_budget(self) -> float:
        """Remaining session budget in dollars. Production agent loops call this to adapt behavior."""
        return max(0.0, self._session_budget - self._session_spent)

    @property
    def session_spent(self) -> float:
        """Total spent this session in dollars."""
        return self._session_spent

    @property
    def locked(self) -> bool:
        """True when session budget exhausted; generate() will raise until reset or new client."""
        return self._locked

    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> ExecutionResult:
        """
        Execute one call within budget. Injects max_tokens from remaining budget.
        Raises BudgetExceededError on per-call or per-session exceed; on session exhaust, locks.
        """
        if self._locked:
            raise BudgetExceededError(
                f"Session budget exhausted (spent ${self._session_spent:.4f}, limit ${self._session_budget:.4f}). Client locked.",
                limit_type="session_locked",
                remaining_budget=0.0,
            )

        token_pred = self._provider.predict_tokens(
            prompt, self._model, output_ratio=self._output_ratio
        )
        input_price_per_1k, output_price_per_1k = self._provider.get_pricing(self._model)

        predicted_input_cost = (token_pred.input_tokens / 1000) * input_price_per_1k
        predicted_output_cost = (token_pred.output_tokens / 1000) * output_price_per_1k
        predicted_total_cost = predicted_input_cost + predicted_output_cost

        if predicted_total_cost > self._per_call_budget:
            raise BudgetExceededError(
                f"Predicted cost ${predicted_total_cost:.4f} exceeds per-call budget ${self._per_call_budget:.4f}",
                limit_type="per_call",
                remaining_budget=self.remaining_budget(),
                allowed_budget=self._per_call_budget,
            )

        remaining_session_budget = self._session_budget - self._session_spent
        allowed_budget = min(self._per_call_budget, remaining_session_budget)

        if allowed_budget <= 0:
            self._locked = True
            raise BudgetExceededError(
                "No remaining session budget.",
                limit_type="per_session",
                remaining_budget=0.0,
            )

        if predicted_input_cost >= allowed_budget:
            raise BudgetExceededError(
                f"Predicted input cost ${predicted_input_cost:.4f} >= allowed budget ${allowed_budget:.4f}",
                limit_type="per_session",
                remaining_budget=remaining_session_budget,
            )

        allowed_output_budget = allowed_budget - predicted_input_cost
        output_cost_per_token = output_price_per_1k / 1000.0
        if output_cost_per_token <= 0:
            max_output_tokens = 8192
        else:
            max_output_tokens = int(allowed_output_budget / output_cost_per_token)
        max_output_tokens = max(1, min(max_output_tokens, 16_384))

        execution = self._provider.execute(
            prompt,
            self._model,
            temperature=temperature,
            max_tokens=max_output_tokens,
            **kwargs,
        )

        if not execution.success:
            return execution

        actual = _actual_from_usage_and_pricing(
            self._model,
            execution.usage,
            input_price_per_1k,
            output_price_per_1k,
        )

        # Post-execution assertion: protect against pricing changes, token drift, rounding.
        if actual.actual_total_cost > allowed_budget + BUDGET_ASSERTION_EPSILON:
            raise BudgetViolationError(
                f"Budget invariant violated: actual cost ${actual.actual_total_cost:.6f} > allowed ${allowed_budget:.6f}",
                actual_cost=actual.actual_total_cost,
                allowed_budget=allowed_budget,
            )

        self._session_spent += actual.actual_total_cost

        if self._session_spent >= self._session_budget:
            self._locked = True

        return execution
