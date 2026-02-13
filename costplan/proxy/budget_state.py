"""In-memory budget state for the proxy server. Asyncio-safe."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


class ProxyBudgetExceeded(Exception):
    """Raised when a proxy request would exceed the budget."""

    def __init__(self, message: str, remaining: float):
        self.remaining = remaining
        super().__init__(message)


@dataclass
class CallRecord:
    """Record of a single proxied call."""
    timestamp: float
    model: str
    actual_cost: float
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class ProxyBudgetState:
    """Async-safe in-memory budget state for the proxy.

    Tracks per-call and per-session budget with an asyncio.Lock for safe
    concurrent access from FastAPI request handlers.
    """

    def __init__(
        self,
        per_call_budget: float,
        session_budget: float,
    ):
        """
        Args:
            per_call_budget: Max dollars per individual API call.
            session_budget: Max dollars for the entire proxy session.
        """
        self._per_call = per_call_budget
        self._session = session_budget
        self._spent = 0.0
        self._locked = False
        self._call_count = 0
        self._history: list[CallRecord] = []
        self._lock = asyncio.Lock()

    async def pre_check(self, estimated_cost: float) -> None:
        """Pre-flight budget check. Raises ProxyBudgetExceeded if over budget.

        Args:
            estimated_cost: Rough estimated cost for the upcoming call.
        """
        async with self._lock:
            if self._locked:
                raise ProxyBudgetExceeded(
                    f"Session budget exhausted (spent ${self._spent:.4f}/{self._session:.4f}). Proxy locked.",
                    remaining=0.0,
                )

            remaining = self._session - self._spent
            if remaining <= 0:
                self._locked = True
                raise ProxyBudgetExceeded(
                    "No remaining session budget.",
                    remaining=0.0,
                )

            if estimated_cost > self._per_call:
                raise ProxyBudgetExceeded(
                    f"Estimated cost ${estimated_cost:.4f} exceeds per-call limit ${self._per_call:.4f}",
                    remaining=remaining,
                )

            if estimated_cost > remaining:
                raise ProxyBudgetExceeded(
                    f"Estimated cost ${estimated_cost:.4f} exceeds remaining budget ${remaining:.4f}",
                    remaining=remaining,
                )

    async def record_actual(
        self,
        actual_cost: float,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        """Record actual cost after a call completes.

        Args:
            actual_cost: The actual cost in dollars.
            model: Model name.
            input_tokens: Actual input tokens.
            output_tokens: Actual output tokens.
            cache_read_tokens: Cache read tokens.
            cache_creation_tokens: Cache creation tokens.
        """
        async with self._lock:
            self._spent += actual_cost
            self._call_count += 1
            self._history.append(CallRecord(
                timestamp=time.time(),
                model=model,
                actual_cost=actual_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
            ))
            if self._spent >= self._session:
                self._locked = True

    async def remaining(self) -> float:
        """Return remaining session budget in dollars."""
        async with self._lock:
            return max(0.0, self._session - self._spent)

    async def reset(self) -> None:
        """Reset session: zero spend, unlock, clear history."""
        async with self._lock:
            self._spent = 0.0
            self._locked = False
            self._call_count = 0
            self._history.clear()

    async def stats(self) -> dict:
        """Return current budget stats."""
        async with self._lock:
            return {
                "per_call_budget": self._per_call,
                "session_budget": self._session,
                "total_spent": round(self._spent, 6),
                "remaining": round(max(0.0, self._session - self._spent), 6),
                "call_count": self._call_count,
                "locked": self._locked,
            }
