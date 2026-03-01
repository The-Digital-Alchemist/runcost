"""In-memory budget state for the proxy server. Asyncio-safe."""

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from costplan.proxy.persistent_store import PersistentCallStore


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

    When reset_every_seconds is set, the budget auto-resets when the
    elapsed time since last_reset_at exceeds that interval (e.g. daily).

    When state_store and budget_window_seconds are set, uses a rolling
    window over persisted data (survives proxy restarts).
    """

    def __init__(
        self,
        per_call_budget: float,
        session_budget: float,
        reset_every_seconds: Optional[float] = None,
        state_store: Optional["PersistentCallStore"] = None,
        budget_window_seconds: Optional[float] = None,
    ):
        """
        Args:
            per_call_budget: Max dollars per individual API call.
            session_budget: Max dollars for the entire proxy session.
            reset_every_seconds: If set, auto-reset budget when this many
                seconds have elapsed since last reset (e.g. 86400 for 24h).
            state_store: If set, persist call records to SQLite.
            budget_window_seconds: If set with state_store, use rolling window
                over persisted data (spent = sum of costs in last N seconds).
        """
        self._per_call = per_call_budget
        self._session = session_budget
        self._reset_every = reset_every_seconds
        self._store = state_store
        self._budget_window = budget_window_seconds
        self._last_reset_at = time.time()
        self._spent = 0.0
        self._locked = False
        self._call_count = 0
        self._history: list[CallRecord] = []
        self._lock = asyncio.Lock()

        if self._budget_window is not None and self._store is None:
            raise ValueError("budget_window_seconds requires state_store")

    def _use_rolling_window(self) -> bool:
        return self._store is not None and self._budget_window is not None

    def _should_reset(self) -> bool:
        """Return True if enough time has passed to trigger auto-reset."""
        if self._reset_every is None or self._use_rolling_window():
            return False
        return (time.time() - self._last_reset_at) >= self._reset_every

    def _do_reset(self) -> None:
        """Reset budget state. Caller must hold _lock."""
        self._spent = 0.0
        self._locked = False
        self._call_count = 0
        self._history.clear()
        self._last_reset_at = time.time()

    async def _maybe_reset(self) -> None:
        """Reset budget if reset_every interval has elapsed. Caller must hold _lock."""
        if self._should_reset():
            self._do_reset()

    def _get_spent(self) -> float:
        """Return current spent amount (from store if rolling window, else in-memory)."""
        if self._use_rolling_window() and self._store:
            return self._store.spent_in_window(self._budget_window)
        return self._spent

    async def pre_check(self, estimated_cost: float) -> None:
        """Pre-flight budget check. Raises ProxyBudgetExceeded if over budget.

        Also performs auto-reset if reset_every_seconds has elapsed (not used
        with rolling window).

        Args:
            estimated_cost: Rough estimated cost for the upcoming call.
        """
        async with self._lock:
            await self._maybe_reset()
            spent = self._get_spent()
            remaining = self._session - spent

            if not self._use_rolling_window() and self._locked:
                raise ProxyBudgetExceeded(
                    f"Session budget exhausted (spent ${self._spent:.4f}/{self._session:.4f}). Proxy locked.",
                    remaining=0.0,
                )

            if remaining <= 0:
                if not self._use_rolling_window():
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
        now = time.time()
        async with self._lock:
            if self._store:
                self._store.insert(
                    timestamp=now,
                    model=model,
                    actual_cost=actual_cost,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_creation_tokens=cache_creation_tokens,
                )
            if not self._use_rolling_window():
                self._spent += actual_cost
                if self._spent >= self._session:
                    self._locked = True
            self._call_count += 1
            self._history.append(CallRecord(
                timestamp=now,
                model=model,
                actual_cost=actual_cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
            ))

    async def remaining(self) -> float:
        """Return remaining session budget in dollars."""
        async with self._lock:
            spent = self._get_spent()
            return max(0.0, self._session - spent)

    async def reset(self) -> None:
        """Reset session: zero spend, unlock, clear history."""
        async with self._lock:
            if self._use_rolling_window() and self._store:
                self._store.clear()
                self._locked = False
                self._call_count = 0
                self._history.clear()
            else:
                self._do_reset()

    async def stats(self) -> dict:
        """Return current budget stats."""
        async with self._lock:
            spent = self._get_spent()
            remaining = max(0.0, self._session - spent)
            out = {
                "per_call_budget": self._per_call,
                "session_budget": self._session,
                "total_spent": round(spent, 6),
                "remaining": round(remaining, 6),
                "call_count": self._call_count,
                "locked": self._locked if not self._use_rolling_window() else (spent >= self._session),
            }
            if self._reset_every is not None:
                out["reset_every_seconds"] = self._reset_every
                out["last_reset_at"] = self._last_reset_at
                out["next_reset_at"] = self._last_reset_at + self._reset_every
            if self._budget_window is not None:
                out["budget_window_seconds"] = self._budget_window
            return out
