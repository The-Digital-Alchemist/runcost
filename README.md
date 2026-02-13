# CostPlan

**Deterministic economics for probabilistic systems.**

CostPlan is an LLM Economic Circuit Breaker. It enforces hard per-call and per-session budget limits on any LLM workflow — so your agent loop doesn't blow up, your CI job doesn't burn $800, and your automation is safe to deploy.

Two integration paths. No complex setup. Just enforcement.

---

## Claude Code Quickstart

Claude Code is the single largest money sink in AI tooling. CostPlan stops the bleed:

```bash
pip install costplan[proxy]

# Terminal 1: Start the circuit breaker
costplan proxy --per-call 1.00 --session 5.00

# Terminal 2: Use Claude Code with budget enforcement
export ANTHROPIC_BASE_URL=http://localhost:8080
claude  # All API calls now budget-enforced. $5 session cap.
```

When the budget runs out, the proxy returns `429` and Claude Code stops. That's it.

---

## Path A: SDK

Drop-in budget enforcement for Python code. Thread-safe. Async-ready.

```python
from costplan import BudgetedLLM, BudgetExceededError

llm = BudgetedLLM(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    per_call_budget=0.50,
    session_budget=5.00,
)

try:
    response = llm.generate("Explain quantum computing in one paragraph.")
    print(response.response_text)
    print(f"Remaining: ${llm.remaining_budget():.2f}")
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
```

### Chat Messages

```python
messages = [
    {"role": "user", "content": "Write a Python function to sort a list."},
    {"role": "assistant", "content": "def sort_list(lst): ..."},
    {"role": "user", "content": "Now add type hints."},
]
response = llm.generate_with_messages(messages)
```

### Async

```python
from costplan import AsyncBudgetedLLM

llm = AsyncBudgetedLLM(
    provider="openai",
    model="gpt-4o",
    per_call_budget=0.10,
    session_budget=2.00,
)

response = await llm.generate("Hello")
```

### Budget Warning Callback

```python
def on_warning(spent, remaining, total):
    print(f"WARNING: ${spent:.2f} spent, ${remaining:.2f} remaining")

llm = BudgetedLLM(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    per_call_budget=0.50,
    session_budget=5.00,
    on_budget_warning=on_warning,
    warning_threshold=0.8,  # Fire at 80% spent
)
```

### Reset

```python
llm.reset()  # Zero spend, unlock, start fresh
```

---

## Path B: Proxy

Transparent HTTP proxy that enforces budgets on any LLM client. Works with any language, any framework.

```bash
pip install costplan[proxy]

costplan proxy \
  --per-call 1.00 \
  --session 10.00 \
  --port 8080
```

Then point your client at it:

```bash
# OpenAI clients
export OPENAI_BASE_URL=http://localhost:8080/v1

# Anthropic clients / Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8080
```

### Proxy Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible proxy |
| `POST /v1/messages` | Anthropic-compatible proxy |
| `GET /v1/budget` | Query remaining budget |
| `POST /v1/budget/reset` | Reset session budget |
| `GET /health` | Health check |

### What the proxy does

- **Pre-check**: Estimates input cost before forwarding. Rejects with `429` if over budget.
- **Streaming**: Forwards SSE chunks in real-time (zero-latency pass-through). Extracts usage from `message_start` and `message_delta` events.
- **Cache-aware**: Tracks Anthropic's `cache_read_input_tokens` and `cache_creation_input_tokens` at their respective price points.
- **Post-track**: Records actual cost after each call. Locks the session when budget is exhausted.
- **Headers**: Adds `X-CostPlan-Budget-Remaining` and `X-CostPlan-Cost` to every response.

---

## How It Works

```
LLMs are non-deterministic, recursive, parallelizable, capable of runaway loops.
Billing is deterministic, real, enforced, financially consequential.

CostPlan closes that gap.
```

- **Per-call limit**: Predicted cost is checked before execution. If it would exceed `per_call_budget`, `BudgetExceededError` is raised (SDK) or `429` is returned (proxy).
- **Session limit**: Total spend is tracked. When `session_budget` is exhausted, the client locks.
- **Dynamic max_tokens**: Output token limit is derived from remaining budget so the call stays within bounds.
- **Post-execution assertion**: If actual cost exceeds allowed budget (pricing drift, rounding), `BudgetViolationError` is raised.

---

## Install

```bash
# Core SDK only
pip install costplan

# With proxy server
pip install costplan[proxy]

# Development
pip install -e ".[dev]"
```

---

## Supported Providers

| Provider | Models | Cache Pricing |
|---|---|---|
| **Anthropic** | claude-sonnet-4, claude-opus-4, claude-sonnet-4.5, claude-haiku-4.5, claude-3.5-sonnet, claude-3-opus, + more | Yes |
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1-preview, o1-mini, + more | No |

---

## CLI

```bash
# Predict cost (no API call)
costplan predict "Your prompt" --provider openai --model gpt-4o

# Execute and compare predicted vs actual
costplan run "Your prompt" --provider anthropic --model claude-sonnet-4-20250514

# Start budget enforcement proxy
costplan proxy --per-call 1.00 --session 5.00

# View history
costplan history
costplan stats
```

---

## License

MIT
