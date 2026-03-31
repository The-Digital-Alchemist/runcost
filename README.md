# CostPlan

**Deterministic economics for probabilistic systems.**

CostPlan is a deterministic budget enforcement layer for production LLM and agent systems. It prevents runaway spend by enforcing hard per-call and per-session ceilings before and after execution.

*v0.1 — Deterministic LLM Budget Enforcement for Agent Systems*

Two integration paths. No complex setup. Just enforcement.

---

## Prerequisites

- **Python 3.10+**
- **API key**: Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment (or in the shell where you run the wrapped command). CostPlan does not store keys; the proxy forwards them to the provider.

---

## Claude Code Quickstart

```bash
pip install costplan
costplan wrap claude
```

That's it. The proxy starts, Claude Code launches with budget enforcement ($1/call, $10/session), and when you exit you get a cost summary. Open [http://localhost:8080](http://localhost:8080) while it's running to see remaining budget and reset the session.

Override defaults if needed:

```bash
costplan wrap --per-call 0.50 --session 5.00 claude
```

---

## OpenClaw Quickstart

OpenClaw is a 24/7 AI gateway (WhatsApp, Telegram, Discord, etc.). Unlike Claude Code, it runs indefinitely — "fire and forget." CostPlan integrates by running the proxy in front of OpenClaw's LLM calls.

```bash
# Terminal 1: Start CostPlan proxy (long-running, $1/call, $50/session, resets daily)
costplan openclaw

# Terminal 2: Run OpenClaw with budget enforcement
export ANTHROPIC_BASE_URL=http://localhost:8080
openclaw gateway --port 18789
```

Override defaults if needed:

```bash
costplan openclaw --per-call 2.00 --session 100.00 --reset-every 7d
```

**OpenClaw config alternative** (custom provider in `~/.openclaw/openclaw.json`):

```json5
{
  models: {
    mode: "merge",
    providers: {
      costplan: {
        baseUrl: "http://localhost:8080",
        apiKey: "${ANTHROPIC_API_KEY}",
        api: "anthropic-messages",
        models: [
          { id: "claude-sonnet-4-20250514", name: "Claude Sonnet" },
          { id: "claude-opus-4-20250514", name: "Claude Opus" },
        ],
      },
    },
  },
  agents: { defaults: { model: { primary: "costplan/claude-sonnet-4-20250514" } } },
}
```

**Rolling window (survives restarts)**: Use `--state-db` and `--budget-window` for a persistent rolling budget that survives proxy restarts:

```bash
costplan proxy --per-call 2.00 --session 20.00 --state-db ~/.costplan/budget.db --budget-window 24h
```

<details>
<summary>Or use two terminals (manual proxy mode)</summary>

```bash
# Terminal 1: Start the circuit breaker
costplan proxy

# Terminal 2: Use Claude Code with budget enforcement
export ANTHROPIC_BASE_URL=http://localhost:8080
claude
```

</details>

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
pip install costplan
costplan proxy
```

Then point your client at it:

```bash
# OpenAI clients
export OPENAI_BASE_URL=http://localhost:8080/v1

# Anthropic clients / Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8080
```

Defaults: $1.00/call, $10.00/session. Override with `--per-call` and `--session`.

### Proxy Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard (budget status page) |
| `GET /health` | Health check |
| `GET /v1/budget` | Query remaining budget |
| `POST /v1/budget/reset` | Reset session budget |
| `POST /v1/chat/completions` | OpenAI-compatible proxy |
| `POST /v1/messages` | Anthropic-compatible proxy |

### What the proxy does

- **Pre-check**: Estimates input cost before forwarding. Rejects with `429` if over budget.
- **Streaming**: Forwards SSE chunks in real-time (zero-latency pass-through). Extracts usage from `message_start` and `message_delta` events.
- **Cache-aware**: Tracks Anthropic's `cache_read_input_tokens` and `cache_creation_input_tokens` at their respective price points.
- **Post-track**: Records actual cost after each call. Locks the session when budget is exhausted.
- **Headers**: Adds `X-CostPlan-Budget-Remaining` and `X-CostPlan-Cost` to every response.

**Concurrency**: Budget state is protected by an asyncio lock; pre-check and record are atomic. Safe for parallel agents, multi-request streams, and concurrent clients hitting the same proxy.

---

## How It Works

```
                    ┌─────────────────────────────────────────┐
   Client  ────────►│  CostPlan Proxy                         │
 (Claude,           │  ┌─────────────────┐  ┌──────────────┐ │
  OpenClaw,         │  │ budget engine   │  │ reservation  │ │
  agents)           │  │ (pre/post check)│  │ ledger       │ │
                    │  └─────────────────┘  └──────────────┘ │
                    └──────────────────────┬─────────────────┘
                                           │
                                           ▼
                                    Provider API
                               (Anthropic, OpenAI)
```

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

## Known Limitations

| Limitation | Notes |
|------------|-------|
| **Static pricing table** | Model pricing is loaded from `pricing.json`. Auto-sync with provider rate cards is planned. |
| **Client routing required** | Proxy enforcement assumes clients route LLM traffic via `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`. Direct-to-provider calls bypass CostPlan. |
| **Single-node scope** | Distributed multi-instance coordination is not supported. One proxy = one budget envelope. |

---

## Install

```bash
pip install costplan

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
# Wrap any command with budget enforcement (sensible defaults, just works)
costplan wrap claude
costplan wrap python my_agent.py

# Customize budgets when needed
costplan wrap --per-call 0.50 --session 5.00 claude

# Start budget enforcement proxy (manual two-terminal mode)
costplan proxy

# OpenClaw integration: proxy with auto-reset for 24/7 VA use
costplan openclaw

# Predict cost (no API call)
costplan predict "Your prompt" --provider openai --model gpt-4o

# Execute and compare predicted vs actual
costplan run "Your prompt" --provider anthropic --model claude-sonnet-4-20250514

# View history
costplan history
costplan stats
```

---

## License

MIT
