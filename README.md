# CostPlan

**CostPlan enforces deterministic per-call and per-session economic limits on LLM executions.**

Not just prediction—hard limits, dynamic `max_tokens` from remaining budget, and post-execution assertion so actual cost never exceeds what you allowed.

---

## Install

```bash
uv pip install -e .
# or: pip install -e .
```

---

## Flagship API: BudgetedLLM

Use `BudgetedLLM` when you need guaranteed cost bounds in code (e.g. agent loops). One client, one call pattern.

```python
from costplan import BudgetedLLM, BudgetExceededError, BudgetViolationError

llm = BudgetedLLM(
    provider="openai",
    model="gpt-4o-mini",
    per_call_budget=0.01,
    session_budget=0.10,
)

response = llm.generate("Reply in one sentence.")
print(response.response_text)
print("Remaining budget:", llm.remaining_budget())
```

- **Per-call limit:** predicted cost is checked before execution; if it would exceed `per_call_budget`, `BudgetExceededError` is raised.
- **Session limit:** total spend is tracked; when `session_budget` is exhausted, the client **locks** and further `generate()` calls raise until you create a new client or reset.
- **Dynamic `max_tokens`:** derived from remaining session budget and output price so the call stays within budget.
- **Post-execution assertion:** if actual cost exceeds allowed (e.g. pricing drift), `BudgetViolationError` is raised.

Catch `BudgetExceededError` in agent loops; use `llm.remaining_budget()` to adapt behavior.

Supports **OpenAI** and **Anthropic** via the `provider` argument.

---

## CLI (secondary)

Predict cost or run a single prompt with optional budget checks. Provider-aware: use `--provider openai` or `--provider anthropic`.

```bash
# Predict (no API call for OpenAI; Anthropic may call count_tokens if key set)
export OPENAI_API_KEY=your_key
costplan predict "Your prompt" --provider openai --model gpt-4o-mini

export ANTHROPIC_API_KEY=your_key
costplan predict "Your prompt" --provider anthropic --model claude-3-5-sonnet-20241022

# Execute and compare predicted vs actual
costplan run "Your prompt"
costplan run "Your prompt" --provider anthropic --model claude-3-5-sonnet-20241022
```

`costplan run` accepts `--per-call` and `--per-session` (dollars) to enforce budgets from the CLI.
