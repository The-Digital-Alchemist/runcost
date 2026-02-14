"""CLI command to start the CostPlan budget enforcement proxy."""

import click


@click.command()
@click.option("--port", default=8080, type=int, help="Port to listen on (default: 8080)")
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
@click.option(
    "--per-call", "per_call", required=True, type=float,
    help="Max dollars per individual API call"
)
@click.option(
    "--session", "session_budget", required=True, type=float,
    help="Max dollars for the entire proxy session"
)
@click.option(
    "--target-openai", default="https://api.openai.com",
    help="Upstream OpenAI API URL"
)
@click.option(
    "--target-anthropic", default="https://api.anthropic.com",
    help="Upstream Anthropic API URL"
)
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def proxy(port, host, per_call, session_budget, target_openai, target_anthropic, log_level):
    """Start the CostPlan budget enforcement proxy.

    Drop-in economic circuit breaker for any LLM workflow.

    \b
    Claude Code quickstart:
        costplan proxy --per-call 1.00 --session 5.00
        export ANTHROPIC_BASE_URL=http://localhost:8080
        claude  # Budget-enforced!

    \b
    OpenAI quickstart:
        costplan proxy --per-call 0.50 --session 5.00
        export OPENAI_BASE_URL=http://localhost:8080/v1
        python my_agent.py  # Budget-enforced!
    """
    import logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        import uvicorn
    except ImportError:
        click.echo(
            "Error: Proxy dependencies not installed. Install with:\n"
            "  pip install costplan[proxy]\n"
            "  # or: pip install fastapi uvicorn httpx",
            err=True,
        )
        raise SystemExit(1)

    if per_call > session_budget:
        click.echo(
            f"Error: Per-call budget (${per_call:.2f}) cannot exceed session budget (${session_budget:.2f}).\n"
            "  No single call can cost more than the session total.",
            err=True,
        )
        raise SystemExit(1)

    from costplan.proxy.budget_state import ProxyBudgetState
    from costplan.proxy.forwarder import Forwarder
    from costplan.proxy.app import create_app

    budget = ProxyBudgetState(per_call_budget=per_call, session_budget=session_budget)
    forwarder = Forwarder(openai_target=target_openai, anthropic_target=target_anthropic)
    app = create_app(budget=budget, forwarder=forwarder)

    click.echo(f"CostPlan Proxy — LLM Economic Circuit Breaker")
    click.echo(f"  Per-call budget:  ${per_call:.2f}")
    click.echo(f"  Session budget:   ${session_budget:.2f}")
    click.echo(f"  OpenAI target:    {target_openai}")
    click.echo(f"  Anthropic target: {target_anthropic}")
    click.echo(f"  Listening on:     http://{host}:{port}")
    click.echo()
    click.echo(f"  Claude Code:  export ANTHROPIC_BASE_URL=http://{host}:{port}")
    click.echo(f"  OpenAI:       export OPENAI_BASE_URL=http://{host}:{port}/v1")
    click.echo(f"  Dashboard:    http://{host}:{port}/")
    click.echo()

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
