"""CLI command to start the CostPlan budget enforcement proxy."""

import logging

import click
import uvicorn

from costplan.utils.helpers import parse_duration_seconds


def _parse_reset_every(_ctx, _param, value):
    """Click callback to parse --reset-every duration."""
    if not value:
        return None
    try:
        return parse_duration_seconds(value)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e


@click.command()
@click.option("--port", default=8080, type=int, help="Port to listen on (default: 8080)")
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
@click.option(
    "--per-call",
    "per_call",
    default=1.00,
    type=float,
    help="Max dollars per individual API call (default: $1.00)",
)
@click.option(
    "--session",
    "session_budget",
    default=10.00,
    type=float,
    help="Max dollars for the entire proxy session (default: $10.00)",
)
@click.option(
    "--reset-every",
    default=None,
    callback=_parse_reset_every,
    help="Auto-reset budget every N (e.g. 24h, 7d, 30d). For OpenClaw / fire-and-forget use.",
)
@click.option(
    "--state-db",
    default=None,
    type=click.Path(path_type=str),
    help="SQLite path to persist call records. Enables --budget-window.",
)
@click.option(
    "--budget-window",
    default=None,
    callback=_parse_reset_every,
    help="Rolling window for budget (e.g. 24h). Requires --state-db. Survives restarts.",
)
@click.option("--target-openai", default="https://api.openai.com", help="Upstream OpenAI API URL")
@click.option(
    "--target-anthropic", default="https://api.anthropic.com", help="Upstream Anthropic API URL"
)
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def proxy(
    port,
    host,
    per_call,
    session_budget,
    reset_every,
    state_db,
    budget_window,
    target_openai,
    target_anthropic,
    log_level,
):
    """Start the CostPlan budget enforcement proxy.

    Drop-in economic circuit breaker for any LLM workflow.

    \b
    Claude Code quickstart:
        costplan proxy
        export ANTHROPIC_BASE_URL=http://localhost:8080
        claude  # Budget-enforced!

    \b
    OpenAI quickstart:
        costplan proxy --per-call 0.50 --session 5.00
        export OPENAI_BASE_URL=http://localhost:8080/v1
        python my_agent.py  # Budget-enforced!
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if per_call > session_budget:
        click.echo(
            f"Error: Per-call budget (${per_call:.2f}) cannot exceed session budget (${session_budget:.2f}).\n"
            "  No single call can cost more than the session total.",
            err=True,
        )
        raise SystemExit(1)

    if budget_window is not None and not state_db:
        click.echo(
            "Error: --budget-window requires --state-db to persist call history.",
            err=True,
        )
        raise SystemExit(1)

    if reset_every is not None and budget_window is not None:
        click.echo(
            "Error: Use either --reset-every or --budget-window, not both.",
            err=True,
        )
        raise SystemExit(1)

    from costplan.proxy.app import create_app
    from costplan.proxy.budget_state import ProxyBudgetState
    from costplan.proxy.forwarder import Forwarder

    state_store = None
    if state_db:
        from costplan.proxy.persistent_store import PersistentCallStore

        state_store = PersistentCallStore(state_db)

    budget = ProxyBudgetState(
        per_call_budget=per_call,
        session_budget=session_budget,
        reset_every_seconds=reset_every,
        state_store=state_store,
        budget_window_seconds=budget_window,
    )
    forwarder = Forwarder(openai_target=target_openai, anthropic_target=target_anthropic)
    app = create_app(budget=budget, forwarder=forwarder)

    click.echo("CostPlan Proxy — LLM Economic Circuit Breaker")
    click.echo(f"  Per-call budget:  ${per_call:.2f}")
    click.echo(f"  Session budget:   ${session_budget:.2f}")
    if reset_every is not None:
        hours = reset_every / 3600
        click.echo(
            f"  Auto-reset:       every {(hours / 24):.1f}d"
            if hours >= 24
            else f"  Auto-reset:       every {hours:.1f}h"
        )
    if budget_window is not None:
        h = budget_window / 3600
        click.echo(
            f"  Budget window:    {(h / 24):.1f}d rolling"
            if h >= 24
            else f"  Budget window:    {h:.1f}h rolling"
        )
    if state_db:
        click.echo(f"  State DB:        {state_db}")
    click.echo(f"  OpenAI target:    {target_openai}")
    click.echo(f"  Anthropic target: {target_anthropic}")
    click.echo(f"  Listening on:     http://{host}:{port}")
    click.echo()
    click.echo(f"  Claude Code:  export ANTHROPIC_BASE_URL=http://{host}:{port}")
    click.echo(f"  OpenAI:       export OPENAI_BASE_URL=http://{host}:{port}/v1")
    click.echo(f"  Dashboard:    http://{host}:{port}/")
    click.echo()

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
