"""CLI command to run CostPlan proxy for OpenClaw integration.

Starts the proxy with OpenClaw-specific setup instructions. Does not wrap
the gateway (which runs as a daemon); user runs OpenClaw in a separate process.
"""

import click

from costplan.utils.helpers import parse_duration_seconds


def _parse_reset_every(ctx, param, value):
    """Click callback to parse --reset-every duration."""
    if not value:
        return None
    try:
        return parse_duration_seconds(value)
    except ValueError as e:
        raise click.BadParameter(str(e))


@click.command()
@click.option("--port", default=8080, type=int, help="Port for CostPlan proxy (default: 8080)")
@click.option("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
@click.option(
    "--per-call", "per_call", required=True, type=float,
    help="Max dollars per individual API call",
)
@click.option(
    "--session", "session_budget", required=True, type=float,
    help="Max dollars for the session (resets automatically if --reset-every set)",
)
@click.option(
    "--reset-every",
    default=None,
    callback=_parse_reset_every,
    help="Auto-reset budget every N (e.g. 24h, 7d). Recommended for OpenClaw.",
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
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def openclaw(port, host, per_call, session_budget, reset_every, state_db, budget_window, log_level):
    """Start CostPlan proxy for OpenClaw.

    Run this in one terminal, then start OpenClaw in another with the
    printed environment variable. The proxy enforces budget on all LLM
    calls from OpenClaw (WhatsApp, Telegram, Discord, etc.).

    \b
    Example:
        costplan openclaw --per-call 1.00 --session 50.00 --reset-every 24h
        # In another terminal:
        export ANTHROPIC_BASE_URL=http://localhost:8080
        openclaw gateway --port 18789
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
            "  pip install costplan[proxy]",
            err=True,
        )
        raise SystemExit(1)

    if per_call > session_budget:
        click.echo(
            f"Error: Per-call budget (${per_call:.2f}) cannot exceed "
            f"session budget (${session_budget:.2f}).",
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

    from costplan.proxy.budget_state import ProxyBudgetState
    from costplan.proxy.forwarder import Forwarder
    from costplan.proxy.app import create_app
    from costplan.proxy.persistent_store import PersistentCallStore

    state_store = PersistentCallStore(state_db) if state_db else None

    budget = ProxyBudgetState(
        per_call_budget=per_call,
        session_budget=session_budget,
        reset_every_seconds=reset_every,
        state_store=state_store,
        budget_window_seconds=budget_window,
    )
    forwarder = Forwarder()
    app = create_app(budget=budget, forwarder=forwarder)

    proxy_url = f"http://{host}:{port}"
    click.echo("CostPlan Proxy — OpenClaw integration")
    click.echo()
    click.echo("Proxy started. In another terminal, run:")
    click.echo()
    click.echo(f"  export ANTHROPIC_BASE_URL={proxy_url}")
    click.echo("  openclaw gateway --port 18789")
    click.echo()
    click.echo("Budget:")
    click.echo(f"  Per-call:  ${per_call:.2f}")
    click.echo(f"  Session:   ${session_budget:.2f}")
    if reset_every is not None:
        hours = reset_every / 3600
        click.echo(f"  Auto-reset: every {(hours / 24):.1f}d" if hours >= 24 else f"  Auto-reset: every {hours:.1f}h")
    click.echo()
    click.echo(f"Dashboard: {proxy_url}/")
    click.echo()

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
