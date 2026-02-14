"""CLI command to wrap any process with CostPlan budget enforcement.

Starts the proxy, sets ANTHROPIC_BASE_URL / OPENAI_BASE_URL, launches the
wrapped command, and prints a cost summary when it exits.
"""

import os
import signal
import subprocess
import sys
import threading
import time

import click


def _find_free_port(preferred: int) -> int:
    """Return *preferred* if it's available, otherwise fail loudly."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            raise click.ClickException(
                f"Port {preferred} is already in use. "
                f"Pick another with --port or stop the process using it."
            )


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--per-call", "per_call", required=True, type=float,
              help="Max dollars per individual API call")
@click.option("--session", "session_budget", required=True, type=float,
              help="Max dollars for the entire session")
@click.option("--port", default=8080, type=int,
              help="Port for the budget proxy (default: 8080)")
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
def wrap(per_call, session_budget, port, command):
    """Wrap any command with budget enforcement.

    Starts the CostPlan proxy, injects ANTHROPIC_BASE_URL and
    OPENAI_BASE_URL into the subprocess, then launches COMMAND.
    When COMMAND exits, the proxy stops and a cost summary is printed.

    \b
    Examples:
        costplan wrap --per-call 1.00 --session 5.00 claude
        costplan wrap --per-call 0.50 --session 10.00 python my_agent.py
        costplan wrap --per-call 1.00 --session 5.00 --port 9090 claude
    """
    # --- Validate -----------------------------------------------------------
    if per_call > session_budget:
        raise click.ClickException(
            f"Per-call budget (${per_call:.2f}) cannot exceed "
            f"session budget (${session_budget:.2f})."
        )

    _find_free_port(port)

    # --- Lazy-import proxy deps --------------------------------------------
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        click.echo(
            "Error: Proxy dependencies not installed. Install with:\n"
            "  pip install costplan[proxy]",
            err=True,
        )
        raise SystemExit(1)

    from costplan.proxy.budget_state import ProxyBudgetState
    from costplan.proxy.forwarder import Forwarder
    from costplan.proxy.app import create_app

    # --- Start the proxy in a daemon thread --------------------------------
    budget = ProxyBudgetState(per_call_budget=per_call, session_budget=session_budget)
    forwarder = Forwarder()
    app = create_app(budget=budget, forwarder=forwarder)

    host = "127.0.0.1"
    proxy_url = f"http://{host}:{port}"

    server_ready = threading.Event()

    class _ReadyServer(uvicorn.Server):
        """Uvicorn server that signals when it's accepting connections."""
        def install_signal_handlers(self):
            pass  # We handle signals ourselves

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = _ReadyServer(config)

    def _run_server():
        server.run()

    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()

    # Wait for the proxy to become ready
    for _ in range(50):  # 5 seconds max
        if server.started:
            break
        time.sleep(0.1)
    else:
        click.echo("Error: Proxy failed to start within 5 seconds.", err=True)
        raise SystemExit(1)

    # --- Print banner ------------------------------------------------------
    click.echo("CostPlan Wrap — budget-enforced subprocess")
    click.echo(f"  Per-call:  ${per_call:.2f}")
    click.echo(f"  Session:   ${session_budget:.2f}")
    click.echo(f"  Proxy:     {proxy_url}")
    click.echo(f"  Dashboard: {proxy_url}/")
    click.echo(f"  Command:   {' '.join(command)}")
    click.echo()

    # --- Launch the wrapped command ----------------------------------------
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = proxy_url
    env["OPENAI_BASE_URL"] = f"{proxy_url}/v1"

    try:
        result = subprocess.run(
            command,
            env=env,
            # Let the child inherit stdin/stdout/stderr so the user can
            # interact with it (e.g. Claude Code is interactive).
        )
        exit_code = result.returncode
    except KeyboardInterrupt:
        exit_code = 130  # Convention for SIGINT
    except FileNotFoundError:
        click.echo(f"Error: command not found: {command[0]}", err=True)
        server.should_exit = True
        thread.join(timeout=3)
        raise SystemExit(127)

    # --- Cost summary ------------------------------------------------------
    import asyncio

    stats = asyncio.get_event_loop().run_until_complete(budget.stats())

    click.echo()
    click.echo("CostPlan — Session summary")
    click.echo(f"  Total spent:  ${stats['total_spent']:.4f}")
    click.echo(f"  Remaining:    ${stats['remaining']:.4f}")
    click.echo(f"  Calls:        {stats['call_count']}")
    click.echo(f"  Circuit:      {'Locked' if stats['locked'] else 'OK'}")
    click.echo(f"  Exit code:    {exit_code}")

    # --- Shutdown proxy ----------------------------------------------------
    server.should_exit = True
    thread.join(timeout=3)

    raise SystemExit(exit_code)
