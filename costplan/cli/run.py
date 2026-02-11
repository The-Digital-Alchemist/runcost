"""Run, history, and stats commands for CostPlan CLI."""

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from costplan.core.calculator import calculate_error_percent
from costplan.core.pricing import PricingNotFoundError
from costplan.core.budget import BudgetPolicy, BudgetSession, BudgetedClient, BudgetExceededError
from costplan.core.providers import OpenAIProvider
from costplan.storage.tracker import RunTracker
from costplan.utils.helpers import format_cost, format_percentage, format_tokens, read_prompt_from_file

console = Console()


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--file", "-f",
    type=click.Path(exists=True),
    help="Read prompt from file"
)
@click.option(
    "--model", "-m",
    default="gpt-3.5-turbo",
    help="Model name (default: gpt-3.5-turbo)"
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="API key (can also use OPENAI_API_KEY env var)"
)
@click.option(
    "--base-url",
    envvar="OPENAI_BASE_URL",
    help="API base URL for compatible providers"
)
@click.option(
    "--output-ratio", "-r",
    type=float,
    help="Override default output token ratio"
)
@click.option(
    "--temperature", "-t",
    type=float,
    default=1.0,
    help="Sampling temperature (default: 1.0)"
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum tokens to generate"
)
@click.option(
    "--show-response/--no-response",
    default=True,
    help="Show/hide LLM response (default: show)"
)
@click.option(
    "--per-call",
    type=float,
    help="Max cost per call (dollars). Abort before running if predicted cost exceeds this."
)
@click.option(
    "--per-session",
    type=float,
    help="Max total cost for this session (dollars). Abort if predicted cost would exceed remaining budget."
)
@click.pass_context
def run(ctx, prompt, file, model, api_key, base_url, output_ratio,
        temperature, max_tokens, show_response, per_call, per_session):
    """Execute a prompt and compare predicted vs actual cost.

    This command will:
    1. Predict the cost before execution
    2. Execute the request via the API
    3. Calculate actual cost
    4. Store run data for calibration
    5. Display cost comparison

    Examples:

        \b
        # Run with environment variable API key
        export OPENAI_API_KEY=your_key
        costplan run "Explain quantum computing"

        \b
        # Run from file with specific model
        costplan run --file prompt.txt --model gpt-4

        \b
        # Run with custom base URL (e.g., local proxy)
        costplan run "Hello" --base-url http://localhost:8000/v1
    """
    settings = ctx.obj.get("settings")

    # Validate API key
    if not api_key:
        api_key = settings.get_api_key()
    if not api_key:
        click.echo(
            "Error: API key not found. Set OPENAI_API_KEY environment variable "
            "or use --api-key option.",
            err=True
        )
        sys.exit(1)

    # Get prompt text
    if file:
        try:
            prompt_text = read_prompt_from_file(file)
        except FileNotFoundError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    elif prompt:
        prompt_text = prompt
    else:
        click.echo("Error: Either provide a prompt or use --file option", err=True)
        click.echo("Try 'costplan run --help' for more information.")
        sys.exit(1)

    # Initialize provider and budgeted client (single path for all runs)
    try:
        provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url or settings.get_base_url(),
            settings=settings,
        )
        policy = BudgetPolicy(per_call=per_call, per_session=per_session) if (per_call is not None or per_session is not None) else None
        session = BudgetSession() if policy else None
        client = BudgetedClient(provider, policy=policy, session=session)
        tracker = RunTracker(settings=settings)
    except Exception as e:
        click.echo(f"Error initializing components: {e}", err=True)
        sys.exit(1)

    # Use Rich Progress for better UX
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task1 = progress.add_task("🔮 Predicting cost...", total=None)
        try:
            prediction, execution, actual = client.execute(
                prompt_text, model,
                output_ratio=output_ratio,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except BudgetExceededError as e:
            progress.update(task1, completed=True)
            console.print(f"[red]Budget exceeded: {e}[/red]")
            sys.exit(1)
        except PricingNotFoundError as e:
            progress.update(task1, completed=True)
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        progress.update(task1, completed=True)
        console.print(f"   [green]Predicted: {format_cost(prediction.predicted_total_cost)}[/green]")

        task2 = progress.add_task("⚡ Executing request...", total=None)
        progress.update(task2, completed=True)
        console.print("   [green]✓ Completed[/green]")

        if not execution.success:
            console.print(f"   [red]Error: {execution.error_message}[/red]")
            tracker.store_run(prediction, actual=None, model=model)
            sys.exit(1)

    error = calculate_error_percent(
        prediction.predicted_total_cost,
        actual.actual_total_cost,
    )

    # Step 4: Store run data
    run_record = tracker.store_run(prediction, actual, model)

    # Step 5: Display comparison with Rich
    console.print("\n")

    # Create comparison table
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Predicted", justify="right", style="yellow")
    table.add_column("Actual", justify="right", style="green")

    table.add_row(
        "Input Tokens",
        format_tokens(prediction.predicted_input_tokens),
        format_tokens(actual.actual_input_tokens),
    )
    table.add_row(
        "Output Tokens",
        format_tokens(prediction.predicted_output_tokens),
        format_tokens(actual.actual_output_tokens),
    )
    table.add_row(
        "Total Cost",
        format_cost(prediction.predicted_total_cost),
        f"[bold]{format_cost(actual.actual_total_cost)}[/bold]",
    )

    error_color = "green" if abs(error) < 15 else "yellow" if abs(error) < 30 else "red"
    table.add_row("Error", "—", f"[{error_color}]{format_percentage(error)}[/{error_color}]")
    table.add_row("Run ID", "—", f"[dim]{run_record.id[:8]}...[/dim]")

    console.print(Panel(table, title="📊 Cost Analysis", border_style="blue"))
    console.print("\n")

    # Show response if requested
    if show_response:
        response_panel = Panel(
            execution.response_text,
            title="🤖 Response",
            border_style="green",
            padding=(1, 2),
        )
        console.print(response_panel)
        console.print("\n")


@click.command()
@click.option(
    "--limit", "-n",
    type=int,
    default=10,
    help="Number of runs to show (default: 10)"
)
@click.option(
    "--model", "-m",
    help="Filter by model name"
)
@click.pass_context
def history(ctx, limit, model):
    """Show recent run history.

    Examples:

        \b
        # Show last 10 runs
        costplan history

        \b
        # Show last 20 runs for gpt-4
        costplan history --limit 20 --model gpt-4
    """
    settings = ctx.obj.get("settings")
    tracker = RunTracker(settings=settings)

    runs = tracker.get_recent_runs(limit=limit, model=model)

    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(
        title=f"📜 Recent Runs (showing {len(runs)})", show_header=True, header_style="bold cyan"
    )
    table.add_column("Timestamp", style="dim")
    table.add_column("Model", style="green")
    table.add_column("Predicted", justify="right", style="yellow")
    table.add_column("Actual", justify="right", style="green")
    table.add_column("Error", justify="right")
    table.add_column("ID", style="dim")

    for run in runs:
        timestamp = run.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        predicted = format_cost(run.predicted_cost)
        actual = format_cost(run.actual_cost) if run.actual_cost else "[dim]N/A[/dim]"

        if run.error_percent:
            error_val = abs(run.error_percent)
            error_color = "green" if error_val < 15 else "yellow" if error_val < 30 else "red"
            error = f"[{error_color}]{format_percentage(run.error_percent, include_sign=True)}[/{error_color}]"
        else:
            error = "[dim]N/A[/dim]"

        run_id = run.id[:8]

        table.add_row(timestamp, run.model, predicted, actual, error, run_id)

    console.print("\n")
    console.print(table)
    console.print("\n")


@click.command()
@click.option(
    "--model", "-m",
    help="Filter by model name"
)
@click.pass_context
def stats(ctx, model):
    """Show statistics and calibration data.

    Examples:

        \b
        # Show overall stats
        costplan stats

        \b
        # Show stats for gpt-4
        costplan stats --model gpt-4
    """
    settings = ctx.obj.get("settings")
    tracker = RunTracker(settings=settings)

    # Get error stats
    error_stats = tracker.get_error_stats(model=model)

    if error_stats["sample_count"] == 0:
        console.print("[yellow]No run data available for statistics.[/yellow]")
        return

    model_name = model if model else "All Models"

    # Create stats table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="bold")

    table.add_row("Model", f"[green]{model_name}[/green]")
    table.add_row("Total Runs", str(error_stats["sample_count"]))

    avg_error = error_stats["avg_error"]
    error_color = "green" if avg_error < 15 else "yellow" if avg_error < 30 else "red"
    table.add_row("Avg Error", f"[{error_color}]{format_percentage(avg_error, include_sign=False)}[/{error_color}]")
    table.add_row("Std Dev", format_percentage(error_stats["std_dev"], include_sign=False))
    table.add_row("Min Error", format_percentage(error_stats["min_error"], include_sign=False))
    table.add_row("Max Error", format_percentage(error_stats["max_error"], include_sign=False))

    # Get total cost
    total_cost = tracker.get_total_cost(model=model)
    if total_cost > 0:
        table.add_row("Total Cost", f"[yellow]{format_cost(total_cost)}[/yellow]")

    # Get rolling average if model specified
    if model:
        rolling_avg = tracker.get_rolling_error_average(model)
        if rolling_avg is not None:
            table.add_row("Rolling Avg", format_percentage(rolling_avg, include_sign=False))

    console.print("\n")
    console.print(Panel(table, title="📈 Statistics", border_style="blue"))

    # Accuracy assessment
    if avg_error < 10:
        assessment = "[green]Excellent[/green] (< 10%)"
        emoji = "🎯"
    elif avg_error < 30:
        assessment = "[yellow]Good[/yellow] (< 30%)"
        emoji = "✅"
    elif avg_error < 50:
        assessment = "[orange]Fair[/orange] (< 50%)"
        emoji = "⚠️"
    else:
        assessment = "[red]Poor[/red] (> 50%)"
        emoji = "❌"

    console.print(f"\n{emoji} Prediction Accuracy: {assessment}\n")
