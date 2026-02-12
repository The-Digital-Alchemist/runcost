"""Predict command for CostPlan CLI. Everything flows through BaseProvider; no direct pricing/token layers."""

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from costplan.core.factory import create
from costplan.core.pricing import PricingNotFoundError
from costplan.utils.helpers import format_cost, format_tokens, read_prompt_from_file

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
    "--provider", "-p",
    default="openai",
    help="Provider name (default: openai)"
)
@click.option(
    "--output-ratio", "-r",
    type=float,
    help="Override default output token ratio"
)
@click.option(
    "--list-models",
    is_flag=True,
    help="List supported models and exit"
)
@click.pass_context
def predict(ctx, prompt, file, model, provider, output_ratio, list_models):
    """Predict cost for a prompt without execution. Uses provider only (no global pricing/token layers)."""
    settings = ctx.obj.get("settings")

    # Create provider via factory (pricing and token logic stay inside provider)
    try:
        prov = create(provider_name=provider, settings=settings)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error creating provider: {e}", err=True)
        sys.exit(1)

    # Handle list models
    if list_models:
        models = prov.list_models()
        if not models:
            console.print("[yellow]This provider does not expose a model list.[/yellow]")
            return

        table = Table(title="Supported Models", show_header=True, header_style="bold cyan")
        table.add_column("Model", style="green", width=30)
        table.add_column("Input ($/1K tokens)", justify="right", style="yellow")
        table.add_column("Output ($/1K tokens)", justify="right", style="yellow")

        for model_name in models:
            try:
                input_cost, output_cost = prov.get_pricing(model_name)
                table.add_row(model_name, format_cost(input_cost), format_cost(output_cost))
            except Exception:
                table.add_row(model_name, "[dim]—[/dim]", "[dim]—[/dim]")

        console.print("\n")
        console.print(table)
        console.print("\n")
        return

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
        click.echo("Try 'costplan predict --help' for more information.")
        sys.exit(1)

    # Predict via provider (no CostPredictor; no direct PricingRegistry/TokenEstimator)
    try:
        result = prov.predict(prompt_text, model, output_ratio=output_ratio)
    except PricingNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("\nUse --list-models to see supported models.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Format and display
    confidence_str = result.confidence_level
    if result.confidence_percent is not None:
        confidence_str += f" (+/-{result.confidence_percent:.1f}%)"
    else:
        confidence_str += f" (+/-{settings.confidence_threshold * 100:.0f}%)"

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value", style="bold green")

    table.add_row("Model", model)
    table.add_row("Input Tokens", f"~{format_tokens(result.predicted_input_tokens)}")
    table.add_row(
        "Output Tokens", f"~{format_tokens(result.predicted_output_tokens)} [dim](estimated)[/dim]"
    )
    table.add_row("Input Cost", format_cost(result.predicted_input_cost))
    table.add_row("Output Cost", format_cost(result.predicted_output_cost))
    table.add_row("Total Cost", f"[bold yellow]{format_cost(result.predicted_total_cost)}[/bold yellow]")
    table.add_row("Confidence", f"[{'green' if result.confidence_level == 'High' else 'yellow' if result.confidence_level == 'Medium' else 'red'}]{confidence_str}[/]")

    console.print("\n")
    console.print(Panel(table, title="Predicted Cost Range", border_style="blue"))
    console.print("\n")

    if file:
        from costplan.utils.helpers import truncate_text
        console.print(f"[dim]Prompt: {truncate_text(prompt_text, 80)}[/dim]\n")
