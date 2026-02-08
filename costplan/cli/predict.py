"""Predict command for CostPlan CLI."""

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from costplan.core.predictor import CostPredictor
from costplan.core.pricing import PricingNotFoundError, PricingRegistry
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
def predict(ctx, prompt, file, model, output_ratio, list_models):
    """Predict cost for a prompt without execution.

    Examples:

        \b
        # Predict from command line argument
        costplan predict "Write a short story about AI"

        \b
        # Predict from file
        costplan predict --file prompt.txt --model gpt-4

        \b
        # List supported models
        costplan predict --list-models
    """
    settings = ctx.obj.get("settings")

    # Handle list models
    if list_models:
        pricing_registry = PricingRegistry()
        models = pricing_registry.list_supported_models()

        table = Table(title="Supported Models", show_header=True, header_style="bold cyan")
        table.add_column("Model", style="green", width=30)
        table.add_column("Input ($/1K tokens)", justify="right", style="yellow")
        table.add_column("Output ($/1K tokens)", justify="right", style="yellow")

        for model_name in models:
            input_cost, output_cost = pricing_registry.get_model_pricing(model_name)
            table.add_row(model_name, format_cost(input_cost), format_cost(output_cost))

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

    # Create predictor
    try:
        predictor = CostPredictor(settings=settings)
    except FileNotFoundError as e:
        click.echo(f"Error: Pricing file not found: {e}", err=True)
        sys.exit(1)

    # Make prediction
    try:
        result = predictor.predict(
            prompt_text,
            model,
            output_ratio=output_ratio
        )
    except PricingNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        click.echo("\nUse --list-models to see supported models.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Format and display output with Rich
    confidence_str = result.confidence_level
    if result.confidence_percent is not None:
        confidence_str += f" (±{result.confidence_percent:.1f}%)"
    else:
        confidence_str += f" (±{settings.confidence_threshold * 100:.0f}%)"

    # Create table for prediction details
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
    console.print(Panel(table, title="💰 Predicted Cost Range", border_style="blue"))
    console.print("\n")

    # Show prompt preview if from file
    if file:
        from costplan.utils.helpers import truncate_text

        console.print(f"[dim]Prompt: {truncate_text(prompt_text, 80)}[/dim]\n")
