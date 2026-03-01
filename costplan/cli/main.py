"""Main CLI entry point for CostPlan."""

import click
from pathlib import Path

from costplan import __version__
from costplan.config.settings import Settings


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.pass_context
def cli(ctx, config):
    """CostPlan - LLM Cost Prediction and Measurement System.

    Predict costs before executing LLM requests and measure actual costs.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Load settings
    if config:
        ctx.obj["settings"] = Settings.load_from_file(config)
    else:
        ctx.obj["settings"] = Settings()


# Import and register commands
from costplan.cli.predict import predict
from costplan.cli.run import run, history, stats
from costplan.cli.proxy_cmd import proxy
from costplan.cli.wrap_cmd import wrap
from costplan.cli.openclaw_cmd import openclaw

cli.add_command(predict)
cli.add_command(run)
cli.add_command(history)
cli.add_command(stats)
cli.add_command(proxy)
cli.add_command(wrap)
cli.add_command(openclaw)


if __name__ == "__main__":
    cli()
