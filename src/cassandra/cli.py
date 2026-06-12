"""Command-line interface for Cassandra.

This module provides the main CLI entry point using Click.
"""

import click
from cassandra.interactive import InteractiveCLI


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """Cassandra - Agentic ML Training Framework.

    A production-ready framework for orchestrating transformer-based model
    development on Databricks with HuggingFace, Optuna/Ray Tune, and Unity AI Gateway.
    """
    if ctx.invoked_subcommand is None:
        # Launch interactive mode by default
        cli = InteractiveCLI()
        cli.run()


@main.command()
def version() -> None:
    """Show Cassandra version."""
    from cassandra import __version__
    from rich.console import Console

    console = Console()
    console.print(f"Cassandra version {__version__}", style="cyan bold")


if __name__ == "__main__":
    main()
