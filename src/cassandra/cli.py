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


@main.command()
@click.option(
    "--experiment-name",
    required=True,
    help="Name of the MLflow experiment",
)
@click.option(
    "--location",
    type=click.Choice(["personal", "shared"]),
    default="personal",
    help="Experiment location (personal or shared)",
)
@click.option(
    "--model",
    type=click.Choice(["prophet", "arima"]),
    default="prophet",
    help="Model type to train",
)
@click.option(
    "--tag",
    multiple=True,
    help="Custom tags in key=value format (can be used multiple times)",
)
def test_mlflow(
    experiment_name: str,
    location: str,
    model: str,
    tag: tuple,
) -> None:
    """Test MLflow integration with synthetic time series data.

    This command generates synthetic time series data, trains a model (Prophet or ARIMA),
    logs parameters/metrics/artifacts to MLflow, and displays the run information.

    Examples:

        \b
        # Test with Prophet model in personal workspace
        cassandra test-mlflow --experiment-name forecast-test --model prophet

        \b
        # Test with ARIMA in shared workspace with custom tags
        cassandra test-mlflow \\
            --experiment-name team-test \\
            --location shared \\
            --model arima \\
            --tag team=ml-platform \\
            --tag version=1.0
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from cassandra.config import CassandraConfig
    from cassandra.mlflow import MLflowIntegration

    console = Console()

    try:
        # Load configuration
        config = CassandraConfig.load()

        # Override experiment location if specified
        config.mlflow_experiment_location = location

        # Parse custom tags
        custom_tags = {}
        for tag_str in tag:
            if "=" not in tag_str:
                console.print(
                    f"[yellow]Warning: Ignoring invalid tag '{tag_str}' (must be key=value)[/yellow]"
                )
                continue
            key, value = tag_str.split("=", 1)
            custom_tags[key.strip()] = value.strip()

        # Display configuration
        console.print(
            Panel(
                f"[cyan]Experiment:[/cyan] {experiment_name}\n"
                f"[cyan]Location:[/cyan] {location}\n"
                f"[cyan]Model:[/cyan] {model}\n"
                f"[cyan]Custom Tags:[/cyan] {len(custom_tags)}",
                title="MLflow Test Configuration",
                border_style="cyan",
            )
        )

        # Initialize MLflow integration
        integration = MLflowIntegration(config)

        # Get experiment path
        experiment_path = config.get_mlflow_experiment_path(experiment_name)
        console.print(f"\n[cyan]Experiment path:[/cyan] {experiment_path}")

        # Run training workflow
        console.print("\n[cyan]Starting training workflow...[/cyan]")
        run_id = integration.run_training_workflow(
            experiment_name=experiment_name,
            model_type=model,
            n_days=730,
            test_size=0.2,
            tags=custom_tags,
        )

        # Get run info
        run = integration.mlflow.get_run(run_id)

        # Display results
        console.print("\n[green bold]✓ Training workflow completed successfully![/green bold]\n")

        # Create results table
        results_table = Table(title="Run Results", show_header=True, header_style="bold cyan")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="white")

        results_table.add_row("Run ID", run_id[:8] + "...")
        results_table.add_row("Experiment Path", experiment_path)
        results_table.add_row("Model Type", model)

        # Add metrics
        metrics = run.data.metrics
        if "mae" in metrics:
            results_table.add_row("MAE", f"{metrics['mae']:.4f}")
        if "rmse" in metrics:
            results_table.add_row("RMSE", f"{metrics['rmse']:.4f}")
        if "mape" in metrics:
            results_table.add_row("MAPE", f"{metrics['mape']:.2f}%")

        console.print(results_table)

        # Display artifacts info
        console.print("\n[cyan]Logged artifacts:[/cyan]")
        console.print("  • 3 visualizations (time series, residuals, forecast)")
        console.print("  • Sample data CSV")
        console.print("  • Predictions CSV")
        console.print("  • Configuration YAML")
        console.print("  • Trained model")

        # Display run URL if available
        if hasattr(run.info, "artifact_uri"):
            # Extract workspace URL from artifact URI
            artifact_uri = run.info.artifact_uri
            if "databricks" in artifact_uri:
                # Try to construct run URL
                console.print(
                    f"\n[cyan]View run in MLflow:[/cyan] "
                    f"Navigate to MLflow UI → {experiment_path} → Run {run_id[:8]}..."
                )

    except Exception as e:
        console.print(f"\n[red bold]Error:[/red bold] {e}")
        raise click.Abort()


if __name__ == "__main__":
    main()
