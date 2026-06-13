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


@main.command()
@click.option("--table", required=True, help="Unity Catalog table (catalog.schema.table)")
@click.option("--limit", type=int, default=10, help="Number of rows to preview")
@click.option("--where", help="SQL WHERE clause (without WHERE keyword)")
@click.option("--columns", help="Comma-separated columns to display")
def preview_table(table: str, limit: int, where: str, columns: str) -> None:
    """Preview Unity Catalog table data.

    Examples:

        \b
        # Preview first 10 rows
        cassandra preview-table --table main.default.reviews

        \b
        # Preview with filter and specific columns
        cassandra preview-table \\
            --table main.default.reviews \\
            --where "rating >= 4" \\
            --columns "text,rating,label" \\
            --limit 20
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from cassandra.config import CassandraConfig
    from cassandra.data import UnityCatalogTables

    console = Console()

    try:
        # Load config
        config = CassandraConfig.load()

        # Display config panel
        console.print(
            Panel(
                f"[cyan]Table:[/cyan] {table}\n"
                f"[cyan]Limit:[/cyan] {limit}\n"
                f"[cyan]Filter:[/cyan] {where or 'None'}\n"
                f"[cyan]Columns:[/cyan] {columns or 'All'}",
                title="Preview Configuration",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        # Parse columns
        column_list = None
        if columns:
            column_list = [c.strip() for c in columns.split(",")]

        # Load data
        console.print("\n[cyan]Loading data...[/cyan]")
        tables = UnityCatalogTables(config)
        df = tables.preview_table(
            table_name=table,
            n_rows=limit,
            columns=column_list,
            where_clause=where,
        )

        # Create Rich table
        rich_table = Table(
            title=f"Preview: {table}",
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
        )

        for col in df.columns:
            rich_table.add_column(col, overflow="fold", max_width=40)

        for _, row in df.iterrows():
            # Truncate long values and convert to string
            values = [str(v)[:100] for v in row]
            rich_table.add_row(*values)

        console.print("\n")
        console.print(rich_table)
        console.print(f"\n[green bold]✓[/green bold] Loaded {len(df)} rows")

    except Exception as e:
        console.print(f"\n[red bold]Error:[/red bold] {e}")
        raise click.Abort()


@main.command()
@click.option("--volume", required=True, help="Volume path (/Volumes/catalog/schema/volume)")
@click.option("--pattern", default="*", help="File pattern (e.g., *.json)")
@click.option("--recursive/--no-recursive", default=False, help="Search subdirectories")
def list_volume(volume: str, pattern: str, recursive: bool) -> None:
    """List files in Unity Catalog volume.

    Examples:

        \b
        # List all files in volume
        cassandra list-volume --volume /Volumes/main/default/data

        \b
        # List JSON files recursively
        cassandra list-volume \\
            --volume /Volumes/main/default/data \\
            --pattern "*.json" \\
            --recursive
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from cassandra.config import CassandraConfig
    from cassandra.data import UnityCatalogVolumes

    console = Console()

    try:
        config = CassandraConfig.load()

        # Display config
        console.print(
            Panel(
                f"[cyan]Volume:[/cyan] {volume}\n"
                f"[cyan]Pattern:[/cyan] {pattern}\n"
                f"[cyan]Recursive:[/cyan] {recursive}",
                title="Volume Listing",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        # List files
        console.print("\n[cyan]Listing files...[/cyan]")
        volumes = UnityCatalogVolumes(config)
        files = volumes.list_files(
            volume_path=volume,
            pattern=pattern,
            recursive=recursive,
        )

        if not files:
            console.print(f"\n[yellow]No files found[/yellow]")
            return

        # Create table
        rich_table = Table(
            title=f"Files in {volume}",
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
        )
        rich_table.add_column("Name", style="white")
        rich_table.add_column("Size", justify="right", style="cyan")
        rich_table.add_column("Type", style="magenta")

        for file_info in files:
            name = file_info["path"].split("/")[-1]
            size = file_info.get("size_bytes", 0)
            ftype = file_info.get("file_type", "file")

            # Format size
            if size > 1024**3:
                size_str = f"{size / 1024**3:.2f} GB"
            elif size > 1024**2:
                size_str = f"{size / 1024**2:.2f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size} B"

            rich_table.add_row(name, size_str, ftype)

        console.print("\n")
        console.print(rich_table)
        console.print(f"\n[green bold]✓[/green bold] Found {len(files)} files")

    except Exception as e:
        console.print(f"\n[red bold]Error:[/red bold] {e}")
        raise click.Abort()


@main.command()
@click.option("--volume", required=True, help="Volume path (/Volumes/catalog/schema/volume)")
@click.option("--file", required=True, help="File name or pattern (e.g., data.csv, *.json)")
@click.option("--limit", type=int, default=10, help="Number of rows to preview")
@click.option(
    "--format",
    type=click.Choice(["csv", "json", "parquet", "text"]),
    help="Force format (auto-detected if omitted)",
)
def preview_volume(volume: str, file: str, limit: int, format: str) -> None:
    """Preview Unity Catalog volume file data.

    Examples:

        \b
        # Preview specific file
        cassandra preview-volume \\
            --volume /Volumes/main/default/data \\
            --file cassandra.csv \\
            --limit 10

        \b
        # Auto-detect format
        cassandra preview-volume \\
            --volume /Volumes/main/default/data \\
            --file data.parquet

        \b
        # Force format
        cassandra preview-volume \\
            --volume /Volumes/catalog/schema/volume \\
            --file data.txt \\
            --format text \\
            --limit 20
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from cassandra.config import CassandraConfig
    from cassandra.data import UnityCatalogVolumes

    console = Console()

    try:
        # Load config
        config = CassandraConfig.load()

        # Detect format for display
        format_display = format if format else "auto-detected"

        # Display config panel
        console.print(
            Panel(
                f"[cyan]Volume:[/cyan] {volume}\n"
                f"[cyan]File:[/cyan] {file}\n"
                f"[cyan]Format:[/cyan] {format_display}\n"
                f"[cyan]Limit:[/cyan] {limit}",
                title="Preview Configuration",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        # Load data
        console.print("\n[cyan]Loading data...[/cyan]")
        volumes = UnityCatalogVolumes(config)
        df = volumes.preview_file(
            volume_path=volume,
            filename=file,
            n_rows=limit,
            file_format=format,
        )

        # Create Rich table
        rich_table = Table(
            title=f"Preview: {file}",
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
        )

        for col in df.columns:
            rich_table.add_column(col, overflow="fold", max_width=40)

        for _, row in df.iterrows():
            # Truncate long values and convert to string
            values = [str(v)[:100] for v in row]
            rich_table.add_row(*values)

        console.print("\n")
        console.print(rich_table)
        console.print(f"\n[green bold]✓[/green bold] Loaded {len(df)} rows")

    except Exception as e:
        console.print(f"\n[red bold]Error:[/red bold] {e}")
        raise click.Abort()


if __name__ == "__main__":
    main()
