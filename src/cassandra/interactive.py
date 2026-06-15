"""Interactive CLI interface for Cassandra.

This module provides a Rich-powered interactive menu system for
configuring and managing the Cassandra framework.
"""

import os
import warnings

# Set langgraph configuration to suppress deprecation warning
os.environ["LANGGRAPH_CHECKPOINT_ALLOWED_OBJECTS"] = "messages"

# Suppress LangChain/LangGraph deprecation warnings globally
warnings.filterwarnings("ignore", message=".*LangChain.*")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, FloatPrompt, IntPrompt
from rich.table import Table
from rich import box
from cassandra.config import CassandraConfig
from typing import Optional

BANNER = """
 ██████╗ █████╗ ███████╗███████╗ █████╗ ███╗   ██╗██████╗ ██████╗  █████╗
██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗████╗  ██║██╔══██╗██╔══██╗██╔══██╗
██║     ███████║███████╗███████╗███████║██╔██╗ ██║██║  ██║██████╔╝███████║
██║     ██╔══██║╚════██║╚════██║██╔══██║██║╚██╗██║██║  ██║██╔══██╗██╔══██║
╚██████╗██║  ██║███████║███████║██║  ██║██║ ╚████║██████╔╝██║  ██║██║  ██║
 ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
"""


class InteractiveCLI:
    """Interactive CLI for Cassandra configuration and management."""

    def __init__(self):
        self.console = Console()
        self.config = CassandraConfig.load()

    def show_banner(self) -> None:
        """Display welcome banner and configuration status."""
        self.console.print(BANNER, style="cyan bold")

        subtitle_text = """
  ✨ Welcome to Cassandra ✨

  Agentic ML Training Framework
  HuggingFace • Optuna • Ray Tune

  Powered by Unity AI Gateway
"""

        subtitle_panel = Panel(
            subtitle_text,
            border_style="cyan",
            box=box.ROUNDED,
        )
        self.console.print(subtitle_panel)
        self.console.print()

        # Show configuration status
        if self.config.is_configured():
            self.console.print("✓ Workspace configured", style="green bold")
        else:
            self.console.print(
                "⚠️  Workspace not configured. Please configure before using.",
                style="yellow bold",
            )
        self.console.print()

    def show_menu(self) -> None:
        """Display main menu options."""
        self.console.print("⚡ MAIN MENU", style="bold yellow")
        self.console.print()
        self.console.print("  1  ⚙️  Configure Workspace & Credentials")
        self.console.print("  2  🔧 Configure AI Gateway")
        self.console.print("  3  🔌 Test Connection")
        self.console.print("  4  📋 Show Current Configuration")
        self.console.print("  5  ❌ Exit")
        self.console.print()

    def configure_workspace(self) -> None:
        """Configure Databricks workspace settings."""
        self.console.print("\n[bold cyan]⚙️  Configure Workspace & Credentials[/bold cyan]\n")

        # Prompt for Databricks CLI profile
        profile = Prompt.ask(
            "Databricks CLI profile name",
            default=self.config.databricks_profile,
        )

        # Prompt for SQL Warehouse ID
        warehouse_id = Prompt.ask(
            "SQL Warehouse ID",
            default=self.config.warehouse_id or "",
        )

        # Update config
        self.config.databricks_profile = profile
        self.config.warehouse_id = warehouse_id

        # Save
        try:
            self.config.save()
            self.console.print("\n✓ Workspace configuration saved", style="green bold")
        except Exception as e:
            self.console.print(f"\n✗ Error saving configuration: {e}", style="red bold")

    def configure_ai_gateway(self) -> None:
        """Configure AI Gateway settings."""
        self.console.print("\n[bold cyan]🔧 Configure AI Gateway[/bold cyan]\n")

        # Prompt for endpoint name
        endpoint = Prompt.ask(
            "AI Gateway endpoint name",
            default=self.config.ai_endpoint,
        )

        # Prompt for temperature
        temperature = FloatPrompt.ask(
            "Temperature (0.0-2.0)",
            default=self.config.temperature,
        )

        # Validate temperature
        if not 0.0 <= temperature <= 2.0:
            self.console.print("✗ Temperature must be between 0.0 and 2.0", style="red")
            return

        # Prompt for max tokens
        max_tokens = IntPrompt.ask(
            "Max tokens",
            default=self.config.max_tokens,
        )

        # Update config
        self.config.ai_endpoint = endpoint
        self.config.temperature = temperature
        self.config.max_tokens = max_tokens

        # Save
        try:
            self.config.save()
            self.console.print("\n✓ AI Gateway configuration saved", style="green bold")
        except Exception as e:
            self.console.print(f"\n✗ Error saving configuration: {e}", style="red bold")

    def test_connection(self) -> None:
        """Test Databricks and AI Gateway connections."""
        self.console.print("\n[bold cyan]🔌 Test Connection[/bold cyan]\n")

        # Test Databricks connection
        self.console.print("[bold]Testing Databricks connection...[/bold]")
        try:
            client = self.config.get_databricks_client()
            user = client.current_user.me()
            self.console.print(
                f"✓ Connected to Databricks as: {user.user_name}",
                style="green",
            )
        except ImportError:
            self.console.print(
                "✗ databricks-sdk not installed. Run: pip install databricks-sdk",
                style="red",
            )
        except Exception as e:
            self.console.print(
                f"✗ Databricks connection failed: {e}",
                style="red",
            )

        self.console.print()

        # Test AI Gateway connection
        self.console.print("[bold]Testing AI Gateway connection...[/bold]")
        try:
            import os
            from databricks.sdk.config import Config

            # Get credentials from Databricks profile
            db_config = Config(profile=self.config.databricks_profile)

            # Set environment variables for authentication
            if db_config.host:
                os.environ["DATABRICKS_HOST"] = db_config.host
                # Set MLflow tracking URI to the workspace
                os.environ["MLFLOW_TRACKING_URI"] = db_config.host
            if db_config.token:
                os.environ["DATABRICKS_TOKEN"] = db_config.token

            # Use databricks-langchain (newer, better maintained)
            # Set environment variable to suppress langgraph deprecation warning
            os.environ["LANGGRAPH_CHECKPOINT_ALLOWED_OBJECTS"] = "messages"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    from databricks_langchain import ChatDatabricks

                    llm = ChatDatabricks(
                        endpoint=self.config.ai_endpoint,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )

                    # Send test message using ChatDatabricks
                    response = llm.invoke("Say 'Hello from Cassandra!' if you can read this.")

                except ImportError:
                    # Fall back to direct API call if ChatDatabricks not available
                    import requests

                    response = requests.post(
                        f"{db_config.host}/serving-endpoints/{self.config.ai_endpoint}/invocations",
                        headers={
                            "Authorization": f"Bearer {db_config.token}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "messages": [{"role": "user", "content": "Say 'Hello from Cassandra!' if you can read this."}],
                            "temperature": self.config.temperature,
                            "max_tokens": self.config.max_tokens,
                        },
                        timeout=30,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        response_text = result.get("choices", [{}])[0].get("message", {}).get("content", str(result))
                        self.console.print(
                            f"✓ AI Gateway response: {response_text}",
                            style="green",
                        )
                        return
                    else:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")

            # Extract content from response
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            self.console.print(
                f"✓ AI Gateway response: {response_text}",
                style="green",
            )
            self.console.print(
                "\n[dim]Note: Deprecation warnings from langgraph are expected and harmless.[/dim]"
            )
        except ImportError as e:
            self.console.print(
                f"✗ ChatDatabricks not available. Error: {e}",
                style="red",
            )
        except Exception as e:
            self.console.print(
                f"✗ AI Gateway connection failed: {e}",
                style="red",
            )

    def show_config(self) -> None:
        """Display current configuration."""
        self.console.print("\n[bold cyan]📋 Current Configuration[/bold cyan]\n")

        # Create table
        table = Table(
            title="Cassandra Configuration",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Setting", style="yellow")
        table.add_column("Value", style="white")

        # Add rows
        table.add_row("Databricks Profile", self.config.databricks_profile)
        table.add_row(
            "Databricks Host",
            self.config.databricks_host or "(from profile)",
        )
        table.add_row("SQL Warehouse ID", self.config.warehouse_id or "(not set)")
        table.add_row("AI Gateway Endpoint", self.config.ai_endpoint)
        table.add_row("Temperature", str(self.config.temperature))
        table.add_row("Max Tokens", str(self.config.max_tokens))
        table.add_row("Lakebase Endpoint", self.config.lakebase_endpoint_url)
        table.add_row("Lakebase Namespace", self.config.lakebase_namespace)

        self.console.print(table)
        self.console.print()

        # Show config file path
        config_path = CassandraConfig.config_path()
        self.console.print(f"Configuration file: {config_path}", style="dim")

    def run(self) -> None:
        """Main interactive loop."""
        try:
            # Show banner once
            self.show_banner()

            while True:
                # Show menu
                self.show_menu()

                # Get user choice
                choice = Prompt.ask(
                    "Select an option",
                    choices=["1", "2", "3", "4", "5"],
                    default="5",
                )

                # Execute action
                if choice == "1":
                    self.configure_workspace()
                elif choice == "2":
                    self.configure_ai_gateway()
                elif choice == "3":
                    self.test_connection()
                elif choice == "4":
                    self.show_config()
                elif choice == "5":
                    self.console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]\n")
                    break

                # Reload config after changes
                self.config = CassandraConfig.load()

        except KeyboardInterrupt:
            self.console.print("\n\n[bold cyan]Interrupted. Goodbye! 👋[/bold cyan]\n")
        except Exception as e:
            self.console.print(f"\n[bold red]Error: {e}[/bold red]\n")
