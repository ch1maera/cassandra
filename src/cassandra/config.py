"""Configuration management for Cassandra.

This module provides configuration management with Databricks CLI profile support,
AI Gateway settings, and YAML persistence.
"""

from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Optional
import yaml
import os


class CassandraConfig(BaseModel):
    """Cassandra configuration with Databricks and AI Gateway settings."""

    # Databricks - use CLI profile by default
    databricks_profile: str = Field(
        default="DEFAULT",
        description="Databricks CLI profile name",
    )
    databricks_host: Optional[str] = Field(
        default=None,
        description="Override workspace URL (optional if using profile)",
    )
    warehouse_id: str = Field(
        default="",
        description="SQL Warehouse ID for data operations",
    )

    # AI Gateway
    ai_endpoint: str = Field(
        default="databricks-claude-3-5-sonnet",
        description="Unity AI Gateway endpoint name",
    )
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)

    # Lakebase
    lakebase_endpoint_url: str = Field(
        default="https://lakebase-prod.cloud.databricks.com"
    )
    lakebase_namespace: str = Field(default="cassandra")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max_tokens is positive."""
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v

    @classmethod
    def config_path(cls) -> Path:
        """Get path to configuration file."""
        return Path.home() / ".config" / "cassandra" / "config.yaml"

    @classmethod
    def load(cls) -> "CassandraConfig":
        """Load configuration from YAML file and environment variables.

        Environment variables take precedence over YAML settings.
        Supported env vars:
            - DATABRICKS_PROFILE
            - DATABRICKS_HOST
            - DATABRICKS_WAREHOUSE_ID
            - CASSANDRA_AI_ENDPOINT
            - CASSANDRA_TEMPERATURE
            - CASSANDRA_MAX_TOKENS

        Returns:
            CassandraConfig instance with loaded settings
        """
        config_file = cls.config_path()
        config_data = {}

        # Load from YAML if exists
        if config_file.exists():
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f) or {}

        # Override with environment variables
        env_overrides = {
            "databricks_profile": os.getenv("DATABRICKS_PROFILE"),
            "databricks_host": os.getenv("DATABRICKS_HOST"),
            "warehouse_id": os.getenv("DATABRICKS_WAREHOUSE_ID"),
            "ai_endpoint": os.getenv("CASSANDRA_AI_ENDPOINT"),
            "temperature": os.getenv("CASSANDRA_TEMPERATURE"),
            "max_tokens": os.getenv("CASSANDRA_MAX_TOKENS"),
        }

        # Apply non-None overrides
        for key, value in env_overrides.items():
            if value is not None:
                # Convert string env vars to appropriate types
                if key == "temperature":
                    value = float(value)
                elif key == "max_tokens":
                    value = int(value)
                config_data[key] = value

        return cls(**config_data)

    def save(self) -> None:
        """Save configuration to YAML file.

        Creates parent directories if they don't exist.
        """
        config_file = self.config_path()
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            yaml.safe_dump(
                self.model_dump(exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def is_configured(self) -> bool:
        """Check if minimum configuration is present.

        Returns:
            True if databricks_profile and warehouse_id are set
        """
        return bool(self.databricks_profile and self.warehouse_id)

    def get_databricks_client(self):
        """Get WorkspaceClient using CLI profile.

        Returns:
            databricks.sdk.WorkspaceClient instance

        Raises:
            ImportError: If databricks-sdk is not installed
            ConfigError: If profile cannot be loaded
        """
        try:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.config import Config
        except ImportError as e:
            raise ImportError(
                "databricks-sdk is required. Install with: pip install databricks-sdk"
            ) from e

        # Load profile from ~/.databrickscfg
        config = Config(profile=self.databricks_profile)
        return WorkspaceClient(config=config)
