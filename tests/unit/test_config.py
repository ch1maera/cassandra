"""Unit tests for configuration management."""

import pytest
from pathlib import Path
from cassandra.config import CassandraConfig
import yaml
import os


def test_config_defaults():
    """Test default configuration values."""
    config = CassandraConfig()

    assert config.databricks_profile == "DEFAULT"
    assert config.databricks_host is None
    assert config.warehouse_id == ""
    assert config.ai_endpoint == "databricks-claude-sonnet-4-6"
    assert config.temperature == 0.7
    assert config.max_tokens == 4096
    assert config.lakebase_endpoint_url == "https://lakebase-prod.cloud.databricks.com"
    assert config.lakebase_namespace == "cassandra"
    assert config.mlflow_tracking_uri == "databricks"
    assert config.mlflow_experiment_location == "personal"
    assert config.databricks_username is None


def test_config_is_configured():
    """Test is_configured check."""
    config = CassandraConfig()
    assert not config.is_configured()

    config.databricks_profile = "DEFAULT"
    config.warehouse_id = "test-warehouse"
    assert config.is_configured()


def test_config_is_configured_missing_profile():
    """Test is_configured returns False when profile is empty."""
    config = CassandraConfig(databricks_profile="", warehouse_id="test-warehouse")
    assert not config.is_configured()


def test_config_is_configured_missing_warehouse():
    """Test is_configured returns False when warehouse_id is empty."""
    config = CassandraConfig(databricks_profile="DEFAULT", warehouse_id="")
    assert not config.is_configured()


def test_config_save_load(tmp_path, monkeypatch):
    """Test save/load roundtrip."""
    # Mock config_path to use tmp_path
    test_config = tmp_path / "config.yaml"
    monkeypatch.setattr(CassandraConfig, "config_path", classmethod(lambda cls: test_config))

    # Create and save config
    config = CassandraConfig(
        databricks_profile="test-profile",
        warehouse_id="test-warehouse",
        ai_endpoint="test-endpoint",
        temperature=0.5,
        max_tokens=2048,
    )
    config.save()

    # Verify file exists
    assert test_config.exists()

    # Load and verify
    loaded = CassandraConfig.load()
    assert loaded.databricks_profile == "test-profile"
    assert loaded.warehouse_id == "test-warehouse"
    assert loaded.ai_endpoint == "test-endpoint"
    assert loaded.temperature == 0.5
    assert loaded.max_tokens == 2048


def test_config_save_creates_directory(tmp_path, monkeypatch):
    """Test that save creates parent directories."""
    # Mock config_path to use nested tmp_path
    test_config = tmp_path / "nested" / "dir" / "config.yaml"
    monkeypatch.setattr(CassandraConfig, "config_path", classmethod(lambda cls: test_config))

    config = CassandraConfig()
    config.save()

    assert test_config.exists()
    assert test_config.parent.exists()


def test_config_load_nonexistent(tmp_path, monkeypatch):
    """Test loading when config file doesn't exist returns defaults."""
    # Mock config_path to non-existent file
    test_config = tmp_path / "nonexistent.yaml"
    monkeypatch.setattr(CassandraConfig, "config_path", classmethod(lambda cls: test_config))

    config = CassandraConfig.load()
    assert config.databricks_profile == "DEFAULT"
    assert config.ai_endpoint == "databricks-claude-sonnet-4-6"


def test_config_env_overrides(tmp_path, monkeypatch):
    """Test environment variable overrides."""
    # Mock config_path
    test_config = tmp_path / "config.yaml"
    monkeypatch.setattr(CassandraConfig, "config_path", classmethod(lambda cls: test_config))

    # Save initial config
    config = CassandraConfig(
        databricks_profile="yaml-profile",
        warehouse_id="yaml-warehouse",
    )
    config.save()

    # Set environment variables
    monkeypatch.setenv("DATABRICKS_PROFILE", "env-profile")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "env-warehouse")
    monkeypatch.setenv("CASSANDRA_AI_ENDPOINT", "env-endpoint")
    monkeypatch.setenv("CASSANDRA_TEMPERATURE", "0.9")
    monkeypatch.setenv("CASSANDRA_MAX_TOKENS", "8192")

    # Load config - env vars should override YAML
    loaded = CassandraConfig.load()
    assert loaded.databricks_profile == "env-profile"
    assert loaded.warehouse_id == "env-warehouse"
    assert loaded.ai_endpoint == "env-endpoint"
    assert loaded.temperature == 0.9
    assert loaded.max_tokens == 8192


def test_config_validate_temperature_valid():
    """Test temperature validation with valid values."""
    config = CassandraConfig(temperature=0.0)
    assert config.temperature == 0.0

    config = CassandraConfig(temperature=1.0)
    assert config.temperature == 1.0

    config = CassandraConfig(temperature=2.0)
    assert config.temperature == 2.0


def test_config_validate_temperature_invalid():
    """Test temperature validation with invalid values."""
    with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
        CassandraConfig(temperature=-0.1)

    with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
        CassandraConfig(temperature=2.1)


def test_config_validate_max_tokens_valid():
    """Test max_tokens validation with valid value."""
    config = CassandraConfig(max_tokens=1000)
    assert config.max_tokens == 1000


def test_config_validate_max_tokens_invalid():
    """Test max_tokens validation with invalid value."""
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        CassandraConfig(max_tokens=0)

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        CassandraConfig(max_tokens=-100)


def test_config_exclude_none_in_save(tmp_path, monkeypatch):
    """Test that None values are excluded from saved YAML."""
    test_config = tmp_path / "config.yaml"
    monkeypatch.setattr(CassandraConfig, "config_path", classmethod(lambda cls: test_config))

    config = CassandraConfig(
        databricks_profile="test-profile",
        databricks_host=None,  # Should be excluded
        warehouse_id="test-warehouse",
    )
    config.save()

    # Load raw YAML and verify databricks_host is not present
    with open(test_config, "r") as f:
        data = yaml.safe_load(f)

    assert "databricks_host" not in data
    assert data["databricks_profile"] == "test-profile"
    assert data["warehouse_id"] == "test-warehouse"


def test_config_mlflow_experiment_path_shared():
    """Test MLflow experiment path generation for shared location."""
    config = CassandraConfig(mlflow_experiment_location="shared")
    path = config.get_mlflow_experiment_path("test-experiment")
    assert path == "/Shared/cassandra-experiments/test-experiment"


def test_config_mlflow_experiment_path_personal_with_username():
    """Test MLflow experiment path generation for personal location with username."""
    config = CassandraConfig(
        mlflow_experiment_location="personal",
        databricks_username="user@example.com",
    )
    path = config.get_mlflow_experiment_path("test-experiment")
    assert path == "/Users/user@example.com/cassandra-experiments/test-experiment"


def test_config_mlflow_experiment_path_personal_no_username():
    """Test MLflow experiment path generation fails without username."""
    config = CassandraConfig(mlflow_experiment_location="personal")
    with pytest.raises(ValueError, match="Cannot determine Databricks username"):
        config.get_mlflow_experiment_path("test-experiment")


def test_config_mlflow_env_overrides(tmp_path, monkeypatch):
    """Test MLflow environment variable overrides."""
    test_config = tmp_path / "config.yaml"
    monkeypatch.setattr(CassandraConfig, "config_path", classmethod(lambda cls: test_config))

    # Save initial config
    config = CassandraConfig(
        mlflow_tracking_uri="databricks",
        mlflow_experiment_location="personal",
    )
    config.save()

    # Set environment variables
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("CASSANDRA_MLFLOW_LOCATION", "shared")
    monkeypatch.setenv("DATABRICKS_USERNAME", "env-user@example.com")

    # Load config - env vars should override YAML
    loaded = CassandraConfig.load()
    assert loaded.mlflow_tracking_uri == "http://localhost:5000"
    assert loaded.mlflow_experiment_location == "shared"
    assert loaded.databricks_username == "env-user@example.com"
