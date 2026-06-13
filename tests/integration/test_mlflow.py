"""Integration tests for MLflow integration.

These tests verify the complete MLflow workflow including experiment creation,
model training, artifact generation, and metric logging.
"""

import pytest
import pandas as pd
from cassandra.mlflow import (
    TimeSeriesGenerator,
    ArtifactGenerator,
    ProphetModel,
    ARIMAModel,
    MLflowIntegration,
)
from cassandra.config import CassandraConfig


# Unit tests (no external dependencies)


def test_timeseries_generator_defaults():
    """Test time series data generation with default parameters."""
    generator = TimeSeriesGenerator(n_days=100)
    data = generator.generate()

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100
    assert "ds" in data.columns
    assert "y" in data.columns
    assert data["ds"].dtype == "datetime64[ns]"
    assert data["y"].dtype == "float64"


def test_timeseries_generator_custom_params():
    """Test time series generation with custom parameters."""
    generator = TimeSeriesGenerator(
        n_days=50,
        trend_slope=0.1,
        seasonality_amplitude=20.0,
        noise_level=2.0,
        seed=123,
    )
    data = generator.generate(start_date="2025-01-01")

    assert len(data) == 50
    assert data["ds"].min() == pd.Timestamp("2025-01-01")
    assert data["ds"].max() == pd.Timestamp("2025-02-19")


def test_timeseries_split():
    """Test train/test splitting."""
    generator = TimeSeriesGenerator(n_days=100)
    data = generator.generate()
    train_df, test_df = generator.split_train_test(data, test_size=0.2)

    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df["ds"].max() < test_df["ds"].min()


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


# Integration tests (require optional dependencies)


@pytest.mark.integration
def test_prophet_model_training():
    """Test Prophet model training workflow."""
    pytest.importorskip("prophet")

    # Generate data
    generator = TimeSeriesGenerator(n_days=100)
    data = generator.generate()
    train_df, test_df = generator.split_train_test(data, test_size=0.2)

    # Train model
    model = ProphetModel()
    model.train(train_df)

    # Generate predictions
    future_df = test_df[["ds"]].copy()
    predictions = model.predict(future_df=future_df)

    assert "yhat" in predictions.columns
    assert "yhat_lower" in predictions.columns
    assert "yhat_upper" in predictions.columns
    assert len(predictions) == len(test_df)


@pytest.mark.integration
def test_arima_model_training():
    """Test ARIMA model training workflow."""
    pytest.importorskip("statsmodels")

    # Generate data
    generator = TimeSeriesGenerator(n_days=100)
    data = generator.generate()
    train_df, test_df = generator.split_train_test(data, test_size=0.2)

    # Train model
    model = ARIMAModel(order=(1, 1, 1))
    model.train(train_df)

    # Generate predictions
    predictions = model.predict(steps=len(test_df))

    assert "yhat" in predictions.columns
    assert len(predictions) == len(test_df)


@pytest.mark.integration
def test_artifact_generation(tmp_path):
    """Test artifact generation (plots)."""
    pytest.importorskip("seaborn")
    pytest.importorskip("matplotlib")

    # Generate data
    generator = TimeSeriesGenerator(n_days=100)
    data = generator.generate()
    train_df, test_df = generator.split_train_test(data, test_size=0.2)

    # Create fake predictions
    predictions = test_df[["ds"]].copy()
    predictions["yhat"] = test_df["y"] * 1.05
    predictions["yhat_lower"] = test_df["y"] * 0.95
    predictions["yhat_upper"] = test_df["y"] * 1.15

    # Generate artifacts
    artifact_gen = ArtifactGenerator()

    # Test time series plot
    ts_plot_path = tmp_path / "timeseries.png"
    artifact_gen.create_timeseries_plot(train_df, test_df, predictions, ts_plot_path)
    assert ts_plot_path.exists()
    assert ts_plot_path.stat().st_size > 0

    # Test residuals plot
    residuals_plot_path = tmp_path / "residuals.png"
    artifact_gen.create_residuals_plot(test_df, predictions, residuals_plot_path)
    assert residuals_plot_path.exists()
    assert residuals_plot_path.stat().st_size > 0

    # Test forecast plot
    forecast_plot_path = tmp_path / "forecast.png"
    artifact_gen.create_forecast_plot(test_df, predictions, forecast_plot_path)
    assert forecast_plot_path.exists()
    assert forecast_plot_path.stat().st_size > 0


@pytest.mark.integration
@pytest.mark.databricks
def test_mlflow_integration_complete_workflow():
    """Test complete MLflow integration workflow.

    This test requires:
    - Databricks workspace access
    - MLflow tracking configured
    - Prophet installed
    """
    pytest.importorskip("prophet")
    pytest.importorskip("mlflow")

    # Load config
    config = CassandraConfig.load()

    # Override for testing
    config.mlflow_experiment_location = "personal"
    config.databricks_username = "test-user@example.com"

    # Initialize integration
    integration = MLflowIntegration(config)

    # Run workflow
    run_id = integration.run_training_workflow(
        experiment_name="test-mlflow-integration",
        model_type="prophet",
        n_days=100,
        test_size=0.2,
        tags={"test": "true", "framework": "cassandra"},
    )

    # Verify run
    assert run_id is not None
    assert len(run_id) > 0

    # Get run and verify contents
    run = integration.mlflow.get_run(run_id)

    # Check tags
    assert run.data.tags.get("framework") == "cassandra"
    assert run.data.tags.get("test") == "true"
    assert run.data.tags.get("model_type") == "prophet"

    # Check params
    assert "n_days" in run.data.params
    assert run.data.params["n_days"] == "100"
    assert run.data.params["model_type"] == "prophet"

    # Check metrics
    assert "mae" in run.data.metrics
    assert "rmse" in run.data.metrics
    assert "mape" in run.data.metrics

    # Verify metrics are reasonable (not NaN, not infinite)
    assert run.data.metrics["mae"] > 0
    assert run.data.metrics["rmse"] > 0
    assert run.data.metrics["mape"] > 0
    assert run.data.metrics["mape"] < 100  # MAPE should be < 100%
