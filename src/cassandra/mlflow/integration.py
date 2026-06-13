"""MLflow integration for Cassandra.

This module provides comprehensive MLflow tracking including experiment management,
model training workflows, artifact generation, and metric logging for time series models.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Literal
import pandas as pd
import numpy as np
import yaml

from cassandra.config import CassandraConfig
from cassandra.mlflow.exceptions import (
    MLflowExperimentError,
    MLflowLoggingError,
    MLflowArtifactError,
)


class TimeSeriesGenerator:
    """Generate synthetic time series data for testing and demonstrations.

    Creates realistic time series with trend, seasonality, and noise components.
    """

    def __init__(
        self,
        n_days: int = 730,
        trend_slope: float = 0.05,
        seasonality_amplitude: float = 10.0,
        noise_level: float = 5.0,
        seed: int = 42,
    ):
        """Initialize time series generator.

        Args:
            n_days: Number of days to generate
            trend_slope: Slope of linear trend
            seasonality_amplitude: Amplitude of seasonal component
            noise_level: Standard deviation of noise
            seed: Random seed for reproducibility
        """
        self.n_days = n_days
        self.trend_slope = trend_slope
        self.seasonality_amplitude = seasonality_amplitude
        self.noise_level = noise_level
        self.seed = seed

    def generate(self, start_date: str = "2024-01-01") -> pd.DataFrame:
        """Generate synthetic time series data.

        Args:
            start_date: Start date for time series

        Returns:
            DataFrame with 'ds' (date) and 'y' (value) columns
        """
        np.random.seed(self.seed)

        # Generate dates
        dates = pd.date_range(start=start_date, periods=self.n_days, freq="D")

        # Generate components
        t = np.arange(self.n_days)
        trend = self.trend_slope * t
        seasonality = self.seasonality_amplitude * np.sin(2 * np.pi * t / 365.25)
        noise = np.random.normal(0, self.noise_level, self.n_days)

        # Combine components
        values = 100 + trend + seasonality + noise

        return pd.DataFrame({"ds": dates, "y": values})

    def split_train_test(
        self, data: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets.

        Args:
            data: Full dataset
            test_size: Fraction of data for testing

        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(data) * (1 - test_size))
        train_df = data.iloc[:split_idx].copy()
        test_df = data.iloc[split_idx:].copy()
        return train_df, test_df


class ArtifactGenerator:
    """Generate visualizations and artifacts for MLflow logging.

    Uses seaborn for high-quality time series visualizations.
    """

    def __init__(self):
        """Initialize artifact generator with seaborn styling."""
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_theme(style="whitegrid")
        self.sns = sns
        self.plt = plt

    def create_timeseries_plot(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        predictions: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Create time series plot with train, test, and predictions.

        Args:
            train_df: Training data with 'ds' and 'y' columns
            test_df: Test data with 'ds' and 'y' columns
            predictions: Predictions with 'ds' and 'yhat' columns
            output_path: Path to save plot
        """
        try:
            fig, ax = self.plt.subplots(figsize=(12, 6))

            # Plot training data
            ax.plot(
                train_df["ds"],
                train_df["y"],
                label="Training Data",
                color="steelblue",
                linewidth=1.5,
            )

            # Plot test data
            ax.plot(
                test_df["ds"],
                test_df["y"],
                label="Test Data",
                color="darkgreen",
                linewidth=1.5,
            )

            # Plot predictions
            ax.plot(
                predictions["ds"],
                predictions["yhat"],
                label="Predictions",
                color="coral",
                linestyle="--",
                linewidth=2,
            )

            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.set_title("Time Series Forecast", fontsize=14, fontweight="bold")
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)

            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            self.plt.close(fig)
        except Exception as e:
            raise MLflowArtifactError(f"Failed to create time series plot: {e}") from e

    def create_residuals_plot(
        self,
        test_df: pd.DataFrame,
        predictions: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Create residuals diagnostic plot (2x2 subplots).

        Args:
            test_df: Test data with 'ds' and 'y' columns
            predictions: Predictions with 'ds' and 'yhat' columns
            output_path: Path to save plot
        """
        try:
            from scipy import stats

            # Calculate residuals
            residuals = test_df["y"].values - predictions["yhat"].values

            fig, axes = self.plt.subplots(2, 2, figsize=(12, 8))

            # Subplot 1: Residuals over time
            axes[0, 0].plot(
                predictions["ds"], residuals, color="steelblue", linewidth=1
            )
            axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=1)
            axes[0, 0].set_xlabel("Date")
            axes[0, 0].set_ylabel("Residuals")
            axes[0, 0].set_title("Residuals Over Time")
            axes[0, 0].grid(True, alpha=0.3)

            # Subplot 2: Histogram
            axes[0, 1].hist(residuals, bins=30, color="steelblue", edgecolor="black")
            axes[0, 1].set_xlabel("Residuals")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Residuals Distribution")
            axes[0, 1].grid(True, alpha=0.3)

            # Subplot 3: Q-Q plot
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title("Q-Q Plot")
            axes[1, 0].grid(True, alpha=0.3)

            # Subplot 4: Predicted vs Actual
            axes[1, 1].scatter(
                predictions["yhat"],
                test_df["y"],
                color="steelblue",
                alpha=0.6,
                edgecolor="black",
            )
            # Add diagonal line
            min_val = min(predictions["yhat"].min(), test_df["y"].min())
            max_val = max(predictions["yhat"].max(), test_df["y"].max())
            axes[1, 1].plot(
                [min_val, max_val], [min_val, max_val], "r--", linewidth=2
            )
            axes[1, 1].set_xlabel("Predicted")
            axes[1, 1].set_ylabel("Actual")
            axes[1, 1].set_title("Predicted vs Actual")
            axes[1, 1].grid(True, alpha=0.3)

            self.plt.tight_layout()
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            self.plt.close(fig)
        except Exception as e:
            raise MLflowArtifactError(f"Failed to create residuals plot: {e}") from e

    def create_forecast_plot(
        self,
        test_df: pd.DataFrame,
        forecast: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Create forecast plot with confidence intervals.

        Args:
            test_df: Test data with 'ds' and 'y' columns
            forecast: Forecast with 'ds', 'yhat', 'yhat_lower', 'yhat_upper' columns
            output_path: Path to save plot
        """
        try:
            fig, ax = self.plt.subplots(figsize=(12, 6))

            # Plot actual values
            ax.plot(
                test_df["ds"],
                test_df["y"],
                label="Actual",
                color="darkgreen",
                linewidth=2,
            )

            # Plot forecast
            ax.plot(
                forecast["ds"],
                forecast["yhat"],
                label="Forecast",
                color="coral",
                linestyle="--",
                linewidth=2,
            )

            # Plot confidence intervals if available
            if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
                ax.fill_between(
                    forecast["ds"],
                    forecast["yhat_lower"],
                    forecast["yhat_upper"],
                    color="coral",
                    alpha=0.3,
                    label="Confidence Interval",
                )

            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.set_title("Forecast with Confidence Intervals", fontsize=14, fontweight="bold")
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)

            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            self.plt.close(fig)
        except Exception as e:
            raise MLflowArtifactError(f"Failed to create forecast plot: {e}") from e


class ProphetModel:
    """Wrapper for Facebook Prophet model."""

    def __init__(self):
        """Initialize Prophet model."""
        from prophet import Prophet

        self.model = Prophet()
        self._trained = False

    def train(self, train_df: pd.DataFrame) -> None:
        """Train Prophet model.

        Args:
            train_df: Training data with 'ds' and 'y' columns
        """
        self.model.fit(train_df)
        self._trained = True

    def predict(self, periods: int = None, future_df: pd.DataFrame = None) -> pd.DataFrame:
        """Generate predictions.

        Args:
            periods: Number of periods to forecast (if future_df not provided)
            future_df: Future dataframe with 'ds' column

        Returns:
            DataFrame with forecast including 'yhat', 'yhat_lower', 'yhat_upper'
        """
        if not self._trained:
            raise ValueError("Model must be trained before prediction")

        if future_df is None:
            future_df = self.model.make_future_dataframe(periods=periods)

        return self.model.predict(future_df)


class ARIMAModel:
    """Wrapper for ARIMA model from statsmodels."""

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """Initialize ARIMA model.

        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order
        self.model = None
        self.model_fit = None

    def train(self, train_df: pd.DataFrame) -> None:
        """Train ARIMA model.

        Args:
            train_df: Training data with 'ds' and 'y' columns
        """
        from statsmodels.tsa.arima.model import ARIMA

        self.model = ARIMA(train_df["y"].values, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, steps: int) -> pd.DataFrame:
        """Generate predictions.

        Args:
            steps: Number of steps to forecast

        Returns:
            DataFrame with forecast ('yhat' column)
        """
        if self.model_fit is None:
            raise ValueError("Model must be trained before prediction")

        forecast = self.model_fit.forecast(steps=steps)
        return pd.DataFrame({"yhat": forecast})


class MLflowIntegration:
    """Main MLflow integration orchestrator.

    Manages experiment setup, model training workflows, and artifact logging.
    """

    def __init__(self, config: CassandraConfig):
        """Initialize MLflow integration.

        Args:
            config: Cassandra configuration
        """
        self.config = config
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking URI."""
        import mlflow

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        self.mlflow = mlflow

    def setup_experiment(self, experiment_name: str) -> str:
        """Create or get MLflow experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Experiment ID

        Raises:
            MLflowExperimentError: If experiment creation fails
        """
        try:
            experiment_path = self.config.get_mlflow_experiment_path(experiment_name)
            experiment = self.mlflow.get_experiment_by_name(experiment_path)

            if experiment is None:
                experiment_id = self.mlflow.create_experiment(experiment_path)
            else:
                experiment_id = experiment.experiment_id

            return experiment_id
        except Exception as e:
            raise MLflowExperimentError(
                f"Failed to setup experiment '{experiment_name}': {e}"
            ) from e

    def run_training_workflow(
        self,
        experiment_name: str,
        model_type: Literal["prophet", "arima"] = "prophet",
        n_days: int = 730,
        test_size: float = 0.2,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Run complete training workflow with MLflow tracking.

        Args:
            experiment_name: Name of the experiment
            model_type: Model type ("prophet" or "arima")
            n_days: Number of days to generate
            test_size: Fraction of data for testing
            tags: Custom tags to log

        Returns:
            MLflow run ID

        Raises:
            MLflowLoggingError: If logging fails
        """
        try:
            # Setup experiment
            experiment_id = self.setup_experiment(experiment_name)
            self.mlflow.set_experiment(experiment_id=experiment_id)

            with self.mlflow.start_run() as run:
                # 1. Generate synthetic data
                generator = TimeSeriesGenerator(n_days=n_days)
                data = generator.generate()
                train_df, test_df = generator.split_train_test(data, test_size=test_size)

                # 2. Train model
                if model_type == "prophet":
                    model = ProphetModel()
                    model.train(train_df)
                    # Predict on test set
                    future_df = test_df[["ds"]].copy()
                    predictions = model.predict(future_df=future_df)
                    predictions = predictions[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                elif model_type == "arima":
                    model = ARIMAModel(order=(1, 1, 1))
                    model.train(train_df)
                    # Predict on test set
                    predictions = model.predict(steps=len(test_df))
                    predictions["ds"] = test_df["ds"].values
                    # ARIMA doesn't provide confidence intervals by default
                    predictions["yhat_lower"] = predictions["yhat"] * 0.95
                    predictions["yhat_upper"] = predictions["yhat"] * 1.05
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                # 3. Log tags
                default_tags = {
                    "framework": "cassandra",
                    "model_type": model_type,
                    "environment": "test",
                }
                if tags:
                    default_tags.update(tags)
                self.mlflow.set_tags(default_tags)

                # 4. Log parameters
                params = {
                    "n_days": n_days,
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "model_type": model_type,
                }
                if model_type == "arima":
                    params["arima_order"] = str(model.order)
                self.mlflow.log_params(params)

                # 5. Calculate and log metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error

                mae = mean_absolute_error(test_df["y"], predictions["yhat"])
                rmse = np.sqrt(mean_squared_error(test_df["y"], predictions["yhat"]))
                mape = np.mean(
                    np.abs((test_df["y"] - predictions["yhat"]) / test_df["y"])
                ) * 100

                metrics = {"mae": mae, "rmse": rmse, "mape": mape}
                self.mlflow.log_metrics(metrics)

                # 6. Generate and log artifacts
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)

                    # Create visualizations
                    artifact_gen = ArtifactGenerator()

                    # Time series plot
                    ts_plot_path = tmpdir_path / "timeseries_plot.png"
                    artifact_gen.create_timeseries_plot(
                        train_df, test_df, predictions, ts_plot_path
                    )
                    self.mlflow.log_artifact(str(ts_plot_path), artifact_path="plots")

                    # Residuals plot
                    residuals_plot_path = tmpdir_path / "residuals_plot.png"
                    artifact_gen.create_residuals_plot(
                        test_df, predictions, residuals_plot_path
                    )
                    self.mlflow.log_artifact(str(residuals_plot_path), artifact_path="plots")

                    # Forecast plot
                    forecast_plot_path = tmpdir_path / "forecast_plot.png"
                    artifact_gen.create_forecast_plot(
                        test_df, predictions, forecast_plot_path
                    )
                    self.mlflow.log_artifact(str(forecast_plot_path), artifact_path="plots")

                    # Save data artifacts
                    sample_data_path = tmpdir_path / "sample_data.csv"
                    data.to_csv(sample_data_path, index=False)
                    self.mlflow.log_artifact(str(sample_data_path), artifact_path="data")

                    predictions_path = tmpdir_path / "predictions.csv"
                    predictions.to_csv(predictions_path, index=False)
                    self.mlflow.log_artifact(
                        str(predictions_path), artifact_path="predictions"
                    )

                    # Save config
                    config_path = tmpdir_path / "config.yaml"
                    config_dict = {
                        "model_type": model_type,
                        "n_days": n_days,
                        "test_size": test_size,
                        "experiment_name": experiment_name,
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(config_path, "w") as f:
                        yaml.safe_dump(config_dict, f)
                    self.mlflow.log_artifact(str(config_path))

                # 7. Log model
                if model_type == "prophet":
                    self.mlflow.prophet.log_model(model.model, artifact_path="model")
                elif model_type == "arima":
                    self.mlflow.statsmodels.log_model(
                        model.model_fit, artifact_path="model"
                    )

                return run.info.run_id

        except Exception as e:
            raise MLflowLoggingError(f"Training workflow failed: {e}") from e
