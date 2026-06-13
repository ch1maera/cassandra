"""Exception hierarchy for MLflow integration.

This module defines custom exceptions for MLflow operations including
experiment management, logging, and artifact generation.
"""


class CassandraMLflowError(Exception):
    """Base exception for Cassandra MLflow integration."""

    pass


class MLflowExperimentError(CassandraMLflowError):
    """Raised when experiment creation or retrieval fails."""

    pass


class MLflowLoggingError(CassandraMLflowError):
    """Raised when logging parameters, metrics, or models fails."""

    pass


class MLflowArtifactError(CassandraMLflowError):
    """Raised when artifact generation or upload fails."""

    pass
