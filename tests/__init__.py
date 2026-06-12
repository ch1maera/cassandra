"""Test suite for Cassandra AutoML CLI.

This package contains unit tests, integration tests, and end-to-end tests for the Cassandra
AutoML CLI tool. Tests are organized by type and marked with pytest markers for selective execution.

Test Organization:
    - tests/unit/: Fast unit tests with mocked dependencies
    - tests/integration/: Integration tests requiring external services (Databricks, Lakebase)

Pytest Markers:
    - @pytest.mark.unit: Fast unit tests (default)
    - @pytest.mark.slow: Tests taking >5s
    - @pytest.mark.integration: Tests requiring external services
    - @pytest.mark.databricks: Tests requiring Databricks workspace

Example:
    Run all tests::

        $ pytest

    Run only unit tests::

        $ pytest tests/unit/

    Run only integration tests::

        $ pytest -m integration

    Skip slow tests::

        $ pytest -m "not slow"

    Run with coverage::

        $ pytest --cov=cassandra --cov-report=html
"""
