"""Integration tests for Unity Catalog data operations.

These tests require:
- Databricks workspace access
- Configured profile in ~/.databrickscfg
- SQL warehouse
- Test table: main.default.cassandra_test_sample
- Test volume: /Volumes/main/default/cassandra_test
"""

import pytest
from cassandra.config import CassandraConfig
from cassandra.data import UnityCatalogTables, UnityCatalogVolumes


@pytest.fixture(scope="module")
def config():
    """Load real configuration.

    Skips tests if not configured.
    """
    config = CassandraConfig.load()
    if not config.is_configured():
        pytest.skip("Databricks not configured. Set DATABRICKS_PROFILE and warehouse_id.")
    return config


@pytest.mark.integration
@pytest.mark.databricks
class TestUnityCatalogTablesIntegration:
    """Integration tests for Unity Catalog table operations."""

    def test_load_table_basic(self, config):
        """Test loading a real table.

        Assumes test table exists with at least some data.
        """
        tables = UnityCatalogTables(config)

        # Load with limit
        dataset = tables.load_table(
            "main.default.cassandra_test_sample",
            limit=10,
        )

        # Verify we got a dataset
        assert len(dataset) <= 10
        assert len(dataset.column_names) > 0

    def test_load_table_with_columns(self, config):
        """Test loading specific columns."""
        tables = UnityCatalogTables(config)

        dataset = tables.load_table(
            "main.default.cassandra_test_sample",
            columns=["id", "text"],
            limit=5,
        )

        assert "id" in dataset.column_names
        assert "text" in dataset.column_names
        assert len(dataset) <= 5

    def test_load_table_with_where(self, config):
        """Test loading with WHERE filter."""
        tables = UnityCatalogTables(config)

        dataset = tables.load_table(
            "main.default.cassandra_test_sample",
            where="id > 0",
            limit=10,
        )

        assert len(dataset) <= 10

    def test_preview_table(self, config):
        """Test table preview."""
        tables = UnityCatalogTables(config)

        df = tables.preview_table(
            "main.default.cassandra_test_sample",
            n_rows=5,
        )

        assert len(df) <= 5
        assert len(df.columns) > 0

    def test_list_tables(self, config):
        """Test listing tables in schema."""
        tables = UnityCatalogTables(config)

        tables_list = tables.list_tables("main", "default")

        # Should have at least some tables
        assert len(tables_list) > 0
        assert all("name" in t for t in tables_list)
        assert all("full_name" in t for t in tables_list)

        # Check that our test table is in the list
        table_names = [t["name"] for t in tables_list]
        assert "cassandra_test_sample" in table_names


@pytest.mark.integration
@pytest.mark.databricks
class TestUnityCatalogVolumesIntegration:
    """Integration tests for Unity Catalog volume operations."""

    def test_list_files_basic(self, config):
        """Test listing files in volume."""
        volumes = UnityCatalogVolumes(config)

        files = volumes.list_files(
            "/Volumes/main/default/cassandra_test",
            pattern="*",
        )

        # May be empty, just verify it doesn't error
        assert isinstance(files, list)
        # If there are files, verify structure
        if files:
            assert "path" in files[0]
            assert "size_bytes" in files[0]

    def test_list_files_with_pattern(self, config):
        """Test listing files with glob pattern."""
        volumes = UnityCatalogVolumes(config)

        files = volumes.list_files(
            "/Volumes/main/default/cassandra_test",
            pattern="*.json",
            recursive=True,
        )

        # Verify all returned files match pattern
        for file_info in files:
            assert file_info["path"].endswith(".json")

    def test_preview_volume_file(self, config):
        """Test previewing a volume file.

        Tests with: /Volumes/ishikone_catalog/cholula/test_volume/cassandra.csv
        """
        volumes = UnityCatalogVolumes(config)

        # Preview CSV file from test volume
        df = volumes.preview_file(
            volume_path="/Volumes/ishikone_catalog/cholula/test_volume",
            filename="cassandra.csv",
            n_rows=10,
        )

        # Verify we got a DataFrame
        assert len(df) <= 10
        assert len(df.columns) > 0

    def test_preview_volume_file_with_format(self, config):
        """Test previewing with explicit format."""
        volumes = UnityCatalogVolumes(config)

        # Preview with auto-detection
        df = volumes.preview_file(
            volume_path="/Volumes/ishikone_catalog/cholula/test_volume",
            filename="cassandra.csv",
            n_rows=5,
            file_format="csv",
        )

        assert len(df) <= 5
