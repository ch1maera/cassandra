"""Unit tests for Unity Catalog data operations."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datasets import Dataset

from cassandra.config import CassandraConfig
from cassandra.data import (
    UnityCatalogTables,
    UnityCatalogVolumes,
    TableNotFoundError,
    DataLoadError,
    VolumeNotFoundError,
)


@pytest.fixture
def mock_config():
    """Mock configuration with Databricks client."""
    return CassandraConfig(
        databricks_profile="test",
        warehouse_id="test-warehouse",
    )


@pytest.fixture
def mock_client():
    """Mock WorkspaceClient."""
    client = Mock()
    client.config.host = "https://test.databricks.com"
    client.config.token = "test-token"
    return client


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": ["text1", "text2", "text3"],
            "label": [0, 1, 0],
        }
    )


# ============================================================================
# UnityCatalogTables Tests
# ============================================================================


class TestUnityCatalogTablesInit:
    """Tests for UnityCatalogTables initialization."""

    @patch("cassandra.data.tables.CassandraConfig.get_databricks_client")
    def test_init_valid_config(self, mock_get_client, mock_config, mock_client):
        """Test initialization with valid config."""
        mock_get_client.return_value = mock_client

        tables = UnityCatalogTables(mock_config)
        assert tables.config == mock_config
        assert tables.client is not None

    def test_init_invalid_config(self):
        """Test initialization fails with invalid config."""
        config = CassandraConfig()  # No warehouse_id
        with pytest.raises(ValueError, match="Configuration incomplete"):
            UnityCatalogTables(config)


class TestTableNameParsing:
    """Tests for table name parsing."""

    def test_parse_valid_table_name(self):
        """Test parsing valid three-part table name."""
        catalog, schema, table = UnityCatalogTables._parse_table_name(
            "main.default.test"
        )
        assert catalog == "main"
        assert schema == "default"
        assert table == "test"

    def test_parse_invalid_table_name_too_few_parts(self):
        """Test parsing fails with too few parts."""
        with pytest.raises(ValueError, match="Invalid table name"):
            UnityCatalogTables._parse_table_name("invalid.table")

    def test_parse_invalid_table_name_one_part(self):
        """Test parsing fails with single part."""
        with pytest.raises(ValueError, match="Invalid table name"):
            UnityCatalogTables._parse_table_name("invalid")


class TestLoadTable:
    """Tests for table loading."""

    @patch("cassandra.data.tables.CassandraConfig.get_databricks_client")
    @patch("databricks.sql.connect")
    def test_load_table_basic(
        self, mock_sql_connect, mock_get_client, mock_config, mock_client, sample_dataframe
    ):
        """Test basic table loading."""
        mock_get_client.return_value = mock_client

        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("text",), ("label",)]
        mock_cursor.fetchall.return_value = sample_dataframe.values.tolist()

        # Mock connection and cursor context managers
        mock_cursor_context = MagicMock()
        mock_cursor_context.__enter__.return_value = mock_cursor
        mock_cursor_context.__exit__.return_value = None

        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor_context
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None

        mock_sql_connect.return_value = mock_connection

        # Mock table existence check
        mock_client.tables.get.return_value = Mock()

        tables = UnityCatalogTables(mock_config)
        dataset = tables.load_table("main.default.test")

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 3
        assert "text" in dataset.column_names

    @patch("cassandra.data.tables.CassandraConfig.get_databricks_client")
    @patch("databricks.sql.connect")
    def test_load_table_with_columns(
        self, mock_sql_connect, mock_get_client, mock_config, mock_client, sample_dataframe
    ):
        """Test table loading with specific columns."""
        mock_get_client.return_value = mock_client
        subset_df = sample_dataframe[["text", "label"]]

        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.description = [("text",), ("label",)]
        mock_cursor.fetchall.return_value = subset_df.values.tolist()

        # Mock connection and cursor context managers
        mock_cursor_context = MagicMock()
        mock_cursor_context.__enter__.return_value = mock_cursor
        mock_cursor_context.__exit__.return_value = None

        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor_context
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None

        mock_sql_connect.return_value = mock_connection

        mock_client.tables.get.return_value = Mock()

        tables = UnityCatalogTables(mock_config)
        dataset = tables.load_table(
            "main.default.test",
            columns=["text", "label"]
        )

        assert "text" in dataset.column_names
        assert "label" in dataset.column_names

    @patch("cassandra.data.tables.CassandraConfig.get_databricks_client")
    @patch("databricks.sql.connect")
    def test_load_table_with_where(
        self, mock_sql_connect, mock_get_client, mock_config, mock_client, sample_dataframe
    ):
        """Test table loading with WHERE clause."""
        mock_get_client.return_value = mock_client
        filtered_df = sample_dataframe[sample_dataframe["label"] == 1]

        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("text",), ("label",)]
        mock_cursor.fetchall.return_value = filtered_df.values.tolist()

        # Mock connection and cursor context managers
        mock_cursor_context = MagicMock()
        mock_cursor_context.__enter__.return_value = mock_cursor
        mock_cursor_context.__exit__.return_value = None

        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor_context
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None

        mock_sql_connect.return_value = mock_connection

        mock_client.tables.get.return_value = Mock()

        tables = UnityCatalogTables(mock_config)
        dataset = tables.load_table(
            "main.default.test",
            where="label = 1"
        )

        assert len(dataset) == 1

    @patch("cassandra.data.tables.CassandraConfig.get_databricks_client")
    def test_load_table_not_found(self, mock_get_client, mock_config, mock_client):
        """Test loading non-existent table."""
        mock_get_client.return_value = mock_client
        mock_client.tables.get.side_effect = Exception("Table not found")

        tables = UnityCatalogTables(mock_config)
        with pytest.raises(TableNotFoundError):
            tables.load_table("main.default.nonexistent")


class TestPreviewTable:
    """Tests for table preview."""

    @patch("cassandra.data.tables.CassandraConfig.get_databricks_client")
    @patch("databricks.sql.connect")
    def test_preview_table(
        self, mock_sql_connect, mock_get_client, mock_config, mock_client, sample_dataframe
    ):
        """Test table preview."""
        mock_get_client.return_value = mock_client

        # Mock cursor
        mock_cursor = MagicMock()
        mock_cursor.description = [("id",), ("text",), ("label",)]
        mock_cursor.fetchall.return_value = sample_dataframe.values.tolist()

        # Mock connection and cursor context managers
        mock_cursor_context = MagicMock()
        mock_cursor_context.__enter__.return_value = mock_cursor
        mock_cursor_context.__exit__.return_value = None

        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor_context
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None

        mock_sql_connect.return_value = mock_connection

        mock_client.tables.get.return_value = Mock()

        tables = UnityCatalogTables(mock_config)
        df = tables.preview_table("main.default.test", n_rows=5)

        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 5


class TestListTables:
    """Tests for listing tables."""

    @patch("cassandra.data.tables.CassandraConfig.get_databricks_client")
    def test_list_tables(self, mock_get_client, mock_config, mock_client):
        """Test listing tables in schema."""
        mock_get_client.return_value = mock_client

        mock_table1 = Mock()
        mock_table1.name = "table1"
        mock_table1.catalog_name = "main"
        mock_table1.schema_name = "default"
        mock_table1.table_type.value = "MANAGED"

        mock_table2 = Mock()
        mock_table2.name = "table2"
        mock_table2.catalog_name = "main"
        mock_table2.schema_name = "default"
        mock_table2.table_type.value = "EXTERNAL"

        mock_client.tables.list.return_value = [
            mock_table1,
            mock_table2,
        ]

        tables = UnityCatalogTables(mock_config)
        result = tables.list_tables("main", "default")

        assert len(result) == 2
        assert result[0]["name"] == "table1"
        assert result[1]["name"] == "table2"


# ============================================================================
# UnityCatalogVolumes Tests
# ============================================================================


class TestUnityCatalogVolumesInit:
    """Tests for UnityCatalogVolumes initialization."""

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_init_valid_config(self, mock_get_client, mock_config, mock_client):
        """Test initialization with valid config."""
        mock_get_client.return_value = mock_client

        volumes = UnityCatalogVolumes(mock_config)
        assert volumes.config == mock_config
        assert volumes.client is not None

    def test_init_invalid_config(self):
        """Test initialization fails with invalid config."""
        config = CassandraConfig()
        with pytest.raises(ValueError, match="Configuration incomplete"):
            UnityCatalogVolumes(config)


class TestVolumePathParsing:
    """Tests for volume path parsing."""

    def test_parse_volume_path_with_prefix(self):
        """Test parsing volume path with /Volumes/ prefix."""
        result = UnityCatalogVolumes._parse_volume_path(
            "/Volumes/main/default/data/file.json"
        )
        assert result["catalog"] == "main"
        assert result["schema"] == "default"
        assert result["volume"] == "data"
        assert result["path"] == "file.json"

    def test_parse_volume_path_without_prefix(self):
        """Test parsing volume path without prefix."""
        result = UnityCatalogVolumes._parse_volume_path(
            "main/default/data/file.json"
        )
        assert result["catalog"] == "main"
        assert result["schema"] == "default"
        assert result["volume"] == "data"
        assert result["path"] == "file.json"

    def test_parse_volume_path_no_file(self):
        """Test parsing volume path without file."""
        result = UnityCatalogVolumes._parse_volume_path(
            "/Volumes/main/default/data"
        )
        assert result["catalog"] == "main"
        assert result["schema"] == "default"
        assert result["volume"] == "data"
        assert result["path"] == ""

    def test_parse_invalid_volume_path(self):
        """Test parsing invalid volume path."""
        with pytest.raises(ValueError, match="Invalid volume path"):
            UnityCatalogVolumes._parse_volume_path("invalid/path")


class TestDetectFormat:
    """Tests for file format detection."""

    def test_detect_csv(self):
        """Test CSV format detection."""
        assert UnityCatalogVolumes._detect_format("file.csv") == "csv"

    def test_detect_json(self):
        """Test JSON format detection."""
        assert UnityCatalogVolumes._detect_format("file.json") == "json"

    def test_detect_jsonl(self):
        """Test JSONL format detection."""
        assert UnityCatalogVolumes._detect_format("file.jsonl") == "json"

    def test_detect_parquet(self):
        """Test Parquet format detection."""
        assert UnityCatalogVolumes._detect_format("file.parquet") == "parquet"

    def test_detect_text(self):
        """Test text format detection."""
        assert UnityCatalogVolumes._detect_format("file.txt") == "text"

    def test_detect_unknown(self):
        """Test unknown format defaults to text."""
        assert UnityCatalogVolumes._detect_format("file.unknown") == "text"


class TestListFiles:
    """Tests for listing files."""

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_list_files_basic(self, mock_get_client, mock_config, mock_client):
        """Test basic file listing."""
        mock_get_client.return_value = mock_client

        mock_file1 = Mock()
        mock_file1.path = "/Volumes/main/default/data/file1.json"
        mock_file1.is_directory = False
        mock_file1.file_size = 1024
        mock_file1.modification_time = 1234567890

        mock_file2 = Mock()
        mock_file2.path = "/Volumes/main/default/data/file2.json"
        mock_file2.is_directory = False
        mock_file2.file_size = 2048
        mock_file2.modification_time = 1234567891

        mock_client.files.list_directory_contents.return_value = [
            mock_file1,
            mock_file2,
        ]

        volumes = UnityCatalogVolumes(mock_config)
        files = volumes.list_files("/Volumes/main/default/data", pattern="*.json")

        assert len(files) == 2
        assert files[0]["path"] == mock_file1.path
        assert files[0]["size_bytes"] == 1024

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_list_files_with_directory(self, mock_get_client, mock_config, mock_client):
        """Test file listing skips directories."""
        mock_get_client.return_value = mock_client

        mock_file = Mock()
        mock_file.path = "/Volumes/main/default/data/file.json"
        mock_file.is_directory = False
        mock_file.file_size = 1024
        mock_file.modification_time = 1234567890

        mock_dir = Mock()
        mock_dir.path = "/Volumes/main/default/data/subdir"
        mock_dir.is_directory = True

        mock_client.files.list_directory_contents.return_value = [
            mock_file,
            mock_dir,
        ]

        volumes = UnityCatalogVolumes(mock_config)
        files = volumes.list_files(
            "/Volumes/main/default/data",
            pattern="*",
            recursive=False
        )

        # Should only include file, not directory
        assert len(files) == 1
        assert files[0]["path"] == mock_file.path

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_list_files_not_found(self, mock_get_client, mock_config, mock_client):
        """Test listing non-existent volume."""
        mock_get_client.return_value = mock_client
        mock_client.files.list_directory_contents.side_effect = Exception(
            "Volume does not exist"
        )

        volumes = UnityCatalogVolumes(mock_config)
        with pytest.raises(VolumeNotFoundError):
            volumes.list_files("/Volumes/main/default/nonexistent")


class TestLoadFiles:
    """Tests for loading files."""

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_load_files_csv(self, mock_get_client, mock_config, mock_client, sample_dataframe):
        """Test loading CSV files."""
        mock_get_client.return_value = mock_client

        mock_file = Mock()
        mock_file.path = "/Volumes/main/default/data/file.csv"
        mock_file.is_directory = False
        mock_file.file_size = 1024
        mock_file.modification_time = 1234567890

        mock_client.files.list_directory_contents.return_value = [mock_file]

        # Mock file download - return DownloadResponse directly
        mock_response = Mock()
        mock_response.contents.read.return_value = sample_dataframe.to_csv(index=False).encode()
        mock_client.files.download.return_value = mock_response

        volumes = UnityCatalogVolumes(mock_config)
        dataset = volumes.load_files("/Volumes/main/default/data", pattern="*.csv")

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 3

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_load_files_no_matches(self, mock_get_client, mock_config, mock_client):
        """Test loading with no matching files."""
        mock_get_client.return_value = mock_client
        mock_client.files.list_directory_contents.return_value = []

        volumes = UnityCatalogVolumes(mock_config)
        with pytest.raises(DataLoadError, match="No files found"):
            volumes.load_files("/Volumes/main/default/data", pattern="*.json")


class TestPreviewFile:
    """Tests for volume file preview."""

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_preview_file_basic(self, mock_get_client, mock_config, mock_client, sample_dataframe):
        """Test basic file preview."""
        mock_get_client.return_value = mock_client

        # Mock file download - return DownloadResponse directly
        mock_response = Mock()
        mock_response.contents.read.return_value = sample_dataframe.to_csv(index=False).encode()
        mock_client.files.download.return_value = mock_response

        volumes = UnityCatalogVolumes(mock_config)
        df = volumes.preview_file(
            volume_path="/Volumes/main/default/data",
            filename="test.csv",
            n_rows=10,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Sample has 3 rows, all returned since n_rows=10

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_preview_file_with_limit(self, mock_get_client, mock_config, mock_client):
        """Test file preview with row limit applied."""
        mock_get_client.return_value = mock_client

        # Create larger dataframe
        large_df = pd.DataFrame(
            {
                "id": list(range(100)),
                "value": [f"val_{i}" for i in range(100)],
            }
        )

        # Mock file download - return DownloadResponse directly
        mock_response = Mock()
        mock_response.contents.read.return_value = large_df.to_csv(index=False).encode()
        mock_client.files.download.return_value = mock_response

        volumes = UnityCatalogVolumes(mock_config)
        df = volumes.preview_file(
            volume_path="/Volumes/main/default/data",
            filename="large.csv",
            n_rows=5,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # Limited to 5 rows

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_preview_file_format_detection(self, mock_get_client, mock_config, mock_client, sample_dataframe):
        """Test auto-format detection works."""
        mock_get_client.return_value = mock_client

        # Mock file download with JSON - return DownloadResponse directly
        json_data = sample_dataframe.to_json(orient="records", lines=True).encode()
        mock_response = Mock()
        mock_response.contents.read.return_value = json_data
        mock_client.files.download.return_value = mock_response

        volumes = UnityCatalogVolumes(mock_config)
        df = volumes.preview_file(
            volume_path="/Volumes/main/default/data",
            filename="test.json",
            n_rows=10,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_preview_file_not_found(self, mock_get_client, mock_config, mock_client):
        """Test file not found raises VolumeNotFoundError."""
        mock_get_client.return_value = mock_client
        mock_client.files.download.side_effect = Exception("File does not exist")

        volumes = UnityCatalogVolumes(mock_config)
        with pytest.raises(VolumeNotFoundError):
            volumes.preview_file(
                volume_path="/Volumes/main/default/data",
                filename="nonexistent.csv",
                n_rows=10,
            )

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_preview_file_empty(self, mock_get_client, mock_config, mock_client):
        """Test empty file returns empty DataFrame."""
        mock_get_client.return_value = mock_client

        # Mock empty CSV file - return DownloadResponse directly
        empty_csv = b"id,text,label\n"
        mock_response = Mock()
        mock_response.contents.read.return_value = empty_csv
        mock_client.files.download.return_value = mock_response

        volumes = UnityCatalogVolumes(mock_config)
        df = volumes.preview_file(
            volume_path="/Volumes/main/default/data",
            filename="empty.csv",
            n_rows=10,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_preview_file_with_pattern(self, mock_get_client, mock_config, mock_client, sample_dataframe):
        """Test preview with wildcard pattern."""
        mock_get_client.return_value = mock_client

        # Mock list_files to return matching files
        mock_file1 = Mock()
        mock_file1.path = "/Volumes/main/default/data/data1.json"
        mock_file1.is_directory = False
        mock_file1.file_size = 1024

        mock_file2 = Mock()
        mock_file2.path = "/Volumes/main/default/data/data2.json"
        mock_file2.is_directory = False
        mock_file2.file_size = 2048

        mock_client.files.list_directory_contents.return_value = [
            mock_file1,
            mock_file2,
        ]

        # Mock file download for first file - return DownloadResponse directly
        json_data = sample_dataframe.to_json(orient="records", lines=True).encode()
        mock_response = Mock()
        mock_response.contents.read.return_value = json_data
        mock_client.files.download.return_value = mock_response

        volumes = UnityCatalogVolumes(mock_config)
        df = volumes.preview_file(
            volume_path="/Volumes/main/default/data",
            filename="*.json",
            n_rows=10,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Preview first matching file

    @patch("cassandra.data.volumes.CassandraConfig.get_databricks_client")
    def test_preview_file_pattern_no_matches(self, mock_get_client, mock_config, mock_client):
        """Test preview with pattern that matches no files."""
        mock_get_client.return_value = mock_client
        mock_client.files.list_directory_contents.return_value = []

        volumes = UnityCatalogVolumes(mock_config)
        with pytest.raises(DataLoadError, match="No files found"):
            volumes.preview_file(
                volume_path="/Volumes/main/default/data",
                filename="*.csv",
                n_rows=10,
            )
