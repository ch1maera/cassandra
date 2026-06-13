"""Unity Catalog table operations for Cassandra.

This module provides functionality to load and save Unity Catalog tables as
HuggingFace Datasets for transformer model training.
"""

from typing import List, Optional, Tuple, Dict, Any, Literal
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from datasets import Dataset

from cassandra.config import CassandraConfig
from cassandra.data.exceptions import TableNotFoundError, DataLoadError, DataSaveError


class UnityCatalogTables:
    """Unity Catalog table operations.

    Provides methods to load, preview, save, and list Unity Catalog tables,
    converting between Delta tables and HuggingFace Datasets.
    """

    def __init__(self, config: CassandraConfig):
        """Initialize with Cassandra configuration.

        Args:
            config: CassandraConfig instance

        Raises:
            ValueError: If configuration is incomplete (missing profile or warehouse_id)
        """
        if not config.is_configured():
            raise ValueError(
                "Configuration incomplete. Required: databricks_profile and warehouse_id. "
                "Set via config file or environment variables."
            )
        self.config = config
        self.client = config.get_databricks_client()

    def _refresh_token(self, refresh_token: str) -> Optional[str]:
        """Refresh the OAuth token using the refresh token.

        Args:
            refresh_token: The refresh token to use

        Returns:
            New access token if refresh succeeds, None otherwise
        """
        try:
            import requests

            # Get the OAuth token endpoint from the workspace
            host = self.config.databricks_host or self.client.config.host
            if not host.startswith("https://"):
                host = f"https://{host}"

            # Databricks OAuth token endpoint
            token_url = f"{host}/oidc/v1/token"

            # Request a new token using the refresh token
            response = requests.post(
                token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": "databricks-cli",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code == 200:
                token_data = response.json()
                new_access_token = token_data.get("access_token")

                # Update the cached token file
                if new_access_token:
                    self._save_token(token_data)
                    return new_access_token

            return None

        except Exception:
            return None

    def _save_token(self, token_data: dict) -> None:
        """Save token data to the cache file.

        Args:
            token_data: Token data from OAuth response
        """
        try:
            token_file = (
                Path.home()
                / ".databricks"
                / f"databricks-cli_{self.config.databricks_profile}.json"
            )

            # Calculate expiration time
            if "expires_in" in token_data:
                from datetime import timedelta

                expires_at = datetime.now() + timedelta(seconds=token_data["expires_in"])
                token_data["expires_at"] = {
                    "__datetime__": expires_at.isoformat()
                }

            with open(token_file, "w") as f:
                json.dump(token_data, f, indent=4)

        except Exception:
            pass  # Ignore errors saving the token

    def _load_cached_token(self) -> Optional[str]:
        """Load cached OAuth token from Databricks CLI cache.

        Automatically refreshes the token if expired.

        Returns:
            Valid access token, or None if unable to get one
        """
        try:
            # Look for the profile-specific token cache file
            token_file = (
                Path.home()
                / ".databricks"
                / f"databricks-cli_{self.config.databricks_profile}.json"
            )

            if not token_file.exists():
                return None

            with open(token_file, "r") as f:
                token_data = json.load(f)

            # Check if token is expired
            is_expired = False
            if "expires_at" in token_data:
                expires_at_data = token_data["expires_at"]
                if isinstance(expires_at_data, dict) and "__datetime__" in expires_at_data:
                    expires_str = expires_at_data["__datetime__"]
                    # Parse as naive datetime (no timezone)
                    expires_at = datetime.fromisoformat(expires_str)

                    # Add 5 minute buffer before expiration
                    from datetime import timedelta
                    if datetime.now() >= (expires_at - timedelta(minutes=5)):
                        is_expired = True

            # If expired, try to refresh using the refresh token
            if is_expired and "refresh_token" in token_data:
                new_token = self._refresh_token(token_data["refresh_token"])
                if new_token:
                    return new_token
                # If refresh failed, fall through to return the expired token
                # The SQL connector might still accept it

            return token_data.get("access_token")

        except Exception:
            # If anything fails, return None
            return None

    @staticmethod
    def _parse_table_name(table_name: str) -> Tuple[str, str, str]:
        """Parse three-part table name.

        Args:
            table_name: Table name in format "catalog.schema.table"

        Returns:
            Tuple of (catalog, schema, table)

        Raises:
            ValueError: If table name format is invalid
        """
        parts = table_name.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid table name '{table_name}'. "
                "Expected format: 'catalog.schema.table'"
            )
        return parts[0], parts[1], parts[2]

    def _validate_table_exists(self, table_name: str) -> None:
        """Validate that table exists in Unity Catalog.

        Args:
            table_name: Three-part table name

        Raises:
            TableNotFoundError: If table doesn't exist
        """
        try:
            catalog, schema, table = self._parse_table_name(table_name)
            self.client.tables.get(f"{catalog}.{schema}.{table}")
        except Exception as e:
            raise TableNotFoundError(
                f"Table '{table_name}' not found in Unity Catalog"
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query via Databricks SQL connector.

        Args:
            query: SQL query to execute

        Returns:
            Query results as pandas DataFrame

        Raises:
            DataLoadError: If query execution fails
        """
        try:
            from databricks import sql
        except ImportError as e:
            raise ImportError(
                "databricks-sql-connector is required. "
                "Install with: pip install databricks-sql-connector"
            ) from e

        try:
            # Get host without https:// prefix
            host = self.config.databricks_host
            if host is None:
                host = self.client.config.host
            if host.startswith("https://"):
                host = host.replace("https://", "")

            # Load cached token (with automatic refresh if expired)
            token = self._load_cached_token()

            if not token:
                raise DataLoadError(
                    f"No valid OAuth token found for profile '{self.config.databricks_profile}'. "
                    "Please authenticate first by running: "
                    f"databricks auth login --profile {self.config.databricks_profile}"
                )

            # Use the token directly - never trigger OAuth popup
            with sql.connect(
                server_hostname=host,
                http_path=f"/sql/1.0/warehouses/{self.config.warehouse_id}",
                access_token=token,
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    df = pd.DataFrame(rows, columns=columns)
                    return df

        except Exception as e:
            raise DataLoadError(f"Query execution failed: {query[:100]}...") from e

    def load_table(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dataset:
        """Load Unity Catalog table as HuggingFace Dataset.

        Args:
            table_name: Three-part table name (catalog.schema.table)
            columns: Specific columns to load (None = all columns)
            where: SQL WHERE clause without 'WHERE' keyword
            limit: Maximum number of rows to load

        Returns:
            HuggingFace Dataset with table data

        Raises:
            TableNotFoundError: If table doesn't exist
            DataLoadError: If data loading fails

        Examples:
            >>> tables = UnityCatalogTables(config)
            >>> dataset = tables.load_table("main.default.reviews")
            >>> dataset = tables.load_table(
            ...     "main.default.reviews",
            ...     columns=["text", "label"],
            ...     where="rating >= 4",
            ...     limit=1000
            ... )
        """
        # Validate table exists
        self._validate_table_exists(table_name)

        # Build SELECT clause
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"

        # Build query
        query = f"SELECT {select_clause} FROM {table_name}"

        if where:
            query += f" WHERE {where}"

        if limit:
            query += f" LIMIT {limit}"

        # Execute and convert to Dataset
        df = self._execute_query(query)
        dataset = Dataset.from_pandas(df, preserve_index=False)

        return dataset

    def preview_table(
        self,
        table_name: str,
        n_rows: int = 10,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
    ) -> pd.DataFrame:
        """Quick preview of table data.

        Args:
            table_name: Three-part table name
            n_rows: Number of rows to preview
            columns: Specific columns to show
            where_clause: Optional WHERE filter

        Returns:
            pandas DataFrame with preview data

        Raises:
            TableNotFoundError: If table doesn't exist
            DataLoadError: If preview fails
        """
        # Validate table exists
        self._validate_table_exists(table_name)

        # Build query
        if columns:
            select_clause = ", ".join(columns)
        else:
            select_clause = "*"

        query = f"SELECT {select_clause} FROM {table_name}"

        if where_clause:
            query += f" WHERE {where_clause}"

        query += f" LIMIT {n_rows}"

        return self._execute_query(query)

    def save_table(
        self,
        dataset: Dataset,
        table_name: str,
        mode: Literal["overwrite", "append"] = "overwrite",
    ) -> None:
        """Save HuggingFace Dataset to Unity Catalog table.

        Args:
            dataset: HuggingFace Dataset to save
            table_name: Three-part table name
            mode: Save mode - 'overwrite' or 'append'

        Raises:
            DataSaveError: If save operation fails

        Note:
            This method requires Spark connectivity and is primarily
            for use in Databricks notebooks.
        """
        try:
            # Convert Dataset to pandas
            df = dataset.to_pandas()

            # This requires Spark session - typically available in notebooks
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()
            spark_df = spark.createDataFrame(df)

            # Save to table
            spark_df.write.mode(mode).saveAsTable(table_name)

        except Exception as e:
            raise DataSaveError(
                f"Failed to save table '{table_name}'. "
                "Note: This operation requires Spark session (typically in notebooks)."
            ) from e

    def list_tables(
        self,
        catalog: str = "main",
        schema: str = "default",
    ) -> List[Dict[str, str]]:
        """List tables in a Unity Catalog schema.

        Args:
            catalog: Catalog name
            schema: Schema name

        Returns:
            List of dictionaries with table metadata:
            [{'name': 'table1', 'catalog': 'main', 'schema': 'default'}, ...]

        Raises:
            DataLoadError: If listing fails
        """
        try:
            tables = list(self.client.tables.list(catalog_name=catalog, schema_name=schema))

            return [
                {
                    "name": table.name,
                    "catalog": table.catalog_name,
                    "schema": table.schema_name,
                    "full_name": f"{table.catalog_name}.{table.schema_name}.{table.name}",
                    "table_type": table.table_type.value if table.table_type else None,
                }
                for table in tables
            ]
        except Exception as e:
            raise DataLoadError(
                f"Failed to list tables in {catalog}.{schema}"
            ) from e
